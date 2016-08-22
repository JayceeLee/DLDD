require 'nn'
require 'dpnn'
require 'optim'
require 'cunn'
local utils = require 'utils'
local class = require 'class'

if opt.cudnn then
  require 'cudnn'
  cudnn.benchmark = opt.cudnn_bench
  cudnn.fastest = true
  cudnn.verbose = false
end
paths.dofile('criterions/TripletEmbedding.lua')
paths.dofile('criterions/TripletEmbeddingRatioCriterion.lua')
paths.dofile('criterions/GlobalCriterionTriplet.lua')
paths.dofile('criterions/CenterCriterion.lua')
paths.dofile('criterions/ContrastiveCriterion.lua')
paths.dofile('criterions/EmptyCriterion.lua')
paths.dofile('criterions/ParallelCriterionS.lua')

paths.dofile('middleBlock/TripletSampling.lua')
paths.dofile('middleBlock/PairSampling.lua')

-- Check if any class have been turn of 
local function ifAny(data)
  for name,value in pairs(data) do
    if value.weight > 0.0 then return true end
  end
  return false
end

-- Check if any class have been turn of 
local function countCriterion(data)
  local num = 0
  for name,value in pairs(data) do
    if value.weight > 0.0 then num = num + 1 end
  end
  return num
end

local CriterionConfig = class('CriterionConfig') -- the table representing the class, which will double as the metatable for the instances

function CriterionConfig:__init(name, func, weight, params)
  self.name   = name
  self.weight = weight
  self.params = params and params or {}
  self.func   = func
  self.loss   = 0.0
  self.gradient = 0.0
  return self
end

-- Create a loger file and criterion function
function CriterionConfig:create()
  self.trainLogger = optim.Logger(paths.concat(opt.save, self.name .. '.log'))
  self.model       = self.func(unpack(self.params))
  return self.model
end

function CriterionConfig:addErrGrad()
  self.loss     = self.loss + self.model.output
  self.gradient = self.gradient + self.model.gradInput:abs():sum()
  print(('%s : %.2e'):format(self.name, self.model.output))
  return self.model
end

function CriterionConfig:zero()
  self.loss   = 0.0
  self.gradient = 0.0
  return self.model
end

function CriterionConfig:logData(batchNumber)
   self.trainLogger:add{
        ['avg ' ..  self.name .. ' (train set)'] = self.loss / batchNumber,
        ['avg gradient from ' .. self.name .. ' (train set)'] = self.gradient / batchNumber,
     }
   print(string.format('Average ' .. self.name .. '  loss (per batch): %.2f', self.loss)) 
end




local M = {}

local ModelConfig = torch.class('ModelConfig', M)

function ModelConfig:generateConfig(opt)
  config = {
    Raw = {
     Center = CriterionConfig('Center', nn.CenterCriterion, opt.Center, {clusterCenters}),
    },
    Classification = {
     SoftMax = CriterionConfig('SoftMax', nn.CrossEntropyCriterion, opt.SoftMax),
    },
    Pair = {
     Constrastive =  CriterionConfig('Constrastive', nn.ContrastiveCriterion, opt.Constr, {margin='auto'}),
    },
    Triplets = {
     Triplet       = CriterionConfig('Triplet', nn.TripletEmbeddingCriterion, opt.Triplet, {alpha=opt.alpha}),
     GlobalTriplet = CriterionConfig('GlobalTriplet', nn.GlobalCriterionTriplet, opt.Global, {alpha=0.4, lambda=0.8}),
     TripletRatio  = CriterionConfig('TripletRatio', nn.TripletEmbeddingRatioCriterion, opt.Triratio, {alpha=0.01}),
    },
    PairSampling  = 'random',
    TripletSampling = 'semi-hard' --'random'
  }
  
end

--define a model with embeding layer
function ModelConfig:modelSetup(continue)

  if continue then
    model = continue
  elseif opt.retrain ~= 'none' then
    assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
    print('Loading model from file: ' .. opt.retrain);
    model = torch.load(opt.retrain)
    print("Using imgDim = ", opt.imgDim)
  else
    paths.dofile(opt.modelDef)
    assert(imgDim, "Model definition must set global variable 'imgDim'")
    assert(imgDim == opt.imgDim, "Model definiton's imgDim must match imgDim option.")
    model = createModel()
    utils.initWeight(model)
    utils.FCinit(model)
  end
  
  -- First remove any DataParallelTable
  if torch.type(model) == 'nn.DataParallelTable' then
    model = model:get(1)
  end
  model = model:cuda()
  
  utils.optimizeNet(model, opt.imgDim)
  
  if opt.cudnn then
    cudnn.convert(model,cudnn)
  end

  model = utils.makeDataParallel(model, opt.nGPU)
  collectgarbage()
  return model
end

-- define any sampling, normaliation or learning classification layer. 
-- [1] = raw embeding [2] = output from classification [3] = pairs [4] = triplets

function ModelConfig:middleBlockSetup(opt)
  local middleBlock = nn.ConcatTable()
  middleBlock:add(nn.Identity()) -- raw embeding
  -- Classification module
  classificationBlock = nn.Identity()
  if ifAny(config.Classification) then
    classificationBlock = nn.Sequential():add(nn.ReLU(true)):add(nn.Linear(opt.embSize,opt.nClasses))
  end
  middleBlock:add(classificationBlock)
  -- Pair module
  pairSampling = nn.Identity()
  if ifAny(config.Pair) then
    pairSampling = nn.PairSampling(opt, config.PairSampling)
  end
  middleBlock:add(pairSampling) -- pairs
  -- Triplet module
  tripletSampling = nn.Identity()
  if ifAny(config.Triplets) then
    tripletSampling = nn.TripletSampling(opt,opt.TripletSampling)
  end
  middleBlock:add(nn.Sequential():add(nn.Normalize(2)):add(tripletSampling))
  
  if opt.cudnn then
    cudnn.convert(middleBlock,cudnn)
  end
  
  return middleBlock:float()
end

-- Setup all Losses for single branch (like Raw, Triplets)
function ModelConfig:setupSingleLoss(branch, criterionsBlock, listActiveCritetion)
   -- Center Loss
   local criterions = nn.EmptyCriterion()
   local weightLoss = 1.0
   if ifAny(branch) then
      if countCriterion(branch) > 1 then
         criterions = nn.ParallelCriterion(true)
         for name,value in pairs(branch) do
            if value.weight > 0.0 then
             criterions:add(value:create(), value.weight)
             table.insert(listActiveCriterion, value)
            end
         end
      else
         for name,value in pairs(branch) do
           if value.weight > 0.0 then
             criterions = value:create()
             weightLoss = value.weight
             table.insert(listActiveCriterion, value)
             break
           end
         end
      end
   end
   criterionsBlock:add(criterions, weightLoss)
end

-- define all criterions
function ModelConfig:critertionSetup()
   local criterionsBlock = nn.ParallelCriterion(true)
   listActiveCriterion = {}
   -- Raw Loss
   self:setupSingleLoss(config.Raw, criterionsBlock, listActiveCriterion)
   -- Classification Loss
   self:setupSingleLoss(config.Classification, criterionsBlock, listActiveCriterion)
   -- Pair Loss
   self:setupSingleLoss(config.Pair, criterionsBlock, listActiveCriterion)
   -- Triplet Loss
   self:setupSingleLoss(config.Triplets, criterionsBlock, listActiveCriterion)
   return criterionsBlock:float()
end  

return M


