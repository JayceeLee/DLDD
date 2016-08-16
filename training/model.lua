require 'nn'
require 'dpnn'
require 'optim'
require 'cunn'
local utils = require 'utils'

if opt.cudnn then
  require 'cudnn'
  cudnn.benchmark = opt.cudnn_bench
  cudnn.fastest = true
  cudnn.verbose = false
end
paths.dofile('criterions/TripletEmbedding.lua')
paths.dofile('criterions/ParallelCriterionS.lua')
paths.dofile('criterions/CenterCriterion.lua')
paths.dofile('criterions/EmptyCriterion.lua')

paths.dofile('middleBlock/TripletSampling.lua')

local M = {}

--define a model with embeding layer
function M.modelSetup(continue)
  if opt.retrain ~= 'none' then
    assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
    print('Loading model from file: ' .. opt.retrain);
    model = torch.load(opt.retrain)
    print("Using imgDim = ", opt.imgDim)
  elseif continue then
    model = continue
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

function M.middleBlockSetup(opt)
  local middleBlock = nn.ConcatTable()
  middleBlock:add(nn.Identity()) -- raw embeding
  -- Classification module
  classificationBlock = nn.Identity()
  if config.SoftMaxLoss then
    classificationBlock = nn.Sequential():add(nn.ReLU(true)):add(nn.Linear(opt.embSize,opt.nClasses))
  end
  middleBlock:add(classificationBlock)
  -- Pair module
  middleBlock:add(nn.Identity()) -- pairs
  -- Triplet module
  tripletSampling = nn.Identity()
  if config.TripletLoss then
    tripletSampling = nn.TripletSampling(opt,opt.TripletSampling)
  end
  middleBlock:add(nn.Sequential():add(nn.Normalize(2)):add(tripletSampling))

  return middleBlock:float()
end

-- define all criterions
function M.critertionSetup(opt)
  local criterionsBlock = nn.ParallelCriterionS(true)
  -- Center Loss
  centerLoss = nn.EmptyCriterion()
  if config.CenterLoss then
    centerLoss  = nn.CenterCriterion(clusterCenters)
  end
  criterionsBlock:add(centerLoss, config.CenterLossWeight)
  -- Classification module
  classificationCriterion = nn.EmptyCriterion()
  if config.SoftMaxLoss then
   classificationCriterion = nn.CrossEntropyCriterion()
  end
  criterionsBlock:add(classificationCriterion, config.SoftMaxLossWeight)
  -- Pair module
  criterionsBlock:add(nn.EmptyCriterion(), 1.0)
  -- Triplet module
  tripletCriterion = nn.EmptyCriterion()
  if config.TripletLoss then
   tripletCriterion = nn.TripletEmbeddingCriterion(opt.alpha)
  end
  criterionsBlock:add(tripletCriterion, config.TripletLossWeight)
  return criterionsBlock:float()
end  

return M


