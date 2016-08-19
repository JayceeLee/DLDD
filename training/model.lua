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
paths.dofile('criterions/GlobalCriterionTriplet.lua')
paths.dofile('criterions/CenterCriterion.lua')
paths.dofile('criterions/ContrastiveCriterion.lua')
paths.dofile('criterions/EmptyCriterion.lua')
paths.dofile('criterions/ParallelCriterionS.lua')

paths.dofile('middleBlock/TripletSampling.lua')
paths.dofile('middleBlock/PairSampling.lua')

local M = {}

--define a model with embeding layer
function M.modelSetup(continue)

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
  pairSampling = nn.Identity()
  if config.ConstrastiveLoss then
    pairSampling = nn.PairSampling(opt, config.PairSampling)
  end
  middleBlock:add(pairSampling) -- pairs
  -- Triplet module
  tripletSampling = nn.Identity()
  if config.TripletLoss then
    tripletSampling = nn.TripletSampling(opt,opt.TripletSampling)
  end
  middleBlock:add(nn.Sequential():add(nn.Normalize(2)):add(tripletSampling))
  
  if opt.cudnn then
    cudnn.convert(middleBlock,cudnn)
  end
  
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
  pairCriterion = nn.EmptyCriterion()
  if config.ConstrastiveLoss then
   pairCriterion = nn.ContrastiveCriterion(config.ConstrastiveLossMargin)
  end
  criterionsBlock:add(pairCriterion, config.ConstrastiveLossWeight)
  -- Triplet module
  tripletCriterion = nn.EmptyCriterion()
  if config.TripletLoss then
   tripletCriterion = nn.TripletEmbeddingCriterion(opt.alpha)
  end
  criterionsBlock:add(tripletCriterion, config.TripletLossWeight)
  return criterionsBlock:float()
end  

return M


