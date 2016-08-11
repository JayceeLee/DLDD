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
paths.dofile('criterions/EmptyCriterion.lua')


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
-- local MiddleBlock, parent = torch.class('nn.MiddleBlock', 'nn.Module')
-- function MiddleBlock:__init(blocks, opt)
--   self.blocks = blocks
--   self.gradInput = {}
--   
-- end
-- 
-- function MiddleBlock:updateOutput(input)
--   self.output = {}
--   for key,block in pairs(self.blocks) do
--     self.output[key] = block:forward(input)
--   end
--   return self.output
-- end
-- 
-- function MiddleBlock:updateGradInput(input, gradOutput)
--   self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
--   nn.utils.recursiveFill(self.gradInput, 0)
--   for key,block in ipairs(self.blocks) do
--     nn.utils.recursiveAdd(self.gradInput, 1.0, block:backward(input, gradOutput[i]))
--   end
--   return self.gradInput
-- end
-- 
-- function MiddleBlock:accGradParameters(input, gradOutput, scale) end
-- 
-- function MiddleBlock:type(type, tensorCache)
--    self.gradInput = {}
--    return parent.type(self, type, tensorCache)
-- end
  
  
  
function M.middleBlockSetup(opt)
--   local middleBlockData = {}
--   middleBlockData[1] = nn.Identity()
--   middleBlockData[2] = nn.Seqential():add(nn.ReLU(true)):add(nn.Linear(opt.embSize,opt.nClasses))
--   middleBlockData[3] = nn.Identity() --nn.Seqential():add(nn.Normalization(2)):add(nn.PairSampling())
--   middleBlockData[3] = nn.Identity() --nn.Seqential():add(nn.Normalization(2)):add(nn.TripletSampling())
--   middleBlock = nn.MiddleBlock(middleBlockData, opt)
  local middleBlock = nn.ConcatTable()
  middleBlock:add(nn.Identity())
  classificationBlock = nn.Sequential():add(nn.ReLU(true)):add(nn.Linear(opt.embSize,opt.nClasses))
  middleBlock:add(classificationBlock)
  middleBlock:add(nn.Identity())
  middleBlock:add(nn.Identity())
  return middleBlock:float()
end

-- define all criterions
function M.critertionSetup(opt)
  local criterionsBlock = nn.ParallelCriterionS(true)
  criterionsBlock:add(nn.EmptyCriterion(), 1.0)
  classificationCriterion = nn.CrossEntropyCriterion()
  criterionsBlock:add(classificationCriterion, 1.0)
  criterionsBlock:add(nn.EmptyCriterion(), 1.0)
  criterionsBlock:add(nn.EmptyCriterion(), 1.0)
  return criterionsBlock:float()
end  

return M


