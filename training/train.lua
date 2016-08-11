require 'optim'
local models = require 'model'

local optimMethod = _G.optim[opt.optimization]
local optimState = {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
local optimator = nil 

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
if opt.nClasses ~= 0 then
  trainLoggerAcc = optim.Logger(paths.concat(opt.save, 'trainSoft.log'))
end

local batchNumber
local triplet_loss
local all_time
local M = {}

function M:train(dataloader)
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   all_time = 0
   batchNumber = 0
   -- init main model
   model = models.modelSetup(model)
   if opt.nClasses ~= 0 then
        cm = optim.ConfusionMatrix(opt.nClasses, torch.range(1,opt.nClasses))
   end
   optimState.learningRate = self:learningRate(epoch)
   -- init model with additional module
   model_middle = nn.Sequential()
   model_middle:add(model):add(middleBlock)
   model_middle:cuda()
   self:initOptim(model_middle, optimState)

   cutorch.synchronize()
   model_middle:training()

   local tm = torch.Timer()
   triplet_loss = 0
   

   for n, sample in dataloader:run() do
      self:copyInputs(sample)
      self:trainBatch(self.input, self.target, self.info)
   end

   cutorch.synchronize()
   triplet_loss = triplet_loss / batchNumber

   trainLogger:add{
      ['avg triplet loss (train set)'] = triplet_loss,
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average triplet loss (per batch): %.2f\t' 
			  .. 'LR:: %.5f',
                       epoch, tm:time().real, triplet_loss, optimState.learningRate))
   if opt.nClasses ~= 0 then
    cm:updateValids()
    print ("Accuracy : " .. cm.totalValid * 100)
    trainLoggerAcc:add{
        ['avg mean acc (train set)'] = cm.totalValid * 100,
      }
   end
   print('\n')
   print('Time elapsed for All: ' .. all_time .. ' seconds')
   collectgarbage()
end -- of train()

function M:initOptim(model, optState)
    assert(model)
    assert(optState)

    self.model = model
    -- Keep this around so we update it in setParameters
    self.originalOptState = optState
    self.params, self.gradParams = self.model:getParameters()
end

function M:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = sample.target
   self.info   = sample.info

   self.input:resize(sample.input:size()):copy(sample.input)
end

function M:saveModel(model)
   -- Check for nans from https://github.com/cmusatyalab/openface/issues/127
   local function checkNans(x, tag)
      local I = torch.ne(x,x)
      if torch.any(I) then
         print("train.lua: Error: NaNs found in: ", tag)
         os.exit(-1)
         -- x[I] = 0.0
      end
   end

   for j, mod in ipairs(model:listModules()) do
      if torch.typename(mod) == 'nn.SpatialBatchNormalization' then
         checkNans(mod.running_mean, string.format("%d-%s-%s", j, mod, 'running_mean'))
         checkNans(mod.running_var, string.format("%d-%s-%s", j, mod, 'running_var'))
      end
   end

   if opt.cudnn then
      cudnn.convert(model, nn)
   end
 
   local dpt
   if torch.type(model) == 'nn.DataParallelTable' then
      dpt   = model
      model = model:get(1)        
   end    
   
   local optnet_loaded, optnet = pcall(require,'optnet')
   if optnet_loaded then
    optnet.removeOptimization(model)
   end
   
   torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'),  model:clearState())
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
  
   if dpt then -- OOM without this
      dpt:clearState()
   end

   collectgarbage()
 
   return model
end


local timer = torch.Timer()
function M:trainBatch(inputs, target, info)
  timer:reset()
  collectgarbage()
  
  if opt.cuda then
    cutorch.synchronize()
  end
  
  local numImages = inputs:size(1)
  local embeddings = self.model:forward(inputs)
  self:toFloat(embeddings)
  if opt.nClasses ~= 0 then
    cm:batchAdd(embeddings[2], target)
  end
  
  self.model:zeroGradParameters()
  local err     = criterion:forward(embeddings,target)
  local df_do   = criterion:backward(embeddings, target)
  self:toCuda(df_do)
  self.model:backward(inputs, df_do)
  self:toFloat(df_do)
  
  local curGrad
  local curParam
  local function fEvalMod()
      return err, self.gradParams
  end
  optimMethod(fEvalMod, self.params, self.originalOptState)

  batchNumber = batchNumber + 1
  print(('Epoch: [%d][%d/%d]\tTime %.3f\ttripErr %.2e'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err[2]))
  all_time = all_time + timer:time().real
  timer:reset()
  triplet_loss = triplet_loss + err[2]
end

function M:toFloat(data)
  for key,value in pairs(data) do
    data[key] = value:float()
  end
end

function M:toCuda(data)
  for key,value in pairs(data) do
    data[key] = value:cuda()
  end
end

function M:learningRate(epoch)
   -- Training schedule
   local   decay = math.floor((epoch - 1) / opt.LRDecay)
   return opt.LR * math.pow(0.1, decay)
end

return M