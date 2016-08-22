require 'optim'

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

if opt.SoftMax ~= 0 then
   trainLoggerAcc = optim.Logger(paths.concat(opt.save, 'AccuracyTrain.log'))
end

local batchNumber
local all_time
local M = {}

function M:train(dataloader, models)
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   all_time = 0
   batchNumber = 0
   -- init main model
   model = models:modelSetup(model)
   if trainLoggerAcc then
        cm = optim.ConfusionMatrix(opt.nClasses, torch.range(1,opt.nClasses))
   end
   optimState.learningRate = self:learningRate(epoch)
   -- init model with additional module
   local model_middle = nn.Sequential()
   model_middle:add(model):add(middleBlock)
   model_middle:cuda()
   self:initOptim(model_middle, optimState)
    
   cutorch.synchronize()
   model_middle:training()
   local tm = torch.Timer()
   
   for n, sample in dataloader:run() do
      self:copyInputs(sample)
      self:trainBatch(self.input, self.target, self.info)
      dataloader.clusterCenters = clusterCenters -- update cluster Centers
      if opt.Center > 0.0 then -- update cluster center in loss function
         config.Raw.Center.model.clusterCenters = clusterCenters
      end
   end

   cutorch.synchronize()
   
   for name, value in pairs(listActiveCriterion) do
      value:logData(batchNumber)
   end

   if trainLoggerAcc then
      cm:updateValids()
      print(string.format('Accuracy: %.2f', cm.totalValid * 100))
      trainLoggerAcc:add{
            ['avg mAP  (train set)'] = cm.totalValid * 100
         }
   end

   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\tLR:: %.5f',
                       epoch, tm:time().real, optimState.learningRate))
   print('\n')
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


local timer = torch.Timer()
function M:trainBatch(inputs, target, info)
  timer:reset()
  collectgarbage()
  
  if opt.cuda then
    cutorch.synchronize()
  end
  
  local numImages = inputs:size(1)

  tripletSampling.numPerClass = info.nSamplesPerClass 
  pairSampling.numPerClass    = info.nSamplesPerClass
  local embeddings            = self.model:forward(inputs)
  self:toFloat(embeddings)
  
  self.model:zeroGradParameters()
  local err     = criterion:forward(embeddings, target)
  print("err: " .. err)
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

  self:logBatch(embeddings, target)
  self:updateClusters(embeddings, target, info)

  all_time = all_time + timer:time().real
  timer:reset()

end

function M:toFloat(data)
  for key,value in pairs(data) do
    if (type(value) == "table") then
      data[key] = {value[1]:float(), value[2]:float()}
    else  
      data[key] = value:float()
    end
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

function M:logBatch(embeddings, target)

   if trainLoggerAcc then
      cm:batchAdd(embeddings[2], target)
   end
   
   -- TODO: log info from turn-on criterion
   for name, value in pairs(listActiveCriterion) do
      value:addErrGrad()
   end
   
   print(('Epoch: [%d][%d/%d]\tTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real))
end

-- Based on paper "A Discriminative Feature Learning Approach for Deep Face Recognition",Yandong Wen, Kaipeng Zhang, Zhifeng Li and Yu Qiao
function M:updateClusters( embeddings, target, info )
   -- Get embeding for each class independent
   local clusterUpdate = torch.FloatTensor(opt.peoplePerBatch, opt.embSize)
   local startIdx      = 1
   for i=1,opt.peoplePerBatch do
      local embClass = embeddings[1][{{startIdx,startIdx + info.nSamplesPerClass[i]-1},{}}]
      local mean     = torch.repeatTensor(clusterCenters[target[startIdx]],embClass:size(1),1):csub(embClass):mean(1)
      clusterCenters[target[startIdx]] = clusterCenters[target[startIdx]] - opt.clusterLR * mean
      startIdx = startIdx + info.nSamplesPerClass[i]
   end
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
   if config.SoftMaxLoss then
      torch.save(paths.concat(opt.save, 'model_classification_' .. epoch .. '.t7'),  classificationBlock:clearState())
   end
  
   if dpt then -- OOM without this
      dpt:clearState()
   end

   collectgarbage()
 
   return model
end



return M