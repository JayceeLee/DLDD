require 'optim'
require 'hdf5'

local M = {}

local Train = torch.class('dldd.Train', M)


function Train:__init(opt)
   self.optimMethod = _G.optim[opt.optimization]
   self.optimState = {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.minLR  = -0.00001
   -- Variable for Auto Learning Decay
   self.decayAuto = 0
   self.decayAutoDiff = 0.01
   self.decayAutoEpoch = 5
   self.lastChange = 0

   if opt.SoftMax ~= 0 then
      self.trainLoggerAcc = optim.Logger(paths.concat(opt.save, 'AccuracyTrain.log'))
   end
   self.LRlogger = optim.Logger(paths.concat(opt.save, 'LR.log'))
   self.opt = opt
end

function Train:train(dataloader, models)
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   self.all_time = 0
   self.batchNumber = 0
   self.embNaN = false
   -- init main model
   model = models:modelSetup(model)
   if self.trainLoggerAcc then
      self.cm = optim.ConfusionMatrix(self.opt.nClasses, torch.range(1,self.opt.nClasses))
   end
   self.optimState.learningRate = self:learningRate(epoch)
   if self.optimState.learningRate < self.minLR then return false end -- LR too small
   -- init model with additional module
   local model_middle = nn.Sequential()
   model_middle:add(model):add(middleBlock)
   model_middle:cuda()
   self:initOptim(model_middle, self.optimState)
    
   cutorch.synchronize()
   model_middle:training()
   local tm = torch.Timer()
   
   for n, sample in dataloader:run() do
      self:copyInputs(sample)
      self:trainBatch(self.input, self.target, self.info)
      dataloader.centerCluster = centerCluster -- update cluster Centers
      if opt.Center > 0.0 then -- update cluster center in loss function
         config.Raw.Center.model.centerCluster = centerCluster
      end
      if self.embNaN then
        print("NaNs at embeding, break experiment")
        return false
      end
   end

   cutorch.synchronize()
   
   for name, value in pairs(listActiveCriterion) do
      value:logData(self.batchNumber)
      value:zero()
   end

   if self.trainLoggerAcc then
      self.cm:updateValids()
      print(string.format('Accuracy: %.2f',  self.cm.totalValid * 100))
      self.trainLoggerAcc:add{
            ['avg mAP  (train set)'] =  self.cm.totalValid * 100
         }
   end
   self.LRlogger:add{['LR'] =  self.optimState.learningRate}

   -- Save cluster data for comaprision between them
   local clusters = hdf5.open(self.opt.save .. '/clusterCenters_' .. epoch .. '.hdf5', 'w')
   clusters:write('clusters', centerCluster)
   clusters:close()

   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\tLR:: %.5f',
                       epoch, tm:time().real,  self.optimState.learningRate))
   print('\n')
   collectgarbage()
   return true
end -- of train()

function Train:initOptim(model, optState)
    assert(model)
    assert(optState)

    self.model = model
    -- Keep this around so we update it in setParameters
    self.originalOptState = optState
    self.params, self.gradParams = self.model:getParameters()
end

function Train:copyInputs(sample)
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
function Train:trainBatch(inputs, target, info)
   timer:reset()
   collectgarbage()

   if opt.cuda then
      cutorch.synchronize()
   end

   local numImages = inputs:size(1)

   tripletSampling.numPerClass = info.nSamplesPerClass 
   pairSampling.numPerClass    = info.nSamplesPerClass
   -- local embeddings2            = model:forward(inputs)
   local embeddings            = self.model:forward(inputs)
   self:toFloat(embeddings)
   if self:checkNans(embeddings[1]) then return end-- Check if output is not NaNs 
   self.model:zeroGradParameters()
   if config.PairSamplingEnable then -- Get targets for Pair Sampling
      target = {target, target, pairSampling.target, target, pairSampling.target}
   end

   local err     = criterion:forward(embeddings, target)
   local df_do   = criterion:backward(embeddings, target)

   self:toCuda(df_do)
   self.model:backward(inputs, df_do)
   self:toFloat(df_do)

   local curGrad
   local curParam
   local function fEvalMod()
      return err, self.gradParams
   end
   self.optimMethod(fEvalMod, self.params, self.originalOptState)
   self.batchNumber = self.batchNumber + 1

   if config.PairSamplingEnable then
      target = target[1]
   end
   self:logBatch(embeddings, target)
   -- self:updateClusters(embeddings, target, info)

   self.all_time = self.all_time + timer:time().real
   timer:reset()
end

function Train:toFloat(data)
   for key,value in pairs(data) do
      if type(value) == "table" then
         data[key] = {value[1]:float(), value[2]:float()}
      else  
         data[key] = value:float()
      end
   end
end

function Train:toCuda(data)
   for key,value in pairs(data) do
      data[key] = value:cuda()
   end
end

function Train:learningRate(epoch)
   -- Training schedule
   if  self.opt.LRDecay > 0.0 then 
      local   decay = math.floor((epoch - 1) /  self.opt.LRDecay)
      return  self.opt.LR * math.pow(0.1, decay)
   else
      if testData.bestEpoch > self.lastChange and testData.diffAcc > self.decayAutoDiff/3 then 
         self.lastChange = testData.bestEpoch 
      end
      if (testData.testVer - testData.bestVerAcc) < self.decayAutoDiff and (epoch - self.lastChange) > self.decayAutoEpoch then
         self.decayAuto = self.decayAuto  + 1
         self.lastChange = epoch
      end
      return  self.opt.LR * math.pow(0.1, self.decayAuto)
   end
end

function Train:logBatch(embeddings, target)
   if self.trainLoggerAcc then
      self.cm:batchAdd(embeddings[2], target)
   end
   
   for name, value in pairs(listActiveCriterion) do
      value:addErrGrad()
   end
   
   print(('Epoch: [%d][%d/%d]\tTime %.3f'):format(
        epoch, self.batchNumber, self.opt.epochSize, timer:time().real))
end

-- Based on paper "A Discriminative Feature Learning Approach for Deep Face Recognition",Yandong Wen, Kaipeng Zhang, Zhifeng Li and Yu Qiao
function Train:updateClusters( embeddings, target, info )
   -- Get embeding for each class independent
   local clusterUpdate = torch.FloatTensor(opt.peoplePerBatch, opt.embSize)
   local startIdx      = 1
   for i=1,opt.peoplePerBatch do
      local embClass = embeddings[1][{{startIdx,startIdx + info.nSamplesPerClass[i]-1},{}}]
      local mean     = torch.repeatTensor(centerCluster[target[startIdx]],embClass:size(1),1):csub(embClass):mean(1)
      centerCluster[target[startIdx]] = centerCluster[target[startIdx]] - opt.clusterLR * mean
      startIdx = startIdx + info.nSamplesPerClass[i]
   end
end

-- Check if model disverge. If the output emb is nans, break training
function Train:checkNans(x)
   local I = torch.ne(x,x)
   if torch.any(I) then
      print("train.lua: Error: NaNs found in: output")
      self.embNaN = true
      return true
   end
   return false
end



return M.Train