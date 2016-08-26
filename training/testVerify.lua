-- Verification Tester for model using Hold-Out Verification set
require 'nn'
local ffi = require 'ffi'
local utils = require 'utils'
testLogger = optim.Logger(paths.concat(opt.save, 'testVer.log'))
local confusion = optim.ConfusionMatrix(2)

local M = {}
local Test       = torch.class('dddl.Test', M)
local normalizer = nn.Normalize(2):float()
local timer = torch.Timer()
if opt.SoftMax ~= 0 then 
  testLoggerAcc = optim.Logger(paths.concat(opt.save, 'AccuracyTest.log'))
  confusionSoft = optim.ConfusionMatrix(opt.nClasses)
  lossSoft      = 0
end

local testLoggerStdDev = optim.Logger(paths.concat(opt.save, 'testStdCenter.log'))
local stdDev = 0

function Test:test(dataLoader)
  print('==> doing epoch on Verification Set:')
  print("==> online epoch # " .. epoch) 
  cutorch.synchronize()
  model:evaluate()
  self:resetLogger()
  local verAcc = self:testVerification(dataLoader)
  collectgarbage()
  return verAcc
end

function Test:testVerification(dataLoader)
  
  for n, sample in dataLoader:run() do
      self:copyInputs(sample)
      self:repBatch(self.input, self.target, self.info, dataLoader.dataset.imageInfo.imagePath)
--       xlua.progress(n, dataLoader:size())
  end

  if testLoggerAcc then
    confusionSoft:updateValids()
    testLoggerAcc:add{
      ['avg mAP  (test set)'] = confusionSoft.totalValid * 100,
      ['avg loss (test set)'] = lossSoft/dataLoader.epochSize,
   }
   print(('Classification ACC: %.2f\tLoss: %.2f'):format(
        confusionSoft.totalValid * 100, lossSoft/dataLoader.epochSize))
  end

  
  -- Compute distance for each pair
  local distances, same_info = self:computeDistances(dataLoader.dataset.pairs)

  -- After having all embeding with pathes, we need to run verification process
  local threshold = torch.linspace(0,torch.max(distances),100) --from 0 to 4, 100 elements
  local best_acc = 0
  local best_thres = 0
  for th=1,threshold:size()[1] do
    local thres = threshold[th]
    local acc = self:evalThresholdAccuracy(distances, same_info, thres)
    if acc > best_acc then
      best_thres = thres
      best_acc   = acc
    end
  end
  print(('Epoch [%d] Verificatin ACC: %.2f\tBest Thres: %.2f\tTime(s): %.3f'):format(
        epoch, best_acc, best_thres, timer:time().real))
  print('\n')
  testLogger:add{
      ['avg mAP (Verification set)'] = best_acc
   }
  testLoggerStdDev:add{
      ['stdDev (Test set)'] = math.sqrt(stdDev / dataLoader:size())
   }
   
  timer:reset()
  return best_acc
end
  
  
function Test:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = sample.target
   self.info   = sample.info

   self.input:resize(sample.input:size()):copy(sample.input)
end

function Test:repBatch(input, labels, info, allPaths)
  -- labels:size(1) is equal to batchSize except for the last iteration if
  -- the number of images isn't equal to the batch size.
  local N = labels:size(1)
  local embeddings = model:forward(input):float()
  cutorch.synchronize()

  for i=1,N do self.mapperEmbName[info.indices[i]] = embeddings[i] end

  if testLoggerAcc then
    local classOutput = classificationBlock:forward(embeddings:cuda()):float()
    confusionSoft:batchAdd(classOutput, labels)
    lossSoft = lossSoft + config.Classification.SoftMax.model:forward(classOutput, labels)
    embeddings:float()
  end
  
  -- log stddev of difference between features and cluster center
  local stdDiff = torch.FloatTensor(N)
  for i=1,N do
    stdDiff[i] = (embeddings[i] - centerCluster[labels[i]]):pow(2):sum() / embeddings:size(2)
  end
  stdDev = stdDev + stdDiff:sum()/N
end
 
function Test:getEmbeddings(pair)
  return  self.mapperEmbName[pair[1]],  self.mapperEmbName[pair[2]], pair[3]
end

function Test:computeDistances(pairs)
  local distances = torch.FloatTensor(pairs:size(1))
  for i=1,pairs:size(1) do
    local x1, x2, actual_same = self:getEmbeddings(pairs[i])
    local diff = x1 - x2
    distances[i] = torch.dot(diff,diff)
  end
  return distances, pairs[{{},{3}}]
end


function Test:evalThresholdAccuracy(distances, same_info, threshold )
  confusion:zero()
  confusion:batchAdd(distances:lt(threshold):add(1), same_info)
  confusion:updateValids()
  return confusion.totalValid
end 

function Test:resetLogger()
  timer:reset()
  self.mapperEmbName = {}
  lossSoft = 0
  stdDev   = 0
  if testLoggerAcc then
    confusionSoft:zero()
  end
end

return M.Test






