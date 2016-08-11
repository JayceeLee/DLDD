-- Verification Tester for model using Hold-Out Verification set
local ffi = require 'ffi'
local utils = require 'utils'
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
local confusion = optim.ConfusionMatrix(2)
require 'nn'

local M = {}
local Test       = torch.class('dddl.Test', M)
local normalizer = nn.Normalize(2):float()
local timer = torch.Timer()
if opt.nClasses ~= 0 then
  testLoggerAcc = optim.Logger(paths.concat(opt.save, 'testSoft.log'))
  confusionSoft = optim.ConfusionMatrix(opt.nClasses)
  lossSoft      = 0
end
function Test:test(dataLoader)
  print('==> doing epoch on Verification Set:')
  print("==> online epoch # " .. epoch) 
  cutorch.synchronize()
  model:evaluate()
  timer:reset()
  self.mapperEmbName = {}
  lossSoft = 0
  self:testVerification(dataLoader)
  collectgarbage()
end

function Test:testVerification(dataLoader)
  
  for n, sample in dataLoader:run() do
      self:copyInputs(sample)
      self:repBatch(self.input, self.target, self.info, dataLoader.dataset.imageInfo.imagePath)
      xlua.progress(n, dataLoader:size())
  end

  if opt.nClasses ~= 0 then
    confusionSoft:updateValids()
    testLoggerAcc:add{
      ['avg mAP (Classification set)'] = confusionSoft.totalValid * 100,
   }
   print(('Epoch [%d] Classification ACC: %.2f\tLoss: %.2f'):format(
        epoch, confusionSoft.totalValid * 100, lossSoft/dataLoader.epochSize))
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
  print(('Epoch [%d] Verificatin ACC: %.2f\tBest Thre:s %.2f\tTime(s): %.3f'):format(
        epoch, best_acc, best_thres, timer:time().real))
  print('\n')
  testLogger:add{
      ['avg mAP (Verification set)'] = best_acc
   }
  timer:reset()
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
  local n = labels:size(1)
  local embeddings = model:forward(input)
  cutorch.synchronize()

  if opt.nClasses ~= 0 then
    local classOutput = classificationBlock:forward(embeddings):float()
    confusionSoft:batchAdd(classOutput, labels)
    lossSoft = lossSoft + classificationCriterion:forward(classOutput, labels)
  end
  embeddings:float()
  for i=1,n do self.mapperEmbName[info.indices[i]] = embeddings[i] end
  
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
  for i=1,distances:size(1) do
    local diff = distances[i]
    if diff < threshold then
      confusion:add(2, same_info[i][1])
    else
      confusion:add(1, same_info[i][1])
    end
  end
  confusion:updateValids()
  return confusion.totalValid
end 

return M.Test






