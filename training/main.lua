#!/usr/bin/env th

require 'torch'
require 'optim'
require 'paths'
require 'xlua'
require 'json'
 require 'hdf5'

function  saveJson(fileName, data)
   local file = assert(io.open(fileName, "w"))
   file:write(json.encode.encode(data))
   file:close()
end

function loadHDF5(fileName, dataName)
   local myFile = hdf5.open(fileName, 'r')
   local data = myFile:read(dataName):all()
   myFile:close()
   return data
end

local opts = paths.dofile('opts.lua')
local checkpoints = require 'checkpoints'
local DataLoader = require 'dataloader'
opt = opts.parse(arg)
print(opt)


if opt.cuda then
   require 'cutorch'
   cutorch.setDevice(opt.device)
end

saveJson(paths.concat(opt.save, 'opts.json'), opt)

print('Saving everything to: ' .. opt.save)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manualSeed)

-- Data loading
if opt.reCluster ~= '' then
   print("Cluster Center loaded from: " .. opt.reCluster)
   centerCluster = loadHDF5(opt.reCluster, 'clusters')
else
   centerCluster = torch.rand(opt.nClasses, opt.embSize)
end

local trainLoader, valLoader = DataLoader.create(opt, centerCluster)
local models = require 'model'
local modelConfig = models.ModelConfig()
modelConfig:generateConfig(opt)
middleBlock = modelConfig:middleBlockSetup(opt)
criterion   = modelConfig:critertionSetup(opt)

local Trainer = require 'train'
local trainer = Trainer(opt)
local Test    = require 'testVerify'
local tester = Test(opt)

epoch = 1
testData = {}
testData.bestVerAcc = 0
testData.bestEpoch = 0
testData.testVer = 0
testData.diffAcc = 0

for e = opt.epochNumber, opt.nEpochs do
   local sucess = trainer:train(trainLoader, modelConfig)
   if not sucess then break end
--    model = Trainer:saveModel(model)
   testData.testVer = tester:test(valLoader)
   local bestModel = false
   if testData.bestVerAcc < testData.testVer then
      print(' * Best model ', testData.testVer)
      bestModel = true
      testData.diffAcc = testData.testVer - testData.bestVerAcc
      testData.bestVerAcc = testData.testVer
      testData.bestEpoch  = epoch
   end
   model = checkpoints.save(epoch, model, optimState, bestModel, opt)
   if opt.checkEpoch > 0 and epoch > opt.checkEpoch then
      if testData.bestVerAcc < opt.checkValue then break end -- model does not converge, break it
   end
   epoch = epoch + 1
end

--[[
th main.lua -data ../../OpenFace/CASIA_FS_VGG_96_lips/ -modelDef models/preresnet.lua -name PreRes18 -peoplePerBatch 16 -imagesPerPerson 10 -imgDim 96  -LR 0.1 
-Triplet 2.0 -Trisim 2.0 -Lifted 2.0 -Constr 2.0 -Mulbatch 2.0 -Center 0.003 -SoftMax 1.0  -nClasses 11487 -LRDecay 40 -nEpochs 140 -nDonkeys 4 -epochSize 1000 -device 2 -cache experiments/DLDD_exp/casia_fs_vgg__exp/ -nGPU 2

]]--

