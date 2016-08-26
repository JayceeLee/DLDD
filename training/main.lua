#!/usr/bin/env th

require 'torch'
require 'optim'
require 'paths'
require 'xlua'
json = require('json')

local opts = paths.dofile('opts.lua')
local checkpoints = require 'checkpoints'
local DataLoader = require 'dataloader'
opt = opts.parse(arg)
print(opt)




if opt.cuda then
   require 'cutorch'
   cutorch.setDevice(opt.device)
end
json.save(paths.concat(opt.save, 'opts.json'), opt)

print('Saving everything to: ' .. opt.save)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manualSeed)

-- Data loading
centerCluster  = torch.rand(opt.nClasses, opt.embSize)
local trainLoader, valLoader = DataLoader.create(opt, centerCluster)
local models = require 'model'
local modelConfig = models.ModelConfig()
modelConfig:generateConfig(opt)
middleBlock = modelConfig:middleBlockSetup(opt)
criterion   = modelConfig:critertionSetup()

local Trainer = require 'train'
local Test    = require 'testVerify'

epoch = 1
local bestVerAcc = 0
for e = opt.epochNumber, opt.nEpochs do
   Trainer:train(trainLoader, modelConfig)
--    model = Trainer:saveModel(model)
   local testVer = Test:test(valLoader)
   local bestModel = false
   if bestVerAcc < testVer then
       print(' * Best model ', testVer)
       bestModel = true
   end
   model = checkpoints.save(epoch, model, optimState, bestModel, opt)
   epoch = epoch + 1
end

