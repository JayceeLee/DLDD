#!/usr/bin/env th

require 'torch'
require 'optim'
require 'paths'
require 'xlua'
json = require('json')

local opts = paths.dofile('opts.lua')
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
clusterCenters  = torch.rand(opt.nClasses, opt.embSize)
local trainLoader, valLoader = DataLoader.create(opt, clusterCenters)
local models = require 'model'
local modelConfig = models.ModelConfig()
modelConfig:generateConfig(opt)
middleBlock = modelConfig:middleBlockSetup(opt)
criterion   = modelConfig:critertionSetup()

local Trainer = require 'train'
local Test    = require 'testVerify'

epoch = 1
for e = opt.epochNumber, opt.nEpochs do
   Trainer:train(trainLoader, modelConfig)
   model = Trainer:saveModel(model)
   Test:test(valLoader)
   epoch = epoch + 1
end

