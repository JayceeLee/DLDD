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

config = {
  SoftMaxLoss = false,
  SoftMaxLossWeight = 1.0,
  TripletLoss = true,	
  TripletLossWeight = 0.1,
  TripletSampling = 'semi-hard' --'random'
}

if opt.cuda then
   require 'cutorch'
   cutorch.setDevice(opt.device)
end
json.save(paths.concat(opt.save, 'opts.json'), opt)
json.save(paths.concat(opt.save, 'config.json'), config)

print('Saving everything to: ' .. opt.save)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manualSeed)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)
local models = require 'model'
middleBlock = models.middleBlockSetup(opt)
criterion   = models.critertionSetup(opt)
local Trainer = require 'train'
local Test    = require 'testVerify'

epoch = 1
for e = opt.epochNumber, opt.nEpochs do
   Trainer:train(trainLoader)
   model = Trainer:saveModel(model)
   Test:test(valLoader)
   epoch = epoch + 1
end

