#!/usr/bin/env th

require 'torch'
require 'optim'

require 'paths'

require 'xlua'

local opts = paths.dofile('opts.lua')
local DataLoader = require 'dataloader'

opt = opts.parse(arg)
print(opt)


if opt.cuda then
   require 'cutorch'
   cutorch.setDevice(opt.device)
end

torch.save(paths.concat(opt.save, 'opts.t7'), opt, 'ascii')
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

