--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('openface.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train','val'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      if split == 'train' then
         if opt.samplePeople then
            _G.dataset.sample = dataset.samplePeople
         else
            _G.dataset.sample = dataset.sampleImagesRandom
         end
      else
         _G.dataset.sample = dataset.sampleImages
      end
      return dataset.imageInfo.labels:size(1)
   end
   local threads, sizes = Threads(opt.nDonkeys, init, main)
   self.threads         = threads
   self.dataset         = dataset
   self.__size          = sizes[1][1]
   self.batchSize       = opt.batchSize
   self.peoplePerBatch  = opt.peoplePerBatch 
   self.imagesPerPerson = opt.imagesPerPerson 
   if split == 'train' then 
      self.epochSize       = opt.epochSize
      if opt.samplePeople then
         self.getInfo = self.infoForSamplingPeople
      else
         self.getInfo = self.infoForRandomExample
      end
   else
     self.epochSize       = math.ceil(self.__size/opt.batchSize)
     self.getInfo = self.infoForNormalEpoch
   end
   self.opt             = opt    
   self.split           = split
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end


function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   self.perm = torch.randperm(size)
   local idx, sample = 1,  nil
   self.idxSample = 1
   local function enqueue()
      while idx <= self.epochSize and threads:acceptsjob() do
	     local infoData = self:getInfo(idxSample) 
        threads:addjob(
            function(infoSampling)
               local indices, additionalInfo = _G.dataset:sample(infoSampling)
               additionalInfo.indices = indices
               local sz = indices:size(1)
               local batch, imageSize
               local target = torch.IntTensor(sz)
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  local input = _G.preprocess(sample.input)
                  if not batch then
                     imageSize = input:size():totable()
                     batch = torch.FloatTensor(sz, table.unpack(imageSize))
                  end
                  batch[i]:copy(input)
                  target[i] = sample.target
               end
               collectgarbage()
               return {
                  input = batch:view(sz, table.unpack(imageSize)),
                  target = target,
                  info  = additionalInfo,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            infoData
         )
         idx = idx + 1
         self.idxSample = self.idxSample + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

function DataLoader:infoForNormalEpoch()
   local info = {}
   info.indices = self.perm:narrow(1, self.idxSample, math.min(self.batchSize, self.__size - self.idxSample + 1))
   return info
end

function DataLoader:infoForRandomExample()
   local info = {}
   info.batchSize = self.batchSize
   return info
end

function DataLoader:infoForSamplingPeople()
   local info = {}
   info.peoplePerBatch = self.peoplePerBatch
   info.imagesPerPerson = self.imagesPerPerson
   return info
end

return M.DataLoader
