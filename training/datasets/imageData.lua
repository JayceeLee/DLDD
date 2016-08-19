
local image = require 'image'
local paths = require 'paths'
local ffi = require 'ffi'
local metrics = require 'metrics' --calculate Distance Matrix
local utils = require '../utils'
local t = require 'datasets/transforms'
local M = {}
local ImageData = torch.class('dldd.ImageData', M)

function ImageData:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.classList = imageInfo.classList
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   self.imgDim = opt.imgDim
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
   self.size    = self.imageInfo.labels:size(1)
   self.indexes = torch.range(1, self.size)
   self.idx = 1 -- idx needed for getting entire epoch of data
   if split == 'val' then
     self:createValidationPairs(opt)
   end
end

function ImageData:createValidationPairs(opt)
  -- load pairs
  local datafile = io.open(opt.data .. "/pairs_val.txt", "r")
  local pairs = {}
  for line in datafile:lines() do
      local splitted = utils.splitString(line, " ")
      table.insert(pairs, splitted)
  end
  datafile:close()
  
  -- Transform image paths to format className/imageName
  local imagePathTransformed = {}
  for i=1,self.imageInfo.imagePath:size(1) do
     local name = utils.splitString(ffi.string(self.imageInfo.imagePath[i]:data()),'/')
     imagePathTransformed[i] = name[#name-1] .. '/' .. name[#name]
  end
  -- Transforms pairs in string to pairs in idx
  self.pairs = {}
  for i=1,table.getn(pairs) do
    local pair = pairs[i]
    if table.getn(pair) == 3 then
      name_1 = ('%s/%s'):format(pair[1], pair[2])
      name_2 = ('%s/%s'):format(pair[1], pair[3])
      actual_same = 2
    elseif table.getn(pair) == 4 then
      name_1 = ('%s/%s'):format(pair[1], pair[2])
      name_2 = ('%s/%s'):format(pair[3], pair[4])
      actual_same = 1
    end
    self.pairs[i] = {AKeyOf(imagePathTransformed,name_1), AKeyOf(imagePathTransformed,name_2),actual_same}
  end
  self.pairs = torch.Tensor(self.pairs)
end

function ImageData:get(i)
   local path = ffi.string(self.imageInfo.imagePath[i]:data())

   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.labels[i]

   return {
      input = image,
      target = class,
   }
end

-- return random idx of images
function ImageData:sampleImagesRandom(info)
  return torch.randperm(self.size)[{{1,info.batchSize}}]:int(), {}
end

-- return next idx of images, generated outside function to be synchronized between threads
function ImageData:sampleImages(info)
  return info.indices, {}
end

function ImageData:sampleImagesGrouped(info)

   local classes = torch.randperm(#self.classList)[{{1,info.peoplePerBatch}}]:int()
   local nSamplesPerClass = torch.Tensor(info.peoplePerBatch)
   for i=1,info.peoplePerBatch do
      local nSample = math.min(self.imageInfo.classSample[classes[i]]:nElement(), info.imagesPerPerson)
      nSamplesPerClass[i] = nSample
   end

   local data = torch.Tensor(nSamplesPerClass:sum())
   local targets = torch.Tensor(nSamplesPerClass:sum())
   local dataIdx = 1
   for i=1,info.peoplePerBatch do
      local cls = classes[i]
      local nSamples = nSamplesPerClass[i]
      local nTotal = self.imageInfo.classSample[classes[i]]:nElement()
      if (nTotal) == 0 then 
         print "zero"
         print(classes[i])
         print(self.classList[classes[i]])
         print(self.imageInfo.classSample[classes[i]])
      end
      local shuffle = torch.randperm(nTotal)
      for j = 1, nSamples do
         data[dataIdx] = self.imageInfo.classSample[cls][shuffle[j]]
         targets[dataIdx] = cls
         dataIdx = dataIdx + 1
      end
   end
   assert(dataIdx - 1 == nSamplesPerClass:sum())
   local infoSampling = {}
   infoSampling.nSamplesPerClass = nSamplesPerClass
   return data, infoSampling
end

-- TODO: add some random classes to have not only hard classes?
function ImageData:sampleImagesFromClusters(info)
   -- Get random class
   local anchorClass      = torch.randperm(#self.classList)[{{1}}]:int()
   local nSamplesPerClass = torch.Tensor(info.peoplePerBatch)
   local clusterCenter    = info.clusterCenters[anchorClass[1]]:view(1, self.opt.embSize)
   -- Get classes close to anchorClass
   local _,classes        = metrics.distancesL2(clusterCenter,info.clusterCenters):float():topk(info.peoplePerBatch)

   -- print(idx:index(1, torch.randperm(idx:size(2)):long()))
   -- local idx              = idx:index(1, torch.randperm(idx:size(1)):long())[{{2,info.peoplePerBatch}}]:int()
   -- local classes          = torch.Tensor(info.peoplePerBatch)
   -- classes[1] = anchorClass
   -- classes[{{2,info.peoplePerBatch}}] = idx
   classes = torch.squeeze(classes)
   for i=1,info.peoplePerBatch do
      local nSample = math.min(self.imageInfo.classSample[classes[i]]:nElement(), info.imagesPerPerson)
      nSamplesPerClass[i] = nSample
   end

   local data = torch.Tensor(nSamplesPerClass:sum())
   local targets = torch.Tensor(nSamplesPerClass:sum())
   local dataIdx = 1
   for i=1,info.peoplePerBatch do
      local cls = classes[i]
      local nSamples = nSamplesPerClass[i]
      local nTotal = self.imageInfo.classSample[classes[i]]:nElement()
      if (nTotal) == 0 then 
         print "zero"
         print(classes[i])
         print(self.classList[classes[i]])
         print(self.imageInfo.classSample[classes[i]])
      end
      local shuffle = torch.randperm(nTotal)
      for j = 1, nSamples do
         data[dataIdx] = self.imageInfo.classSample[cls][shuffle[j]]
         targets[dataIdx] = cls
         dataIdx = dataIdx + 1
      end
   end
   assert(dataIdx - 1 == nSamplesPerClass:sum())
   local infoSampling = {}
   infoSampling.nSamplesPerClass = nSamplesPerClass
   return data, infoSampling
end



function ImageData:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function ImageData:size()
   return self.imageInfo.labels:size(1)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function ImageData:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.Scale(self.imgDim),
         -- t.ColorJitter({
         --    brightness = 0.4,
         --    contrast = 0.4,
         --    saturation = 0.4,
         -- }),
         -- t.Lighting(0.1, pca.eigval, pca.eigvec),
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(self.imgDim),
         t.ColorNormalize(meanstd),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

-- return any key holding the value 
function AKeyOf(t,val)
    for k,v in pairs(t) do 
        if v == val then return k end
    end
end

return M.ImageData
