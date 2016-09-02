--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'transforms'
local ffi = require 'ffi'

function splitString(inputstr, sep)
  if sep == nil then
          sep = "%s"
  end
  local t={} ; i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
          t[i] = str
          i = i + 1
  end
  return t
end

-- see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

--preprocess txt file to extract paths and target of each example
function lines_from(file)
  if not file_exists(file) then return {} end
  paths = {}
  targets = {}
  for line in io.lines(file) do 
    data = splitString(line," ")
    paths[#paths + 1] = data[1]
    targets[#targets + 1] = tonumber(data[2]) + 1 --add 1 to labels, as idx start from 1, not 0
  end
  return paths,targets
end

local ImageDataset = torch.class('ImageDataset')

function ImageDataset:__init(paths_txt, opt, name)
   paths ,targets = lines_from(paths_txt)
   self.paths   = paths
   self.targets = torch.Tensor(targets):float()
   self.imgDim = opt.imgDim
   self.opt   = opt
   self.split = name
end

function ImageDataset:get(i)
   local image = self:_loadImage(self.paths[i])
   local class = self.targets[i]

   return {
      input  = image,
      target = class,
   }
end

function ImageDataset:_loadImage(path)
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

function ImageDataset:size()
   return self.imageInfo.imageClass:size(1)
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

function ImageDataset:preprocess()
   if self.split == 'aug' then
      return t.Compose{
         t.Scale(imgDimScale), 
--          t.Lighting(0.1, pca.eigval, pca.eigvec),
--          t.ColorJitter({
--             brightness = 0.4,
--             contrast = 0.4,
--             saturation = 0.4,
--          }),
         -- t.Warp(nil),
         t.RandomSizedCrop(imgDim),
         t.HorizontalFlip(0.5),
         t.ColorNormalize(meanstd),
         t.CenterCrop(imgDim),
         
--          t.RandomSizedCrop(imgDim),
--          t.ColorJitter({
--             brightness = 0.4,
--             contrast = 0.4,
--             saturation = 0.4,
--          }),
--          t.Lighting(0.1, pca.eigval, pca.eigvec),

--          t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(self.imgDim),
         t.ColorNormalize(meanstd)
      }
   else
      error('invalid split: ' .. self.split)
   end
end
