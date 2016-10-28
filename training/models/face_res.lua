--  Model from https://github.com/ydwen/caffe-face which implement "A Discriminative Feature Learning Approach for Deep Face Recognition"
-- , but use different model (with residuals but without Local-Convolutions

local nn = require 'nn'
require 'cunn'
require 'cudnn'

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.PReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

imgDim = 100

function createModel()
   local depth = 27
   local shortcutType = 'B'
   local iChannels


   -- The basic residual layer block for Res-Face
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local block = nn.Sequential()
      local s = nn.Sequential()

      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(ReLU())
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(ReLU())
      
      return block
         :add(nn.ConcatTable()
            :add(s)
            :add(nn.Identity()))
         :add(nn.CAddTable(true))
   end

   -- CONV-RELU-MAXPOOL with increase fetures in conv
   local function increaseFeatures(n_in, n_out, stride)

      local block = nn.Sequential()
      block:add(Convolution(n_in,n_out,3,3,stride,stride,0,0))
      block:add(ReLU())
      block:add(Max(2,2,2,2,0,0))

      return block
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      if count < 1 then
        return s
      end
      for i=1,count do
         s:add(block(features, stride))
      end
      return s
   end

   local model = nn.Sequential()

    -- Configurations for ResNet:
    --  num. residual blocks, num features, residual block function
    local cfg = {
	[27]  = {{1, 2, 5, 3}, 512*5*4, basicblock},
    }

--     assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
    local def, nFeatures, block = table.unpack(cfg[depth])
    iChannels = 64
    print(' | ResNet-' .. depth .. ' ImageNet')

    -- The ResNet ImageNet model
    model:add(Convolution(3,32,3,3,1,1,0,0))
    model:add(ReLU())
    model:add(increaseFeatures(32,64,1))
    model:add(layer(block,64,def[1],1))
    model:add(increaseFeatures(64,128,1))
    model:add(layer(block,128,def[2],1))
    model:add(increaseFeatures(128,256,1))
    model:add(layer(block,256,def[3],1))
    model:add(increaseFeatures(256,512,1))
    model:add(layer(block,512,def[4],1))
    model:add(nn.View(nFeatures):setNumInputDims(3))
    model:add(nn.Linear(nFeatures, opt.embSize))
  

    model:get(1).gradInput = nil

   return model
end

