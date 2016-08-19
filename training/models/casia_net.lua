require 'nn'
require 'cunn'
require 'cudnn'
		 
local SpatialConvolution = nn.SpatialConvolution
local SpatialMaxPooling = nn.SpatialMaxPooling
local SpatialAveragePooling = nn.SpatialAveragePooling
local ReLU = nn.ReLU


imgDim = 96
function createModel()
   local net = nn.Sequential()
   --stage 1
   net:add(SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1))
--    net:add(nn.SpatialBatchNormalization(32))
   net:add(ReLU(true))
   net:add(SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
--    net:add(nn.SpatialBatchNormalization(64))
   net:add(ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2, 1, 1))
   
   --stage 2
   net:add(SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
--    net:add(nn.SpatialBatchNormalization(64))
   net:add(ReLU(true))
   net:add(SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
--    net:add(nn.SpatialBatchNormalization(128))
   net:add(ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2, 1, 1))
   
   --stage 3
   net:add(SpatialConvolution(128, 96, 3, 3, 1, 1, 1, 1))
--    net:add(nn.SpatialBatchNormalization(96))
   net:add(ReLU(true))
   net:add(SpatialConvolution(96, 192, 3, 3, 1, 1, 1, 1))
--    net:add(nn.SpatialBatchNormalization(192))
   net:add(ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2, 1, 1))
   
   --stage 4
   net:add(SpatialConvolution(192, 128, 3, 3, 1, 1, 1, 1))
--    net:add(nn.SpatialBatchNormalization(128))
   net:add(ReLU(true))
   net:add(SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
--    net:add(nn.SpatialBatchNormalization(256))
   net:add(ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2, 1, 1))
   
   --stage 5
   net:add(SpatialConvolution(256, 160, 3, 3, 1, 1, 1, 1))
--    net:add(nn.SpatialBatchNormalization(160))
   net:add(ReLU(true))
   net:add(SpatialConvolution(160, 320, 3, 3, 1, 1, 1, 1))
--    net:add(nn.SpatialBatchNormalization(320))
   net:add(ReLU(true))
--    net:add(SpatialMaxPooling(2, 2, 2, 2, 1, 1))
-- 
   net:add(SpatialAveragePooling(7, 7))

   -- Validate shape with:
   -- net:add(nn.Reshape(320))

   net:add(nn.View(320))
   net:add(nn.Linear(320, opt.embSize))

    
   
   -- print(#net:cuda():forward(torch.CudaTensor(1,3,96,96)))
   
   
   return net
end

