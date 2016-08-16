-- Model from https://github.com/vbalnt/pnnet, 
-- paper: "PN-Net: Conjoined Triple Deep Network for Learning Local Image Descriptors", 
-- Vassileios Balntas, Edward Johns, Lilian Tang, Krystian Mikolajczyk, 	arXiv:1601.05030

imgDim = 28
function createModel()
  local model = nn.Sequential() 
  model:add(nn.SpatialConvolution(3, 32, 7, 7))
  model:add(nn.Tanh(true))
  model:add(nn.SpatialMaxPooling(2,2,2,2)) 
  model:add(nn.SpatialConvolution(32, 64, 6, 6))
  model:add(nn.Tanh(true))
  model:add(nn.View(64*6*6))
  model:add(nn.Linear(64*6*6, opt.embSize))
--   model:add(nn.Tanh(true))
  
  return model
end

