local nninit = require 'nninit'
local utils = {}

--Reduce the memory consumption by model by sharing the buffers
function utils.optimizeNet( model, inputSize )
   local optnet_loaded, optnet = pcall(require,'optnet')
   if optnet_loaded then
      local opts   = {inplace=true, mode='training', removeGradParams=false}
      local input  = torch.rand(2,3,inputSize,inputSize)
      if opt.cuda then
          input = input:cuda()
      end
      optnet.optimizeMemory(model, input, opts)
   else
      print("'optnet' package not found, install it to reduce the memory consumption.")
      print("Repo: https://github.com/fmassa/optimize-net")
   end
end

function utils.makeDataParallel(model, nGPU)
   -- Wrap the model with DataParallelTable, if using more than one GPU
   if nGPU > 1 then
      local gpus = torch.range(1, nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            require ("dpnn")
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end
   return model
end

function utils.initWeight(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
--       local n = v.kW*v.kH*v.nOutputPlane
--       v.weight:normal(0,math.sqrt(2/n))
--       if opt.init == "orthogonal" then
--          v:init('weight', nninit.orthogonal, {gain = 'relu'})
--       elseif opt.init == "msra"  then
--          v:init('weight', nninit.kaiming, {dist = 'uniform',gain = 'relu'})
--       elseif opt.init == "xavier" then
--          v:init('weight', nninit.xavier, {dist = 'normal'})
--       elseif opt.init == "gaussian" then
--          v:init('weight', nninit.addNormal, 0.0, 0.01) -- mean and std-dev
--       end
      -- if v.bias then v.bias:zero() end

--       local n = v.kW*v.kH*v.nOutputPlane
--       v.weight:normal(0,math.sqrt(2/n))
--       if cudnn.version >= 4000 then
--          v.bias = nil
--          v.gradBias = nil
--       else
--          v.bias:zero()
--       end
   end

   for k,v in pairs(model:findModules(nn.SpatialBatchNormalization)) do
      v.weight:fill(1)
      v.bias:zero()
   end
end

function utils.FCinit(model)
   for k,v in pairs(model:findModules'nn.Linear') do
     v.bias:zero()
   end
end

function utils.DisableBias(model)
   for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
      v.bias = nil
      v.gradBias = nil
   end
end

function utils.testModel(model)
   model:float()
   local imageSize = opt and opt.imgDim or 96
   local input = torch.randn(1,3,imageSize,imageSize):type(model._type)
   print('forward output',{model:forward(input)})
   print('backward output',{model:backward(input,model.output)})
   model:reset()
end

function utils.splitString(inputstr, sep)
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

return utils
