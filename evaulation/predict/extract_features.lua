require 'xlua'
require 'optim'
require 'torchx'
require 'cunn'
require 'cutorch'
require 'nn'
require 'cudnn'
require 'dataset'
require 'image'
require 'hdf5'
optnet = require 'optnet'
local nCrops = 1

opt = lapp[[
   -v,--valData               (default "val.txt")       path to txt file <path> <label>
   -b,--batchSize             (default 64)          batch size
   --model                    (default lenet)     trained model
   --manualSeed               (default 10)        seed to reproduce results
   --GPU                      (default 1)         GPU ID
   --aug                      (default 1)         get results by TTA, set number of augumentation
   --save                     (default 'predictions.h5')         name of prediction
   --normalize                (default false)         normalize features by L2
   --imgDim                   (default 96)         size of image input to network
]]
torch.manualSeed(opt.manualSeed)
cutorch.manualSeed(opt.manualSeed)
cutorch.setDevice(opt.GPU)
print(opt)

local  data = ImageDataset(opt.valData,opt,'val')
data.preprocess_image = data:preprocess()
local  model = torch.load(opt.model):clearState():cuda()
local normalize = nn.Identity():float()
if opt.normalize == 'true' then
  print ("Normalize features by L2")
  normalize = nn.Normalize(2):float()
end

optnet.optimizeMemory(model, torch.CudaTensor(10,3,opt.imgDim,opt.imgDim))

function loadImages(indices, data )
  local sz = indices:size(1)
  local batch, imageSize
  local target = torch.IntTensor(sz)
  for i, idx in ipairs(indices:totable()) do
    local sample = data:get(idx)
    local input = data.preprocess_image(sample.input)
    if not batch then
      imageSize = input:size():totable()
      batch = torch.FloatTensor(sz , table.unpack(imageSize))
    end
    batch[i]:copy(input)
    target[i] = sample.target
  end
  return batch, target
end



function testData(data)
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print('==>'.." testing")
  local N = 0
  local outputs = {}
  local target = {}
  local indices     = torch.linspace(1,data.targets:size(1), data.targets:size(1)):long():split(math.floor(opt.batchSize / opt.aug))
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
    batch, tar = loadImages(v,data)
    local out      = model:forward(batch:cuda()):float()
    table.insert(outputs, out)
    table.insert(target,  tar)
  end
  outputs            = torch.concat(outputs):float()
  target             = torch.concat(target):float()
  return normalize:forward(outputs), target
end

local out,target = testData(data)

local myFile = hdf5.open(opt.save, 'w')
myFile:write('prediction',out)
myFile:write('target',target)
myFile:close()




