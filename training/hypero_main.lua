require 'hypero'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('DLDD')
cmd:text()
cmd:text('Options:')

------------ General options --------------------
cmd:option('-manualSeed',  22, 'Manually set RNG seed')
cmd:option('-cuda',     true, 'Use cuda.')
cmd:option('-device',      1, 'Cuda device to use.')
cmd:option('-nGPU',        1,  'Number of GPUs to use by default')
cmd:option('-cudnn',       true, 'Convert the model to cudnn.')
cmd:option('-cudnn_bench', true, 'Run cudnn to choose fastest option. Increase memory usage')

------------- Data options ------------------------
cmd:option('-nDonkeys',     1, 'number of donkeys to initialize (data loading threads)')
cmd:option('-dataset',      'imageData', 'Options: imageData | fileData')
cmd:option('-data',         '../MNIST/', 'Home of dataset. Images separated by train/val or two hdf5 files')
cmd:option('-cache',        'hyper_mnist', 'Directory to cache experiments and data.')
cmd:option('-name',         'Hyper', 'Name of experiment')
cmd:option('-save',         '', 'Directory to save experiment.')
cmd:option('-nClasses',     10,      'Number of classes in the dataset')
cmd:option('-samplePeople', 'grouped', 'Sample people using peoplePerBatch and imagesPerPerson, grouped or cluster')
cmd:option('-clusterLR',     0.5,      'Learning rate for cluster center')
------------- Training options --------------------
cmd:option('-nEpochs',     50, 'Number of total epochs to run')
cmd:option('-epochSize',   300, 'Number of batches per epoch')
cmd:option('-epochNumber', 1, 'Manual epoch number (useful on restarts)')
-- GPU memory usage depends on peoplePerBatch and imagesPerPerson.
cmd:option('-peoplePerBatch', 10, 'Number of people to sample in each mini-batch.')
cmd:option('-imagesPerPerson', 6, 'Number of images to sample per person in each mini-batch.')
cmd:option('-testing',         true, 'Test using verification pairs')
cmd:option('-init', 'msra', 'Algorithm used for initialize the weight of Conv Layers: msra,xavier,gaussian,orthogonal')
 ---------- Optimization options ----------------------
cmd:option('-optimization',    'sgd',  'optimization method')
cmd:option('-LR',              0.01,  'initial learning rate')
cmd:option('-LRDecay',         25,     'get 10x smaller LR after period')
cmd:option('-LRDecay',         30,     'get 10x smaller LR after period or automatic decay after period of no progress (set negative value)')
cmd:option('-momentum',        0.9,   'momentum')
cmd:option('-weightDecay',     5e-4,  'weight decay')
---------- Model options ----------------------------------
cmd:option('-retrain',  'none', 'provide path to model to retrain with')
cmd:option('-modelDef', 'models/pnnet.lua', 'path to model definiton')
cmd:option('-imgDim',    28, 'Image dimension')
cmd:option('-embSize',   128, 'size of embedding from model')
cmd:option('-checkEpoch', 15, 'check if model get > -checkValue after this epoch. If not, mean that no reason to learn more')
cmd:option('-checkValue', 0.7, 'minimul value after some tries to continue')
---------- Loss options ---------------
---------- Raw Features module
cmd:option('-Center', '{0.0,1.0}', 'CenterLoss, Uniform')
----------Classification  module
cmd:option('-SoftMax', '{0.0,2.0}', 'SoftMax for Classification, need nClasses, Uniform')
----------Pair module
cmd:option('-Constr', '{0.0,10.0}', 'Contrastive Loss Uniform')
cmd:option('-Mulbatch', '{0.0,10.0}', 'MultiBatch Loss Uniform' )
cmd:option('-Cosine', 0.0, 'CosineEmbeding Loss')
-- cmd:option('-Cosine', '{0.0,10.0}',, 'CosineEmbeding Loss')
----------Triplet module
cmd:option('-Triplet', '{0.0,10.0}', 'Triplet Loss Uniform')
cmd:option('-alpha', 0.2, 'margin in TripletLoss Uniform')
-- cmd:option('-Triratio', '{0.0,10.0}', 'Triplet Embedding-Ratio Loss')
cmd:option('-Global', '{0.0,10.0}', 'Global Triplet Loss Uniform')
cmd:option('-Trisim', '{0.0,10.0}', 'TripletSimilarity Loss Uniform')
cmd:option('-Lifted', '{0.0,10.0}', 'LiftedStructured Loss Uniform')
cmd:option('-Triprob', '{0.0,10.0}', 'TripletProbability Loss Uniform')
cmd:option('-Triratio', 0.0, 'Triplet Embedding-Ratio Loss')
-- HyperOpt
cmd:option('--batteryName', 'hypero mnist', "name of battery of experiments to be run")
cmd:option('--maxHex', 100, 'maximum number of hyper-experiments to train (from this script)')
cmd:text()

hopt = cmd:parse(arg or {})


function returnString(str)
   return loadstring(" return "..str)()
end


hopt.Center   = returnString(hopt.Center)
hopt.SoftMax  = returnString(hopt.SoftMax)
hopt.Constr   = returnString(hopt.Constr)
hopt.Mulbatch = returnString(hopt.Mulbatch)
hopt.Triplet  = returnString(hopt.Triplet)
hopt.Global   = returnString(hopt.Global)
hopt.Trisim   = returnString(hopt.Trisim)
hopt.Lifted   = returnString(hopt.Lifted)
hopt.Triprob  = returnString(hopt.Triprob)

hopt.versionDesc = "Mnist Experiments"

local checkpoints = require 'checkpoints'
local DataLoader = require 'dataloader'


if hopt.cuda then
   require 'cutorch'
   cutorch.setDevice(hopt.device)
end
torch.setdefaulttensortype('torch.FloatTensor')
os.execute('mkdir -p ' .. hopt.cache)

opt = hopt


function  saveJson(fileName, data)
   local file = assert(io.open(fileName, "w"))
   file:write(json.encode.encode(data))
   file:close()
end

-- this allows the hyper-param sampler to be bypassed via cmd-line
function ntbl(param)
   return torch.type(param) ~= 'table' and param
end

function buildExperiment(opt)
   -- Data loading
   centerCluster  = torch.rand(opt.nClasses, opt.embSize)
   local trainLoader, valLoader = DataLoader.create(opt, centerCluster)
   local models = require 'model'
   local modelConfig = models.ModelConfig()
   modelConfig:generateConfig(opt)
   middleBlock = modelConfig:middleBlockSetup(opt)
   criterion   = modelConfig:critertionSetup(opt)

   return trainLoader, valLoader, modelConfig
end

function runExperiment( trainLoader, valLoader, modelConfig, opt)
   local Trainer = require 'train'
   local trainer = Trainer(opt)
   local Test    = require 'testVerify'
   local tester = Test(opt)
   model = nil
   epoch = 1
   testData = {}
   testData.bestVerAcc = 0
   testData.bestEpoch = 0
   testData.testVer = 0
   for e = opt.epochNumber, opt.nEpochs do
      local sucess = trainer:train(trainLoader, modelConfig)
      if not sucess then return false, Test.testVerTable end
   --    model = Trainer:saveModel(model)
      testData.testVer = tester:test(valLoader)
      local bestModel = false
      if testData.bestVerAcc < testData.testVer then
         print(' * Best model ', testData.testVer)
         bestModel = true
         testData.bestVerAcc = testData.testVer
         testData.bestEpoch  = epoch
      end
      model = checkpoints.save(epoch, model, optimState, bestModel, opt)
      if opt.checkEpoch > 0 and epoch > opt.checkEpoch then
         if testData.bestVerAcc < opt.checkValue then return false,Test.testVerTable end -- model does not converge, break it
      end
      epoch = epoch + 1
   end
   return true, Test.testVerTable
end

--[[hypero]]--

conn = hypero.connect()
bat = conn:battery(hopt.batteryName, hopt.versionDesc)
hs = hypero.Sampler()
local optData = _
-- loop over experiments
for i=1,hopt.maxHex do
   collectgarbage()
   local hex = bat:experiment()
   opt = optData.clone(hopt) 

   -- hyper-parameters
   local hp = {}

   hp.Center  = ntbl(opt.Center)  or hs:uniform(tonumber(opt.Center[1]), tonumber(opt.Center[2]))
   hp.SoftMax = ntbl(opt.SoftMax) or hs:uniform(tonumber(opt.SoftMax[1]), tonumber(opt.SoftMax[2]))
   hp.Constr  = ntbl(opt.Constr)  or hs:randint(tonumber(opt.Constr[1]), tonumber(opt.Constr[2]))
   hp.Mulbatch = ntbl(opt.Mulbatch) or hs:randint(tonumber(opt.Mulbatch[1]), tonumber(opt.Mulbatch[2]))
   hp.Triplet  = ntbl(opt.Triplet) or hs:randint(tonumber(opt.Triplet[1]), tonumber(opt.Triplet[2]))
   hp.Global   = ntbl(opt.Global)  or hs:randint(tonumber(opt.Global[1]), tonumber(opt.Global[2]))
   hp.Trisim   = ntbl(opt.Trisim)  or hs:randint(tonumber(opt.Trisim[1]), tonumber(opt.Trisim[2]))
   hp.Lifted   = ntbl(opt.Lifted)  or hs:randint(tonumber(opt.Lifted[1]), tonumber(opt.Lifted[2]))
   hp.Triprob  = ntbl(opt.Triprob) or hs:randint(tonumber(opt.Triprob[1]), tonumber(opt.Triprob[2]))
   hp.save = paths.concat(hopt.cache, hopt.name .. "_" .. os.date("%Y-%m-%d_%H-%M-%S"))
   for k,v in pairs(hp) do opt[k] = v end
   print('Saving everything to: ' .. opt.save)
   torch.manualSeed(opt.manualSeed)
   opt.batchSize = opt.peoplePerBatch * opt.imagesPerPerson
   os.execute('mkdir -p ' .. opt.save)
   saveJson(paths.concat(opt.save, 'opts.json'), opt)

   -- meta-data
   local md = {}
   -- md.hostname = os.hostname() --require dp
   md.modelstr = tostring(model)
   hex:setMeta(md)

   local trainLoader, valLoader, modelConfig = buildExperiment(opt)
   local success, testVerTable = runExperiment(trainLoader, valLoader, modelConfig, opt)
   -- results
   -- if success then
      local res = {}
      res.tesVer = testVerTable
      hex:setResult(res)
   -- end


end