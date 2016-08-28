local M = { }

-- http://stackoverflow.com/questions/6380820/get-containing-path-of-lua-file
function script_path()
   local str = debug.getinfo(2, "S").source:sub(2)
   return str:match("(.*/)")
end

function M.parse(arg)

   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('DLDD')
   cmd:text()
   cmd:text('Options:')

   ------------ General options --------------------
   cmd:option('-manualSeed', 2, 'Manually set RNG seed')
   cmd:option('-cuda', true, 'Use cuda.')
   cmd:option('-device', 1, 'Cuda device to use.')
   cmd:option('-nGPU',   1,  'Number of GPUs to use by default')
   cmd:option('-cudnn', true, 'Convert the model to cudnn.')
   cmd:option('-cudnn_bench', true, 'Run cudnn to choose fastest option. Increase memory usage')

   ------------- Data options ------------------------
   cmd:option('-nDonkeys', 1, 'number of donkeys to initialize (data loading threads)')
   cmd:option('-dataset',    'imageData', 'Options: imageData | fileData')
   cmd:option('-data', '', 'Home of dataset. Images separated by train/val or two hdf5 files')
   cmd:option('-cache', paths.concat(script_path(), 'work'), 'Directory to cache experiments and data.')
   cmd:option('-name', '', 'Name of experiment')
   cmd:option('-save', '', 'Directory to save experiment.')
   cmd:option('-nClasses',         0,      'Number of classes in the dataset')
   cmd:option('-samplePeople',  'grouped', 'Sample people using peoplePerBatch and imagesPerPerson, grouped or cluster')
   cmd:option('-clusterLR',         0.5,      'Learning rate for cluster center')
   ------------- Training options --------------------
   cmd:option('-nEpochs', 100, 'Number of total epochs to run')
   cmd:option('-epochSize', 250, 'Number of batches per epoch')
   cmd:option('-epochNumber', 1, 'Manual epoch number (useful on restarts)')
   -- GPU memory usage depends on peoplePerBatch and imagesPerPerson.
   cmd:option('-peoplePerBatch', 10, 'Number of people to sample in each mini-batch.')
   cmd:option('-imagesPerPerson', 6, 'Number of images to sample per person in each mini-batch.')
   cmd:option('-testing', true, 'Test using verification pairs')
   cmd:option('-init', 'msra', 'Algorithm used for initialize the weight of Conv Layers: msra,xavier,gaussian,orthogonal')
     ---------- Optimization options ----------------------
   cmd:option('-optimization',    'sgd',  'optimization method')
   cmd:option('-LR',              0.001,  'initial learning rate')
   cmd:option('-LRDecay',         30,     'get 10x smaller LR after period or automatic decay after period of no progress (set negative value)')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-weightDecay',     5e-4,  'weight decay')
   ---------- Model options ----------------------------------
   cmd:option('-retrain', 'none', 'provide path to model to retrain with')
   cmd:option('-modelDef', 'models/pnnet.lua', 'path to model definiton')
   cmd:option('-imgDim', 28, 'Image dimension')
   cmd:option('-embSize', 128, 'size of embedding from model')
   cmd:option('-checkEpoch', 15, 'check if model get > -checkValue after this epoch. If not, mean that no reason to learn more')
   cmd:option('-checkValue', 0.7, 'minimul value after some tries to continue')
   ---------- Loss options ---------------
   ---------- Raw Features module
   cmd:option('-Center', 0.0, 'CenterLoss')
   ----------Classification  module
   cmd:option('-SoftMax', 0.0, 'SoftMax for Classification, need nClasses')
   ----------Pair module
   cmd:option('-Constr', 0.0, 'Contrastive Loss')
   cmd:option('-Mulbatch', 0.0, 'MultiBatch Loss')
   cmd:option('-Cosine', 0.0, 'CosineEmbeding Loss')
   ----------Triplet module
   cmd:option('-Triplet', 0.0, 'Triplet Loss')
   cmd:option('-alpha', 0.2, 'margin in TripletLoss')
   cmd:option('-Triratio', 0.0, 'Triplet Embedding-Ratio Loss')
   cmd:option('-Global', 0.0, 'Global Triplet Loss')
   cmd:option('-Trisim', 0.0, 'TripletSimilarity Loss')
   cmd:option('-Lifted', 0.0, 'LiftedStructured Loss')
   cmd:option('-Triprob', 0.0, 'TripletProbability Loss')
   cmd:text()
  
   
   
   local opt = cmd:parse(arg or {})
   opt.batchSize = opt.peoplePerBatch * opt.imagesPerPerson
   os.execute('mkdir -p ' .. opt.cache)

   if opt.save == '' then
      opt.save = paths.concat(opt.cache, opt.name .. "_" .. os.date("%Y-%m-%d_%H-%M-%S"))
   end
   os.execute('mkdir -p ' .. opt.save)

   return opt
end

return M
