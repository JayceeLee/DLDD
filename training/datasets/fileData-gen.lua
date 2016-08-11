require 'hdf5'
local M = {}

function unique(input)
  local b = {}
  for i=1,input:numel() do
     b[input[i]] = true
   end
  local out = {}
  for i in pairs(b) do
      table.insert(out,i)
   end
  return out
end

function M.exec(opt, cacheFile)
   print("=> Loading the HDF5 files")
   
   local trainData = {}
   local dataFile = hdf5.open(paths.concat(opt.data, 'train.hdf5'), 'r')
   trainData.data   = dataFile:read('data'):all()
   trainData.labels = dataFile:read('labels'):all()
   local testData = {}
   local dataFile = hdf5.open(paths.concat(opt.data, 'val.hdf5'), 'r')
   trainData.data   = dataFile:read('data'):all()
   trainData.labels = dataFile:read('labels'):all()
   
   print(" | saving HDF5 dataset to " .. cacheFile)
   local info = {
      basedir = opt.data,
      classList = unique(trainData.labels),
      train = trainData,
      val = testData
   }
   torch.save(cacheFile, info)
end

return M