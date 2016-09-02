require 'nn'
require 'cunn'
require 'cudnn'
imgDim = 96
function createModel()
    local depth = 19
    if (depth - 4 ) % 3 ~= 0 then
      error("Depth must be 3N + 4!")
    end

    --#layers in each denseblock
    local N = (depth - 4)/3

    --growth rate
    local growthRate = 12

    --dropout rate, set it to nil to disable dropout, non-zero number to enable dropout and set drop rate
    local dropRate = nil

    --#channels before entering the first denseblock
    --set it to be comparable with growth rate
    local nChannels = 16

    local function addLayer(model, nChannels, nOutChannels, dropRate)
      concate = nn.Concat(2)
      concate:add(nn.Identity())

      convFactory = nn.Sequential()
      convFactory:add(cudnn.SpatialBatchNormalization(nChannels))
      convFactory:add(cudnn.ReLU(true))
      convFactory:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 3, 3, 1, 1, 1,1))
      if dropRate then
        convFactory:add(nn.Dropout(dropRate))
      end
      concate:add(convFactory)
      model:add(concate)
    end

    local function addTransition(model, nChannels, nOutChannels, dropRate)
      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))
      model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
      if dropRate then
        model:add(nn.Dropout(dropRate))
      end
      model:add(cudnn.SpatialAveragePooling(2, 2))
    end

    print("Building model")
    model = nn.Sequential()
    model:add(cudnn.SpatialConvolution(3,nChannels,7,7,2,2,3,3))
--     model:add(cudnn.SpatialConvolution(3, nChannels, 3,3, 1,1, 1,1))

    for i=1, N do 
      addLayer(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, nChannels, dropRate)

    for i=1, N do
      addLayer(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, nChannels, dropRate)

    for i=1, N do
      addLayer(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end

    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialAveragePooling(8,8))
    model:add(nn.View(nChannels):setNumInputDims(3))
    model:add(nn.Linear(nChannels, opt.embSize))
   
   
    return model
end
