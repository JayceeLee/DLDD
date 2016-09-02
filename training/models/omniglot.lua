-- "Siamese Neural Networks for One-shot Image Recognition", Gregory Koch


imgDim = 90 --105
local sizeOut = 4 --6
function createModel()
    local lenet = nn.Sequential()
    -- stage 1 
    lenet:add(nn.SpatialConvolution(3, 64, 10, 10))
    lenet:add(nn.ReLU())
    lenet:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    -- stage 2 
    lenet:add(nn.SpatialConvolution(64, 128, 7, 7))
    lenet:add(nn.ReLU())
    lenet:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    -- stage 3 
    lenet:add(nn.SpatialConvolution(128, 128, 4, 4))
    lenet:add(nn.ReLU())
    lenet:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    -- stage 4: embeding
    lenet:add(nn.SpatialConvolution(128, 256, 4, 4)) --256x9x9
    lenet:add(nn.ReLU())
    lenet:add(nn.View(256*sizeOut*sizeOut))
    lenet:add(nn.Linear(256*sizeOut*sizeOut, opt.embSize))

    return lenet
end