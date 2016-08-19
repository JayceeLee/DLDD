-- "Deep Learning Face Representation by Joint Identification-Verification",Yi Sun

--model:forward(torch.FloatTensor(10,3,55,55)):size()
imgDim = 55
function createModel()
    local model = nn.Sequential()
    -- stage 1 
    model:add(nn.SpatialConvolution(3, 20, 4, 4, 1, 1, 0, 0))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1,0,0))
    -- stage 2 
    model:add(nn.SpatialConvolution(20, 40, 3, 3, 1, 1,0,0))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1,0,0))
    -- stage 3 
    model:add(nn.SpatialConvolution(40, 60, 3, 3, 1, 1,0,0))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1,0,0))
    -- stage 4: embeding
    local featureBlock = nn.ConcatTable()
    featureBlock:add(nn.View(2160))
    featureBlock:add(nn.Sequential():add(nn.SpatialConvolution(60, 80, 2, 2, 1, 1,0,0)):add(nn.ReLU()):add(nn.View(2000)))
    model:add(featureBlock)
    model:add(nn.JoinTable(2))
    
    model:add(nn.Linear(4160, opt.embSize))

    return model
end 
