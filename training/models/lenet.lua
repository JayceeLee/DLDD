-- LeNet++ from "A Discriminative Feature Learning Approach for Deep Face Recognition",Yandong Wen, Kaipeng Zhang, Zhifeng Li and Yu Qiao


imgDim = 28
function createModel()
    local lenet = nn.Sequential()
    -- stage 1 
    lenet:add(nn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
    lenet:add(nn.PReLU())
    lenet:add(nn.SpatialConvolution(32, 32, 5, 5, 1, 1, 2, 2))
    lenet:add(nn.PReLU())
    lenet:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
    -- stage 2 
    lenet:add(nn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
    lenet:add(nn.PReLU())
    lenet:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, 2, 2))
    lenet:add(nn.PReLU())
    lenet:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
    -- stage 3 
    lenet:add(nn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
    lenet:add(nn.PReLU())
    lenet:add(nn.SpatialConvolution(128, 128, 5, 5, 1, 1, 2, 2))
    lenet:add(nn.PReLU())
    lenet:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
    -- stage 4: embeding
    lenet:add(nn.View(128*5*5))
    lenet:add(nn.Linear(128*5*5, opt.embSize))

    return lenet
end