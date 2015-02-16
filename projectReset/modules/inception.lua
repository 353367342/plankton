-- inception 1x1,1x1+3x3,1x1+5x5,poolproj module
-- inputSize is first dimension
-- depthDim is the index of the dimension it concatenates, zero pads the other dimensions if size mismatch
-- generally want depthDim to be 2
-- 1st dim is batch
function inception_module(depthDim, inputSize, config)
   local conv1 = nil   
   local conv3 = nil
   local conv5 = nil
   local pool = nil
   local depthConcat = nn.DepthConcat(depthDim)
   conv1 = nn.Sequential()
   conv1:add(nn.SpatialConvolutionMM(inputSize, config[1][1], 1, 1))
   conv1:add(nn.ReLU())
   depthConcat:add(conv1)
   conv3 = nn.Sequential()
   conv3:add(nn.SpatialConvolutionMM(inputSize, config[2][1], 1, 1))
   conv3:add(nn.ReLU())
   conv3:add(nn.SpatialConvolutionMM(config[2][1], config[2][2], 3, 3))
   conv3:add(nn.ReLU())
   depthConcat:add(conv3)
   conv5 = nn.Sequential()
   conv5:add(nn.SpatialConvolutionMM(inputSize, config[3][1], 1, 1))
   conv5:add(nn.ReLU())
   conv5:add(nn.SpatialConvolutionMM(config[3][1], config[3][2], 5, 5))
   conv5:add(nn.ReLU())
   depthConcat:add(conv5)
   pool = nn.Sequential()
   pool:add(nn.SpatialMaxPooling(config[4][1], config[4][1], 1, 1))
   pool:add(nn.SpatialConvolutionMM(inputSize, config[4][2], 1, 1))
   pool:add(nn.ReLU())
   depthConcat:add(pool)
   
   return depthConcat
end

-- ex: model:add(inception_module(2, 64, {{64}, {96, 128}, {16, 32}, {3, 32}}))
