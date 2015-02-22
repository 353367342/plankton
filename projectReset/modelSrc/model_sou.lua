require 'nn'
require 'cutorch'
require 'cunn'

fSize = {1, 96, 256, 256, 256}
featuresOut = fSize[5] * 2 * 2

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(fSize[1], fSize[2], 8, 8, 2, 2)) -- (120 - 8 + 2)/2 = 57
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2)) -- 28
features:add(nn.SpatialConvolutionMM(fSize[2], fSize[3], 5, 5)) -- 24
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 12
features:add(nn.SpatialConvolutionMM(fSize[3], fSize[4], 4, 4)) -- 9
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[4], fSize[5], 3, 3)) -- 8
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(3,3)) -- 2
features:add(nn.View(featuresOut))
features:cuda()

dgraph = nn.Sequential()
dgraph:add(nn.Dropout(0.5))
dgraph:add(nn.Linear(featuresOut,featuresOut))
dgraph:add(nn.Dropout(0.5))
dgraph:add(nn.ReLU())
dgraph:add(nn.Linear(featuresOut,121))
dgraph:add(nn.LogSoftMax())
dgraph:cuda()

mdl = nn.Sequential()
mdl:cuda()
mdl:add(features)
mdl:add(dgraph)

