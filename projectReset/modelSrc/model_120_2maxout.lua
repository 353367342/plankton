require 'nn'
require 'cunn'
require 'cutorch'
require 'image'

fSize = {1,96,256,256,512,512,512}
featuresOut = fSize[7]*2*2

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(fSize[1],fSize[2],2,2,2,2)) -- (120 - 2 + 2)/2 = 60
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2)) -- 30
features:add(nn.SpatialConvolutionMM(fSize[2],fSize[3],3,3)) -- 28
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2)) -- 14
features:add(nn.SpatialConvolutionMM(fSize[3],fSize[4],4,4)) -- 11 
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[4],fSize[5],4,4)) -- 8
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[5],fSize[6],3,3)) -- 6
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[6],fSize[7],3,3)) -- 4
features:add(nn.SpatialMaxPooling(2,2)) -- 2
features:add(nn.ReLU())
features:add(nn.View(featuresOut))
features:cuda()

dgraph = nn.Sequential()
dgraph:add(nn.Dropout(0.5))
dgraph:add(mnn(featuresOut,featuresOut,3))
dgraph:add(nn.Dropout(0.5))
dgraph:add(mnn(featuresOut,121,3))
dgraph:cuda()

mdl = nn.Sequential()
mdl:cuda()
mdl:add(features)
mdl:add(dgraph)
mdl:add(nn.LogSoftMax():cuda())
