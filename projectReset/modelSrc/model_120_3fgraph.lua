require 'nn'
require 'cunn'
require 'cutorch'
require 'image'

fSize = {1,96,128,256,256}

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(fSize[1],fSize[2],8,8,2,2)) -- (120 - 8+ 2)/2 = 57
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2)) -- 28
features:add(nn.SpatialConvolutionMM(fSize[2],fSize[3],3,3)) -- 26
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2)) -- 13
features:add(nn.SpatialConvolutionMM(fSize[3],fSize[4],4,4)) -- 10 
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[4],fSize[5],3,3)) -- 8
features:cuda()

dropout_p = 0.5

dgraph = fgraph(fSize[5],256,64,0.5)
dgraph:cuda()

mdl = nn.Sequential()
mdl:cuda()
mdl:add(features)
mdl:add(dgraph)
mdl:add(nn.LogSoftMax():cuda())
