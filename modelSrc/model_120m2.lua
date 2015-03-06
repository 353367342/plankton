require 'nn'
require 'cunn'
require 'cutorch'
require 'image'

fSize = {1,512,512,564,600,632,700,762,800,892,962}
featuresOut = fSize[11]

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(fSize[1],fSize[2],2,2,2,2)) -- (120 - 2 + 2)/2 = 60
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[2],fSize[3],2,2,2,2)) -- 30
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[3],fSize[4],3,3)) -- 28
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[4],fSize[5],2,2,2,2)) --14
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[5],fSize[6],4,4)) -- 11 
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[6],fSize[7],4,4)) -- 8
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[7],fSize[8],3,3)) -- 6
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[8],fSize[9],3,3)) -- 4
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[9],fSize[10],3,3)) -- 2
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[10],fSize[11],2,2)) -- 1
features:add(nn.ReLU())
features:add(nn.View(featuresOut))
features:cuda()

dropout_p = 0.5

dgraph = nn.Sequential()
dgraph:add(nn.Linear(featuresOut,featuresOut))
dgraph:add(nn.Dropout(dropout_p))
dgraph:add(nn.ReLU())
dgraph:add(nn.Linear(featuresOut,121))
dgraph:cuda()

mdl = nn.Sequential()
mdl:cuda()
mdl:add(features)
mdl:add(dgraph)
mdl:add(nn.LogSoftMax():cuda())
