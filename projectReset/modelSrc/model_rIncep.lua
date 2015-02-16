require 'inception'
require 'nn'
require 'cutorch'
require 'cunn'

fSize = {1,256,300,324,512,640,692,1024}
featuresOut = fSize[8]*2*2

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(fSize[1],fSize[2],2,2,2,2)) -- (120 - 2 + 2)/2 = 60
features:add(nn.Threshold())
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2)) -- 30
features:add(nn.SpatialConvolutionMM(fSize[2],fSize[3],3,3)) -- 28
features:add(nn.Threshold())
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2)) -- 14
features:add(inception_module(2,fSize[3],{{64}, {96, 164}, {16, 48}, {3, 48}})) -- fSize[4] x 14 x 14 
features:add(nn.SpatialConvolutionMM(fSize[4],fSize[4],4,4)) -- 11 
features:add(nn.Threshold())
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[4],fSize[5],4,4)) -- 8
features:add(nn.Threshold())
features:add(nn.ReLU()) -- 512 8 8 
features:add(inception_module(2,fSize[5],{{128}, {164, 256}, {64, 128}, {3, 128}})) -- 640 x 8 x 8 
features:add(nn.SpatialConvolutionMM(fSize[6],fSize[6],2,2)) -- 7
features:add(nn.Threshold())
features:add(nn.ReLU()) -- 640 6 6
features:add(nn.SpatialConvolutionMM(fSize[6],fSize[7],2,2)) -- 6
features:add(inception_module(2,fSize[7],{{256}, {164, 256}, {128, 256}, {3, 256}})) -- 1024 x 4 x4
features:add(nn.SpatialMaxPooling(3,3)) -- 2
features:add(nn.ReLU())
features:add(nn.View(featuresOut))
features:cuda()

mdl = nn.Sequential()
mdl:add(features)
mdl:add(nn.Dropout(0.5))
mdl:add(nn.Linear(featuresOut,featuresOut))
mdl:add(nn.Dropout(0.5))
mdl:add(nn.ReLU())
mdl:add(nn.Linear(featuresOut,121))
mdl:add(nn.LogSoftMax())
mdl:cuda()
