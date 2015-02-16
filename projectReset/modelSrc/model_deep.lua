
require 'nn'
require 'cunn'
require 'cutorch'
require 'image'

-- start with 1 x 52 x 52
fSize = {16,32,64,128,128,512,1024,121}

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(1,fSize[1],10,10,5,5)) -- 9
features:add(nn.Dropout(0.75))
features:add(nn.SpatialConvolutionMM(fSize[1],fSize[2],2,2)) -- 8
features:add(nn.Dropout(0.75))
features:add(nn.SpatialConvolutionMM(fSize[2],fSize[3],2,2)) -- 7
features:add(nn.Dropout(0.75))
features:add(nn.SpatialConvolutionMM(fSize[3],fSize[4],3,3)) -- 5
features:add(nn.Dropout(0.75))
features:add(nn.SpatialConvolutionMM(fSize[4],fSize[5],2,2)) -- 4
features:add(nn.Dropout(0.75))
features:add(nn.SpatialConvolutionMM(fSize[5],fSize[6],2,2)) -- 3
features:add(nn.Dropout(0.75))
features:add(nn.SpatialConvolutionMM(fSize[6],fSize[7],2,2)) -- 2
features:add(nn.Dropout(0.5))
features:add(nn.Linear(fSize[7],fSize[8])) -- 1
features:add(nn.View(121))
features:add(nn.LogSoftMax())

features:cuda()

mdl = features

print(features:forward(torch.randn(2,1,52,52):cuda()))
