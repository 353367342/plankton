require 'nn'
require 'cutorch'
require 'cunn'

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(1,96,7,7,2,2)) -- 61 
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 30
features:add(nn.SpatialConvolutionMM(96,256,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 15
features:add(nn.SpatialConvolutionMM(256,512,3,3,1,1,1)) --15
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --15
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 7
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --7
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --7
features:add(nn.ReLU())
features:add(nn.SpatialAdaptiveMaxPooling(4,4)) -- 4
features:cuda()

featuresOut = 512*4*4

graph = nn.Sequential()
graph:add(nn.View(512*16))
graph:add(nn.Dropout(0.5))
graph:add(nn.Linear(512*16,6000))
graph:add(nn.View(2,3000))
graph:add(nn.TemporalMaxPooling(2))
graph:add(nn.View(3000))
graph:add(nn.Linear(3000,121))
graph:add(nn.LogSoftMax())
graph:cuda()

mdl = nn.Sequential()
mdl:cuda()
mdl:add(features)
mdl:add(graph)
mdl:add(nn.LogSoftMax():cuda())

--x = mdl:forward(torch.randn(64,1,256,256):cuda())
--print(x:size())
