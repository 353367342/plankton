require 'nn'
require 'cutorch'
require 'cunn'

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(1,96,7,7,2,2)) -- 61 
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 30
features:add(nn.Dropout(0.05))
features:add(nn.SpatialConvolutionMM(96,256,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 15
features:add(nn.Dropout(0.05))
features:add(nn.SpatialConvolutionMM(256,512,3,3,1,1,1)) --15
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --15
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --15
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --15
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --15
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --15
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 7
features:add(nn.Dropout(0.05))
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --7
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --7
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --7
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --7
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --7
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --7
features:add(nn.ReLU())
features:add(nn.SpatialAdaptiveMaxPooling(4,4)) -- 4
features:cuda()

dgraph = nn.Sequential()
dgraph:add(nn.Dropout(0.5))
dgraph:add(nn.Linear(512*16,512*16))
features:add(nn.ReLU():cuda())
dgraph:add(nn.Linear(512*16,121))
dgraph:add(nn.LogSoftMax())
dgraph:cuda()

mdl = nn.Sequential()
mdl:cuda()
mdl:add(features)
mdl:add(nn.View(512*16):cuda())
mdl:add(dgraph)

--x = mdl:forward(torch.randn(64,1,256,256):cuda())
--print(x:size())