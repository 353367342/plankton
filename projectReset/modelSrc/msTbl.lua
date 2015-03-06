require 'nn'
require 'cunn'

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(1,128,7,7,2,2)) -- 61 
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 30
features:add(nn.Dropout(0.1))
features:add(nn.SpatialConvolutionMM(128,512,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.Dropout(0.1))
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 15
features:add(nn.Dropout(0.1))
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --15
features:add(nn.ReLU())
features:add(nn.Dropout(0.1))
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --15
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 7
features:add(nn.Dropout(0.1))
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --7
features:add(nn.ReLU())
features:add(nn.Dropout(0.1))
features:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) --7
features:add(nn.ReLU())
features:add(nn.SpatialAdaptiveMaxPooling(4,4)) -- 4
features:cuda()

featuresOut = 512*4*4

p1 = nn.Sequential()
p1:cuda()
p1:add(features)

p2 = p1:clone()
p2:share(p1, "weight", "bias", "gradWeight", "gradBias")
p3 = p1:clone()
p3:share(p1, "weight", "bias", "gradWeight", "gradBias")

tbl = nn.ParallelTable()
tbl:cuda()
tbl:add(p1)
tbl:add(p2)
tbl:add(p3)

graph = nn.Sequential()
graph:add(nn.Dropout(0.5))
graph:add(nn.Linear(featuresOut*3,6000))
graph:add(nn.Transpose({2,1}))
graph:add(nn.TemporalMaxPooling(2))
graph:add(nn.Transpose({2,1}))
graph:add(nn.Dropout(0.5))
graph:add(nn.Linear(3000,6000))
graph:add(nn.Transpose({2,1}))
graph:add(nn.TemporalMaxPooling(2))
graph:add(nn.Transpose({2,1}))
graph:add(nn.Dropout(0.5))
graph:add(nn.Linear(3000,121))
graph:add(nn.LogSoftMax())
graph:cuda()

mdl = nn.Sequential()
mdl:add(tbl)
mdl:add(nn.JoinTable(2))
mdl:add(nn.View(featuresOut*3))
mdl:add(graph)
mdl:cuda()

-- x = torch.randn(32,1,128,128):cuda()
-- y = torch.randn(32,1,128,128):cuda()
-- z = torch.randn(32,1,128,128):cuda()

-- t = {x,y,z}

-- out = mdl:forward(t)
-- print(out:size())