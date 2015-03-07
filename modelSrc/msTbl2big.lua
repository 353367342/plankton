require 'nn'
require 'cunn'

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(1,128,7,7,2,2)) -- 61 
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 30
features:add(nn.Dropout(0.1))
features:add(nn.SpatialConvolutionMM(128,256,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.Dropout(0.1))
features:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1)) --30
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 15
features:add(nn.Dropout(0.1))
features:add(nn.SpatialConvolutionMM(256,512,3,3,1,1,1)) --15
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

feat = {}
feat[1] = nn.Sequential()
feat[1]:cuda()
feat[1]:add(features)

tbl = nn.ParallelTable()
tbl:cuda()
tbl:add(feat[1])

for i = 2,5 do
	feat[i] = feat[1]:clone()
	feat[i]:share(feat[1], "weight", "bias", "gradWeight", "gradBias")
	tbl:add(feat[i])
end

graph = nn.Sequential()
graph:add(nn.View(featuresOut))
graph:add(nn.Dropout(0.5))
graph:add(nn.Linear(featuresOut,featuresOut))
graph:add(nn.ReLU())
graph:add(nn.Dropout(0.5))
graph:add(nn.Linear(featuresOut,121))
graph:add(nn.LogSoftMax())
graph:cuda()

mdl = nn.Sequential()
mdl:add(tbl)
mdl:add(nn.JoinTable(3):cuda())
mdl:add(nn.View(5,featuresOut):cuda()) -- (#2 pool,w*h*depth)
mdl:add(nn.TemporalMaxPooling(5):cuda())
mdl:add(graph)
mdl:cuda()