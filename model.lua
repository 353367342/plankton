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

dgraph = nn.Sequential()
dgraph:add(nn.View(featuresOut))
dgraph:add(nn.Dropout(0.5))
dgraph:add(nn.Linear(featuresOut,3000))
dgraph:add(nn.SoftSign())
dgraph:add(nn.Linear(3000,121))
dgraph:cuda()

mdl = nn.Sequential()
mdl:cuda()
mdl:add(features)
mdl:add(dgraph)
mdl:add(nn.LogSoftMax():cuda())