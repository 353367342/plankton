require 'nn'
require('inception.lua')
require 'cunn'
require 'cutorch'
require('ensembleBranch.lua')

features = {}

features[1] = nn.Sequential()
features[1]:add(nn.SpatialConvolutionMM(1,64,7,7,2,2))
features[1]:add(nn.SpatialMaxPooling(3,3,2,2))
features[1]:add(nn.SpatialConvolutionMM(64,192,1,1,1,1))    --added
features[1]:add(nn.SpatialConvolutionMM(64,192,3,3,1,1))
features[1]:add(nn.SpatialMaxPooling(3,3,2,2))
features[1]:cuda()

features[2] = nn.Sequential()
features[2]:add(inception_module(2,192,{{64}, {96, 128}, {16, 32}, {3, 32}}))
features[2]:add(inception_module(2,256,{{128}, {128, 192}, {32, 96}, {3, 64}}))
features[2]:add(nn.SpatialMaxPooling(2,2,2,2))
features[2]:add(inception_module(2,480,{{192}, {96, 208}, {16, 48}, {3, 64}}))
features[2]:cuda()

-- features[3] = nn.Sequential()
-- depthConcat1 = nn.depthConcat(2)    --depth dim = 2
-- depthConcat1:add(ensemble_branch(512,5,3))
-- features[3]:add(inception_module(2,512,{{160}, {112, 224}, {24, 64}, {3, 64}}))
-- features[3]:add(inception_module(2,512,{{128}, {128, 256}, {24, 64}, {3, 64}}))
-- features[3]:add(inception_module(2,512,{{112}, {144, 288}, {32, 64}, {3, 64}}))
-- depthConcat2 = nn.depthConcat(2)
-- depthConcat1:add(features[3])
-- depthConcat1:cuda()

-- features[3]:cuda()

-- features[4] = nn.Sequential()
-- features[4]:add(inception_module(2,512,{{128}, {128, 256}, {24, 64}, {3, 64}}))
-- features[4]:add(inception_module(2,512,{{112}, {144, 288}, {32, 64}, {3, 64}}))
-- features[4]:cuda()

-- features[5] = nn.Sequential()
-- features[5]:add(inception_module(2,528,{{256}, {160, 320}, {32, 128}, {3, 128}}))
-- features[5]:add(inception_module(2,832,{{256}, {160, 320}, {32, 128}, {3, 128}}))
-- --features[5]:add(nn.SpatialMaxPooling(3,3,2,2))
-- features[5]:cuda()

-- features[6] = nn.Sequential()
-- features[6]:add(inception_module(2,832,{{384}, {192, 384}, {48, 128}, {3, 128}}))
-- features[6]:add(nn.SpatialAveragePooling(6,6))
-- features[6]:cuda()



mdl = nn.Sequential()
mdl:cuda()

mdl:add(features[1])
mdl:add(features[2])
depthConcat1 = nn.depthConcat(2)
depthConcat1:add(ensemble_branch(512,5,3))
features[3] = nn.Sequential()
features[3]:add(inception_module(2,512,{{160}, {112, 224}, {24, 64}, {3, 64}}))
features[3]:add(inception_module(2,512,{{128}, {128, 256}, {24, 64}, {3, 64}}))
features[3]:add(inception_module(2,512,{{112}, {144, 288}, {32, 64}, {3, 64}}))

depthConcat2 = nn.depthConcat(2)
depthConcat2:add(ensemble_branch(528, 5, 3))
features[4] = nn.Sequential()
features[4]:add(inception_module(2,528,{{256}, {160, 320}, {32, 128}, {3, 128}}))
features[4]:add(inception_module(2,832,{{256}, {160, 320}, {32, 128}, {3, 128}}))
features[4]:add(inception_module(2,832,{{384}, {192, 384}, {48, 128}, {3, 128}}))
features[4]:add(nn.SpatialAveragePooling(6,6))
features[4]:add(nn.Dropout(0.4))
features[4]:add(nn.View(1024))
features[4]:add(nn.Linear(1024,121))
features[4]:add(nn.LogSoftMax())
depthConcat2:add(features[4])

features[3]:add(depthConcat2)
depthConcat1:add(features[3])
mdl:add(depthConcat1)

--need to repmat output x3
--output should be 363


--x = mdl:forward(torch.randn(32,1,120,120):cuda())
--print(x:size())