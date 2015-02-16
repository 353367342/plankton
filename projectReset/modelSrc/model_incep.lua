require 'nn'
require('inception.lua')
require 'cunn'
require 'cutorch'

features = {}

features[1] = nn.Sequential()
features[1]:add(nn.SpatialConvolutionMM(1,64,7,7,2,2))
features[1]:add(nn.SpatialMaxPooling(3,3,2,2))
features[1]:add(nn.SpatialConvolutionMM(64,192,3,3,1,1))
features[1]:add(nn.SpatialMaxPooling(3,3,2,2))
features[1]:add(inception_module(2,192,{{64}, {96, 128}, {16, 32}, {3, 32}}))
features[1]:cuda()

features[2] = nn.Sequential()
features[2]:add(inception_module(2,256,{{128}, {128, 192}, {32, 96}, {3, 64}}))
features[2]:add(nn.SpatialMaxPooling(2,2,2,2))
features[2]:add(inception_module(2,480,{{192}, {96, 208}, {16, 48}, {3, 64}}))
features[2]:add(nn.View(512*6*6))
features[2]:cuda()

mdl = nn.Sequential()
mdl:cuda()
mdl:add(features[1])
mdl:add(features[2])
mdl:add(nn.Dropout(0.6):cuda())
mdl:add(nn.Linear(512*6*6,512*4*6):cuda())
mdl:add(nn.Linear(512*4*6,121):cuda())
mdl:add(nn.LogSoftMax():cuda())

-- x = mdl:forward(torch.randn(32,1,120,120):cuda())
-- print(x:size())
