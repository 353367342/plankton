require 'nn'
require('inception.lua')
require 'cunn'
require 'cutorch'

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(1,64,7,7,2,2))
features:add(nn.SpatialMaxPooling(3,3,2,2))
features:add(nn.SpatialConvolutionMM(64,192,3,3,1,1))
features:add(nn.SpatialMaxPooling(3,3,2,2))
features:add(inception_module(2,192,{{64}, {96, 128}, {16, 32}, {3, 32}}))
features:add(inception_module(2,256,{{128}, {128, 192}, {32, 96}, {3, 64}}))
features:add(nn.SpatialMaxPooling(2,2,2,2))
features:add(inception_module(2,480,{{192}, {96, 208}, {16, 48}, {3, 64}}))
features:add(nn.View(512*6*6))

mdl = nn.Sequential()
mdl:add(features)
mdl:add(nn.Dropout(0.6))
mdl:add(nn.Linear(512*6*6,512*4*6))
mdl:add(nn.Linear(512*4*6,121))
mdl:add(nn.LogSoftMax())

mdl:cuda()

-- x = mdl:forward(torch.randn(32,1,120,120):cuda())
-- print(x:size())