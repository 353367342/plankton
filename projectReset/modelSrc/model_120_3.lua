require 'nn'
require 'cunn'
require 'cutorch'
require 'image'

--fSize = {1,16,16,32}
fSize = {1,96,300,400,512,562,600,692}
featuresOut = fSize[7]
hiddenNodes = {512,256}
--hiddenNodes = {64,32}

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(fSize[1],fSize[2],8,8,2,2)) -- (120 - 8+ 2)/2 = 57
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2)) -- 28
features:add(nn.SpatialConvolutionMM(fSize[2],fSize[3],3,3)) -- 26
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2)) -- 13
features:add(nn.SpatialConvolutionMM(fSize[3],fSize[4],4,4)) -- 10 
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[4],fSize[5],4,4)) -- 7
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[5],fSize[6],3,3)) -- 5
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[6],fSize[7],3,3)) -- 3
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(3,3)) -- 1
features:add(nn.View(featuresOut))
features:cuda()

dropout_p = 0.5

dgraph = nn.Sequential()
dgraph:add(nn.Linear(featuresOut,featuresOut))
dgraph:add(nn.Dropout(0.5))
dgraph:add(nn.ReLU())
dgraph:add(nn.Linear(featuresOut,121))
dgraph:cuda()

mdl = nn.Sequential()
mdl:cuda()
mdl:add(features)
mdl:add(dgraph)
mdl:add(nn.LogSoftMax():cuda())

--output = torch.CudaTensor(10):fill(1)
--
--criterion = nn.ClassNLLCriterion()
--criterion:cuda()
--
--for i=1,30 do
--  local currentError = 0
--  input = torch.randn(1,1,512,512)
--  input = randomTransform(input[1][1],10):cuda()
--  oHat = mdl:forward(input)
--  currentError = currentError + criterion:forward(oHat,output)
--  mdl:zeroGradParameters()
--  mdl:backward(input,criterion:backward(mdl.output,output))
--  mdl:updateParameters(6e-1)
--  if i % 10 == 0 then
--    print('Batch:',i,'Error:',currentError/58)
--  end
--  collectgarbage()
--end
--
--print(torch.pow(10,oHat:float()))
