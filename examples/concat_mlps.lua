require 'nn'
require 'cunn'
require 'cutorch'
require('../project/randTransform.lua')

output = torch.CudaTensor(10):fill(1)

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(1,10,7,7))
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2))
features:add(nn.SpatialConvolutionMM(10,1,10,10))
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2)) -- 122,122
features:add(nn.View(122*122))

branch = {}

branch[1] = nn.Sequential()
branch[1]:add(nn.Linear(122*122,512))
branch[1]:add(nn.Linear(512,2))
branch[1]:add(nn.LogSoftMax())

branch[2] = nn.Sequential()
branch[2]:add(nn.Linear(122*122,512))
branch[2]:add(nn.Linear(512,2))
branch[2]:add(nn.LogSoftMax())

dgraph = nn.Concat(2)
dgraph:add(branch[1])
dgraph:add(branch[2])

mdl = nn.Sequential()
mdl:add(features)
mdl:add(dgraph)
mdl:cuda()

criterion = nn.ClassNLLCriterion()
criterion:cuda()

for i=1,30 do
  local currentError = 0
  input = torch.randn(512,512)
  input = randomTransform(input,15):cuda()
  oHat = mdl:forward(input)
  currentError = currentError + criterion:forward(oHat,output)
  mdl:zeroGradParameters()
  mdl:backward(input,criterion:backward(mdl.output,output))
  mdl:updateParameters(1e-3)
  if i % 10 == 0 then
    print('Batch:',i,'Error:',currentError/58)
  end
  collectgarbage()
end

print(torch.pow(10,oHat:float()))
