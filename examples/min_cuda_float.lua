require 'nn'
require 'cunn'
require 'cutorch'

output = torch.CudaTensor(58):fill(1)

mdl = nn.Sequential()
mdl:add(nn.SpatialConvolutionMM(1,10,7,7))
mdl:add(nn.SpatialMaxPooling(2,2))
mdl:add(nn.SpatialConvolutionMM(10,1,10,10))
mdl:add(nn.Reshape(244*244))
mdl:add(nn.Linear(244*244,2))
mdl:add(nn.LogSoftMax())
mdl:cuda()

criterion = nn.ClassNLLCriterion()
criterion:cuda()

for i=1,512 do
  local currentError = 0
  input = torch.randn(58,1,512,512):cuda()
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
