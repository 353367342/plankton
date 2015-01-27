require 'nn'
require 'cunn'
require 'cutorch'
require 'image'
require('loadData.lua')
require('randTransform.lua')

dofile('model.lua')

dataset = readTrainFiles('../data/train')

criterion = nn.ClassNLLCriterion()
criterion:cuda()

batchSize = 10

nModel = os.time()
for epoch = 1,2 do
  local ind = torch.randperm(#dataset)
  for i=1,2 do
    local currentError = 0
    local output = torch.CudaTensor(batchSize):fill(dataset[ind[i]].classNum)
    local input = image.load(dataset[ind[i]].relPath)
    input = randomTransform(input[1],batchSize):cuda()
    oHat = mdl:forward(input)
    currentError = currentError + criterion:forward(oHat,output)
    mdl:zeroGradParameters()
    mdl:backward(input,criterion:backward(mdl.output,output))
    mdl:updateParameters(6e-1)
    if i % 10 == 0 then
      print('Batch:',i,'Error:',currentError/batchSize)
    end
    collectgarbage()
  end
  local mdlFileName = string.format('models/model%d_epoch%g.th',nModel,epoch)
  local mdlFile = torch.DiskFile(mdlFileName,'w')
  mdlFile:binary()
  mdlFile:writeObject(mdl:float())
  mdlFile:close()
end

