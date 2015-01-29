require 'nn'
require 'cunn'
require 'cutorch'
require 'image'
require('loadData.lua')
require('randTransform.lua')

dofile('model_128.lua')

dataset = readTrainFiles('../data/train_128')

criterion = nn.ClassNLLCriterion()
criterion:cuda()

augSize = 1
miniBatchSize = 16

splitInd = torch.randperm(#dataset)
trainEnd = torch.floor(0.9*#dataset)
valBegin = torch.floor(0.9*#dataset) + 1
--splitInd = torch.sort(splitInd)

learningRate = 0.03
learningDecay = 1e-7

nModel = os.time()
for epoch = 1,100 do
   --   local ind = torch.randperm(trainEnd)
   learningRate = learningRate - learningDecay
   mdl:training()
   for miniBatch=1,torch.floor(trainEnd/miniBatchSize)*miniBatchSize,miniBatchSize do
    local miniBatchInd = splitInd:narrow(1,miniBatch,miniBatchSize)
    local currentError = 0
    output = torch.CudaTensor(augSize*miniBatchSize)
    local input  = torch.CudaTensor(augSize*miniBatchSize,1,128,128)
    for j=1,miniBatchSize do
       local rawExample = image.load(dataset[miniBatchInd[j]].relPath)
       local augExample   = randomTransform(rawExample[1],augSize)
       input[{{1 + (j-1)*augSize,j*augSize},{1,1},{1,128},{1,128}}] = augExample
       output:narrow(1,1 +(j-1)*augSize,augSize):fill(dataset[miniBatchInd[j]].classNum)
    end
    oHat = mdl:forward(input)
    currentError = currentError + criterion:forward(oHat,output)
    mdl:zeroGradParameters()
    mdl:backward(input,criterion:backward(oHat,output))
    mdl:updateParameters(0.03)
    if (miniBatch - 1) % miniBatchSize == 0 then
       print('# of unAugmented Examples:',miniBatch*augSize,'Error:',currentError)
    end
   end
   mdl:evaluate()
  local valError = 0
  for i=valBegin,#dataset do
    local output = torch.CudaTensor(1):fill(dataset[splitInd[i]].classNum)
    local input = image.load(dataset[splitInd[i]].relPath)
    local oHat = mdl:forward(input:resize(1,1,128,128):cuda())
    valError = valError + criterion:forward(oHat,output)
     if i == #dataset then
        local errStr = string.format('Epoch: %g, Cross Val Error: %g\n',epoch,valError/torch.floor(0.1*#dataset))
        print(errStr)
        local mdlErrFileName = string.format('models/model%d.err',nModel)
        local errFile = io.open(mdlErrFileName,'a')
        errFile:write(errStr)
        errFile:close()
     end
  end
  -- local mdlFileName = string.format('models/model%d_epoch%g.th',nModel,epoch)
  -- local mdlFile = torch.DiskFile(mdlFileName,'w')
  -- mdlFile:binary()
  -- mdlFile:writeObject(mdl:float())
  -- mdlFile:close()
end
