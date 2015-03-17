require 'nn'
require 'image'
require 'lfs'
require 'cutorch'
require 'cunn'
require 'optim'
require 'gnuplot'
require('sampleAq/loadData.lua')
require('sampleAq/sampleAq.lua')
require('sampleAq/writeData.lua')
require('augFuncs/affine.lua')
require('paramUpdate.lua')
dofile('/usr/local/lua/opencv/init.lua')

noaug = false 

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

loadSize = {1,128,128}
sampleSize = {1,120,120}

batchSize = 32
valBatchSize = 32
testBatchSize = 32

epoch = 1
epochSize = 30e3/batchSize
nEpochs = 100
cvError = torch.Tensor(nEpochs)
nModel = os.time()
confusion = optim.ConfusionMatrix(121)
criterion = nn.ClassNLLCriterion()
criterion:cuda()

optimState = {
    learningRate = 1e-2,
    weightDecay = 1e-4, 
    momentum = 0.9,
    learningRateDecay = 0
}

optimMethod = optim.nag

trainFiles = '/mnt/plankton_data/train_128gtn'
trainSet, valSet = readTrainAndCrossValFiles(trainFiles,20)

dofile('model.lua')

logFile = io.open(string.format('modelLogs/model%d.err',nModel),'a')
logFile:write(trainFiles)
logFile:write('\n')
s = torch.initialSeed()
logFile:write(string.format('Seed: %d\n',s))

plotFile = string.format('modelLogs/model%d.pdf',nModel)
for epoch = 1,nEpochs do
    dofile('train.lua')
    dofile('val.lua')
    dofile('plot.lua')
    optimState.learningRate = setRate(1)
    optimState.weightDecay = setDecay(1)
    noaug = setAug(1)
    if epoch % 5 == 0 then
    	optimState.learningRate = optimState.learningRate*0.8
    end
    if file_exists('save') then
        os.remove('save')
        fileName = string.format('models/model%d_epoch%g.th',nModel,epoch-1)
        torch.save(fileName, mdl)
    end
end
