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
require('augFuncs/affine5.lua')
require('modules/inception')
require('modules/ensembleBranch.lua')
require('paramUpdates/rate.lua')
require('paramUpdates/decay.lua')
require('modules/graph.lua')
dofile('/usr/local/lua/opencv/init.lua')
--dofile('modules/MsSpatialConvolutionMM.lua')

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

loadSize = {1,128,128}
sampleSize = {1,120,120}

batchSize = 32
valBatchSize = 32
valAugSize = 32
testAugSize = 32
testBatchSize = 50

augSize = 1
epoch = 1
epochSize = 30e3/batchSize/augSize --30e3/batchSize/augSize
nEpochs = 10000
cvError = torch.Tensor(nEpochs)
nModel = os.time()
confusion = optim.ConfusionMatrix(121)
criterion = nn.ClassNLLCriterion()
criterion:cuda()

optimState = {
    learningRate = 0.02, -- 1e-3, --0.03,
    weightDecay = 0, -- play with
    momentum = 0.9,
    learningRateDecay = 5e-4,
    dampening = 0,
    nesterov = true
}

optimMethod = optim.nag

torch.manualSeed(31415)
trainFiles = '/mnt/plankton_data/train_128gtn'
trainSet, valSet = readTrainAndCrossValFiles(trainFiles,9)
torch.seed()

mdlFile = 'modelSrc/model_sou2.lua'

logFile = io.open(string.format('modelLogs/model%d.err',nModel),'a')
logFile:write(trainFiles)
logFile:write('\n')
logFile:write(mdlFile)
logFile:write('\n')
logFile:close()

mdl = torch.load('modelLogs/model1424102763_epoch12.th')
mdl:cuda()
mdl:evaluate()

--dofile(mdlFile) -- ?

--share = true

for epoch = 1,nEpochs do
    confusion:zero()
    dofile('train.lua')
    dofile('val.lua')
    optimState.learningRate = setRate()
    optimState.weightDecay = setDecay()
    gnuplot.plot(cvError[{{1,epoch}}],'-')
    gnuplot.axis({1,epoch+100,0.5,5})
--    torch.save('confusionMat.th',confusion)
    if file_exists('save') then
        fileName = string.format('models/model%d_epoch%g.th',nModel,epoch-1)
        torch.save(fileName, mdl)
        os.remove('save')
    end
    if file_exists('test') then
        testset = readTestFiles('/mnt/plankton_data/test_128gtn')
        dofile('test.lua')
        os.remove('test')
    end
    if file_exists('break') then
        os.remove('break')
        break
    end
end

-- -- fileName = string.format('models/model%d_epoch%g.th',nModel,epoch)
-- -- torch.save(fileName, mdl)
