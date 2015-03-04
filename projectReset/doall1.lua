require 'nn'
require 'image'
require 'lfs'
require 'cutorch'
require 'cunn'
require 'fbcunn'
require 'optim'
require 'gnuplot'
require('sampleAq/loadData.lua')
require('sampleAq/sampleAq.lua')
require('sampleAq/writeData.lua')
require('augFuncs/affine5.lua')
require('modules/inception')
require('modules/mnn')
require('modules/fgraph')
require('modules/ensembleBranch2.lua')
require('paramUpdates/rate.lua')
require('paramUpdates/decay.lua')
require('modules/graph.lua')
dofile('/usr/local/lua/opencv/init.lua')

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

loadSize = {1,128,128}
sampleSize = {1,120,120}

batchSize = 32 --32
valBatchSize = 32
valAugSize = 32
testAugSize = 32
testBatchSize = 32

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
    learningRate = 0.01, -- 1e-2, --0.01, -- 1e-3, --0.03,
    weightDecay = 1e-4, -- play with
    momentum = 0.9,
    learningRateDecay = 0
}

optimMethod = optim.nag

cutorch.setDevice(1) -- setgtx
--torch.manualSeed(31415)
--torch.manualSeed(21718)
trainFiles = '/mnt/plankton_data/train_128gtn'
trainSet, valSet = readTrainAndCrossValFiles(trainFiles,5)
--torch.seed()

mdlFile = 'modelSrc/ms3thresh.lua'

logFile = io.open(string.format('modelLogs/model%d.err',nModel),'a')
logFile:write(trainFiles)
logFile:write('\n')
logFile:write(mdlFile)
logFile:write('\n')
s = torch.initialSeed()
logFile:write(string.format('Seed: %d\n',s))
logFile:close()

--mdl = torch.load('models/model1424395071_epoch37.th')
--mdl:cuda()
--mdl:evaluate()

dofile(mdlFile) -- ?
--mdl:float()
--mdl:cuda()

--share = true
plotFile = string.format('modelLogs/model%d.pdf',nModel)
for epoch = 1,nEpochs do
    mdl_last = mdl:clone()
    confusion:zero()
    dofile('train.lua')
    dofile('val.lua')
    optimState.learningRate = setRate()
    optimState.weightDecay = setDecay()
    gnuplot.pdffigure(plotFile)
    gnuplot.axis({1,epoch+5,0.5,2.5})
    gnuplot.grid(true)
    gnuplot.title('Cross Validation Error')
    gnuplot.xlabel('Epoch (30e3 Images per Epoch)')
    gnuplot.ylabel('Multi-Class Negative Log Loss')
    gnuplot.plot({cvError[{{1,epoch}}],'-'})
    gnuplot.plotflush()
--    torch.save('confusionMat.th',confusion)
    if epoch % 5 == 0 then
    	optimState.learningRate = optimState.learningRate*0.8
    end
    if file_exists('save1') then
        os.remove('save1')
        fileName = string.format('models/model%d_epoch%g.th',nModel,epoch-1)
        torch.save(fileName, mdl_last)
    end
    if file_exists('feat1') then
        os.remove('feat1')
        fileName = string.format('models/feat%d_epoch%g.th',nModel,epoch-1)
        torch.save(fileName, mdl)
    end
    if file_exists('test1') then
        os.remove('test1')
        testset = readTestFiles('/mnt/plankton_data/test_128gtn')
        dofile('test.lua')
    end
    if file_exists('break1') then
        os.remove('break1')
        break
    end
end
