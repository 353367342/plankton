require 'nn'
require 'image'
require 'lfs'
require 'cutorch'
require 'cunn'
require 'optim'
require 'gnuplot'
require('loadData.lua')
require('randTransform.lua')
require('sampleAq.lua')
require('writeData.lua')
require('jitter')
require('shareTrans')
require('inception')
require('ensembleBranch.lua')

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

loadSize = {1,128,128}
sampleSize = {1,120,120}

batchSize = 32
valBatchSize = 32
testBatchSize = 32

augSize = 1
epoch = 1
epochSize = 30e3/batchSize/augSize
nEpochs = 10000
cvError = torch.Tensor(nEpochs)
nModel = os.time()
confusion = optim.ConfusionMatrix(121)
criterion = nn.ClassNLLCriterion()
criterion:cuda()

optimState = {
    learningRate = 0.01, -- 1e-3, --0.03,
    weightDecay = 1e-5, -- play with
    momentum = 0.1,
    learningRateDecay = 5e-4
}

optimMethod = optim.adagrad

trainSet, valSet = readTrainAndCrossValFiles('/mnt/plankton_data/train_128gthn/',9)

--mdl = torch.load('models/model1422714991_epoch129.th')
--mdl:cuda()
--mdl:evaluate()

dofile('model_ultraIncep.lua') -- ?

--share = true

for epoch = 1,nEpochs do
    confusion:zero()
    dofile('train.lua')
    dofile('val.lua')
    gnuplot.plot(cvError[{{1,epoch}}],'-')
    gnuplot.axis({1,epoch+100,0.5,5})
--    torch.save('confusionMat.th',confusion)
--    mdl_last = mdl:clone():float()
    if file_exists('save') then
        fileName = string.format('models/model%d_epoch%g.th',nModel,epoch-1)
        torch.save(fileName, mdl_last)
        os.remove('save')
    end
    if file_exists('test') then
        testset = readTestFiles('/mnt/plankton_data/test_128gthn')
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
