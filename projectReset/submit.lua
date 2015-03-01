require 'nn'
require 'image'
require 'lfs'
require 'cutorch'
require 'cunn'
require 'fbcunn'
require('sampleAq/loadData.lua')
require('sampleAq/sampleAq.lua')
require('sampleAq/writeData.lua')
require('augFuncs/affine5.lua')
require('modules/inception')
require('modules/mnn')
require('modules/ensembleBranch.lua')
require('paramUpdates/rate.lua')
require('paramUpdates/decay.lua')
require('modules/graph.lua')
dofile('/usr/local/lua/opencv/init.lua')

cutorch.setDevice(2)

loadSize = {1,128,128}
sampleSize = {1,120,120}
testBatchSize = 32
mdl = torch.load('models/model1425142487_epoch59.th')
mdl:cuda()
mdl:evaluate()

testset = readTestFiles('/mnt/plankton_data/test_128tn')

for i =1,1 do
    subFileName = string.format('submissions/model1425142487_epoch59_%g.csv',i)
    subFile = openFile(subFileName)
    for n = 1,#testset/testBatchSize do
        batch,labels = getTestSample(n)
        pred = mdl:forward(batch)
        pred = torch.pow(torch.exp(1),pred:float())
        writeBatch(pred,labels,subFile)
        print('Run',i,'Batch',n)
        collectgarbage()
    end
    subFile:close()
end
