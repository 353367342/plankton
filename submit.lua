require 'nn'
require 'image'
require 'lfs'
require 'cutorch'
require 'cunn'
require('sampleAq/loadData.lua')
require('sampleAq/sampleAq.lua')
require('sampleAq/writeData.lua')
require('augFuncs/affine.lua')
dofile('/usr/local/lua/opencv/init.lua')

cutorch.setDevice(1)

noaug = true 

loadSize = {1,128,128}
sampleSize = {1,120,120}
testBatchSize = 32
mdl = torch.load('models/model1426513041_epoch41.th')
mdl:cuda()
mdl:evaluate()
testset = readTestFiles('/mnt/plankton_data/test_128gtn')

for i =1,1 do
    subFileName = string.format('submissions/model1426513041_epoch41_%g.csv',i)
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
