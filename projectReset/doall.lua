require 'nn'
require 'image'
require 'lfs'
require 'cutorch'
require 'cunn'
require 'optim'
require('loadData.lua')
require('randTransform.lua')
require('sampleAq.lua')
require('writeData.lua')

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

sampleSize = {1,96,96}
batchSize = 32
valBatchSize = 128
testBatchSize = 100
augSize = 1
epoch = 1
epochSize = 30e3/batchSize/augSize
nEpochs = 500
nModel = os.time()


dataset = readTrainFiles('../data/raw/train_96ht')
torch.manualSeed(22)
splitInd = torch.randperm(#dataset)
trainEnd = torch.floor(0.98*#dataset)
valBegin = trainEnd + 1
torch.seed()

confusion = optim.ConfusionMatrix(121)
mdl = torch.load('models/model1422714991_epoch129.th')
--mdl:cuda()
--mdl:evaluate()
--dofile('model_96.lua')

criterion = nn.ClassNLLCriterion()
criterion:cuda()

for epoch = 1,nEpochs do
   dofile('train.lua')
   mdl_last = mdl:clone():float()
   dofile('val.lua')
   if file_exists('save') then
      fileName = string.format('models/model%d_epoch%g.th',nModel,epoch-1)
      torch.save(fileName, mdl_last)
      os.remove('save')
   end
   
   if file_exists('test') then
      testset = readTestFiles('/tmp/test_96')
      dofile('test.lua')
      os.remove('test')
   end
end
-- fileName = string.format('models/model%d_epoch%g.th',nModel,epoch)
-- torch.save(fileName, mdl)
