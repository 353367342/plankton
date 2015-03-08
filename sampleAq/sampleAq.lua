function getTrainSample()
   local sampleList = torch.randperm(#trainSet):narrow(1,1,batchSize)
   local batch = torch.CudaTensor(augSize*batchSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local targets = torch.CudaTensor(augSize*batchSize)
   for j=1,batchSize do
      local rawExample = image.load(trainSet[sampleList[j]].relPath)
      --      local augExample   = randomTransform(rawExample[1],augSize)
      --local augExample = image.scale(rawExample,sampleSize[2],sampleSize[3]):resize(1,sampleSize[1],sampleSize[2],sampleSize[3])  
--      if epoch > 20 then
         local augExample = jitter(rawExample)
--      else
--         augExample = image.scale(rawExample, sampleSize[2], sampleSize[3])
--         augExample = augExample:resize(1,sampleSize[1],sampleSize[2],sampleSize[3])
--      end
      batch[{{1 + (j-1)*augSize,j*augSize},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = augExample
      targets:narrow(1,1 +(j-1)*augSize,augSize):fill(trainSet[sampleList[j]].classNum)
   end
   return batch, targets
end

function getCrValSample(n)
   --      local sampleList = torch.randperm(#dataset - trainEnd -1):narrow(1,1,valBatchSize):add(valBegin)
   valList = torch.Tensor(#valSet)
   i = 0; valList:apply(function() i = i + 1; return i end)
   local sampleList = valList:narrow(1,1 + (n-1)*valBatchSize,valBatchSize)
   local batch = torch.CudaTensor(valBatchSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local targets = torch.CudaTensor(valBatchSize)
   for j=1,valBatchSize do
      local rawExample = torch.CudaTensor(1,3,sampleSize[2],sampleSize[3])
      --rawExample[{{1,1},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = image.scale(image.load(valSet[sampleList[j]].relPath),sampleSize[2],sampleSize[3])
      rawExample[{{1,1},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = jitter(image.load(valSet[sampleList[j]].relPath))
      batch[{{1 + (j-1),j},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = rawExample[1]:resize(1,sampleSize[1],sampleSize[2],sampleSize[3])
      targets:narrow(1,1 +(j-1),1):fill(valSet[sampleList[j]].classNum)
   end
   return batch, targets
end

function getAugCrValSample(n)
   local batch = torch.CudaTensor(valAugSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local targets = torch.CudaTensor(valAugSize)
   local img = image.load(valSet[n].relPath)
   for j=1,valAugSize do
      batch[{{1 + (j-1),j},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = jitter(img):cuda()
      targets:narrow(1,1 +(j-1),1):fill(valSet[n].classNum)
   end
   return batch, targets
end

function getTestSample(n)
   local batch = torch.CudaTensor(testBatchSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local imId = {}
   for j=1,testBatchSize do
      local testExample = torch.CudaTensor(1,3,sampleSize[2],sampleSize[3])
      -- testExample[{{1,1},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = image.scale(image.load(testset[(n-1)*testBatchSize + j].relPath),sampleSize[2],sampleSize[3])
      testExample[{{1,1},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = jitter(image.load(testset[(n-1)*testBatchSize + j].relPath))
      imId[j] = testset[(n-1)*testBatchSize + j].name
      batch[{{j,j},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = testExample[1]:resize(1,sampleSize[1],sampleSize[2],sampleSize[3])
   end
   return batch, imId
end

function getAugTestSample(n)
   local batch = torch.CudaTensor(testAugSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local imId = {}
   local img = image.load(testset[n].relPath)
   for j=1,testAugSize do
      imId[j] = testset[n].name
      batch[{{j,j},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = jitter(img):cuda() --testExample
   end
   return batch, imId
end

function getTrainTableSample()
   local sampleList = torch.randperm(#trainSet):narrow(1,1,batchSize)
   local batchTbl = {}
   local targets = torch.CudaTensor(augSize*batchSize)
   for i = 1,5 do 
      local batch = torch.CudaTensor(augSize*batchSize,sampleSize[1],sampleSize[2],sampleSize[3])
      for j=1,batchSize do
         local rawExample = image.load(trainSet[sampleList[j]].relPath)
         local augExample = jitter(rawExample)
         batch[{{1 + (j-1)*augSize,j*augSize},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = augExample
         targets:narrow(1,1 +(j-1)*augSize,augSize):fill(trainSet[sampleList[j]].classNum)
      end
      batchTbl[i] = batch
   end
   return batchTbl, targets
end

function getCrValTableSample(n)
   valList = torch.Tensor(#valSet)
   i = 0; valList:apply(function() i = i + 1; return i end)
   local sampleList = valList:narrow(1,1 + (n-1)*valBatchSize,valBatchSize)
   local targets = torch.CudaTensor(valBatchSize)
   local batchTbl = {}
   for i=1,5 do
      local batch = torch.CudaTensor(valBatchSize,sampleSize[1],sampleSize[2],sampleSize[3])
      for j=1,valBatchSize do
         local rawExample = torch.CudaTensor(1,3,sampleSize[2],sampleSize[3])
         rawExample[{{1,1},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = jitter(image.load(valSet[sampleList[j]].relPath))
         batch[{{1 + (j-1),j},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = rawExample[1]:resize(1,sampleSize[1],sampleSize[2],sampleSize[3])
         targets:narrow(1,1 +(j-1),1):fill(valSet[sampleList[j]].classNum)
      end
      batchTbl[i] = batch
   end
   return batchTbl, targets
end