function getTrainSample()
   local sampleList = torch.randperm(#trainSet):narrow(1,1,batchSize)
   local batch = torch.CudaTensor(batchSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local targets = torch.CudaTensor(batchSize)
   for j=1,batchSize do
      local rawExample = image.load(trainSet[sampleList[j]].relPath)
      local augExample = jitter(rawExample)
      batch[j] = augExample
      targets:narrow(1,j,1):fill(trainSet[sampleList[j]].classNum)
   end
   return batch, targets
end

function getCrValSample(n)
   valList = torch.Tensor(#valSet)
   i = 0; valList:apply(function() i = i + 1; return i end)
   local sampleList = valList:narrow(1,1 + (n-1)*valBatchSize,valBatchSize)
   local batch = torch.CudaTensor(valBatchSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local targets = torch.CudaTensor(valBatchSize)
   for j=1,valBatchSize do
      local rawExample = torch.CudaTensor(1,3,sampleSize[2],sampleSize[3])
      rawExample[{{1,1},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = image.scale(image.load(valSet[sampleList[j]].relPath),sampleSize[2],sampleSize[3])
      batch[{{1 + (j-1),j},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = rawExample[1]:resize(1,sampleSize[1],sampleSize[2],sampleSize[3])
      targets:narrow(1,1 +(j-1),1):fill(valSet[sampleList[j]].classNum)
   end
   return batch, targets
end

function getTestSample(n)
   local batch = torch.CudaTensor(testBatchSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local imId = {}
   for j=1,testBatchSize do
      local testExample = torch.CudaTensor(1,3,sampleSize[2],sampleSize[3])
      testExample[{{1,1},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = jitter(image.load(testset[(n-1)*testBatchSize + j].relPath))
      imId[j] = testset[(n-1)*testBatchSize + j].name
      batch[{{j,j},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = testExample[1]:resize(1,sampleSize[1],sampleSize[2],sampleSize[3])
   end
   return batch, imId
end

