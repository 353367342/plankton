function getTrainSample()
   local sampleList = torch.randperm(trainEnd):narrow(1,1,batchSize)
   local batch = torch.CudaTensor(augSize*batchSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local targets = torch.CudaTensor(augSize*batchSize)
   for j=1,batchSize do
      local rawExample = image.load(dataset[splitInd[sampleList[j]]].relPath)
      --      local augExample   = randomTransform(rawExample[1],augSize)
      local augExample = jitter(rawExample)
      batch[{{1 + (j-1)*augSize,j*augSize},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = augExample
      targets:narrow(1,1 +(j-1)*augSize,augSize):fill(dataset[splitInd[sampleList[j]]].classNum)
   end
   return batch, targets
end

function getCrValSample(n)
   --      local sampleList = torch.randperm(#dataset - trainEnd -1):narrow(1,1,valBatchSize):add(valBegin)
   local sampleList = splitInd[{{valBegin,#dataset}}]:narrow(1,1 + (n-1)*valBatchSize,valBatchSize)
   local batch = torch.CudaTensor(valBatchSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local targets = torch.CudaTensor(valBatchSize)
   for j=1,valBatchSize do
      local rawExample = torch.CudaTensor(1,3,sampleSize[2],sampleSize[3])
      rawExample[{{1,1},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = image.load(dataset[splitInd[sampleList[j]]].relPath)
      batch[{{1 + (j-1),j},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = rawExample[1]:resize(1,sampleSize[1],sampleSize[2],sampleSize[3])
      targets:narrow(1,1 +(j-1),1):fill(dataset[splitInd[sampleList[j]]].classNum)
   end
   return batch, targets
end

function getTestSample(n)
   local batch = torch.CudaTensor(testBatchSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local imId = {}
   for j=1,testBatchSize do
      local testExample = torch.CudaTensor(1,3,sampleSize[2],sampleSize[3])
      testExample[{{1,1},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = image.load(testset[(n-1)*testBatchSize + j].relPath)
      imId[j] = testset[(n-1)*testBatchSize + j].name
      batch[{{j,j},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = testExample[1]:resize(1,sampleSize[1],sampleSize[2],sampleSize[3])
   end
   return batch, imId
   end
