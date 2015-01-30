function getTrainSample()
   local sampleList = torch.randperm(trainEnd):narrow(1,1,batchSize)
   local batch = torch.CudaTensor(augSize*batchSize,sampleSize[1],sampleSize[2],sampleSize[3])
   local targets = torch.CudaTensor(augSize*batchSize)
   for j=1,batchSize do
      local rawExample = image.load(dataset[splitInd[sampleList[j]]].relPath)
      local augExample   = randomTransform(rawExample[1],augSize)
      batch[{{1 + (j-1)*augSize,j*augSize},{1,1},{1,sampleSize[2]},{1,sampleSize[3]}}] = augExample
      targets:narrow(1,1 +(j-1)*augSize,augSize):fill(dataset[splitInd[sampleList[j]]].classNum)
      end
   return batch, targets
   end

function getCrValSample()
   local sampleList = torch.randperm(#dataset - trainEnd -1):narrow(1,1,valBatchSize):add(valBegin)
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
