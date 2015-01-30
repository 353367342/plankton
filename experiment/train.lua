require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'

----------------------------------------------------------------------
print '==> train.lua'
print '==> defining some tools'

-- trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

-- if model then
--    if opt.retrain ~= "none" then
--       local parameters,gradParameters = model:getParameters()
--       local mod2 = torch.load(opt.retrain):float()
--       local p2,gp2 = mod2:getParameters()
--       parameters:copy(p2)
--       gradParameters:copy(gp2)
--    end
--    model:cuda()
--    parameters,gradParameters = model:getParameters()
--    collectgarbage()
-- end

mdl:cuda()


augSize = 1
criterion = nn.ClassNLLCriterion()
criterion:cuda()
parameters,gradParameters = mdl:getParameters()
epochSize = 10
batchSize = 16
trainEnd = torch.floor(0.9*#dataset)
print '==> configuring optimizer'

optimState = {
   learningRate = 0.5,
   weightDecay = 1e-5,
   momentum = 0.9,
   learningRateDecay = 5e-7
}

asgdSate = {
   eta0 = 1,
   lambda = 1e-3,
   alpha = 1,
   t0 = 1e5
}

optimMethod = optim.sgd

print '==> defining training procedure'
function train()
   -- epoch tracker
   epoch = epoch or 1
   -- local vars
   local time = sys.clock()
   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')

splitInd = torch.randperm(#dataset)

for t = 1,epochSize do

   for miniBatch=1,torch.floor(trainEnd/batchSize)*batchSize,batchSize do
      -- create mini batch
      local output = torch.CudaTensor(augSize*batchSize)
      local input  = torch.CudaTensor(augSize*batchSize,1,128,128)
      local miniBatchInd = splitInd:narrow(1,miniBatch,batchSize)
      for j=1,batchSize do
         local rawExample = image.load(dataset[miniBatchInd[j]].relPath)
         local augExample = randomTransform(rawExample[1],augSize)
         input[{{1 + (j-1)*augSize,j*augSize},{1,1},{1,128},{1,128}}] = augExample:cuda()
         output:narrow(1,1 +(j-1)*augSize,augSize):fill(dataset[miniBatchInd[j]].classNum)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- f is the average of all criterions
         -- f same as currentError
         local f = 0;

         -- evaluate function for complete mini batch    
         -- estimate f
         local oHat = mdl:forward(input)
         --oHat = oHat:float()
         f = f + criterion:forward(oHat,output)
         mdl:backward(input,criterion:backward(oHat,output))

         -- normalize gradients and f(X)
         -- gradParameters --:div(batchSize)
         -- fgradParameters:mul(#branch)
         --f = f --/batchSize

         if (miniBatch - 1) % batchSize == 0 then
            print('# of unAugmented Examples:',miniBatch*augSize,'Error:',f)
         end

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      optim.sgd(feval, parameters, optimState)
--      optim.asgd(feval,parameters,asgdState)
      collectgarbage()
   end

   -- time taken
   -- time = sys.clock() - time
   -- time = time / epochSize
   -- print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

end
end
train()
