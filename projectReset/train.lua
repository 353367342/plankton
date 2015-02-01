----------------------------------------------------------------------
print '==> train.lua'
print '==> defining some tools'

mdl:cuda()
mdl:training()


parameters,gradParameters = mdl:getParameters()
print '==> configuring optimizer'

--- sgd test params
 optimState = {
    learningRate = 1e-5,
    weightDecay = 1e-5,
    momentum = 0.6,
    learningRateDecay = 5e-4
 }


--optimMethod = optim.nag

print '==> defining training procedure'
function train()
   -- epoch tracker
   --epoch = epoch or 1
   -- local vars
   local time = sys.clock()
   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')

for t = 1,epochSize do

   batch, targets = getTrainSample()
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end
         gradParameters:zero()
         local f = 0;
         local oHat = mdl:forward(batch)
         f = f + criterion:forward(oHat,targets)
         mdl:backward(batch,criterion:backward(oHat,targets)) --problem line
         -- normalize gradients and f(X)
         -- gradParameters --:div(batchSize)
         -- fgradParameters:mul(#branch)
         --f = f --/batchSize

            print('# of Examples:',t*batchSize*augSize,'Error:',f)
            return f,gradParameters
      end
      optim.adagrad(feval, parameters, optimState)
      collectgarbage()
   end
   -- time taken
time = sys.clock() - time
print("<trainer> time for 1 Epoch = " .. (time) .. 's')

end
train()
