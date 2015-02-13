----------------------------------------------------------------------
print '==> train.lua'
print '==> defining some tools'

mdl:training()

parameters,gradParameters = mdl:getParameters()
print '==> configuring optimizer'

print '==> defining training procedure'
function train()
   local time = sys.clock()
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')

local epochError = 0 
for t = 1,epochSize do

   local batch, targets = getTrainSample()
   if share then
      batch = shareTrans(batch:float()):cuda()
   end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end
         gradParameters:zero()
         local oHat = mdl:forward(batch)
         oHat = oHat:reshape(batchSize,3,121)
         local f = {0,0,0}
         local grad = {}
         gradAcc = torch.CudaTensor(32,3*121)
         local w = {1,1,1} -- {0.3,0.3,1}
         for aux = 1,3 do
            f[aux] = f[aux] + criterion:forward(oHat[{{1,32},{aux,aux},{1,121}}]:reshape(32,121),targets)
            grad[aux] = criterion:backward(oHat[{{1,32},{aux,aux},{1,121}}]:reshape(32,121),targets)
            gradAcc[{{1,32},{1 + (aux-1)*121,121 + (aux-1)*121}}] = grad[aux]:mul(w[aux])
         end
         mdl:backward(batch,gradAcc) --problem line
         epochError = epochError + f[3]
         print('# of Examples:',t*batchSize*augSize,'Error:',f)
         return f,gradParameters
      end
      optimMethod(feval, parameters, optimState)
      collectgarbage()
end
   -- time taken
time = sys.clock() - time
print("<trainer> time for 1 Epoch = " .. (time) .. 's')
epochError = epochError/epochSize
--torch.save('confusionMat.th',confusion)
local errStr = string.format('Epoch: %g, Epoch Error: %g, Learning Rate: %g',epoch,epochError,optimState.learningRate)
print(errStr)
local mdlErrFileName = string.format('models/model%d.err',nModel)
local errFile = io.open(mdlErrFileName,'a')
errFile:write(errStr)
errFile:close()
end
train()
