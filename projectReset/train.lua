----------------------------------------------------------------------
print '==> train.lua'
print '==> defining some tools'

mdl:cuda()
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

   batch, targets = getTrainSample()
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
         local f = 0;
         local oHat = mdl:forward(batch)
         f = f + criterion:forward(oHat,targets)
         mdl:backward(batch,criterion:backward(oHat,targets)) --problem line
         --confusion:batchAdd(oHat:float(),targets:float())
	 epochError = epochError + f
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
local errStr = string.format('Epoch: %g, Epoch Error: %g,',epoch,epochError)
print(errStr)
local mdlErrFileName = string.format('models/model%d.err',nModel)
local errFile = io.open(mdlErrFileName,'a')
errFile:write(errStr)
errFile:close()
end
train()
