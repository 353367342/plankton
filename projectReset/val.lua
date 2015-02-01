mdl:evaluate()
local valError = 0
local valSize = torch.floor(0.02*#dataset/valBatchSize)
local time = sys.clock()
for i=1,valSize do
   local batch, targets = getCrValSample(i)
   local oHat = mdl:forward(batch)
   valError = valError + criterion:forward(oHat,targets)
   confusion:batchAdd(oHat:float(),targets:float())
   if i == valSize then
      local errStr = string.format('Epoch: %g, Cross Val Error: %g\n',epoch,valError/valSize)
      print(errStr)
      local mdlErrFileName = string.format('models/model%d.err',nModel)
      local errFile = io.open(mdlErrFileName,'a')
      errFile:write(errStr)
      errFile:close()
   end
   end
time = sys.clock() - time
print("<validation> time for CrosVal = " .. (time) .. 's')
