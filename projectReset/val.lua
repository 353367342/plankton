mdl:evaluate()
local valError = 0
local valSize = torch.floor(#valSet/valBatchSize)
local time = sys.clock()
for i=1,valSize do
   local batch, targets = getCrValSample(i)
   if share then
      batch = shareTrans(batch:float()):cuda()
   end
   local oHat = mdl:forward(batch)
   valError = valError + criterion:forward(oHat,targets)
   confusion:batchAdd(oHat:float(),targets:float())
   if i == valSize then
      cvError[epoch] = valError/valSize
--      logger:add{valError/valSize}
      local errStr = string.format(' Cross Val Error: %g\n',valError/valSize)
      print(errStr)
      local mdlErrFileName = string.format('models/model%d.err',nModel)
      local errFile = io.open(mdlErrFileName,'a')
      errFile:write(errStr)
      errFile:close()
   end
   end
time = sys.clock() - time
print("<validation> time for CrosVal = " .. (time) .. 's')
epoch = epoch + 1
