mdl:cuda()
mdl:evaluate()
subFile = openFile('sub.csv')
for n = 1,#testset do
   local batch,labels = getAugTestSample(n)
   local pred = mdl:forward(batch)
   pred = torch.pow(torch.exp(1),pred:float())
   pred = pred:sum(1):div(testAugSize)
   writeBatch(pred,labels,subFile)
   print(n)
   collectgarbage()
end
subFile:close()

