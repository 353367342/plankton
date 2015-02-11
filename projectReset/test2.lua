mdl:cuda()
mdl:evaluate()
subFile = openFile('sub.csv')
for n = 1,#testset/testBatchSize do
   batch,labels = getTestSample(n)
   pred = mdl:forward(batch)
   pred = pred:reshape(32,3,121)[3]
   pred = torch.pow(torch.exp(1),pred:float())
   writeBatch(pred,labels,subFile)
   print(n)
end
subFile:close()
