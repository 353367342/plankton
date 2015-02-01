mdl:cuda()
mdl:evaluate()
subFile = openFile('sub.csv')
for n = 1,#testset/testBatchSize do
   batch,labels = getTestSample(n)
   pred = mdl:forward(batch)
   pred = torch.pow(10,pred:float())
   writeBatch(pred,labels,subFile)
   print(n)
end
subFile:close()
