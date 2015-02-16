mdl:cuda()
mdl:evaluate()
subFile = openFile('submissions/sub.csv')
for n = 1,#testset/testBatchSize do
   batch,labels = getTestSample(n)
   pred = mdl:forward(batch)
   pred = torch.pow(torch.exp(1),pred:float())
   writeBatch(pred,labels,subFile)
   print(n)
end
subFile:close()
