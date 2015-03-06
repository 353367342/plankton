mdl:cuda()
mdl:evaluate()
subFileName = string.format('submissions/model%d_epoch%g.csv',nModel,epoch)
-- subFile = openFile('submissions/sub.csv')
subFile = openFile(subFileName)
for n = 1,#testset/testBatchSize do
   batch,labels = getTestSample(n)
   pred = mdl:forward(batch)
   pred = torch.pow(torch.exp(1),pred:float())
   writeBatch(pred,labels,subFile)
   print(n)
end
subFile:close()
