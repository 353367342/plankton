require 'nn'
--require 'xlua'

dataset={};
function dataset:size() return 1000 end -- 1000 examples

mu = torch.Tensor(100);

i = 0
mu:apply(function()
           i = i +1
           return i
        end)

for i=1,dataset:size() do
  local class = torch.round(torch.uniform(1,mu:nElement()))
  local input = torch.randn(1,100,100) + mu[class];
  local output = torch.Tensor(100):fill(0)
  output[class] = 1
  
  dataset[i] = {input, output}
end

mdl = nn.Sequential() -- Setup Feedforward connection
mdl:add(nn.SpatialConvolution(1,10,4,4)) -- output 10 x 97 x 97
mdl:add(nn.SpatialConvolution(10,1,20,20)) -- output 1 x 78 x 78
mdl:add(nn.Reshape(78*78)) -- reshape into Tensor(16)
mdl:add(nn.Linear(78*78,100))
mdl:add(nn.SoftMax()) -- normalize


-- criterion = nn.MSECriterion()

-- epochs = 12;
-- for epoch= 1,epochs do 
--    for i = 1,dataset:size() do
--       local input  = dataset[i][1]
--       local output = dataset[i][2]
      
--       criterion:forward(mdl:forward(input),output)
--       mdl:zeroGradParameters()
--       mdl:backward(input,criterion:backward(mdl.output,output))
--       mdl:updateParameters(0.001)
--       --print(i,"of",dataset:size())
--       xlua.progress(i,dataset:size())
--    end
--    print("Epoch",epoch,"of",epochs)
-- end

criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mdl, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 2;
trainer:train(dataset)

for i=1,10 do -- Show that it worked
   local x = torch.Tensor(100,2)
   x:select(2,1):fill(0):add(mdl:forward(dataset[i][1]):t())
   x:select(2,2):fill(0):add(dataset[i][2])
   print(x)
end

-- print(mdl:forward(torch.randn(1,10,10))) -- Show that it "guesses" example right in middle


