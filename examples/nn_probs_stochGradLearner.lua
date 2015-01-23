require 'nn'

dataset={};
function dataset:size() return 1000 end -- 1000 examples

eps = 5

for i=1,dataset:size() do 
  local input = torch.randn(1,10,10) + eps;     -- normally distributed example in 2d
  local sign = torch.Tensor(1)  
  local output = torch.Tensor(2)
  
  if torch.round(torch.uniform(0,1)) then
     output[1] = 0
     output[2] = 1
     sign = -1
  else
     output[1] = 1
     output[2] = 0
     sign = 1
  end
  
  input = input:mul(input,sign)
  
  dataset[i] = {input, output}
end


mdl = nn.Sequential() -- Setup Feedforward connection
mdl:add(nn.SpatialConvolution(1,10,4,4)) -- output 10 x 7 x 7
mdl:add(nn.SpatialConvolution(10,1,4,4)) -- output 1 x 4 x 4
mdl:add(nn.Reshape(16)) -- reshape into Tensor(16)
mdl:add(nn.Linear(16,2))
mdl:add(nn.Sigmoid())

criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mdl, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)

for i=1,dataset:size() do -- Show that it worked 
  print(mdl:forward(dataset[i][1]))
  print(dataset[i][2])
end

print(mdl:forward(torch.randn(1,10,10))) -- Show that it "guesses" example right in middle


