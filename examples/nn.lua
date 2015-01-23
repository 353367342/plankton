require 'nn'

dataset={};
function dataset:size() return 1000 end -- 1000 examples

for i=1,dataset:size() do 
  local input = torch.randn(1,10,10) + 5;     -- normally distributed example in 2d
  local bit = torch.Tensor(1,2)
  
  bit[1][1] = -1
  bit[1][2] = 1
  
  local output = torch.Tensor(1) 
  output[1] = torch.round(torch.uniform(1,2))
  local sign = bit[1][output[1]] -- Create two classes centered around \pm 5
  
  input = input:mul(input,sign)
  dataset[i] = {input, output}
end


mdl = nn.Sequential() -- Setup Feedforward connection
mdl:add(nn.SpatialConvolution(1,10,4,4)) -- output 10 x 7 x 7
mdl:add(nn.SpatialConvolution(10,1,4,4)) -- output 1 x 4 x 4
mdl:add(nn.Reshape(16)) -- reshape into Tensor(16)
mdl:add(nn.Linear(16,1))

--criterion = nn.ClassNLLCriterion()
criterion = nn.MSECriterion()

for i = 1,dataset:size() do
	local input  = dataset[i][1]
	local output = dataset[i][2]
	criterion:forward(mdl:forward(input),output)
	mdl:zeroGradParameters()
	mdl:backward(input,criterion:backward(mdl.output,output))
	mdl:updateParameters(0.001)
end

for i=1,dataset:size() do -- Show that it worked 
	print(mdl:forward(dataset[i][1]))
	print(dataset[i][2])
end
