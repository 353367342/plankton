require 'nn'
require 'cunn'
require 'cutorch'
require 'image'
require('randTransform.lua')

--fSize = {1,16,16,32}
fSize = {1,256,256,256,512,512,600,692}
featuresOut = fSize[7]*2*2
hiddenNodes = {512,256}
--hiddenNodes = {64,32}

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(fSize[1],fSize[2],2,2,2,2)) -- (120 - 2 + 2)/2 = 60
features:add(nn.Threshold(0,1e-6))
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2)) -- 30
features:add(nn.SpatialConvolutionMM(fSize[2],fSize[3],3,3)) -- 28
features:add(nn.Threshold(0,1e-6))
features:add(nn.ReLU())
features:add(nn.SpatialMaxPooling(2,2)) -- 14
features:add(nn.SpatialConvolutionMM(fSize[3],fSize[4],4,4)) -- 11 
features:add(nn.Threshold(0,1e-6))
features:add(nn.ReLU())
--features:add(nn.SpatialMaxPooling(2,2)) -- 4
features:add(nn.SpatialConvolutionMM(fSize[4],fSize[5],4,4)) -- 8
features:add(nn.Threshold(0,1e-6))
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[5],fSize[6],3,3)) -- 6
features:add(nn.Threshold(0,1e-6))
features:add(nn.ReLU())
features:add(nn.SpatialConvolutionMM(fSize[6],fSize[7],3,3)) -- 4
features:add(nn.SpatialMaxPooling(2,2)) -- 2
features:add(nn.View(featuresOut))

dropout_p = 0.5

-- 1-10
protists = nn.Sequential()
protists:add(nn.Dropout(dropout_p))
protists:add(nn.Linear(featuresOut,hiddenNodes[1]))
protists:add(nn.Threshold(0,1e-6))
protists:add(nn.Dropout(dropout_p))
protists:add(nn.Linear(hiddenNodes[1],10))

-- 11-14
trichodesmium = nn.Sequential()
trichodesmium:add(nn.Dropout(dropout_p))
trichodesmium:add(nn.Linear(featuresOut,hiddenNodes[1]))
trichodesmium:add(nn.Threshold(0,1e-6))
trichodesmium:add(nn.Dropout(dropout_p))
trichodesmium:add(nn.Linear(hiddenNodes[1],4))

-- 15-16
diatoms = nn.Sequential()
diatoms:add(nn.Dropout(dropout_p))
diatoms:add(nn.Linear(featuresOut,hiddenNodes[1]))
diatoms:add(nn.Threshold(0,1e-6))
diatoms:add(nn.Dropout(dropout_p))
diatoms:add(nn.Linear(hiddenNodes[1],2))

-- 17-62
gelatZoo ={}
gelatZoo[1] = nn.Sequential()
gelatZoo[1]:add(nn.Dropout(dropout_p))
gelatZoo[1]:add(nn.Linear(featuresOut,hiddenNodes[1]))
gelatZoo[1]:add(nn.Threshold(0,1e-6))
gelatZoo[1]:add(nn.Dropout(dropout_p))
gelatZoo[1]:add(nn.Linear(hiddenNodes[1],hiddenNodes[2]))

--17-25
pelagicTunicate = nn.Sequential()
pelagicTunicate:add(nn.Dropout(dropout_p))
pelagicTunicate:add(nn.Threshold(0,1e-6))
pelagicTunicate:add(nn.Linear(hiddenNodes[2],9))

--26-35
siphonophore = nn.Sequential()
siphonophore:add(nn.Dropout(dropout_p))
siphonophore:add(nn.Threshold(0,1e-6))
siphonophore:add(nn.Linear(hiddenNodes[2],10))

--36-56
hydromedusae = nn.Sequential()
hydromedusae:add(nn.Dropout(dropout_p))
hydromedusae:add(nn.Threshold(0,1e-6))
hydromedusae:add(nn.Linear(hiddenNodes[2],21))

--57-60
ctenophore = nn.Sequential()
ctenophore:add(nn.Dropout(dropout_p))
ctenophore:add(nn.Threshold(0,1e-6))
ctenophore:add(nn.Linear(hiddenNodes[2],4))

--61-62
otherGelatZoo = nn.Sequential()
otherGelatZoo:add(nn.Dropout(dropout_p))
otherGelatZoo:add(nn.Threshold(0,1e-6))
otherGelatZoo:add(nn.Linear(hiddenNodes[2],2))

gelatZoo[2] = nn.Concat(2)
gelatZoo[2]:add(pelagicTunicate)
gelatZoo[2]:add(siphonophore)
gelatZoo[2]:add(hydromedusae)
gelatZoo[2]:add(ctenophore)
gelatZoo[2]:add(otherGelatZoo)

gelatZoo[1]:add(gelatZoo[2])

--63-68
fish = nn.Sequential()
fish:add(nn.Dropout(dropout_p))
fish:add(nn.Linear(featuresOut,hiddenNodes[1]))
fish:add(nn.Threshold(0,1e-6))
fish:add(nn.Dropout(dropout_p))
fish:add(nn.Linear(hiddenNodes[1],6))

--69-91
crustaceans = {}
crustaceans[1] = nn.Sequential()
crustaceans[1]:add(nn.Dropout(dropout_p))
crustaceans[1]:add(nn.Linear(featuresOut,hiddenNodes[1]))
crustaceans[1]:add(nn.Threshold(0,1e-6))
crustaceans[1]:add(nn.Dropout(dropout_p))
crustaceans[1]:add(nn.Linear(hiddenNodes[1],hiddenNodes[2]))

--69-81
copepods = nn.Sequential()
copepods:add(nn.Dropout(dropout_p))
copepods:add(nn.Threshold(0,1e-6))
copepods:add(nn.Linear(hiddenNodes[2],13))

--82-88
shrimpLike = nn.Sequential()
shrimpLike:add(nn.Dropout(dropout_p))
shrimpLike:add(nn.Threshold(0,1e-6))
shrimpLike:add(nn.Linear(hiddenNodes[2],7))

--89-91
otherCrust = nn.Sequential()
otherCrust:add(nn.Dropout(dropout_p))
otherCrust:add(nn.Threshold(0,1e-6))
otherCrust:add(nn.Linear(hiddenNodes[2],3))

crustaceans[2] = nn.Concat(2)
crustaceans[2]:add(copepods)
crustaceans[2]:add(shrimpLike)
crustaceans[2]:add(otherCrust)

crustaceans[1]:add(crustaceans[2])

--92-94
chaetognath = nn.Sequential()
chaetognath:add(nn.Dropout(dropout_p))
chaetognath:add(nn.Linear(featuresOut,hiddenNodes[1]))
chaetognath:add(nn.Threshold(0,1e-6))
chaetognath:add(nn.Dropout(dropout_p))
chaetognath:add(nn.Linear(hiddenNodes[1],3))

--95-98
gastropod = nn.Sequential()
gastropod:add(nn.Dropout(dropout_p))
gastropod:add(nn.Linear(featuresOut,hiddenNodes[1]))
gastropod:add(nn.Threshold(0,1e-6))
gastropod:add(nn.Dropout(dropout_p))
gastropod:add(nn.Linear(hiddenNodes[1],4))

--99-110
otherInvertLarvae = {}
otherInvertLarvae[1] = nn.Sequential()
otherInvertLarvae[1]:add(nn.Dropout(dropout_p))
otherInvertLarvae[1]:add(nn.Linear(featuresOut,hiddenNodes[1]))
otherInvertLarvae[1]:add(nn.Threshold(0,1e-6))
otherInvertLarvae[1]:add(nn.Dropout(dropout_p))
otherInvertLarvae[1]:add(nn.Linear(hiddenNodes[1],hiddenNodes[2]))

--99-106
echinoderms = nn.Sequential()
echinoderms:add(nn.Dropout(dropout_p))
echinoderms:add(nn.Threshold(0,1e-6))
echinoderms:add(nn.Linear(hiddenNodes[2],8))

otherOtherInvertLarvae = nn.Sequential()
otherOtherInvertLarvae:add(nn.Dropout(dropout_p))
otherOtherInvertLarvae:add(nn.Threshold(0,1e-6))
otherOtherInvertLarvae:add(nn.Linear(hiddenNodes[2],4))

otherInvertLarvae[2] = nn.Concat(2)
otherInvertLarvae[2]:add(echinoderms)
otherInvertLarvae[2]:add(otherOtherInvertLarvae)

otherInvertLarvae[1]:add(otherInvertLarvae[2])

--111-114
detritus = nn.Sequential()
detritus:add(nn.Dropout(dropout_p))
detritus:add(nn.Linear(featuresOut,hiddenNodes[1]))
detritus:add(nn.Threshold(0,1e-6))
detritus:add(nn.Dropout(dropout_p))
detritus:add(nn.Linear(hiddenNodes[1],4))

--115-116
otherPlankton = nn.Sequential()
otherPlankton:add(nn.Dropout(dropout_p))
otherPlankton:add(nn.Linear(featuresOut,hiddenNodes[1]))
otherPlankton:add(nn.Threshold(0,1e-6))
otherPlankton:add(nn.Dropout(dropout_p))
otherPlankton:add(nn.Linear(hiddenNodes[1],2))

--117-121
unknownOrArtifacts = nn.Sequential()
unknownOrArtifacts:add(nn.Dropout(dropout_p))
unknownOrArtifacts:add(nn.Linear(featuresOut,hiddenNodes[1]))
unknownOrArtifacts:add(nn.Threshold(0,1e-6))
unknownOrArtifacts:add(nn.Dropout(dropout_p))
unknownOrArtifacts:add(nn.Linear(hiddenNodes[1],5))

dgraph = nn.Concat(2)
dgraph:add(protists)
dgraph:add(trichodesmium)
dgraph:add(diatoms)
dgraph:add(gelatZoo[1])
dgraph:add(fish)
dgraph:add(crustaceans[1])
dgraph:add(chaetognath)
dgraph:add(gastropod)
dgraph:add(otherInvertLarvae[1])
dgraph:add(detritus)
dgraph:add(otherPlankton)
dgraph:add(unknownOrArtifacts)

mdl = nn.Sequential()
mdl:add(features)
mdl:add(dgraph)
mdl:add(nn.LogSoftMax())
mdl:cuda()

--output = torch.CudaTensor(10):fill(1)
--
--criterion = nn.ClassNLLCriterion()
--criterion:cuda()
--
--for i=1,30 do
--  local currentError = 0
--  input = torch.randn(1,1,512,512)
--  input = randomTransform(input[1][1],10):cuda()
--  oHat = mdl:forward(input)
--  currentError = currentError + criterion:forward(oHat,output)
--  mdl:zeroGradParameters()
--  mdl:backward(input,criterion:backward(mdl.output,output))
--  mdl:updateParameters(6e-1)
--  if i % 10 == 0 then
--    print('Batch:',i,'Error:',currentError/58)
--  end
--  collectgarbage()
--end
--
--print(torch.pow(10,oHat:float()))
