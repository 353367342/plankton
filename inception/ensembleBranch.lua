function ensemble_branch(inputSz, avgPoolSz, avgPoolStride)
    ensemble = nn.Sequential()
--    ensemble:add(nn.SpatialAveragePooling(avgPoolSz,avgPoolSz,avgPoolStride,avgPoolStride))
    ensemble:add(nn.SpatialAdaptiveMaxPooling(1,1))
    ensemble:add(nn.SpatialConvolutionMM(inputSz,512,1,1,1,1))
    ensemble:add(nn.View(512))
    ensemble:add(nn.Linear(512,422))
    ensemble:add(nn.ReLU())
    ensemble:add(nn.Dropout(0.7))
    ensemble:add(nn.Linear(422,121))
    ensemble:add(nn.LogSoftMax())
    ensemble:cuda()
    return ensemble
end