```
ingest = dofile('loadData.lua')
dataset = ingest.readTrainFiles('[path to data_128/train]');
dofile('model.lua')
dofile('train.lua')

```