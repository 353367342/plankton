Sample Acquisition Files
-----------------------

What each file does:
- loadData.lua - script when required, returns the following table:
```
{
  readTrainFiles: gets all image urls along with class name and class index from training directory
  readTestFiles: gets all image urls for test files
  classesToNum: returns the mapping of classes to class indicies
}
```

- writeData.lua - script when required, returns the following table:
```
{
    openFile: creates and returns a file, writes the class header to the file,
    writeBatch: writes a batch of predictions to file
}
```