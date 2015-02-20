usage of merge_multiple_files.cpp (arbitrarily many files)
```
g++ --std=c++11 merge_output_files.cpp
./a.out <input file 1> <input file 2> ... <input file n> <output file>
```

usage of merge_output_files.cpp (2 files)
```
g++ --std=c++11 merge_output_files.cpp
./a.out <input file 1> <input file 2> <output file>
```

usage of threshold.cpp (thresholds 1 file)
```
g++ --std=c++11 threshold.cpp
./a.out <input file> <output file> <threshold>
```