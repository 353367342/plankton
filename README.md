MUST RENAME shrimp-like_other to shrimp_like_other
the dash matters for lua tables

```
Array Indicies:
Plankton: 1-116
  Protists: 1-10
  Trichodesmium: 11-14
  Diatoms: 15-16
  Gelatinous_Zooplankton: 17-62
    Pelagic_Tunicate: 17-25
    Siphonophore: 26-35
    Hydromedusae: 36-56
    Ctenophore: 57-60
  Fish: 63-68
  Crustaceans: 69-91
    Copepods: 69-81
    Shrimp-like: 82-88
  Chaetognath: 92-94
  Gastropod: 95-98
  Other_Invert_Larvae: 99-110
    Echinoderms: 99-106
  Detritus: 111-114
Unknown/Artifacts: 117-121
```

What each file does:
- project/loadData.lua - script when required, loads the following table:
```
{
  readTrainFiles: gets all image urls along with class name and class index from training directory
  readTestFiles: gets all image urls for test files
  classesToNum: returns the mapping of classes to class indicies
}
```
- luaPreprocess/classTreeOrig.txt - tabbed tree pulled from kaggle forum
- luaPreprocess/classTree.txt - reorganized classTreeOrig.txt file (red boxes that are on the same level as blue boxes go to the end of the group)
- luaPreprocess/classesToInt - text file with lua table for mapping class names to array indicies (also copied and pasted in loadData.lua)
