MUST RENAME shrimp-like_other to shrimp_like_other
the dash matters for lua tables

Note unknown and artifacts are grouped together (last group)
Red boxes that are on the same level as blue boxes go to the end of the group
The exception is unknown/artifacts (which is the absolute end group)

Only Blue boxes: (Pulled from categoriesOnly.txt and pdf found at
https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3978/plankton_identification.pdf?sv=2012-02-12&se=2015-01-30T05%3A18%3A09Z&sr=b&sp=r&sig=uIVvBgdnf2UqhWSza28QT3YxGF%2B4YRxezDPXwSQwLpE%3D)
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
- loadData.lua - script when required, loads the following table:
```
{
  readTrainFiles: gets all image urls along with class name and class index from training directory
  readTestFiles: gets all image urls for test files
  classesToNum: returns the mapping of classes to class indicies
}
```
- tableparser.js - outputs to stdout json for mapping classnames to integers
- removeBroadCategories.js - writes to categoriesOnly.txt (used for counting class groups to produce the above array indicies)
- classTreeOrig.txt - tabbed tree pulled from kaggle forum
- classTree.txt - reorganized classTreeOrig.txt file (red boxes that are on the same level as blue boxes go to the end of the group)
- classesToInt - text file with lua table for mapping class names to array indicies (also copied and pasted in loadData.lua)

Note to run javascript files:
```
npm install
node [filename]
```