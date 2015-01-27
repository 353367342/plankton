var lineReader = require('line-reader');
var fs = require('fs');
var categoriesOnlyFile = 'categoriesOnly.txt';
fs.truncate(categoriesOnlyFile, 0, function(){
  var classesTreeFile = 'classTree.txt';
  var l;
  lineReader.eachLine(classesTreeFile, function(line) {
    l = line.replace(/\t/g,'');
    l = l.charAt(0);
    if (l.toLowerCase() === l) {
      fs.appendFile(categoriesOnlyFile, line+'\n', function (err) {
        if (err) {
          console.log(err);
        }
      });
    }
  });
})