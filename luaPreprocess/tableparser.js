var tabdown = require('tabdown');
var lineReader = require('line-reader');

var classesTreeFile = 'classTree.txt';
var numClasses = 121;
var isFirstCharUpperCase = function(str) {
  var c = str.charAt(0);
  return c.toLowerCase() !== c;
};

var lines = [];
var tree = {};
lineReader.eachLine(classesTreeFile, function(line, isLast) {
  lines.push(line);
  if (isLast) {
    tree = tabdown.parse(lines,'\t');
    var counter = 1;
    var strToIntMap = {};
    var data;
    tabdown.traverse(tree, function(node) {
      data = node.data.replace('\t','');
      if (!isFirstCharUpperCase(data)) {
        strToIntMap[data] = counter;
        ++counter;
      }
      if (counter === numClasses+1) { //edge case
        //just need to replace each ':' with an '=' to make it a lua table
        console.log(strToIntMap);
      }
    });
  }
});