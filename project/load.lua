local lfs = require 'lfs'
--TODO: make it a module

-- returns 2 arrays: fileNames, classMap
-- only traverses 2 levels down
-- assumes first level is directory of class names
-- 2nd level is image files
-- modifies fileNames to store image file names (relative to your current directory)
-- and classIds (integer >=1)
-- modifies classMap to store an array mapping classIds to classNames
function readTrainFiles(path)
    --array that stores tables 
    -- {name: [relative path to image], class: [integer class to index into classMap]}
    local fileNames = {}
    --array that stores classmap from int to string i.e. index classMap[i]
    local classMap = {}
    local classCounter = 1
    local imageCounter = 1
    for dir in lfs.dir(path) do
        if dir ~= "." and dir ~= ".." then
            local d = path..'/'..dir
            local attr = lfs.attributes (d)
            if attr.mode == "directory" then
                classMap[classCounter] = dir
                for image in lfs.dir(d) do
                    if image ~= "." and image ~= ".." then
                        fileNames[imageCounter] = {}
                        fileNames[imageCounter].class = classCounter
                        fileNames[imageCounter].name = d..'/'..image
                        imageCounter = imageCounter + 1
                    end
                end
                classCounter = classCounter + 1
            end
        end
    end
    return fileNames, classMap
end

function readTestFiles(path)
    --array that stores all the file names relative to current path
    local files = {}
    local counter = 0
    for image in lfs.dir(path) do
        if image ~= "." and image ~= ".." then
            files[counter] = path..'/'..image
            counter = counter + 1
        end
    end
    return files
end

-- table.getn(table)
-- ex. table.getn(fileNames) - will output the number of files
-- use image.load to actually load files

-- readFiles('/users/h/downloads/data_512/train')