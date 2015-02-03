local lfs = require 'lfs'
local torch = require 'torch'

local classesToNum = { 
    acantharia_protist= 1,
    acantharia_protist_big_center= 2,
    acantharia_protist_halo= 3,
    protist_other= 4,
    protist_star= 5,
    protist_fuzzy_olive= 6,
    protist_dark_center= 7,
    radiolarian_colony= 8,
    radiolarian_chain= 9,
    protist_noctiluca= 10,
    trichodesmium_tuft= 11,
    trichodesmium_bowtie= 12,
    trichodesmium_puff= 13,
    trichodesmium_multiple= 14,
    diatom_chain_string= 15,
    diatom_chain_tube= 16,
    appendicularian_fritillaridae= 17,
    appendicularian_s_shape= 18,
    appendicularian_slight_curve= 19,
    appendicularian_straight= 20,
    tunicate_doliolid= 21,
    tunicate_doliolid_nurse= 22,
    tunicate_salp= 23,
    tunicate_salp_chains= 24,
    tunicate_partial= 25,
    siphonophore_calycophoran_abylidae= 26,
    siphonophore_calycophoran_rocketship_adult= 27,
    siphonophore_calycophoran_rocketship_young= 28,
    siphonophore_calycophoran_sphaeronectes= 29,
    siphonophore_calycophoran_sphaeronectes_young= 30,
    siphonophore_calycophoran_sphaeronectes_stem= 31,
    siphonophore_physonect= 32,
    siphonophore_physonect_young= 33,
    siphonophore_other_parts= 34,
    siphonophore_partial= 35,
    hydromedusae_narcomedusae= 36,
    hydromedusae_narco_dark= 37,
    hydromedusae_solmaris= 38,
    hydromedusae_narco_young= 39,
    hydromedusae_solmundella= 40,
    hydromedusae_aglaura= 41,
    hydromedusae_liriope= 42,
    hydromedusae_haliscera= 43,
    hydromedusae_haliscera_small_sideview= 44,
    hydromedusae_bell_and_tentacles= 45,
    hydromedusae_typeD_bell_and_tentacles= 46,
    hydromedusae_typeD= 47,
    hydromedusae_shapeA= 48,
    hydromedusae_shapeA_sideview_small= 49,
    hydromedusae_sideview_big= 50,
    hydromedusae_typeE= 51,
    hydromedusae_shapeB= 52,
    hydromedusae_typeF= 53,
    hydromedusae_h15= 54,
    hydromedusae_other= 55,
    hydromedusae_partial_dark= 56,
    ctenophore_cestid= 57,
    ctenophore_cydippid_tentacles= 58,
    ctenophore_cydippid_no_tentacles= 59,
    ctenophore_lobate= 60,
    jellies_tentacles= 61,
    ephyra= 62,
    fish_larvae_leptocephali= 63,
    fish_larvae_myctophids= 64,
    fish_larvae_very_thin_body= 65,
    fish_larvae_thin_body= 66,
    fish_larvae_medium_body= 67,
    fish_larvae_deep_body= 68,
    copepod_cyclopoid_oithona= 69,
    copepod_cyclopoid_oithona_eggs= 70,
    copepod_cyclopoid_copilia= 71,
    copepod_calanoid= 72,
    copepod_calanoid_small_longantennae= 73,
    copepod_calanoid_frillyAntennae= 74,
    copepod_calanoid_flatheads= 75,
    copepod_calanoid_eggs= 76,
    copepod_calanoid_octomoms= 77,
    copepod_calanoid_large= 78,
    copepod_calanoid_large_side_antennatucked= 79,
    copepod_calanoid_eucalanus= 80,
    copepod_other= 81,
    shrimp_like_other= 82,
    euphausiids= 83,
    euphausiids_young= 84,
    decapods= 85,
    shrimp_zoea= 86,
    shrimp_caridean= 87,
    shrimp_sergestidae= 88,
    crustacean_other= 89,
    stomatopod= 90,
    amphipods= 91,
    chaetognath_sagitta= 92,
    chaetognath_non_sagitta= 93,
    chaetognath_other= 94,
    heteropod= 95,
    pteropod_butterfly= 96,
    pteropod_triangle= 97,
    pteropod_theco_dev_seq= 98,
    echinoderm_larva_seastar_bipinnaria= 99,
    echinoderm_larva_seastar_brachiolaria= 100,
    echinoderm_larva_pluteus_early= 101,
    echinoderm_larva_pluteus_urchin= 102,
    echinoderm_larva_pluteus_typeC= 103,
    echinoderm_larva_pluteus_brittlestar= 104,
    echinopluteus= 105,
    echinoderm_seacucumber_auricularia_larva= 106,
    trochophore_larvae= 107,
    tornaria_acorn_worm_larvae= 108,
    invertebrate_larvae_other_A= 109,
    invertebrate_larvae_other_B= 110,
    fecal_pellet= 111,
    detritus_blob= 112,
    detritus_filamentous= 113,
    detritus_other= 114,
    polychaete= 115,
    chordate_type1= 116,
    unknown_blobs_and_smudges= 117,
    unknown_sticks= 118,
    unknown_unclassified= 119,
    artifacts= 120,
    artifacts_edge= 121
}

function setValue(arr, index, d, dir, image)
    arr[index] = {}
    arr[index].classNum = classesToNum[dir]
    arr[index].className = dir
    arr[index].relPath = d..'/'..image
end

-- RETURNS array of tables storing the following:
    -- classNum: array index for the given class name see classesToNum table above
    -- className: actual name of the class
    -- relPath: image relative path
-- OTHER INFO
    -- only traverses 2 levels down
    -- assumes first level is directory of class names
    -- 2nd level is image files
function readTrainFiles(path)
    --array that stores information about the images
    local fileNames = {}
    local imageCounter = 1
    for dir in lfs.dir(path) do -- iterate through train directory (contains classNames)
        if dir ~= "." and dir ~= ".." then
            if string.match(dir, '-') then
                print 'Rename shrimp-like_other directory to shrimp_like_other!'
                return
            end
            local d = path..'/'..dir
            local attr = lfs.attributes (d)
            if attr.mode == "directory" then
                for image in lfs.dir(d) do
                    if image ~= "." and image ~= ".." then
                        setValue(fileNames, imageCounter, d, dir, image)
                        imageCounter = imageCounter + 1
                    end
                end
            end
        end
    end
    return fileNames
end

-- pct is percent of set for cross val
function readTrainAndCrossValFiles(path, pct)
    local trainFiles = {}
    local crossValFiles = {}
    local trainCounter = 1
    local crossValCounter = 1
    local categoryStartCounter = 0 --actually category start index - 1
    local categoryEndCounter = 0

    for dir in lfs.dir(path) do -- iterate through train directory (contains classNames)
        if dir ~= "." and dir ~= ".." then
            if string.match(dir, '-') then
                print 'Rename shrimp-like_other directory to shrimp_like_other!'
                return
            end
            local d = path..'/'..dir
            local attr = lfs.attributes (d)
            if attr.mode == "directory" then
                --count num files in dir
                for image in lfs.dir(d) do
                    if image ~= "." and image ~= ".." then
                        categoryEndCounter = categoryEndCounter + 1
                    end
                end
                local counter = 1
                local randomPermCounter = 1
                local catNumEl = categoryEndCounter-categoryStartCounter
                local randomPerm = torch.randperm(catNumEl):narrow(1,1,math.ceil(catNumEl/pct))
                randomPerm = torch.sort(randomPerm)
                local randomPermSize = randomPerm:size(1)
                for image in lfs.dir(d) do
                    if image ~= "." and image ~=".." then
                        if (randomPermCounter <= randomPermSize) and (counter == randomPerm[randomPermCounter]) then
                            randomPermCounter = randomPermCounter + 1
                            setValue(crossValFiles, crossValCounter, d, dir, image)
                            crossValCounter = crossValCounter + 1
                        else
                            setValue(trainFiles, trainCounter, d, dir, image)
                            trainCounter = trainCounter + 1
                        end
                        counter = counter + 1
                    end
                end
                categoryStartCounter = categoryEndCounter;
            end
        end
    end
    return trainFiles, crossValFiles
end

-- returns array containing all the test data file paths relative to current directory
function readTestFiles(path)
    local files = {}
    local counter = 1
    for image in lfs.dir(path) do
        if image ~= "." and image ~= ".." then
            files[counter] = {}
            files[counter].relPath = path..'/'..image
            files[counter].name = image
            counter = counter + 1
        end
    end
    return files
end

return {
    readTrainFiles = readTrainFiles,
    readTrainAndCrossValFiles = readTrainAndCrossValFiles,
    readTestFiles = readTestFiles,
    classesToNum = classesToNum
}

-- randomnotes about require and table length
-- ex for requiring in different directory: 
-- require('project/loadData.lua')
-- if in same directory can do: require 'loadData.lua'
-- #table - # is the length operator
-- ex. #fileNames - will output the number of files
-- for torch, use image.load to actually load files
-- readTrainFiles('/users/h/downloads/data_512/train')