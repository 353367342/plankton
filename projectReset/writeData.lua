-- classHeader is generated from the table defined in loadData.lua via flipping key value pairs
-- image is also appended at the beginning of the header
-- creates a file/clears existing file and writes the header
function openFile(path)
    local classHeader = 'image,acantharia_protist,acantharia_protist_big_center,acantharia_protist_halo,protist_other,protist_star,protist_fuzzy_olive,protist_dark_center,radiolarian_colony,radiolarian_chain,protist_noctiluca,trichodesmium_tuft,trichodesmium_bowtie,trichodesmium_puff,trichodesmium_multiple,diatom_chain_string,diatom_chain_tube,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,tunicate_doliolid,tunicate_doliolid_nurse,tunicate_salp,tunicate_salp_chains,tunicate_partial,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_physonect,siphonophore_physonect_young,siphonophore_other_parts,siphonophore_partial,hydromedusae_narcomedusae,hydromedusae_narco_dark,hydromedusae_solmaris,hydromedusae_narco_young,hydromedusae_solmundella,hydromedusae_aglaura,hydromedusae_liriope,hydromedusae_haliscera,hydromedusae_haliscera_small_sideview,hydromedusae_bell_and_tentacles,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_shapeA,hydromedusae_shapeA_sideview_small,hydromedusae_sideview_big,hydromedusae_typeE,hydromedusae_shapeB,hydromedusae_typeF,hydromedusae_h15,hydromedusae_other,hydromedusae_partial_dark,ctenophore_cestid,ctenophore_cydippid_tentacles,ctenophore_cydippid_no_tentacles,ctenophore_lobate,jellies_tentacles,ephyra,fish_larvae_leptocephali,fish_larvae_myctophids,fish_larvae_very_thin_body,fish_larvae_thin_body,fish_larvae_medium_body,fish_larvae_deep_body,copepod_cyclopoid_oithona,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_copilia,copepod_calanoid,copepod_calanoid_small_longantennae,copepod_calanoid_frillyAntennae,copepod_calanoid_flatheads,copepod_calanoid_eggs,copepod_calanoid_octomoms,copepod_calanoid_large,copepod_calanoid_large_side_antennatucked,copepod_calanoid_eucalanus,copepod_other,shrimp_like_other,euphausiids,euphausiids_young,decapods,shrimp_zoea,shrimp_caridean,shrimp_sergestidae,crustacean_other,stomatopod,amphipods,chaetognath_sagitta,chaetognath_non_sagitta,chaetognath_other,heteropod,pteropod_butterfly,pteropod_triangle,pteropod_theco_dev_seq,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_urchin,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_brittlestar,echinopluteus,echinoderm_seacucumber_auricularia_larva,trochophore_larvae,tornaria_acorn_worm_larvae,invertebrate_larvae_other_A,invertebrate_larvae_other_B,fecal_pellet,detritus_blob,detritus_filamentous,detritus_other,polychaete,chordate_type1,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified,artifacts,artifacts_edge'
    local file = assert(io.open(path, 'w+'))
    file:write(classHeader)
    file:write('\n')
    return file
end

-- batchPrediction is a 2D torch tensor (1st dimension is example, 2nd is classes)
-- fileNames is a table of 1st dimension batchPrediction length long
-- file is a file object
function writeBatch(batchPrediction, fileNames, file)
    local i,j
    local size = #batchPrediction
    for i=1,size[1] do
        file:write(fileNames[i])
        for j=1,size[2] do
            file:write(',' .. batchPrediction[i][j])
        end
        file:write('\n')
    end
end


-- close file is call: file:close()

return {
    openFile = openFile,
    writeBatch = writeBatch
}