workflow classification_workflow {

    take:
        path_data_level5

    main:
        // Run Level 6 aggregate by compounds
        path_data_level6 = level6_aggregate_compound(path_data_level5)


    emit:
        path_data_level6
}


process level6_aggregate_compound{
    publishDir "02_processed_data", mode: 'copy'

    tag { "${input_file.baseName}" }

    input:
    path input_file

    output:
    path "level6_${params.input_name}.parquet"

    script:
    """
    python ${projectDir}/01_analysis/2F_data_processing_level6_aggregation_compounds.py \\
        --input $input_file \\
        --output level6_${params.input_name}.parquet
    """
}

process axiom_assay_hitcall{

    input:
        path inputs//f"outputs/{features}/{name}/curves/mttpods.parquet",
        //f"outputs/{features}/{name}/curves/ldhpods.parquet",
        //f"outputs/{features}/{name}/curves/ccpods.parquet",
    output:
        path "axiom_hits.parquet",
    script:
    """
    python ${projectDir}/01_analysis/7_script_classifier/7_data_axiom_hitcalls.py \
         --input_pod_mtt 03_tables/pods_axiom_Metadata_mtt_normalized_original_conc.parquet\
         --input_pod_ldh 03_tables/pods_axiom_Metadata_ldh_normalized_original_conc.parquet \
         --input_pod_cc 03_tables/pods_axiom_Metadata_cc_normalized_original_conc.parquet \ !! CC POD
         --output 03_tables/test.parquet 

    """
}
/*
process toxcast_cellbased_binary{ // Check again if to use OASIS_ID or compound ID
    input:
        path input //f"outputs/{features}/{name}/aggregated_profiles/agg.parquet",
        //f"inputs/annotations/toxcast_cellbased_binary.parquet",
    output:
        f"axiom_toxcast_cellbased_binary_predictions.parquet",
    script:
    """
    python ${projectDir}/01_analysis/7_script_classifier/7_data_classify.py \
         --input_path 02_processed_data/level6_axiom.parquet\
         --label_path 03_tables/test.parquet \
         --output 03_tables/test_class.parquet 
    """
}
*/