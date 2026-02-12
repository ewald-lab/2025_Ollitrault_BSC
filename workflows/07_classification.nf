nextflow.enable.dsl = 2

workflow classification_workflow {
    take:
        path_data_level6
        compute_meta_pods
    
    main:
        pods_map_ch = compute_meta_pods
            .toList()
            .map { tuples ->
                tuples.collectEntries { label, file -> 
                    [(label): file] 
                }
            }
        
        axiom_assay_hitcall(pods_map_ch)
    
    emit:
        path_data_level6
}


process axiom_assay_hitcall {

    input:
        val pods

    output:
        path "axiom_hits.parquet"

    script:
    """
    python ${projectDir}/modules/07_classification/scripts/hitcalls.py \
        --input_pod_mtt ${pods["Metadata_mtt_normalized"]} \
        --input_pod_ldh ${pods["Metadata_ldh_normalized"]} \
        --input_pod_cc ${pods["Metadata_Count_Cells"]} \
        --output axiom_hits.parquet
    """
}

/*
process toxcast_cellbased_binary{ // Check again if to use OASIS_ID or compound ID
label gpu !!
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