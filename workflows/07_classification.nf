nextflow.enable.dsl = 2

workflow classification_workflow {
    take:
        path_data_level4
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
        // Hit call axiom
        path_axiom_hits = axiom_assay_hitcall(pods_map_ch)

        // Create channel for label files with their label column
        label_files_ch = Channel.fromList(params.classifier.label_files)
            .map { label_file -> 
                tuple(
                    file("${projectDir}/modules/07_classification/inputs/${label_file}"),
                    "OASIS_ID"
                )
            }

        // Add axiom_hits with "ID" label column
        all_labels_ch = label_files_ch
            .concat(
                path_axiom_hits.map { file -> tuple(file, "ID") }
            )


        // Run classification for all label files
        path_output_classification = path_data_level6
                    .combine(path_data_level4)
                    .combine(all_labels_ch)
                    | binary_classification
       
    emit:
        path_data_level6
}


process axiom_assay_hitcall {
    tag { "${params.input_name}" }

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


process binary_classification {

    tag { "${label_file.baseName}" }

    label 'gpu'

    publishDir "modules/07_classification/outputs/", mode: 'copy', pattern: '*.parquet'
    publishDir "modules/07_classification/outputs", mode: 'copy', pattern: 'models/*.pkl'  // Changed!
    
    input:
        tuple path(input_level6), path(input_level4), path(label_file), val(label_column)
    
    output:
        path "axiom_*_predictions.parquet"
        path "models/*.pkl"  // Make sure this line exists!
    
    script:
    def label_name = label_file.baseName
    """
    python ${projectDir}/modules/07_classification/scripts/classify.py \
        --input_path_level6 ${input_level6} \
        --input_path_level4 ${input_level4} \
        --label_path ${label_file.name} \
        --label_column ${label_column} \
        --output axiom_${label_name}_predictions.parquet \
        --n_repeats ${params.classifier.n_repeats} \
        --filter_label_size ${params.classifier.filter_label_size}
    """
}
