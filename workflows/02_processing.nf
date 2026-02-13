nextflow.enable.dsl=2

workflow processing_workflow {
    take:
    path_data_level2

    main:
    path_data_level3 = level3_aggregate_annotate(path_data_level2)
    path_data_level4a = level4a_normalize(path_data_level3)

    if( params.feature_selection ) {
        path_data_level4b = level4b_feature_selection(path_data_level4a)
        // Create tuples with labels
        all_features = path_data_level4a.map { file -> tuple('all_features', file) }
        selected_features = path_data_level4b.map { file -> tuple('selected_features', file) }
        // Mix both paths into one channel
        combined = all_features.mix(selected_features)
        // Process both in parallel
        path_data_level5_combined = level5_consensus(combined)
        path_data_level6_combined = level6_aggregate_profiles(path_data_level5_combined)
        // Split level5 outputs by feature type
        path_data_level5_combined
            .branch {
                all: it[0] == 'all_features'
                selected: it[0] == 'selected_features'
            }
            .set { level5_outputs }
        // Split level6 outputs by feature type
        path_data_level6_combined
            .branch {
                all: it[0] == 'all_features'
                selected: it[0] == 'selected_features'
            }
            .set { level6_outputs }
        // Extract files from tuples
        path_data_level5_all = level5_outputs.all.map { it[1] }
        path_data_level5_selected = level5_outputs.selected.map { it[1] }
        path_data_level6_all = level6_outputs.all.map { it[1] }
        path_data_level6_selected = level6_outputs.selected.map { it[1] }
        // Set defaults (selected features)
        path_data_level4 = path_data_level4b
        path_data_level5 = path_data_level5_selected
        path_data_level6 = path_data_level6_selected
    } else {
        all_features = path_data_level4a.map { file -> tuple('all_features', file) }
        path_data_level5_tuples = level5_consensus(all_features)
        path_data_level6_tuples = level6_aggregate_profiles(path_data_level5_tuples)
        // Extract files from tuples
        path_data_level5 = path_data_level5_tuples.map { it[1] }
        path_data_level6 = path_data_level6_tuples.map { it[1] }
        // Set defaults
        path_data_level4 = path_data_level4a
    }

    emit:
    path_data_level2
    path_data_level3
    path_data_level4
    path_data_level5
    path_data_level6
}

process level3_aggregate_annotate {
    publishDir "modules/02_processing/outputs", mode: 'copy'

    tag { "${params.input_name}" }

    input:
    path input_file

    output:
    path "level3_${params.input_name}.parquet"

    script:
    """
    python ${projectDir}/modules/02_processing/scripts/aggregate_annotate.py \\
        --input $input_file \\
        --output level3_${params.input_name}.parquet
    """
}

process level4a_normalize {
    publishDir "modules/02_processing/outputs", mode: 'copy'

    tag { "${params.input_name}" }

    input:
    path input_file

    output:
    path "level4a_${params.input_name}.parquet"

    script:
    """
    python ${projectDir}/modules/02_processing/scripts/normalize.py \\
        --input $input_file \\
        --output level4a_${params.input_name}.parquet \\
        --control ${params.control}
    """
}

process level4b_feature_selection {
    publishDir "modules/02_processing/outputs", mode: 'copy'

    tag { "${params.input_name}" }

    input:
    path input_file

    output:
    path "level4b_${params.input_name}.parquet"

    script:
    """
    python ${projectDir}/modules/02_processing/scripts/feature_selection.py \\
        --input $input_file \\
        --output level4b_${params.input_name}.parquet
    """
}

process level5_consensus {
    publishDir "modules/02_processing/outputs", mode: 'copy'

    tag { "${params.input_name}" }

    input:
    tuple val(feature_type), path(input_file)

    output:
    tuple val(feature_type), path("level5_${params.input_name}_${feature_type}.parquet")

    script:
    """
    python ${projectDir}/modules/02_processing/scripts/consensus.py \\
        --input $input_file \\
        --output level5_${params.input_name}_${feature_type}.parquet
    """
}

process level6_aggregate_profiles {
    publishDir "modules/02_processing/outputs", mode: 'copy'

    tag { "${params.input_name}" }

    input:
    tuple val(feature_type), path(input_file)

    output:
    tuple val(feature_type), path("level6_${params.input_name}_${feature_type}.parquet")

    script:
    """
    python ${projectDir}/modules/02_processing/scripts/aggregate_profiles.py \\
        --input $input_file \\
        --output level6_${params.input_name}_${feature_type}.parquet
    """
}
