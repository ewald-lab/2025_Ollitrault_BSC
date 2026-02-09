nextflow.enable.dsl=2

workflow processing_workflow {
    take:
    path_data_level2

    main:
    path_data_level3 = level3_aggregate_annotate(path_data_level2)
    path_data_level4a = level4a_normalize(path_data_level3)

    if( params.feature_selection ) {
        path_data_level4 = level4b_feature_selection(path_data_level4a)
    } else {
        path_data_level4 = path_data_level4a
    }

    path_data_level5 = level5_consensus(path_data_level4)

    path_data_level6 = level6_aggregate_profiles(path_data_level5)

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
    path input_file

    output:
    path "level5_${params.input_name}.parquet"

    script:
    """
    python ${projectDir}/modules/02_processing/scripts/consensus.py \\
        --input $input_file \\
        --output level5_${params.input_name}.parquet
    """
}

process level6_aggregate_profiles {
    publishDir "modules/02_processing/outputs", mode: 'copy'

    tag { "${params.input_name}" }

    input:
    path input_file

    output:
    path "level6_${params.input_name}.parquet"
    script:
    """
    python ${projectDir}/modules/02_processing/scripts/aggregate_profiles.py \\
        --input $input_file \\
        --output level6_${params.input_name}.parquet
    """
}
