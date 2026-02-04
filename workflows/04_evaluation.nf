nextflow.enable.dsl=2

// ------------------- Evaluation Workflow -------------------
nextflow.enable.dsl=2

workflow evaluation_workflow {
    take:
        input_data_level4
        input_data_level5

    main:
        // path_data_3a = evaluate_3a_replicate_reproducibility(input_data_level4)
        path_data_3c = evaluate_3c_grit(input_data_level4)
        path_data_3d = evaluate_3d_ap_map(input_data_level4)

}

process evaluate_3a_replicate_reproducibility {
    errorStrategy 'finish'
    memory '128 GB'
    cpus 4
    time '2h'

    publishDir "04_figures", mode: 'copy', pattern: '*.png'
    publishDir "03_tables", mode: 'copy', pattern: '*.csv'

    tag { "${input_file.baseName}" }

    input:
        path input_file

    output:
        path "3a_replicate_reproducibility_${params.input_name}.png"
        path "3a_replicate_reproducibility_${params.input_name}.csv"

    script:
    """
    python ${projectDir}/01_analysis/3A_data_evaluation_replicate_reproducibility.py \\
        --input $input_file \\
        --output 3a_replicate_reproducibility_${params.input_name} 
    """
}



/*
process evaluate_3b_precision_recall {
    errorStrategy 'finish'
    memory '128 GB'
    cpus 4
    time '2h'

    publishDir "04_figures", mode: 'copy', pattern: "*.png"
    publishDir "03_tables",  mode: 'copy', pattern: "*.parquet"

    tag { "${input_file.baseName}_${grouping}_${prefix_feature}" }

    input:
        tuple val(grouping), val(prefix_feature), path(input_file)

    output:
    path "3b_precision_recall_grp_${grouping}_feature_${prefix_feature}.png"
    path "3b_precision_recall_grp_${grouping}_feature_${prefix_feature}.parquet"

    script:
    """
    echo test 2
    python ${projectDir}/01_analysis/3B_data_evaluation_precision_recall.py \\
        --input $input_file \\
        --output 3b_precision_recall_grp_${grouping}_feature_${prefix_feature} \\
        --grouping ${grouping} \\
        --prefix_feature ${prefix_feature}

    """
}
*/
process evaluate_3c_grit {
    errorStrategy 'finish'
    cpus 4
    time '2h'

    // Tables to 03_tables
    publishDir "03_tables", mode: 'copy', pattern: '*.csv'

    tag { "${input_file.baseName}" }

    input:
        path input_file

    output:
        path "3c_grit_${params.input_name}.csv"

    script:
    """
    python ${projectDir}/01_analysis/3C_data_evaluation_grit.py \\
        --input $input_file \\
        --output 3c_grit_${params.input_name} \\
        --control ${params.control}
    """
}

process evaluate_3d_ap_map {
    errorStrategy 'finish'
    memory '128 GB'
    cpus 4
    time '2h'

    // Tables to 03_tables
    publishDir "03_tables", mode: 'copy', pattern: '*.csv'

    tag { "${input_file.baseName}" }

    input:
        path input_file

    output:
    path "3d_map_${params.input_name}_ap.csv"
    path "3d_map_${params.input_name}_map.csv"

    script:
    """
    python ${projectDir}/01_analysis/3D_data_evaluation_map.py \\
        --input $input_file \\
        --output 3d_map_${params.input_name} \\
        --control ${params.control}
    """
}
