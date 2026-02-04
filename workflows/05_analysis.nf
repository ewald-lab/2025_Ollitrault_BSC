nextflow.enable.dsl = 2

workflow analysis_workflow {
    take:
        path_data_level2
        path_data_level3
        path_data_level4
        path_data_level5
    
    main:
        // 1. Raw analysis level 2
        raw_analysis_results = level2_analysis(path_data_level2)
        // 2. Raw analysis level 3
        raw_analysis_results = level3_analysis(path_data_level3)
        // 3. Transformed analysis
        level4_analysis_results = level4_analysis(path_data_level4)
}
// Create a second workflow for processed data analysis
workflow analysis_workflow_processed {
    take:
        path_data_level2
        path_data_level3
        path_data_level4
        path_data_level5
    main:
        level2_analysis(path_data_level2)
        level3_analysis(path_data_level3)
        level4_analysis(path_data_level4)
    emit:
        level2_analysis.out
        level3_analysis.out
}

// 1. Raw analysis
// marimo export html modules/08_reporting/notebooks/1_analysis_formated_data.marimo.py -o modules/08_reporting/outputs/1_analysis_formated_data_marimo_bsc.html -- --name bsc --input_path 02_processed_data/level2_bsc.parquet

process level2_analysis {
    publishDir "modules/08_reporting/outputs", mode: 'copy'


    tag { "${path_data_level2.baseName}" }

    input:
    path path_data_level2

    output:
    path "1_analysis_formated_data_marimo_${path_data_level2.baseName}.html"

    script:
    """
    marimo export html ${projectDir}/modules/08_reporting/notebooks/1_analysis_formated_data.marimo.py \
    -o 1_analysis_formated_data_marimo_${path_data_level2.baseName}.html \
    -- \
    --name ${params.input_name} \
    --input_path ${path_data_level2} \
    --level 2 \
    --control ${params.control}
    """
}

process level3_analysis {
    publishDir "modules/08_reporting/outputs", mode: 'copy'


    tag { "${path_data_level3.baseName}" }

    input:
    path path_data_level3

    output:
    path "1_analysis_formated_data_marimo_${path_data_level3.baseName}.html"

    script:
    """
    marimo export html ${projectDir}/modules/08_reporting/notebooks/1_analysis_formated_data.marimo.py \
    -o 1_analysis_formated_data_marimo_${path_data_level3.baseName}.html \
    -- \
    --name ${params.input_name} \
    --input_path ${path_data_level3} \
    --level 3 \
    --control ${params.control}
    """
}

process level4_analysis {
    publishDir "modules/08_reporting/outputs", mode: 'copy'

    tag { "${path_data_level4.baseName}" }

    input:
    path path_data_level4

    output:
    path "2F_analysis_processed_data_marimo_${path_data_level4.baseName}.html"

    script:
    """
    marimo export html \
    ${projectDir}/modules/08_reporting/notebooks/2F_analysis_processed_data.marimo.py \
    -o 2F_analysis_processed_data_marimo_${path_data_level4.baseName}.html \
    -- \
    --name ${params.input_name} \
    --input_path ${path_data_level4} \
    --control ${params.control} \
    --metadata_cell_viability ${params.concentration_response.compute_meta_pods.join(' ')} \
    """
}