nextflow.enable.dsl=2

workflow postprocessing_workflow {

    take:
        path_data_level2
        path_data_level3
        path_data_level4
        path_data_level5
        path_data_level6
        path_data_level5_all
        path_data_level6_all


    main:

        /*
         * Levels for first BSC filtering
         * level2 -> level4
         */
        levels_postprocess =
            path_data_level2.map { tuple('level2', it) }
                .mix( path_data_level3.map { tuple('level3', it) } )
                .mix( path_data_level4.map { tuple('level4', it) } )

        /*
         * First filtering step
         */
        postprocessed_filter = bsc_postprocessing(levels_postprocess)

        /*
         * Levels for concentration correction
         * level2 -> level5_all
         */
        levels_concentration =
            postprocessed_filter
                .mix( path_data_level5.map { tuple('level5', it) } )
                .mix( path_data_level5_all.map { tuple('level5_all', it) } )

        /*
         * Concentration correction
         */
        postprocessed =
            bsc_postprocessing_correct_botanicals_concentration_units(
                levels_concentration
            )

        /*
         * Split outputs
         */

        level2_post =
            postprocessed
                .filter { it[0] == 'level2' }
                .map { it[1] }

        level3_post =
            postprocessed
                .filter { it[0] == 'level3' }
                .map { it[1] }

        level4_post =
            postprocessed
                .filter { it[0] == 'level4' }
                .map { it[1] }

        level5_post =
            postprocessed
                .filter { it[0] == 'level5' }
                .map { it[1] }

        level5_all_post =
            postprocessed
                .filter { it[0] == 'level5_all' }
                .map { it[1] }

        // untouched
        level6_post = path_data_level6
        level6_all_post = path_data_level6_all

    emit:
        path_data_level2 = level2_post
        path_data_level3 = level3_post
        path_data_level4 = level4_post
        path_data_level5 = level5_post
        path_data_level6 = level6_post
        path_data_level5_all = level5_all_post
        path_data_level6_all = level6_all_post
}

process bsc_postprocessing {

    publishDir "modules/03_postprocessing/outputs", mode: 'copy'
    tag { "${level}:${input_file.baseName}" }

    beforeScript = """
    export PYTHONPATH="${params.base_dir}/modules"
    """

    input:
        tuple val(level), path(input_file)

    output:
        tuple val(level), path("${input_file.baseName}_filtered.parquet")

    script:
    """
    python ${params.base_dir}/modules/03_postprocessing/scripts/${params.post_processing.script} \
        --input ${input_file} \
        --output "${input_file.baseName}_filtered.parquet" \
        --bsc_annotation "${params.base_dir}/${params.post_processing.bsc_annotation}"
    """
}

process bsc_postprocessing_correct_botanicals_concentration_units  {

    publishDir "modules/03_postprocessing/outputs", mode: 'copy'
    tag { "${level}:${input_file.baseName}" }

    beforeScript = """
    export PYTHONPATH="${params.base_dir}/modules"
    """

    input:
        tuple val(level), path(input_file)

    output:
        tuple val(level), path("${input_file.baseName}_filtered_converted.parquet")

    script:
    """
    python ${params.base_dir}/modules/03_postprocessing/scripts/postprocessing_bsc_botanical_shared_units.py \
        --input ${input_file} \
        --output "${input_file.baseName}_filtered_converted.parquet" \
        --bsc_annotation "${params.base_dir}/${params.post_processing.bsc_annotation}"
    """
}

