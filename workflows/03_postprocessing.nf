nextflow.enable.dsl=2

workflow postprocessing_workflow {

    take:
        path_data_level2
        path_data_level3
        path_data_level4
        path_data_level5

    main:
        // 1. Tag inputs by level
        all_levels =
            path_data_level2.map { tuple('level2', it) }
                .mix( path_data_level3.map { tuple('level3', it) } )
                .mix( path_data_level4.map { tuple('level4', it) } )
                .mix( path_data_level5.map { tuple('level5', it) } )

        // 2. Single postprocessing step
        postprocessed = bsc_postprocessing(all_levels)

        // 3. Split back by level
        level2_post = postprocessed.filter { it[0] == 'level2' }.map { it[1] }
        level3_post = postprocessed.filter { it[0] == 'level3' }.map { it[1] }
        level4_post = postprocessed.filter { it[0] == 'level4' }.map { it[1] }
        level5_post = postprocessed.filter { it[0] == 'level5' }.map { it[1] }

    emit:
        path_data_level2 = level2_post
        path_data_level3 = level3_post
        path_data_level4 = level4_post
        path_data_level5 = level5_post
}

process bsc_postprocessing {

    publishDir "modules/03_postprocessing/outputs", mode: 'copy'
    tag { "${level}:${input_file.baseName}" }

    beforeScript = """
    export PYTHONPATH="${projectDir}/modules"
    """

    input:
        tuple val(level), path(input_file)

    output:
        tuple val(level), path("${input_file.baseName}_filtered.parquet")

    script:
    """
    python ${projectDir}/modules/03_postprocessing/scripts/${params.post_processing.script} \
        --input ${input_file} \
        --output "${input_file.baseName}_filtered.parquet" \
        --bsc_annotation "${projectDir}/${params.post_processing.bsc_annotation}"
    """
}

