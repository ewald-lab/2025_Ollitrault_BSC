nextflow.enable.dsl=2

// ------------------- Preprocessing Workflow -------------------
workflow preprocessing_workflow {
    take:
        input_path

    main:
        // Create metadata channel - either the file or a dummy empty value
        if (params.metadata_path) {
            metadata_ch = Channel.fromPath(params.metadata_path)
        } else {
            metadata_ch = Channel.fromPath("NO_FILE")
        }

        path_data_level2     = format_dino(input_path, metadata_ch)
        path_data_level2_log = compute_log10conc(path_data_level2)

    emit:
        path_data_level2_log
}

process format_dino {

    publishDir "modules/01_preprocessing/outputs", mode: 'copy'

    tag { "${params.input_name}" }

    input:
        path input_file
        val  metadata_file

    output:
        path "level2_${params.input_name}.parquet"

    script:
        def script_path
        def metadata_arg = ""

        if (params.input_name == 'bsc') {
            script_path = "${params.base_dir}/modules/01_preprocessing/scripts/format_bsc_dino.py"
        }
        else if (params.input_name == 'axiom') {
            script_path = "${params.base_dir}/modules/01_preprocessing/scripts/format_axiom_oasis_dino.py"
            if (metadata_file != "NO_FILE") {
                metadata_arg = "--metadata ${metadata_file}"
            }
        }
        else {
            error "Unsupported input_name: ${params.input_name}"
        }

        """
        python ${script_path} \\
            --input ${input_file} \\
            ${metadata_arg} \\
            --output level2_${params.input_name}.parquet
        """
}

process compute_log10conc {

    publishDir "modules/01_preprocessing/outputs", mode: 'copy'

    tag { "${params.input_name}" }

    input:
        path input_file

    output:
        path "level2_${params.input_name}_log10conc.parquet"

    script:
        """
        python ${params.base_dir}/modules/01_preprocessing/scripts/log10_concentration.py \\
            --input ${input_file} \\
            --output level2_${params.input_name}_log10conc.parquet \\
            --control ${params.control}
        """
}