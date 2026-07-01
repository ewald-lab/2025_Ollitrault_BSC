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
workflow preprocessing_workflow_bsc {

    take:
        input_path_level2

    main:
        result = bsc_metadata_preparation(input_path_level2)

    emit:
        html_file        = result[0]
        annotations_file = result[1]
        compounds_file   = result[2]
}

workflow preprocessing_workflow_axiom {

    take:
        input_path_level2

    main:
        axiom_metadata_preparation(input_path_level2)

    emit:
        compounds_file = axiom_metadata_preparation.out.compounds_file
}

process axiom_metadata_preparation {

    publishDir "modules/01_preprocessing/outputs", mode: 'copy'

    tag "axiom_metadata_preparation"

    input:
        path input_file

    output:
        path "compounds_axiom.txt", emit: compounds_file

    script:
    """
    python - <<'PY'
    import pandas as pd

    input_file = "${input_file}"
    output_file = "compounds_axiom.txt"

    df = pd.read_parquet(input_file)

    if "Metadata_compound" not in df.columns:
        raise ValueError("Column 'Metadata_compound' not found in parquet file")

    compounds = (
        df["Metadata_compound"]
        .dropna()
        .drop_duplicates()
    )

    compounds.to_csv(output_file, index=False, header=False)
    PY
    """
}


process bsc_metadata_preparation {

    publishDir "modules/01_preprocessing/outputs", mode: 'copy'

    tag "bsc_metadata_preparation"

    input:
        path input_file

    output:
        path "bsc_metadata_preparation_marimo.html"
        path "cellpainting_bsc_annotations_merged.tsv"
        path "compounds_bsc.txt"

    script:
    """
    marimo export html \
        ${params.base_dir}/modules/01_preprocessing/scripts/chemical_annotations.marimo.py \
        -o bsc_metadata_preparation_marimo.html \
        -- \
        python ${params.base_dir}/modules/01_preprocessing/scripts/chemical_annotations.marimo.py \
            --input_bsc_cp ${input_file} \
            --input_bsc_annotations "${params.bsc_annotation}" \
            --output_annotations cellpainting_bsc_annotations_merged.tsv \
            --output_compounds compounds_bsc.txt \
            --path_utils_standardizer "${params.base_dir}modules/common/utils"
    """
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
        def bsc_annot_arg = ""

        if (params.input_name == 'bsc') {
            script_path = "${params.base_dir}/modules/01_preprocessing/scripts/format_bsc_dino.py"
            if (params.bsc_annotation) {
                bsc_annot_arg = "--bsc_annotation '${params.bsc_annotation}'"
            }
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
            ${bsc_annot_arg} \\
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