nextflow.enable.dsl=2

// To RUN: nextflow run entrypoints/07_downstream_analysis.nf -profile slurm -resume

params.base_dir = params.base_dir ?: "${projectDir}/.."
params.downstream_notebook_dir = params.downstream_notebook_dir ?: "${params.base_dir}/modules/07_downstream_analysis/notebook_manuscript"
params.downstream_html_dir = params.downstream_html_dir ?: "${params.downstream_notebook_dir}/__marimo__"

def manuscript_notebooks = [
    "0_analysis_raw_data.marimo.py",
    "1_analysis_processed_data.marimo.py",
    "2_analysis_chemicals.marimo.py",
    "3A_axiom_analysis_concresponse.marimo.py",
    "3B_BSC_analysis_concresponse.marimo.py",
    "4_bsc_analysis.marimo.py",
    "5_axiom_bsc_analysis.marimo.py",
    "6_analysis_axiom_bsc.marimo.py",
    "7_analysis_classificstion.marimo.py",
    "8_analysis_bsc_prediction.marimo.py",
]

workflow {
    Channel
        .fromList(manuscript_notebooks)
        .set { manuscript_notebooks_ch }

    EXPORT_MANUSCRIPT_NOTEBOOK(manuscript_notebooks_ch)
}

process EXPORT_MANUSCRIPT_NOTEBOOK {
    tag { notebook_name }
    maxForks 1
    publishDir { params.downstream_html_dir }, mode: 'copy'

    input:
    val notebook_name

    output:
    tuple val(notebook_name), path("*.html"), emit: html

    script:
    def html_name = notebook_name.replaceFirst(/\.py$/, ".html")
    """
    set -euo pipefail

    work_dir="\$PWD"
    cd "${params.base_dir}"

    marimo export html \\
        "${params.downstream_notebook_dir}/${notebook_name}" \\
        -o "\${work_dir}/${html_name}"
    """
}
