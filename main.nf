nextflow.enable.dsl=2

/*
 * Ordered entrypoint launcher.
 *
 * This main workflow intentionally delegates to the numbered scripts in
 * entrypoints/ so their current stage-specific code remains the source of
 * truth. The list below follows the sorted entrypoint order.
 *
 * Example:
 *   nextflow run main.nf -profile local --entrypoint_profile slurm -resume
 */

params.entrypoint_profile = params.entrypoint_profile ?: 'slurm'
params.entrypoint_resume = params.entrypoint_resume == null ? true : params.entrypoint_resume

def ordered_entrypoints = [
    [id: '01_preprocessing_axiom', script: 'entrypoints/01_preprocessing_axiom.nf', params_file: 'configs/config_base_axiom.json'],
    [id: '01_preprocessing_bsc', script: 'entrypoints/01_preprocessing_bsc.nf', params_file: 'configs/config_base_bsc.json'],
    [id: '02_processing_axiom', script: 'entrypoints/02_processing_axiom.nf', params_file: 'configs/config_base_axiom.json'],
    [id: '02_processing_bsc', script: 'entrypoints/02_processing_bsc.nf', params_file: 'configs/config_base_bsc.json'],
    [id: '03_postprocessing_bsc', script: 'entrypoints/03_postprocessing_bsc.nf', params_file: 'configs/config_base_bsc.json'],
    [id: '04_concresponse_axiom', script: 'entrypoints/04_concresponse_axiom.nf', params_file: 'configs/config_base_axiom.json'],
    [id: '04_concresponse_bsc', script: 'entrypoints/04_concresponse_bsc.nf', params_file: 'configs/config_base_bsc.json'],
    [id: '05_batch_correction_bsc_axiom', script: 'entrypoints/05_batch_correction_bsc_axiom.nf', params_file: 'configs/config_base_bsc.json'],
    [id: '06_classification_axiom', script: 'entrypoints/06_classification_axiom.nf', params_file: 'configs/config_base_axiom.json'],
    [id: '07_downstream_analysis', script: 'entrypoints/07_downstream_analysis.nf', params_file: null],
]

workflow {
    RUN_ORDERED_ENTRYPOINTS(Channel.value(ordered_entrypoints))
}

process RUN_ORDERED_ENTRYPOINTS {
    tag 'entrypoints'
    executor 'local'

    input:
        val entrypoints

    output:
        path 'entrypoint-run-order.tsv'

    script:
        def profile_args = params.entrypoint_profile ? "-profile ${params.entrypoint_profile}" : ''
        def resume_args = params.entrypoint_resume.toString().toBoolean() ? '-resume' : ''
        def commands = entrypoints.collect { entry ->
            def params_args = entry.params_file ? "-params-file ${entry.params_file}" : ''
            """
            echo "${entry.id}\t${entry.script}" | tee -a "\$log_file"
            nextflow run ${entry.script} ${params_args} ${profile_args} ${resume_args}
            """
        }.join('\n')

        """
        set -euo pipefail

        log_file="\$PWD/entrypoint-run-order.tsv"
        : > "\$log_file"

        cd "${projectDir}"

        ${commands}
        """
}
