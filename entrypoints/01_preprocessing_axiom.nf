nextflow.enable.dsl=2

// To RUN Axiom: nextflow run entrypoints/01_preprocessing_axiom.nf -params-file configs/config_base_axiom.json -resume

include { preprocessing_workflow } from '../workflows/01_preprocessing.nf'
include { preprocessing_workflow_axiom } from '../workflows/01_preprocessing.nf'

params.base_dir = params.base_dir ?: "${projectDir}/.."

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {
    input_path = Channel.fromPath(params.input_path)
    preprocessed_data = preprocessing_workflow(input_path)
    if (params.input_name == "axiom") {
        preprocessed_data_axiom = preprocessing_workflow_axiom(preprocessed_data.path_data_level2_log)
    }
}
