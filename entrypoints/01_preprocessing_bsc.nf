nextflow.enable.dsl=2

// To RUN BSC: nextflow run entrypoints/01_preprocessing_bsc.nf -params-file entrypoints/config_base_bsc.json -profile slurm -resume

include { preprocessing_workflow } from './workflows/01_preprocessing.nf'
include { preprocessing_workflow } from './workflows/01_preprocessing.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {
    input_path = Channel.fromPath(params.input_path)
    preprocessed_data = preprocessing_workflow(input_path)
}
