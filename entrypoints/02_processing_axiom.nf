nextflow.enable.dsl=2

// To RUN Axiom: nextflow run entrypoints/02_processing_axiom.nf -params-file configs/config_base_axiom.json -resume

include { processing_workflow } from '../workflows/02_processing.nf'

params.base_dir = params.base_dir ?: "${projectDir}/.."

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {
    path_data_level2_log = Channel.fromPath("${params.base_dir}/modules/01_preprocessing/outputs/level2_axiom_log10conc.parquet")
    processed_data = processing_workflow(path_data_level2_log)
}
