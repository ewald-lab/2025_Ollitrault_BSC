nextflow.enable.dsl=2

// To RUN BSC: nextflow run entrypoints/02_processing_bsc.nf -params-file entrypoints/config_base_bsc.json -profile local -resume

include { processing_workflow } from './workflows/02_processing.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {
    path_data_level2_log = Channel.fromPath("./modules/01_preprocessing/output/level2_bsc_log10conc.parquet")
    processed_data = processing_workflow(path_data_level2_log)
}

