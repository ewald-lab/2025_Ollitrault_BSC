nextflow.enable.dsl=2

// To RUN BSC: nextflow run entrypoints/02_processing_bsc.nf -params-file entrypoints/config_base_bsc.json -profile local -resume

include { concresponse_workflow_feature_wise } from './workflows/06_concresponse.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {

    path_data_level4 = Channel.fromPath("./modules/03_postprocessing/output/level4b_bsc_filtered_converted.parquet")
    path_data_level5 = Channel.fromPath("./modules/03_postprocessing/output/level5_bsc_selected_features_filtered_converted.parquet")

    concresponse_results = concresponse_workflow_feature_wise(path_data_level4, path_data_level5)
}

