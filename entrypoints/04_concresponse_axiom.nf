nextflow.enable.dsl=2

// To RUN Axiom: nextflow run entrypoints/06_concresponse_axiom.nf -params-file entrypoints/config_base_axiom.json -resume

include { concresponse_workflow_feature_wise } from './workflows/06_concresponse.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {

    path_data_level4 = Channel.fromPath("./modules/03_postprocessing/output/level4b_axiom_filtered.parquet")
    path_data_level5 = Channel.fromPath("./modules/03_postprocessing/output/level5_axiom_selected_features_filtered.parquet")

    concresponse_results = concresponse_workflow_feature_wise(path_data_level4, path_data_level5)
}

