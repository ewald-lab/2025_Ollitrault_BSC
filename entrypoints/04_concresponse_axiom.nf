nextflow.enable.dsl=2

// To RUN Axiom: nextflow run entrypoints/04_concresponse_axiom.nf -params-file configs/config_base_axiom.json -resume

include { concresponse_workflow_metadata } from '../workflows/04_concresponse.nf'
include { concresponse_workflow_feature_wise } from '../workflows/04_concresponse.nf'

params.base_dir = params.base_dir ?: "${projectDir}/.."

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {

    path_data_level4 = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level4b_axiom.parquet")
    path_data_level5 = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level5_axiom_selected_features.parquet")
    // Keep the list as a file input; feature-wise tasks resolve compounds by line number.
    path_list_compounds = Channel.fromPath("${params.base_dir}/modules/01_preprocessing/outputs/compounds_axiom.txt", checkIfExists: true)

    concresponse_results_metadata = concresponse_workflow_metadata(path_data_level4, path_data_level5)
    concresponse_results = concresponse_workflow_feature_wise(path_data_level4, path_data_level5, path_list_compounds)
}
