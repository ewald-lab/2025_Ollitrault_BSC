nextflow.enable.dsl=2

// To RUN BSC: nextflow run entrypoints/04_concresponse_bsc.nf -params-file configs/config_base_bsc.json -profile slurm -resume

include { concresponse_workflow_metadata } from '../workflows/04_concresponse.nf'
include { concresponse_workflow_feature_wise } from '../workflows/04_concresponse.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {

    path_data_level4 = Channel.fromPath("${params.base_dir}/modules/03_postprocessing/outputs/level4b_bsc_filtered.parquet")
    path_data_level5 = Channel.fromPath("${params.base_dir}/modules/03_postprocessing/outputs/level5_bsc_selected_features_filtered.parquet")
    // Keep the list as a file input; metadata and feature-wise tasks resolve compounds by line number.
    path_list_compounds = Channel.fromPath("${params.base_dir}/modules/01_preprocessing/outputs/compounds_bsc.txt", checkIfExists: true)

    concresponse_results_metadata = concresponse_workflow_metadata(path_data_level4, path_data_level5, path_list_compounds)
    concresponse_results = concresponse_workflow_feature_wise(path_data_level4, path_data_level5, path_list_compounds)
}
