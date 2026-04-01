nextflow.enable.dsl=2

// To RUN BSC: nextflow run entrypoints/02_processing_bsc.nf -params-file entrypoints/config_base_bsc.json -profile local -resume

include { concresponse_workflow_feature_wise } from './workflows/06_concresponse.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {

    path_data_level4 = Channel.fromPath("./modules/03_postprocessing/output/level4b_axiom_filtered.parquet")
    path_data_level5 = Channel.fromPath("./modules/03_postprocessing/output/level5_axiom_selected_features_filtered.parquet")
    path_data_level6 = Channel.fromPath("./modules/03_postprocessing/output/level6_axiom_selected_features_filtered.parquet")
    compute_meta_pods = Channel.of(
        tuple("Metadata_mtt_normalized", file("./modules/03_postprocessing/output/pods_axiom_Metadata_mtt_normalized_original_conc.parquet")),
        tuple("Metadata_ldh_normalized", file("./modules/03_postprocessing/output/pods_axiom_Metadata_ldh_normalized_original_conc.parquet")),
        tuple("Metadata_Count_Cells", file("./modules/03_postprocessing/output/pods_axiom_Metadata_Count_Cells_original_conc.parquet"))
    )

    classification_workflow(path_data_level4, path_data_level6, compute_meta_pods)
}

