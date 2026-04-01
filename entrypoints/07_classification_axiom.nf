nextflow.enable.dsl=2

// To RUN Axiom: nextflow run entrypoints/07_classification_axiom.nf -params-file configs/config_base_axiom.json -profile local

include { classification_workflow } from '../workflows/07_classification.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 

params.base_dir = "${projectDir}/.."

workflow {

    if (false) { // default no batch correction
        path_data_level4 = Channel.fromPath("./modules/03_postprocessing/output/level4b_axiom_filtered.parquet")
        path_data_level5 = Channel.fromPath("./modules/03_postprocessing/output/level5_axiom_selected_features_filtered.parquet")
        path_data_level6 = Channel.fromPath("./modules/03_postprocessing/output/level6_axiom_selected_features_filtered.parquet")
        compute_meta_pods = Channel.of(
            tuple("Metadata_mtt_normalized", file("./modules/03_postprocessing/output/pods_axiom_Metadata_mtt_normalized_original_conc.parquet")),
            tuple("Metadata_ldh_normalized", file("./modules/03_postprocessing/output/pods_axiom_Metadata_ldh_normalized_original_conc.parquet")),
            tuple("Metadata_Count_Cells", file("./modules/03_postprocessing/output/pods_axiom_Metadata_Count_Cells_original_conc.parquet"))
        )
    }
    if (true) { // batch correction
        path_data_level4 = Channel.fromPath("./modules/03_postprocessing/output/level4b_axiom_batch.parquet")
        path_data_level5 = Channel.fromPath("./modules/03_postprocessing/output/level5_axiom_selected_features_batch.parquet")
        path_data_level6 = Channel.fromPath("./modules/03_postprocessing/output/level6_axiom_selected_features_batch.parquet")
        compute_meta_pods = Channel.of(
            tuple("Metadata_mtt_normalized", file("./modules/03_postprocessing/output/pods_axiom_Metadata_mtt_normalized_original_conc.parquet")),
            tuple("Metadata_ldh_normalized", file("./modules/03_postprocessing/output/pods_axiom_Metadata_ldh_normalized_original_conc.parquet")),
            tuple("Metadata_Count_Cells", file("./modules/03_postprocessing/output/pods_axiom_Metadata_Count_Cells_original_conc.parquet"))
        )
    }

    classification_workflow(path_data_level4, path_data_level6, compute_meta_pods)
}

