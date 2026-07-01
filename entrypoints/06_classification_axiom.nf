nextflow.enable.dsl=2
params.base_dir = params.base_dir ?: "${projectDir}/.."
// To RUN Axiom: nextflow run entrypoints/06_classification_axiom.nf -params-file configs/config_base_axiom.json -profile slurm

include { classification_workflow } from '../workflows/05_classification.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 

workflow {
    println "BASE DIR: ${params.base_dir}"
    if (false) { // default no batch correction
        path_data_level4 = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level4b_axiom.parquet")
        path_data_level5 = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level5_axiom_selected_features.parquet")
        path_data_level6 = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level6_axiom_selected_features.parquet")
        compute_meta_pods = Channel.of(
            tuple("Metadata_mtt_normalized", file("${params.base_dir}/modules/04_concresponse/outputs/pods/pods_axiom_Metadata_mtt_normalized_original_conc.parquet")),
            tuple("Metadata_ldh_normalized", file("${params.base_dir}/modules/04_concresponse/outputs/pods/pods_axiom_Metadata_ldh_normalized_original_conc.parquet")),
            tuple("Metadata_Count_Cells", file("${params.base_dir}/modules/04_concresponse/outputs/pods/pods_axiom_Metadata_Count_Cells_original_conc.parquet"))
        )
    }
    if (true) { // batch correction
        path_data_level4 = Channel.fromPath("${params.base_dir}/modules/05_batch_correction_bsc_axiom/outputs/level4b_axiom_batch.parquet")
        path_data_level5 = Channel.fromPath("${params.base_dir}/modules/05_batch_correction_bsc_axiom/outputs/level5_axiom_selected_features_batch.parquet")
        path_data_level6 = Channel.fromPath("${params.base_dir}/modules/05_batch_correction_bsc_axiom/outputs/level6_axiom_selected_features_batch.parquet")
        compute_meta_pods = Channel.of(
            tuple("Metadata_mtt_normalized", file("${params.base_dir}/modules/04_concresponse/outputs/pods/pods_axiom_Metadata_mtt_normalized_original_conc.parquet")),
            tuple("Metadata_ldh_normalized", file("${params.base_dir}/modules/04_concresponse/outputs/pods/pods_axiom_Metadata_ldh_normalized_original_conc.parquet")),
            tuple("Metadata_Count_Cells", file("${params.base_dir}/modules/04_concresponse/outputs/pods/pods_axiom_Metadata_Count_Cells_original_conc.parquet"))
        )
    }

    classification_workflow(path_data_level4, path_data_level6, compute_meta_pods)
}
