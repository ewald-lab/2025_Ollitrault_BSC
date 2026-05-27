nextflow.enable.dsl=2

// To RUN BSC: nextflow run entrypoints/03_postprocessing_bsc.nf -params-file configs/config_base_bsc.json -profile slurm -resume

include { postprocessing_workflow } from '../workflows/03_postprocessing.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {
    path_data_level2_log = Channel.fromPath("${params.base_dir}/modules/01_preprocessing/outputs/level2_bsc_log10conc.parquet")
    path_data_level3 = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level3_bsc.parquet")
    // path_data_level4 = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level4a_bsc.parquet")
    path_data_level4 = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level4b_bsc.parquet")
    path_data_level5 = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level5_bsc_selected_features.parquet")
    path_data_level6 = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level6_bsc_selected_features.parquet")
    path_data_level5_all = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level5_bsc_all_features.parquet")
    path_data_level6_all = Channel.fromPath("${params.base_dir}/modules/02_processing/outputs/level6_bsc_all_features.parquet")

    postprocessing_workflow(
            path_data_level2_log,
            path_data_level3,
            path_data_level4,
            path_data_level5,
            path_data_level6,
            path_data_level5_all,
            path_data_level6_all
        )
}

