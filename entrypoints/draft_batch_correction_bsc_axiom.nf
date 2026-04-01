nextflow.enable.dsl=2

// To RUN BSC: nextflow run entrypoints/03_postprocessing_bsc.nf -params-file entrypoints/config_base_bsc.json -profile local -resume

include { postprocessing_workflow } from './workflows/03_postprocessing.nf'
include { level5_consensus } from './workflows/02_processing.nf'
include { level6_aggregate_profiles } from './workflows/02_processing.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {

    // Inputs
    path_data_level4       = Channel.fromPath("./modules/02_processing/output/level4b_bsc.parquet")
    path_data_level4_axiom = Channel.fromPath("./modules/02_processing/output/level4b_axiom.parquet")

    // Combine into one tuple (IMPORTANT)
    paired_input = path_data_level4.combine(path_data_level4_axiom)

    // Run Harmony
    result = HARMONY_BATCH_CORRECTION(paired_input)

    features_tuple = Channel.of('selected_features_batch', path_data_level4b)
    path_data_level5_tuples = level5_consensus(features_tuple)
    path_data_level6_tuples = level6_aggregate_profiles(features_tuple)

}


process HARMONY_BATCH_CORRECTION {
    publishDir "modules/03_postprocessing/outputs", mode: 'copy'
    tag "harmony"

    input:
    path bsc_file
    path axiom_file

    output:
    path "bsc_corrected.parquet", emit: bsc
    path "axiom_corrected.parquet", emit: axiom

    script:
    """
    python3 harmony.py \
        --bsc ${bsc_file} \
        --axiom ${axiom_file} \
        --out-bsc level4b_bsc.parquet \
        --out-axiom level4b_axiom.parquet
    """
}