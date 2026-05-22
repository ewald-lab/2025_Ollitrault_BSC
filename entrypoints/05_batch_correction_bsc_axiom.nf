nextflow.enable.dsl=2

// To RUN BSC: nextflow run entrypoints/draft_batch_correction_bsc_axiom.nf -params-file configs/config_base_bsc.json -profile slurm -resume

include { postprocessing_workflow } from '../workflows/03_postprocessing.nf'
// include { level5_consensus } from '../workflows/02_processing.nf'
// include { level6_aggregate_profiles } from '../workflows/02_processing.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 

params.base_dir = "${projectDir}/.."

workflow {

    // Inputs
    path_data_level4_bsc       = Channel.fromPath("${projectDir}/../modules/03_postprocessing/outputs/level4b_bsc_filtered.parquet")
    path_data_level4_axiom = Channel.fromPath("${projectDir}/../modules/02_processing/outputs/level4a_axiom.parquet")

    // Run Harmony
    result = HARMONY_BATCH_CORRECTION(path_data_level4_bsc, path_data_level4_axiom)

    // process
    // path_data_level4_bsc_batch = Channel.fromPath("${projectDir}/../modules/03_postprocessing/outputs/level4b_bsc_batch.parquet")
    // path_data_level4_axiom_batch = Channel.fromPath("${projectDir}/../modules/03_postprocessing/outputs/level4b_axiom_batch.parquet")

    features_tuple = result.bsc
        .mix(result.axiom)
        .map { file -> tuple('selected_features_batch', file) }

    features_tuple.view()

    path_data_level5_tuples = level5_consensus(features_tuple)
    path_data_level6_tuples = level6_aggregate_profiles(path_data_level5_tuples)

}


process HARMONY_BATCH_CORRECTION {
    publishDir "modules/05_batch_correction_bsc_axiom/outputs", mode: 'copy'
    tag "harmony"

    input:
    path bsc_file
    path axiom_file

    output:
    path "level4b_bsc_batch.parquet", emit: bsc
    path "level4b_axiom_batch.parquet", emit: axiom

    script:
    """
    python3 ${projectDir}/../modules/03_postprocessing/scripts/draft_batch_correction.py \
        --bsc ${bsc_file} \
        --axiom ${axiom_file} \
        --out-bsc level4b_bsc_batch.parquet \
        --out-axiom level4b_axiom_batch.parquet
    """
}

process level5_consensus {
    publishDir "modules/05_batch_correction_bsc_axiom/outputs", mode: 'copy'
    tag { "${dataset_type}" }
    input:
    tuple val(feature_type), path(input_file)
    output:
    tuple val(feature_type), path("level5_${dataset_type}_${feature_type}.parquet")
    script:
    dataset_type = input_file.name.toString().contains('bsc') ? 'bsc' : 'axiom'
    """
    python ${projectDir}/../modules/02_processing/scripts/consensus.py \\
        --input $input_file \\
        --output level5_${dataset_type}_${feature_type}.parquet
    """
}
process level6_aggregate_profiles {
    publishDir "modules/05_batch_correction_bsc_axiom/outputs", mode: 'copy'
    tag { "${dataset_type}" }
    input:
    tuple val(feature_type), path(input_file)
    output:
    tuple val(feature_type), path("level6_${dataset_type}_${feature_type}.parquet")
    script:
    dataset_type = input_file.name.toString().contains('bsc') ? 'bsc' : 'axiom'
    """
    python ${projectDir}/../modules/02_processing/scripts/aggregate_profiles.py \\
        --input $input_file \\
        --output level6_${dataset_type}_${feature_type}.parquet
    """
}
