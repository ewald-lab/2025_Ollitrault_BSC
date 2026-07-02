nextflow.enable.dsl = 2

workflow classification_workflow {
    take:
        path_data_level4
        path_data_level6
        compute_meta_pods
    
    main:
        classifier_params = params.classifier ?: [:]
        classifier_device = (classifier_params.device ?: 'cpu').toString().toLowerCase()
        if (classifier_device == 'cuda') {
            classifier_device = 'gpu'
        }
        if (!(classifier_device in ['cpu', 'gpu'])) {
            throw new IllegalArgumentException("params.classifier.device must be 'cpu' or 'gpu' (got: ${classifier_device})")
        }

        classifier_n_repeats = (classifier_params.n_repeats ?: 1) as int
        if (classifier_n_repeats < 1) {
            throw new IllegalArgumentException("params.classifier.n_repeats must be >= 1")
        }

        pods_map_ch = compute_meta_pods
            .toList()
            .map { tuples ->
                tuples.collectEntries { label, file -> 
                    [(label): file] 
                }
            }
        // Hit call axiom
        path_axiom_hits = axiom_assay_hitcall(pods_map_ch)

        // Create channel for label files with their label column
        label_files_ch = Channel.fromList(params.classifier.label_files)
            .map { label_file ->
                def label_path = file("${params.base_dir}/modules/06_classification/inputs/${label_file}")
                tuple(
                    label_path,
                    "OASIS_ID",
                    label_path.baseName
                )
            }

        // Add axiom_hits with "ID" label column
        all_labels_ch = label_files_ch
            .concat(
                path_axiom_hits.map { file -> tuple(file, "ID", file.baseName) }
            )

        filtered_labels_input_ch = all_labels_ch
            .combine(path_data_level6)
            .combine(path_data_level4)
            .map { label_file, label_column, label_name, input_level6, input_level4 ->
                tuple(input_level6, input_level4, label_file, label_column, label_name)
            }

        filtered_labels_ch = filter_classification_labels(filtered_labels_input_ch).labels

        // Split labels into configurable chunks before CV fan-out. Set
        // params.classifier.label_chunk_size = 1 for one Slurm task per assay.
        label_chunks_ch = list_classification_label_chunks(filtered_labels_ch).chunks
            .splitCsv(sep: '\t')
            .map { row ->
                tuple(row[0], row[1] as int, row[2])
            }

        // Run each repeated CV repeat and label chunk as its own Nextflow task,
        // then concatenate chunks/repeats back to the historical outputs.
        classification_jobs_ch = path_data_level6
            .combine(path_data_level4)
            .combine(filtered_labels_ch)
            .map { input_level6, input_level4, label_file, label_column, label_name ->
                tuple(label_name, input_level6, input_level4, label_file, label_column)
            }

        classification_label_jobs_ch = classification_jobs_ch
            .join(label_chunks_ch)
            .map { label_name, input_level6, input_level4, label_file, label_column, label_chunk_idx, include_labels ->
                tuple(input_level6, input_level4, label_file, label_column, label_name, label_chunk_idx, include_labels)
            }

        repeat_index_ch = Channel.fromList(1..classifier_n_repeats)

        classification_repeat_jobs_ch = classification_label_jobs_ch
            .combine(repeat_index_ch)
            .map { input_level6, input_level4, label_file, label_column, label_name, label_chunk_idx, include_labels, repeat_idx ->
                tuple(input_level6, input_level4, label_file, label_column, label_name, label_chunk_idx, include_labels, repeat_idx)
            }

        if (classifier_device == 'gpu') {
            classification_repeat_predictions = binary_classification_repeat_gpu(classification_repeat_jobs_ch).predictions
        } else {
            classification_repeat_predictions = binary_classification_repeat_cpu(classification_repeat_jobs_ch).predictions
        }

        path_output_classification = classification_repeat_predictions
            .groupTuple()
            | concatenate_classification_repeats
       
    emit:
        path_data_level6
}


process axiom_assay_hitcall {
    tag { "${params.input_name}" }

    input:
        val pods

    output:
        path "axiom_hits.parquet"

    script:
    """
    python ${params.base_dir}/modules/06_classification/scripts/hitcalls.py \
        --input_pod_mtt ${pods["Metadata_mtt_normalized"]} \
        --input_pod_ldh ${pods["Metadata_ldh_normalized"]} \
        --input_pod_cc ${pods["Metadata_Count_Cells"]} \
        --output axiom_hits.parquet
    """
}


process filter_classification_labels {

    tag { "${label_name}" }

    publishDir "modules/06_classification/outputs/label_filters", mode: 'copy'

    input:
        tuple path(input_level6), path(input_level4), path(label_file), val(label_column), val(label_name)

    output:
        tuple path("${label_name}_filtered_labels.parquet"), val(label_column), val(label_name), emit: labels
        path "${label_name}_label_filter_audit.tsv", emit: audit

    script:
    def classifier_params = params.classifier ?: [:]
    def filter_label_size = classifier_params.containsKey('filter_label_size') ? classifier_params.filter_label_size : 50
    def filter_label_size_active = classifier_params.containsKey('filter_label_size_active') ? classifier_params.filter_label_size_active : 20
    """
    python ${params.base_dir}/modules/06_classification/scripts/filter_classification_labels.py \
        --input_path_level6 ${input_level6} \
        --input_path_level4 ${input_level4} \
        --label_path ${label_file.name} \
        --label_column ${label_column} \
        --output ${label_name}_filtered_labels.parquet \
        --audit_output ${label_name}_label_filter_audit.tsv \
        --label_group_name ${label_name} \
        --n_splits 5 \
        --filter_label_size ${filter_label_size} \
        --filter_label_size_active ${filter_label_size_active}
    """
}


process list_classification_label_chunks {

    tag { "${label_name}" }

    input:
        tuple path(label_file), val(label_column), val(label_name)

    output:
        path "label_chunks.tsv", emit: chunks

    script:
    def label_chunk_size = params.classifier.label_chunk_size ?: 0
    """
    python ${params.base_dir}/modules/06_classification/scripts/classify.py \
        --label_path ${label_file.name} \
        --label_column ${label_column} \
        --list_label_chunks \
        --label_group_name ${label_name} \
        --label_chunk_size ${label_chunk_size} \
        > label_chunks.tsv
    """
}


process binary_classification_repeat_cpu {

    tag { "${label_name}:chunk_${label_chunk_idx}:repeat_${repeat_idx}:cpu" }

    label 'classification_cpu'
    cpus { params.classifier.cpus ?: 8 }
    memory { params.classifier.memory ?: '64 GB' }
    time '24h'
    maxForks ((params.classifier.max_cpu_jobs ?: 125) as int)

    publishDir "modules/06_classification/outputs/models", mode: 'copy', pattern: 'models/*.pkl', saveAs: { file -> file.toString().tokenize('/').last() }, failOnError: true
    
    input:
        tuple path(input_level6), path(input_level4), path(label_file), val(label_column), val(label_name), val(label_chunk_idx), val(include_labels), val(repeat_idx)
    
    output:
        tuple val(label_name), path("axiom_${label_name}_chunk${label_chunk_idx}_repeat${repeat_idx}_predictions.parquet"), emit: predictions
        tuple val(label_name), path("models/*.pkl"), emit: models, optional: true
    
    script:
    def total_repeats = params.classifier.n_repeats ?: 1
    def label_workers = params.classifier.label_workers ?: 1
    def save_models_arg = repeat_idx == 1 ? "--save_models" : "--no_save_models"
    """
    python ${params.base_dir}/modules/06_classification/scripts/classify.py \
        --input_path_level6 ${input_level6} \
        --input_path_level4 ${input_level4} \
        --label_path ${label_file.name} \
        --label_column ${label_column} \
        --output axiom_${label_name}_chunk${label_chunk_idx}_repeat${repeat_idx}_predictions.parquet \
        --n_repeats ${total_repeats} \
        --repeat_index ${repeat_idx} \
        --device cpu \
        --n_jobs ${task.cpus} \
        --label_workers ${label_workers} \
        --include_labels "${include_labels}" \
        ${save_models_arg}
    """
}


process binary_classification_repeat_gpu {

    tag { "${label_name}:chunk_${label_chunk_idx}:repeat_${repeat_idx}:gpu" }

    label 'gpu'
    cpus { params.classifier.gpu_cpus ?: params.classifier.cpus ?: 8 }
    memory { params.classifier.memory ?: '64 GB' }
    time '24h'
    maxForks ((params.classifier.max_gpu_jobs ?: 16) as int)

    publishDir "modules/06_classification/outputs/models", mode: 'copy', pattern: 'models/*.pkl', saveAs: { file -> file.toString().tokenize('/').last() }, failOnError: true
    
    input:
        tuple path(input_level6), path(input_level4), path(label_file), val(label_column), val(label_name), val(label_chunk_idx), val(include_labels), val(repeat_idx)
    
    output:
        tuple val(label_name), path("axiom_${label_name}_chunk${label_chunk_idx}_repeat${repeat_idx}_predictions.parquet"), emit: predictions
        tuple val(label_name), path("models/*.pkl"), emit: models, optional: true
    
    script:
    def total_repeats = params.classifier.n_repeats ?: 1
    def label_workers = params.classifier.label_workers ?: 1
    def save_models_arg = repeat_idx == 1 ? "--save_models" : "--no_save_models"
    """
    python ${params.base_dir}/modules/06_classification/scripts/classify.py \
        --input_path_level6 ${input_level6} \
        --input_path_level4 ${input_level4} \
        --label_path ${label_file.name} \
        --label_column ${label_column} \
        --output axiom_${label_name}_chunk${label_chunk_idx}_repeat${repeat_idx}_predictions.parquet \
        --n_repeats ${total_repeats} \
        --repeat_index ${repeat_idx} \
        --device gpu \
        --n_jobs ${task.cpus} \
        --label_workers ${label_workers} \
        --include_labels "${include_labels}" \
        ${save_models_arg}
    """
}


process concatenate_classification_repeats {

    tag { "${label_name}" }

    publishDir "modules/06_classification/outputs/", mode: 'copy', pattern: '*.parquet'

    input:
        tuple val(label_name), path(prediction_files)

    output:
        path "axiom_${label_name}_predictions.parquet"

    script:
    """
python - <<'PY'
from pathlib import Path
import polars as pl

files = sorted(Path(".").glob("axiom_*_chunk*_repeat*_predictions.parquet"))
frames = [pl.read_parquet(path) for path in files]
combined = pl.concat(frames, how="vertical")

if {"repeat", "k_fold"}.issubset(combined.columns):
    combined = combined.sort(["repeat", "k_fold", "Model_type", "Metadata_Label", "Metadata_compound"])

combined.write_parquet("axiom_${label_name}_predictions.parquet")
PY
    """
}
