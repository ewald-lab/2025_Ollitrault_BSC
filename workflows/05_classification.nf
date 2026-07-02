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
                tuple(
                    file("${params.base_dir}/modules/06_classification/inputs/${label_file}"),
                    "OASIS_ID"
                )
            }

        // Add axiom_hits with "ID" label column
        all_labels_ch = label_files_ch
            .concat(
                path_axiom_hits.map { file -> tuple(file, "ID") }
            )


        // Run each repeated CV repeat as its own Nextflow task, then concatenate
        // repeats back to the historical axiom_*_predictions.parquet outputs.
        classification_jobs_ch = path_data_level6
            .combine(path_data_level4)
            .combine(all_labels_ch)
            .map { input_level6, input_level4, label_file, label_column ->
                tuple(input_level6, input_level4, label_file, label_column, label_file.baseName)
            }

        repeat_index_ch = Channel.fromList(1..classifier_n_repeats)

        classification_repeat_jobs_ch = classification_jobs_ch
            .combine(repeat_index_ch)
            .map { input_level6, input_level4, label_file, label_column, label_name, repeat_idx ->
                tuple(input_level6, input_level4, label_file, label_column, label_name, repeat_idx)
            }

        if (classifier_device == 'gpu') {
            classification_repeat_predictions = binary_classification_repeat_gpu(classification_repeat_jobs_ch).predictions
        } else {
            classification_repeat_predictions = binary_classification_repeat_cpu(classification_repeat_jobs_ch).predictions
        }

        path_output_classification = classification_repeat_predictions
            .groupTuple(size: classifier_n_repeats)
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


process binary_classification_repeat_cpu {

    tag { "${label_name}:repeat_${repeat_idx}:cpu" }

    label 'classification_cpu'
    cpus { params.classifier.cpus ?: 8 }
    memory { params.classifier.memory ?: '64 GB' }
    time '24h'

    publishDir "modules/06_classification/outputs", mode: 'copy', pattern: 'models/*.pkl'
    
    input:
        tuple path(input_level6), path(input_level4), path(label_file), val(label_column), val(label_name), val(repeat_idx)
    
    output:
        tuple val(label_name), path("axiom_${label_name}_repeat${repeat_idx}_predictions.parquet"), emit: predictions
        tuple val(label_name), path("models/*.pkl"), emit: models, optional: true
    
    script:
    def total_repeats = params.classifier.n_repeats ?: 1
    def filter_label_size = params.classifier.filter_label_size ?: 50
    def filter_label_size_active = params.classifier.filter_label_size_active ?: 20
    def label_workers = params.classifier.label_workers ?: 1
    def save_models_arg = repeat_idx == 1 ? "--save_models" : "--no_save_models"
    """
    python ${params.base_dir}/modules/06_classification/scripts/classify.py \
        --input_path_level6 ${input_level6} \
        --input_path_level4 ${input_level4} \
        --label_path ${label_file.name} \
        --label_column ${label_column} \
        --output axiom_${label_name}_repeat${repeat_idx}_predictions.parquet \
        --n_repeats ${total_repeats} \
        --repeat_index ${repeat_idx} \
        --device cpu \
        --n_jobs ${task.cpus} \
        --label_workers ${label_workers} \
        --filter_label_size ${filter_label_size} \
        --filter_label_size_active ${filter_label_size_active} \
        ${save_models_arg}
    """
}


process binary_classification_repeat_gpu {

    tag { "${label_name}:repeat_${repeat_idx}:gpu" }

    label 'gpu'
    cpus { params.classifier.gpu_cpus ?: params.classifier.cpus ?: 8 }
    memory { params.classifier.memory ?: '64 GB' }
    time '24h'

    publishDir "modules/06_classification/outputs", mode: 'copy', pattern: 'models/*.pkl'
    
    input:
        tuple path(input_level6), path(input_level4), path(label_file), val(label_column), val(label_name), val(repeat_idx)
    
    output:
        tuple val(label_name), path("axiom_${label_name}_repeat${repeat_idx}_predictions.parquet"), emit: predictions
        tuple val(label_name), path("models/*.pkl"), emit: models, optional: true
    
    script:
    def total_repeats = params.classifier.n_repeats ?: 1
    def filter_label_size = params.classifier.filter_label_size ?: 50
    def filter_label_size_active = params.classifier.filter_label_size_active ?: 20
    def label_workers = params.classifier.label_workers ?: 1
    def save_models_arg = repeat_idx == 1 ? "--save_models" : "--no_save_models"
    """
    python ${params.base_dir}/modules/06_classification/scripts/classify.py \
        --input_path_level6 ${input_level6} \
        --input_path_level4 ${input_level4} \
        --label_path ${label_file.name} \
        --label_column ${label_column} \
        --output axiom_${label_name}_repeat${repeat_idx}_predictions.parquet \
        --n_repeats ${total_repeats} \
        --repeat_index ${repeat_idx} \
        --device gpu \
        --n_jobs ${task.cpus} \
        --label_workers ${label_workers} \
        --filter_label_size ${filter_label_size} \
        --filter_label_size_active ${filter_label_size_active} \
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
    def files_literal = prediction_files.collect { "'${it}'" }.join(", ")
    """
python - <<'PY'
import polars as pl

files = [${files_literal}]
frames = [pl.read_parquet(path) for path in files]
combined = pl.concat(frames, how="vertical")

if {"repeat", "k_fold"}.issubset(combined.columns):
    combined = combined.sort(["repeat", "k_fold", "Model_type", "Metadata_Label", "Metadata_compound"])

combined.write_parquet("axiom_${label_name}_predictions.parquet")
PY
    """
}
