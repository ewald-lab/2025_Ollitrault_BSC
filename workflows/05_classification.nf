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

        filtered_labels = filter_classification_labels(filtered_labels_input_ch)
        filtered_labels_ch = filtered_labels.labels
        filtered_assays_ch = filtered_labels.assays
            .splitCsv(sep: '\t', header: true)
            .map { row ->
                tuple(row.label_group, row.assay_index as int, row.assay)
            }

        // Parallelize only by filtered assay. Each assay task runs all repeated
        // cross-validation internally, then assay outputs are concatenated.
        classification_jobs_ch = path_data_level6
            .combine(path_data_level4)
            .combine(filtered_labels_ch)
            .map { input_level6, input_level4, label_file, label_column, label_name ->
                tuple(label_name, input_level6, input_level4, label_file, label_column)
            }
            .join(filtered_assays_ch)
            .map { label_name, input_level6, input_level4, label_file, label_column, assay_index, assay_label ->
                tuple(input_level6, input_level4, label_file, label_column, label_name, assay_index, assay_label)
            }

        if (classifier_device == 'gpu') {
            classification_predictions = binary_classification_assay_gpu(classification_jobs_ch).predictions
        } else {
            classification_predictions = binary_classification_assay_cpu(classification_jobs_ch).predictions
        }

        path_output_classification = classification_predictions
            .groupTuple()
            | concatenate_classification_assays
       
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
        path "${label_name}_assays.tsv", emit: assays

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
        --assay_output ${label_name}_assays.tsv \
        --label_group_name ${label_name} \
        --n_splits 5 \
        --filter_label_size ${filter_label_size} \
        --filter_label_size_active ${filter_label_size_active}
    """
}


process binary_classification_assay_cpu {

    tag { "${label_name}:assay_${assay_index}:cpu" }

    label 'classification_cpu'
    cpus { params.classifier.cpus ?: 8 }
    memory { params.classifier.memory ?: '64 GB' }
    time '24h'
    maxForks ((params.classifier.max_cpu_jobs ?: 125) as int)

    publishDir "modules/06_classification/outputs/models", mode: 'copy', pattern: 'models/*.pkl', saveAs: { file -> file.toString().tokenize('/').last() }, failOnError: true
    
    input:
        tuple path(input_level6), path(input_level4), path(label_file), val(label_column), val(label_name), val(assay_index), val(assay_label)
    
    output:
        tuple val(label_name), path("axiom_${label_name}_assay${assay_index}_predictions.parquet"), emit: predictions
        tuple val(label_name), path("models/*.pkl"), emit: models, optional: true
    
    script:
    def total_repeats = params.classifier.n_repeats ?: 1
    def assay_arg = assay_label.toString().replace("'", "'\"'\"'")
    """
    python ${params.base_dir}/modules/06_classification/scripts/classify.py \
        --input_path_level6 ${input_level6} \
        --input_path_level4 ${input_level4} \
        --label_path ${label_file.name} \
        --label_column ${label_column} \
        --output axiom_${label_name}_assay${assay_index}_predictions.parquet \
        --n_repeats ${total_repeats} \
        --device cpu \
        --n_jobs ${task.cpus} \
        --include_label '${assay_arg}' \
        --save_models
    """
}


process binary_classification_assay_gpu {

    tag { "${label_name}:assay_${assay_index}:gpu" }

    label 'gpu'
    cpus { params.classifier.gpu_cpus ?: params.classifier.cpus ?: 8 }
    memory { params.classifier.memory ?: '64 GB' }
    time '24h'
    maxForks ((params.classifier.max_gpu_jobs ?: 16) as int)

    publishDir "modules/06_classification/outputs/models", mode: 'copy', pattern: 'models/*.pkl', saveAs: { file -> file.toString().tokenize('/').last() }, failOnError: true
    
    input:
        tuple path(input_level6), path(input_level4), path(label_file), val(label_column), val(label_name), val(assay_index), val(assay_label)
    
    output:
        tuple val(label_name), path("axiom_${label_name}_assay${assay_index}_predictions.parquet"), emit: predictions
        tuple val(label_name), path("models/*.pkl"), emit: models, optional: true
    
    script:
    def total_repeats = params.classifier.n_repeats ?: 1
    def assay_arg = assay_label.toString().replace("'", "'\"'\"'")
    """
    python ${params.base_dir}/modules/06_classification/scripts/classify.py \
        --input_path_level6 ${input_level6} \
        --input_path_level4 ${input_level4} \
        --label_path ${label_file.name} \
        --label_column ${label_column} \
        --output axiom_${label_name}_assay${assay_index}_predictions.parquet \
        --n_repeats ${total_repeats} \
        --device gpu \
        --n_jobs ${task.cpus} \
        --include_label '${assay_arg}' \
        --save_models
    """
}


process concatenate_classification_assays {

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

files = sorted(Path(".").glob("axiom_*_assay*_predictions.parquet"))
frames = [pl.read_parquet(path) for path in files]
combined = pl.concat(frames, how="vertical")

if {"repeat", "k_fold"}.issubset(combined.columns):
    combined = combined.sort(["Metadata_Label", "repeat", "k_fold", "Model_type", "Metadata_compound"])

combined.write_parquet("axiom_${label_name}_predictions.parquet")
PY
    """
}
