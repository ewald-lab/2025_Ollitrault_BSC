nextflow.enable.dsl = 2

workflow concresponse_workflow {
    take:
        input_data_level4
        input_data_level5
        input_list_compounds

    main:
        // 1. compute distance
        channel_5a_distances_r = Channel
            .fromList(params.concentration_response.distances_R)
        path_data_5a = compute_distances_R(
            channel_5a_distances_r.combine(input_data_level4))
        // 2. compile distances
        path_data_5b = compile_distances(path_data_5a.collect())
        // 3a. fit curves (bmds)
        path_data_5c = fit_curves(path_data_5b)
        // 4. fit curves meta (CC, MTT, LDH) by compound-list line
        ch_compound_lines_5d = input_list_compounds
            .flatMap { compound_file ->
                def records = []
                compound_file.toFile().eachLine { line, line_number ->
                    if (line.trim()) {
                        records << tuple(line_number, compound_file)
                    }
                }
                records
            }

        channel_5d_meta_channel = Channel
            .fromList(params.concentration_response.compute_meta_pods)

        path_data_5d_by_line = fit_curves_meta(
            channel_5d_meta_channel
                .combine(input_data_level4)
                .combine(ch_compound_lines_5d)
                .map { meta_nm, input, compound_line, compound_list_file ->
                    tuple(meta_nm, input, compound_line, compound_list_file)
                }
        ) // emits: (meta_nm, compound_line, meta_pods_path)

        path_data_5d = path_data_5d_by_line
            .map { meta_nm, compound_line, meta_pods_path ->
                tuple(meta_nm, meta_pods_path)
            }
            .groupTuple()
            | concatenate_metadata_pod_files

        // 5. select PODs
        path_data_5e = select_pod(path_data_5c)
        
        // 6. Transform back log10 concentration - combine ALL inputs into ONE channel
        // Combine BMD, metadata, and CP data into a single channel
        transform_input_channel = path_data_5c
            .combine(input_data_level5)
            .map { pods_path, metadata_path ->
                tuple("cp_all", pods_path, metadata_path, true)
            }
            .mix(
                path_data_5d
                    .combine(input_data_level5)
                    .map { meta_nm, meta_pods_path, input_file ->
                        tuple(meta_nm, meta_pods_path, input_file, true)
                    }
            )
            .mix(
                path_data_5e
                    .combine(input_data_level5)
                    .map { pods_path, input_file ->
                        tuple("cp", pods_path, input_file, false)
                    }
            )
        
        // Single call to transform_back_log10concentration with all data
        path_data_transformed = transform_back_log10concentration(transform_input_channel)

        // 7. plot Metadata curve fits [OUTDATED ]
        // path_data_5g = plot_meta_curve_fits(
        //    path_data_5d
        //        .combine(input_data_level4)
        //        .map { meta_nm, meta_pods, input_file ->
        //            tuple(meta_nm, meta_pods, input_file)
        //        }
        //)
        // 8. plot CP curve fits (FINAL)
        // Collect metadata
        path_data_5d
            .collect()
            .map { flat ->
                flat.collate(2)
            }
            .map { rows ->
                def meta_nm_list        = rows.collect { it[0] }
                def meta_pods_path_list = rows.collect { it[1] }
                tuple(meta_nm_list, meta_pods_path_list)
            }
            .set { meta_lists_5d }

        // Collect other inputs
        path_data_5e.collect().set { data_5e_all }
        path_data_5b.collect().set { data_5b_all }

        final_plot_input =
            data_5e_all
                .combine(data_5b_all)
                .combine(meta_lists_5d)
                .map { it ->
                    def cp_pod_path = it[0]
                    def dist_path   = it[1]
                    def meta_meta_nm  = it[2]
                    def meta_meta_pods_path = it[3]

                    tuple(
                        cp_pod_path,
                        dist_path,
                        meta_meta_nm,   // meta_nm list
                        meta_meta_pods_path    // meta_pods_path list
                    )
                }
        // Final plot runs ONCE
        path_data_5h = plot_cp_curve_fits(final_plot_input)

        // Analysis concreponse report

        // all_pods_files is a channel of all files
        all_pods_files = path_data_transformed.collect()

        // Split the files into separate channels by pattern
        input_pods_cp_all_ch = all_pods_files.map { files ->
            files.findAll { it.name.contains("cp_all") }
        }.flatten()

        input_pods_cp_ch = all_pods_files.map { files ->
            files.findAll { it.name.contains("cp_original") }
        }.flatten()

        input_pods_metadata_ch = all_pods_files.map { files ->
            files.findAll { it.name.contains("Metadata") }
        }.collect()

        analysis_concresponse_report(
                    input_data_level5,
                    input_pods_cp_all_ch,
                    input_pods_cp_ch,
                    input_pods_metadata_ch
                )
    emit: 
        all_pods_files
}
workflow concresponse_workflow_metadata {

    take:
        input_data_level4
        input_data_level5
        input_list_compounds

    main:
        ch_compound_lines_meta = input_list_compounds
            .flatMap { compound_file ->
                def records = []
                compound_file.toFile().eachLine { line, line_number ->
                    if (line.trim()) {
                        records << tuple(line_number, compound_file)
                    }
                }
                records
            }

        // 1. Fit curves for metadata assays (CC, MTT, LDH, ...) by compound-list line
        channel_meta = Channel
            .fromList(params.concentration_response.compute_meta_pods)

        path_data_meta_by_line = fit_curves_meta(
            channel_meta
                .combine(input_data_level4)
                .combine(ch_compound_lines_meta)
                .map { meta_nm, input, compound_line, compound_list_file ->
                    tuple(meta_nm, input, compound_line, compound_list_file)
                }
        ) // emits: (meta_nm, compound_line, meta_pods_path)

        path_data_meta = path_data_meta_by_line
            .map { meta_nm, compound_line, meta_pods_path ->
                tuple(meta_nm, meta_pods_path)
            }
            .groupTuple()
            | concatenate_metadata_pod_files

        // 2. Transform back log10 concentration — metadata branch only
        transform_input_meta = path_data_meta
            .combine(input_data_level5)
            .map { meta_nm, meta_pods_path, input_file ->
                tuple(meta_nm, meta_pods_path, input_file, true)
            }

        path_data_transformed_meta = transform_back_log10concentration(transform_input_meta)

        // 3. Collect transformed metadata pod files
        all_pods_files_meta = path_data_transformed_meta.collect()

    emit:
        all_pods_files_meta
}

workflow concresponse_workflow_feature_wise {
    take:
        input_data_level4
        input_data_level5
        input_list_compounds
    main:
        ch_compound_lines = input_list_compounds
            .flatMap { compound_file ->
                def records = []
                compound_file.toFile().eachLine { line, line_number ->
                    if (line.trim()) {
                        records << tuple(line_number, compound_file)
                    }
                }
                records
            }
        
        ch_feature_selection_interdependence_score = select_feature_selection_interdependence_score(
            input_data_level4.first(),
            ch_compound_lines
        )

        ch_feature_selection_score_by_line = ch_feature_selection_interdependence_score.score_by_compound
            .join(ch_compound_lines)

        ch_permutation_n_active = fit_curves_feature_wise_permutation(
            input_data_level4.first(),
            ch_feature_selection_score_by_line.randomSample(100),
            "TRUE"
        )

        // Collect permutation active-feature counts after interdependence-score filtering.
        ch_permutation_n_active_threshold = ch_permutation_n_active.n_active_by_compound_permutation
            .map { compound_line, permutation, n_active_file -> n_active_file }
            .collectFile(name: "all_n_actives_permutation_${params.input_name}.txt", newLine: true)
            | compute_percentile_permutation_active_features

        // Run feature-wise curve fitting after interdependence-score filtering, with threshold.
        ch_fit_curves_feature_wise = fit_curves_feature_wise(
            input_data_level4.first(),
            ch_feature_selection_score_by_line,
            ch_permutation_n_active_threshold.median_val.first()
        )

        ch_fit_curves_feature_wise_log10 = transform_log10concentration(input_data_level4.first(), ch_fit_curves_feature_wise.pod_feature_wise, "cp", "modules/04_concresponse/outputs/pods/feature_wise")

        // Collect all parquet outputs and concatenate
        ch_concatenated = ch_fit_curves_feature_wise.pod
            .collect()
            | concatenate_parquet_files
}

workflow concresponse_workflow_bayesian_inference {
    take:
        input_data_level4
        input_data_level5
    main:
        ch_compounds = Channel.fromPath("${params.base_dir}/modules/01_preprocessing/outputs/compounds_bsc.txt")
                            .splitText()
                            .map { it.trim() }
        // 1. compute distance
        channel_5a_distances_r = Channel
            .fromList(params.concentration_response.distances_R)
        path_data_5a = compute_distances_R(
            channel_5a_distances_r.combine(input_data_level4))
        // 2. compile distances
        path_data_5b = compile_distances(path_data_5a.collect())
        // 3a. fit curves (bmds)
        path_data_5c = fit_curves_bayesian_inferences(path_data_5b.collect().first(), ch_compounds)

        ch_fit_curves_feature_wise_log10 = transform_log10concentration(input_data_level4.first(), path_data_5c.pod, "cp", "modules/04_concresponse/outputs/pods/bayesian_inference")

        // Collect all parquet outputs and concatenate
        ch_concatenated_pods = path_data_5c.pod
            .collect()
            | concatenate_parquet_files

        
        // 6. Transform back log10 concentration - combine ALL inputs into ONE channel
        // Combine BMD, metadata, and CP data into a single channel
        /*
        transform_input_channel = path_data_5c
            .combine(input_data_level5)
            .map { pods_path, metadata_path ->
                tuple("cp_all", pods_path, metadata_path, true)
            }
            .mix(
                path_data_5e
                    .combine(input_data_level5)
                    .map { pods_path, input_file ->
                        tuple("cp", pods_path, input_file, false)
                    }
            )
        
        // Single call to transform_back_log10concentration with all data
        ch_fit_curves_bayesian_log10 = transform_log10concentration(input_data_level4.first(), path_data_5c, "cp", "modules/04_concresponse/outputs/pods")
        */
}

// ---------------------------------------------------------------------------
// compute_distances_R
// ---------------------------------------------------------------------------
// Script test : Rscript modules/04_concresponse/scripts/compute_distances.R 02_processed_data/level4b_bsc.parquet 03_tables/gmd.parquet 0.95 Metadata_perturbation DNA,ER,AGP,RNA,Mito,Brightfield gmd s0617825 modules/04_concresponse/scripts/cmd_functions.R
process compute_distances_R {
    tag { "${input_file.baseName}" }

    label 'r'

    publishDir "modules/04_concresponse/outputs/distances", mode: 'symlink', pattern: '*.parquet'

    input:
    tuple val(distance_R), path(input_file)

    output:
    path "distance_${params.input_name}_${distance_R}.parquet"

    script:
    def gmd_path = "${params.base_dir}/modules/04_concresponse/scripts/gmd_functions.R"
    def cmd_path = "${params.base_dir}/modules/04_concresponse/scripts/cmd_functions.R"
    def categories = params.categories.join(',')
    """
    Rscript ${params.base_dir}/modules/04_concresponse/scripts/compute_distances.R \
        ${input_file} \
        distance_${params.input_name}_${distance_R}.parquet \
        ${params.concentration_response.cover_var} \
        ${params.treatment} \
        "${categories}" \
        ${distance_R} \
        ${params.control} \
        ${gmd_path} \
        ${cmd_path}
    """
}

/*
// ---------------------------------------------------------------------------
// compute_distances_python [TODO ?]
// ---------------------------------------------------------------------------
process compute_distances_python {
    tag "${method}"

    input:
    val method from ch_dist_py
    path profiles from profiles_file

    output:
    path "outputs/${params.features}/${params.name}/distances/${method}.parquet"

    script:
    """
    python3 - <<'EOF'
import cr
cr.ap.calculate_distances(
    "${profiles}",
    "outputs/${params.features}/${params.name}/distances/${method}.parquet",
    "${method}",
)
EOF
    """
}
*/

// ---------------------------------------------------------------------------
// compile_distances
// ---------------------------------------------------------------------------
process compile_distances {
    tag { "${input_files.collect{ it.baseName }.join('_')}" }


    publishDir "modules/04_concresponse/outputs/distances", mode: 'symlink', pattern: '*.parquet'

    input:
        path input_files

    output:
        path "distances_${params.input_name}.parquet"

    script:
    """
    python ${params.base_dir}/modules/04_concresponse/scripts/compile_dist.py \
        --inputs ${input_files.collect { it }.join(' ')} \
        --output "distances_${params.input_name}.parquet" \
        --transform "${params.concentration_response.transform}"
    """
}


// ---------------------------------------------------------------------------
// fit_curves (bmds)
// ---------------------------------------------------------------------------
process fit_curves {
    tag { "${input_file.baseName}" }

    label 'r'

    publishDir "modules/04_concresponse/outputs/pods", mode: 'symlink', pattern: '*.parquet'

    input:
        path input_file

    output:
        path "pods_${params.input_name}_cp.parquet"

    """
    Rscript ${params.base_dir}/modules/04_concresponse/scripts/fit_curves.R \
        ${input_file} \
        pods_${params.input_name}_cp.parquet \
        ${params.concentration_response.num_sds} \
        ${params.control} \
        ${params.concentration_response.dmso_mode}
    """
}


// ---------------------------------------------------------------------------
// fit_curves_meta: CC, MTT, LDH
// ---------------------------------------------------------------------------
// Rscript modules/04_concresponse/scripts/fit_curves_meta.R 02_processed_data/level4b_bsc.parquet pods_Metadata_mtt_normalized_line1.parquet 2 Metadata_ldh_normalized s0617825 closest 1 compounds_bsc.txt
// Rscript /Users/ollitrault/Documents/projects/2025_Ollitrault_BSC/00.first_analysis_module/modules/04_concresponse/scripts/compute_distances.R 02_processed_data/level4b_axiom.parquet 03_tables/distance_axiom_cmd.parquet 0.95 Metadata_Perturbation "DNA,ER,AGP,RNA,Mito,Brightfield" cmd DMSO /Users/ollitrault/Documents/projects/2025_Ollitrault_BSC/00.first_analysis_module/modules/04_concresponse/scripts/gmd_functions.R /Users/ollitrault/Documents/projects/2025_Ollitrault_BSC/00.first_analysis_module/modules/04_concresponse/scripts/cmd_functions.R
process fit_curves_meta {
    tag { "${input_file.baseName} ${meta_nm} line=${compound_line}" }

    label 'r'

    input:
        tuple val(meta_nm), path(input_file), val(compound_line), path(compound_list_file)

    output:
        tuple val(meta_nm), val(compound_line), path("pods_${params.input_name}_${meta_nm}_line${compound_line}.parquet")

    """
    Rscript ${params.base_dir}/modules/04_concresponse/scripts/fit_curves_meta.R \
        ${input_file} \
        pods_${params.input_name}_${meta_nm}_line${compound_line}.parquet \
        ${params.concentration_response.num_sds} \
        ${meta_nm} \
        ${params.control} \
        ${params.concentration_response.dmso_mode} \
        ${compound_line} \
        ${compound_list_file}
    """
}


// ---------------------------------------------------------------------------
// concatenate_metadata_pod_files
// ---------------------------------------------------------------------------
process concatenate_metadata_pod_files {
    tag { "${meta_nm}" }

    label 'r'

    publishDir "modules/04_concresponse/outputs/pods", mode: 'symlink', pattern: '*.parquet'

    input:
        tuple val(meta_nm), path(input_files)

    output:
        tuple val(meta_nm), path("pods_${params.input_name}_${meta_nm}.parquet")

    script:
    def r_quote = { x -> '"' + x.toString().replace('\\', '\\\\').replace('"', '\\"') + '"' }
    def r_files = input_files.collect { r_quote(it) }.join(', ')

    """
    Rscript - <<'EOF'
    library(arrow)
    library(dplyr)

    files <- c(${r_files})
    if (length(files) == 0) {
      stop("No metadata POD files were provided for ${meta_nm}.")
    }

    line_number <- suppressWarnings(as.integer(gsub("[^0-9]", "", sub(".*_line", "", basename(files)))))
    files <- files[order(line_number, na.last = TRUE)]

    combined <- dplyr::bind_rows(lapply(files, arrow::read_parquet))
    arrow::write_parquet(combined, "pods_${params.input_name}_${meta_nm}.parquet")
EOF
    """
}


// ---------------------------------------------------------------------------
// select_pod
// ---------------------------------------------------------------------------
// Rscript modules/04_concresponse/scripts/select_pod.R 03_tables/bmds_bsc.parquet 03_tables/meta_pods_Metadata_mtt_normalized.parquet pods_bsc_Metadata_mtt_normalized.parquet         Metadata_mtt_normalized
process select_pod {
    tag { "${input_file_bmds.baseName}" }

    label 'r'

    publishDir "modules/04_concresponse/outputs/pods", mode: 'copy', pattern: '*.parquet'

    input:
    // tuple val(meta_nm), path(input_file_meta_pods), path(input_file_bmds)
    path(input_file_bmds)
    
    output:
    // tuple val(meta_nm), path(input_file_meta_pods), path("pods_${params.input_name}_${meta_nm}.parquet")
    path("pods_${params.input_name}_cp_filter.parquet")

    """
    Rscript ${params.base_dir}/modules/04_concresponse/scripts/select_pod.R \
        ${input_file_bmds} \
        "pods_${params.input_name}_cp_filter.parquet" \
        ${params.concentration_response.filter_sd_residuals} # whether to filter by the SD of the residuals (TRUE or FALSE)
    """
}

// ---------------------------------------------------------------------------
// Transform back log10 concentration to original concentration
// ---------------------------------------------------------------------------
// python3 modules/04_concresponse/scripts/transform_back_log10concentration.py --input_pods 03_tables/meta_pods_Metadata_mtt_normalized.parquet --input_metadata 02_processed_data/level5_bsc.parquet --output pods_bsc_Metadata_mtt_normalized_original_conc.parquet --control s0617825 --pod_col Metadata_mtt_normalized_POD
// python3 modules/04_concresponse/scripts/transform_back_log10concentration.py --input_pods 03_tables/pods_bsc_meta_Metadata_mtt_normalized.parquet --input_metadata 02_processed_data/level5_bsc.parquet --output pods_bsc_meta_Metadata_mtt_normalized_original_conc.parquet --control s0617825 --pod_col bmd
process transform_back_log10concentration {
    tag { "${input_file.baseName} ${meta_nm}" }

    publishDir "modules/04_concresponse/outputs/pods", mode: 'symlink', pattern: '*.parquet'

    input:
    tuple val(meta_nm), path(input_file), path(input_file_metadata), val(is_metadata_pods)

    output:
    path "pods_${params.input_name}_${meta_nm}_original_conc.parquet"

    script:
    def metadata_flag = is_metadata_pods ? "--is_metadata_pods" : ""

    """
    python3 ${params.base_dir}/modules/04_concresponse/scripts/transform_back_log10concentration.py \
        --input_pods ${input_file} \
        --input_metadata ${input_file_metadata} \
        --output pods_${params.input_name}_${meta_nm}_original_conc.parquet \
        --control ${params.control} \
        --pod_col ${meta_nm} \
        ${metadata_flag}
    """
}

// ---------------------------------------------------------------------------
// Transform back log10 concentration to original concentration simplified function
// ---------------------------------------------------------------------------
// python3 modules/04_concresponse/scripts/transform_log10concentration.py --input_pods 03_tables/meta_pods_Metadata_mtt_normalized.parquet --input_metadata 02_processed_data/level5_bsc.parquet --output pods_bsc_Metadata_mtt_normalized_original_conc.parquet --control s0617825 --pod_col Metadata_mtt_normalized_POD
// python3 modules/04_concresponse/scripts/transform_log10concentration.py --input_pods 03_tables/pods_bsc_meta_Metadata_mtt_normalized.parquet --input_metadata 02_processed_data/level5_bsc.parquet --output pods_bsc_meta_Metadata_mtt_normalized_original_conc.parquet --control s0617825 --pod_col bmd
process transform_log10concentration {
    tag { "${input_file.baseName} ${path_dt_pod}" }

    publishDir "${publishdir}", mode: 'symlink', pattern: '*.parquet'

    input:
        path input_file
        path path_dt_pod
        val bmd_col
        val publishdir

    output:
    path "${path_dt_pod.baseName}_original_conc.parquet"

    """
    python3 ${params.base_dir}/modules/04_concresponse/scripts/transform_log10concentration.py \
        --input_level4 ${input_file} \
        --input_file ${path_dt_pod} \
        --output ${path_dt_pod.baseName}_original_conc.parquet \
        --control ${params.control} \
        --pod_col ${bmd_col}
    """
}

// ---------------------------------------------------------------------------
// Plotting processes
// ---------------------------------------------------------------------------
// Rscript modules/04_concresponse/scripts/plot_meta_curve.R 03_tables/meta_pods_Metadata_mtt_normalized.parquet 02_processed_data/level4b_bsc.parquet plots_bsc_Metadata_mtt_normalized_plots.pdf Metadata_mtt_normalized s0617825
/*
process plot_meta_curve_fits {
    tag { "${input_file.baseName} ${meta_nm}" }

    publishDir { "reports/${params.run_code}_${params.input_name}" }, mode: 'copy', pattern: '*.pdf'

    input:
    tuple val(meta_nm), path(input_file_meta_pods), path(input_file)

    output:
    path "plots_pods_meta_${params.input_name}_${meta_nm}.pdf"

    """
    Rscript ${params.base_dir}/modules/04_concresponse/scripts/plot_meta_curve.R \
        ${input_file_meta_pods} \
        ${input_file} \
        "plots_pods_meta_${params.input_name}_${meta_nm}.pdf" \
        ${meta_nm} \
        ${params.control} \
        ${params.concentration_response.dmso_mode}
    """
}

process plot_mtt_curve_fits {
    input:
    path mttpods from ch_mttpods
    path profiles from profiles_file

    output:
    path "outputs/${params.features}/${params.name}/curves/plots/mtt_plots.pdf"

    """
    Rscript concresponse/plot_meta_curve.R \
        ${mttpods} \
        ${profiles} \
        outputs/${params.features}/${params.name}/curves/plots/mtt_plots.pdf \
        Metadata_mtt_normalized
    """
}


process plot_ldh_curve_fits {
    input:
    path ldhpods from ch_ldhpods
    path profiles from profiles_file

    output:
    path "outputs/${params.features}/${params.name}/curves/plots/ldh_plots.pdf"

    """
    Rscript concresponse/plot_meta_curve.R \
        ${ldhpods} \
        ${profiles} \
        outputs/${params.features}/${params.name}/curves/plots/ldh_plots.pdf \
        Metadata_ldh_abs_signal
    """
}
*/

// Rscript modules/04_concresponse/scripts/plot_cp_curve.R 03_tables/pods_bsc_Metadata_ldh_normalized.parquet 03_tables/meta_pods_Metadata_ldh_normalized.parquet 03_tables/distances_bsc.parquet "plots_pods_cp_bsc_Metadata_ldh_normalized.pdf" Metadata_ldh_normalized s0617825
process plot_cp_curve_fits {
    tag { "${dist_path.baseName} ${meta_nm}" }

    label 'r'

    publishDir { "reports/${params.run_code}_${params.input_name}" }, mode: 'symlink', pattern: '*.pdf'

    input:
    tuple path(cp_pod_path), path(dist_path), val(meta_nm), val(meta_pods_path)

    output:
    path "plots_pods_cp_${params.input_name}.pdf"

    """
    Rscript ${params.base_dir}/modules/04_concresponse/scripts/plot_cp_curve.R \
        ${cp_pod_path} \
        '${meta_pods_path.join(",")}' \
        ${dist_path} \
        "plots_pods_cp_${params.input_name}.pdf" \
        '${meta_nm.join(",")}' \
        ${params.control} \
        ${params.concentration_response.dmso_mode}
    """
}


process analysis_concresponse_report {
    tag { "${params.input_name}" }
    publishDir { "reports/${params.run_code}_${params.input_name}" }, mode: 'symlink'

    input:
    path input_data_level5            // your level5 parquet
    path input_pods_cp_all            // the CP_all parquet
    path input_pods_cp                // the CP parquet
    path input_pods_metadata          // list of metadata parquet files

    output:
    path "*.html"                     // output Marimo report

    script:
    """
    marimo export html \
    ${params.base_dir}/modules/08_reporting/notebooks/5_analysis_concresponse.marimo.py \
    -o 5_analysis_concresponse_marimo_${params.input_name}.html \
    -- \
        --name ${params.input_name} \\
        --input ${input_data_level5} \\
        --input_pods_cp_all ${input_pods_cp_all} \\
        --input_pods_cp ${input_pods_cp} \\
        --input_pods_metadata ${input_pods_metadata.join(' ')} \\
        --control ${params.control} \\
        --metadata_cell_viability ${params.concentration_response.compute_meta_pods.join(' ')}
    """
}

process select_feature_selection_interdependence_score {
    tag { "${input_file.baseName} compound_line=${compound_line}" }
    publishDir "modules/04_concresponse/outputs/interdependence_score",
        mode: 'symlink',
        pattern: '*.txt'
    input:
        path input_file
        tuple val(compound_line), path(compound_list_file)
    output:
        tuple val(compound_line), path("feature_selection_interdependence_score_${params.input_name}_line*.txt"), emit: score_by_compound
    script:
        def output_file = "feature_selection_interdependence_score_${params.input_name}_line${compound_line}.txt"

    """
    python "${params.base_dir}/modules/04_concresponse/scripts/select_features_interdependence_scores.py" \\
        --input_path "${input_file}" \\
        --output_path "${output_file}" \\
        --ctrl "${params.control}" \\
        --dmso_mode "${params.concentration_response.dmso_mode}" \\
        --compound_list_path "${compound_list_file}" \\
        --compound_line "${compound_line}" \\
        --num_tests "${params.concentration_response.num_tests}" \\
        --p_value_correction "${params.concentration_response.p_value_correction}"
    """
}

process fit_curves_feature_wise_permutation {

    tag { "compound_line=${compound_line} permutation=${permutation}" }

    label 'r'

    input:
        path input_file
        tuple val(compound_line), path(feature_selection_score_file), path(compound_list_file)
        val permutation

    output:
        tuple val(compound_line), val(permutation), path("n_active_permutation_${params.input_name}_line*.txt"), emit: n_active_by_compound_permutation

    script:
        def output_file = "n_active_permutation_${params.input_name}_line${compound_line}.txt"

    """
    set -euo pipefail

    Rscript "${params.base_dir}/modules/04_concresponse/scripts/fit_curves_features_wise.R" \\
        "${input_file}" \\
        "${output_file}" \\
        "${params.concentration_response.num_sds}" \\
        "${params.control}" \\
        "${params.concentration_response.dmso_mode}" \\
        "${params.categories.join(',')}" \\
        "TRUE" \\
        "${compound_line}" \\
        "${permutation}" \\
        "NA" \\
        "NA" \\
        "${feature_selection_score_file}" \\
        "${compound_list_file}"
    """
}
process compute_percentile_permutation_active_features {
    label 'r'
    publishDir { "modules/04_concresponse/outputs/pods/feature_wise" }, mode: 'symlink', pattern: 'feature_wise_median_permutation_threshold_*.txt'
    input:
        path threshold_active_features_file
    output:
        stdout emit: median_val
        path "feature_wise_median_permutation_threshold_${params.input_name}.txt", emit: median_file
    script:
        def output_file = "feature_wise_median_permutation_threshold_${params.input_name}.txt"
    """
    Rscript - <<'EOF'
    vals <- as.numeric(readLines("${threshold_active_features_file}"))
    med <- median(vals, na.rm = TRUE)
    cat(med)
    writeLines(as.character(med), "${output_file}")
EOF
"""
}

process fit_curves_feature_wise {
    tag { "compound_line=${compound_line}" }
    publishDir "modules/04_concresponse/outputs/pods/feature_wise",
        mode: 'symlink',
        pattern: '*_feature_wise.parquet'
    label 'r'
    input:
        path input_file
        tuple val(compound_line), path(feature_selection_score_file), path(compound_list_file)
        val thresholds_n_active
    output:
        path "pod_${params.input_name}_*_summary.parquet", emit: pod
        path "pod_${params.input_name}_*_feature_wise.parquet", emit: pod_feature_wise
    script:
        def pod_file = "pod_${params.input_name}_line${compound_line}_summary.parquet"
        def pod_feature_wise_file = "pod_${params.input_name}_line${compound_line}_feature_wise.parquet"

        // Bash-safe single quoting for all arguments
        def q = { x -> "'${x.toString().replace("'", "'\"'\"'")}'" }

    """
    Rscript ${q(params.base_dir + "/modules/04_concresponse/scripts/fit_curves_features_wise.R")} \\
        ${q(input_file)} \\
        ${q(pod_file)} \\
        ${q(params.concentration_response.num_sds)} \\
        ${q(params.control)} \\
        ${q(params.concentration_response.dmso_mode)} \\
        ${q(params.categories.join(','))} \\
        TRUE \\
        ${q(compound_line)} \\
        FALSE \\
        ${q(thresholds_n_active)} \\
        ${q(pod_feature_wise_file)} \\
        ${q(feature_selection_score_file)} \\
        ${q(compound_list_file)}
    """
}

process concatenate_parquet_files {
    label 'r'
    publishDir { "modules/04_concresponse/outputs/pods/" }, mode: 'symlink', pattern: '*.parquet'
    input:
        path parquet_files
    output:
        path "pods_${params.input_name}_cp.parquet"
    """
    Rscript - <<'EOF'
    library(arrow)
    
    files <- strsplit("${parquet_files}", " ")[[1]]
    
    combined <- do.call(rbind, lapply(files, arrow::read_parquet))
    
    arrow::write_parquet(combined, "pods_${params.input_name}_cp.parquet")
EOF
    """
}

// ---------------------------------------------------------------------------
// fit_curves (bmds) BAYESIAN INFERENCE
// ---------------------------------------------------------------------------
process fit_curves_bayesian_inferences {
    tag { "${input_file.baseName}" }

    memory = '8 GB'

    label 'r'

    publishDir "modules/04_concresponse/outputs/pods/bayesian_inference", mode: 'symlink', pattern: '*.parquet'
    publishDir "modules/04_concresponse/outputs/pods/bayesian_inference", mode: 'symlink', pattern: '*.png'

    input:
        path input_file
        val compound

    output:
        path "pods_${params.input_name}_cp_${compound}.parquet", emit: pod
        path "pods_${params.input_name}_cp_bayese_inference_plot_${compound}_*.png", emit: plot

    """
    echo yes
    Rscript ${params.base_dir}/modules/04_concresponse/scripts/bayesian_inference_toxicr.R \
        ${input_file} \
        pods_${params.input_name}_cp_${compound}.parquet \
        ${params.control} \
        ${params.concentration_response.dmso_mode} \
        pods_${params.input_name}_cp_bayese_inference_plot_${compound} \
        TRUE \
        ${params.concentration_response.num_sds} \
        ${compound} 
    """
}
