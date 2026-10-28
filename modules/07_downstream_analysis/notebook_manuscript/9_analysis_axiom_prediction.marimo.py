import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pickle
    from pathlib import Path

    import marimo as mo
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm
    return Path, mo, mpatches, np, pd, pickle, plt, sns, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Axiom Prediction With Axiom ToxCast Models

    Predict activity for all Axiom compounds with the batch-trained ToxCast
    models, retain significant assays, and visualize correlation between
    predicted assay activities.
    """
    )
    return


@app.cell
def _(Path):
    project_root = Path(".")
    classification_output_dir = project_root / "modules/06_classification/outputs"
    model_dir = classification_output_dir / "models"
    figure_output_dir = project_root / "modules/07_downstream_analysis/figures"
    table_output_dir = project_root / "modules/07_downstream_analysis/tables"
    downloads_dir = Path.home() / "Downloads"

    level6_axiom_path = (
        project_root
        / "modules/05_batch_correction_bsc_axiom/outputs/level6_axiom_selected_features_batch.parquet"
    )
    performance_path = classification_output_dir / "results_performance.csv"
    toxcast_category_path = (
        project_root
        / "modules/07_downstream_analysis/outputs/df_toxcast_cellbased_assay_info_significant_llm_categories.csv"
    )

    n_active_filter = 20
    correlation_method = "pearson"
    return (
        correlation_method,
        downloads_dir,
        figure_output_dir,
        level6_axiom_path,
        model_dir,
        n_active_filter,
        performance_path,
        table_output_dir,
        toxcast_category_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Axiom Profiles""")
    return


@app.cell
def _(level6_axiom_path, pd):
    df_level6_axiom_all = pd.read_parquet(level6_axiom_path)

    _candidate_id_cols = [
        _column
        for _column in ["Metadata_compound", "Metadata_OASIS_ID", "OASIS_ID"]
        if _column in df_level6_axiom_all.columns
    ]
    if not _candidate_id_cols:
        raise ValueError(
            "No compound identifier column found. Expected one of "
            "Metadata_compound, Metadata_OASIS_ID, or OASIS_ID."
        )

    compound_id_col = _candidate_id_cols[0]
    if (
        "Metadata_compound" not in df_level6_axiom_all.columns
        or df_level6_axiom_all["Metadata_compound"].isna().all()
    ):
        df_level6_axiom_all["Metadata_compound"] = (
            df_level6_axiom_all[compound_id_col].astype(str)
        )

    metadata_cols = [
        _column
        for _column in df_level6_axiom_all.columns
        if _column.startswith("Metadata_")
    ]
    feature_cols = [
        _column
        for _column in df_level6_axiom_all.columns
        if _column not in metadata_cols
    ]

    print(f"Axiom feature matrix: {df_level6_axiom_all.shape}")
    print(f"Compound identifier: {compound_id_col}")
    print(
        "Axiom compounds:",
        df_level6_axiom_all["Metadata_compound"].nunique(dropna=True),
    )
    print(f"Feature columns: {len(feature_cols)}")
    return compound_id_col, df_level6_axiom_all, feature_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Batch-Trained Models""")
    return


@app.cell
def _(model_dir, pickle):
    _model_files = sorted(model_dir.glob("*_actual_model.pkl"))
    if not _model_files:
        raise FileNotFoundError(f"No trained models found in {model_dir}")

    loaded_models = {}
    for _model_file in _model_files:
        _model_name = _model_file.name.replace("_actual_model.pkl", "")
        with open(_model_file, "rb") as _handle:
            loaded_models[_model_name] = pickle.load(_handle)

    print(f"Loaded models: {len(loaded_models)}")
    print("First models:", ", ".join(list(loaded_models)[:10]))
    return (loaded_models,)


@app.cell
def _(np, pd, tqdm):
    def predict_with_models(
        feature_df: pd.DataFrame,
        models: dict,
        *,
        id_col: str = "Metadata_compound",
        fallback_feature_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Predict class labels and activity probabilities for every model."""
        if fallback_feature_cols is None:
            fallback_feature_cols = [
                _column
                for _column in feature_df.columns
                if not _column.startswith("Metadata_")
                and _column != "Cell_Count"
            ]

        _compound_ids = feature_df[id_col].astype(str).values
        _class_matrix = pd.DataFrame(
            {"Metadata_compound": _compound_ids},
            index=feature_df.index,
        )
        _probability_matrix = pd.DataFrame(
            {"Metadata_compound": _compound_ids},
            index=feature_df.index,
        )
        _feature_audit = []

        for _model_name, _model in tqdm(
            models.items(),
            total=len(models),
            desc="Predicting",
        ):
            _booster_feature_names = _model.get_booster().feature_names
            _use_positional_features = _booster_feature_names is None
            _feature_name_source = "booster"
            if _use_positional_features:
                _trained_features = list(fallback_feature_cols)
                _feature_name_source = "fallback_order"
            else:
                _trained_features = list(_booster_feature_names)

            _missing_features = [
                _feature
                for _feature in _trained_features
                if _feature not in feature_df.columns
            ]

            if (
                len(_missing_features) == len(_trained_features)
                and len(_trained_features) == len(fallback_feature_cols)
            ):
                _trained_features = list(fallback_feature_cols)
                _missing_features = []
                _use_positional_features = True
                _feature_name_source = "fallback_order"

            _X_pred = feature_df.reindex(columns=_trained_features)
            _X_model = (
                _X_pred.to_numpy()
                if _use_positional_features
                else _X_pred
            )

            _class_predictions = np.asarray(_model.predict(_X_model))
            _class_matrix[_model_name] = _class_predictions

            if hasattr(_model, "predict_proba"):
                _probabilities = np.asarray(
                    _model.predict_proba(_X_model)
                )[:, 1]
            else:
                _probabilities = _class_predictions.astype(float)
            _probability_matrix[_model_name] = _probabilities

            _feature_audit.append(
                {
                    "assay": _model_name,
                    "n_trained_features": len(_trained_features),
                    "n_missing_features": len(_missing_features),
                    "missing_features": ", ".join(_missing_features[:10]),
                    "feature_name_source": _feature_name_source,
                }
            )

        return (
            _class_matrix.reset_index(drop=True),
            _probability_matrix.reset_index(drop=True),
            pd.DataFrame(_feature_audit),
        )
    return (predict_with_models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Predict All Axiom Compounds""")
    return


@app.cell
def _(
    compound_id_col,
    df_level6_axiom_all,
    feature_cols,
    loaded_models,
    predict_with_models,
):
    (
        pred_matrix_class_all,
        pred_matrix_probability_all,
        df_model_feature_audit,
    ) = predict_with_models(
        df_level6_axiom_all.copy(),
        loaded_models,
        id_col=compound_id_col,
        fallback_feature_cols=feature_cols,
    )

    _models_with_missing_features = (
        df_model_feature_audit["n_missing_features"].gt(0).sum()
    )
    print(f"Axiom rows predicted: {len(pred_matrix_class_all)}")
    print(f"Prediction assays: {pred_matrix_class_all.shape[1] - 1}")
    print(f"Models with missing features: {_models_with_missing_features}")
    df_model_feature_audit.head()
    return (
        df_model_feature_audit,
        pred_matrix_class_all,
        pred_matrix_probability_all,
    )


@app.cell
def _(df_model_feature_audit):
    df_model_feature_audit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Select Significant Prediction Assays""")
    return


@app.cell
def _(pd, performance_path, toxcast_category_path):
    df_ttest_results = pd.read_csv(performance_path)

    _significant_mask = pd.Series(True, index=df_ttest_results.index)
    if {"AUROC_significant", "AUPRG_significant"}.issubset(
        df_ttest_results.columns
    ):
        _significant_mask &= (
            df_ttest_results["AUROC_significant"]
            & df_ttest_results["AUPRG_significant"]
        )
    if {
        "AUROC_mean_Actual",
        "AUROC_mean_Random_baseline",
        "AUPRG_mean_Actual",
        "AUPRG_mean_Random_baseline",
    }.issubset(df_ttest_results.columns):
        _significant_mask &= (
            df_ttest_results["AUROC_mean_Actual"]
            > df_ttest_results["AUROC_mean_Random_baseline"]
        )
        _significant_mask &= (
            df_ttest_results["AUPRG_mean_Actual"]
            > df_ttest_results["AUPRG_mean_Random_baseline"]
        )

    df_ttest_results_significant = df_ttest_results[_significant_mask].copy()
    if df_ttest_results_significant.empty:
        df_ttest_results_significant = df_ttest_results.copy()

    df_toxcast_cellbased_assay_info_significant = pd.read_csv(
        toxcast_category_path
    )

    print(f"Performance rows: {len(df_ttest_results)}")
    print(f"Significant performance rows: {len(df_ttest_results_significant)}")
    print(
        "Cell-based category rows:",
        len(df_toxcast_cellbased_assay_info_significant),
    )
    return (
        df_toxcast_cellbased_assay_info_significant,
        df_ttest_results_significant,
    )


@app.cell
def _(n_active_filter, pd):
    def filter_prediction_assays(
        pred_matrix: pd.DataFrame,
        performance_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Keep prediction columns that correspond to significant assays."""
        _pred_matrix = pred_matrix.rename(columns={"MTT": "MT"})
        _valid_labels = set(performance_df["Metadata_Label"].values)

        _prediction_cols = [
            _column
            for _column in _pred_matrix.columns
            if _column != "Metadata_compound" and _column in _valid_labels
        ]

        if "Metadata_Count_1" in performance_df.columns:
            _active_lookup = performance_df.set_index("Metadata_Label")[
                "Metadata_Count_1"
            ]
            _prediction_cols = [
                _column
                for _column in _prediction_cols
                if _active_lookup.get(_column, 0) > n_active_filter
            ]

        return _pred_matrix[["Metadata_compound"] + _prediction_cols].copy()
    return (filter_prediction_assays,)


@app.cell
def _(
    df_ttest_results_significant,
    filter_prediction_assays,
    pred_matrix_class_all,
    pred_matrix_probability_all,
):
    pred_matrix_class_filtered = filter_prediction_assays(
        pred_matrix_class_all,
        df_ttest_results_significant,
    )
    pred_matrix_probability_filtered = filter_prediction_assays(
        pred_matrix_probability_all,
        df_ttest_results_significant,
    )

    print(
        "Filtered binary prediction matrix:",
        pred_matrix_class_filtered.shape,
    )
    print(
        "Filtered probability prediction matrix:",
        pred_matrix_probability_filtered.shape,
    )
    if pred_matrix_probability_filtered.shape[1] > 1:
        print(
            "Assays retained:",
            ", ".join(pred_matrix_probability_filtered.columns[1:]),
        )
    return pred_matrix_class_filtered, pred_matrix_probability_filtered


@app.cell
def _(df_ttest_results_significant):
    df_ttest_results_significant
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## Load prediction tables""")
    return


@app.cell
def _(pd):
    df_axiom_hits_prediction = pd.read_parquet("modules/06_classification/outputs/axiom_axiom_hits_predictions.parquet")
    df_axiom_cellbased_binary_prediction = pd.read_parquet("modules/06_classification/outputs/axiom_toxcast_cellbased_binary_predictions.parquet")
    df_axiom_cellfree_binary_prediction = pd.read_parquet("modules/06_classification/outputs/axiom_toxcast_cellfree_binary_predictions.parquet")
    df_axiom_cytotox_binary_prediction = pd.read_parquet("modules/06_classification/outputs/axiom_toxcast_cytotox_binary_predictions.parquet")
    print(f"Axiom cytotoxicity predictions: {df_axiom_hits_prediction.shape}")
    print(f"ToxCast cell-based predictions: {df_axiom_cellbased_binary_prediction.shape}")
    print(f"ToxCast cell-free predictions: {df_axiom_cellfree_binary_prediction.shape}")
    print(f"ToxCast cytotox predictions: {df_axiom_cytotox_binary_prediction.shape}")
    return (
        df_axiom_cellbased_binary_prediction,
        df_axiom_cellfree_binary_prediction,
        df_axiom_cytotox_binary_prediction,
        df_axiom_hits_prediction,
    )


@app.cell
def _(df_axiom_hits_prediction):
    df_axiom_hits_prediction
    return


@app.cell
def _(
    df_axiom_cellbased_binary_prediction,
    df_axiom_cellfree_binary_prediction,
    df_axiom_cytotox_binary_prediction,
    df_axiom_hits_prediction,
    df_ttest_results_significant,
):
    significant_labels = set(df_ttest_results_significant["Metadata_Label"].dropna())

    # Filter predictions to significant compounds
    _df_axiom_hits_prediction_filtered = df_axiom_hits_prediction[
        df_axiom_hits_prediction["Metadata_Label"].isin(significant_labels)
    ].copy()

    _df_axiom_cellbased_binary_prediction_filtered = df_axiom_cellbased_binary_prediction[
        df_axiom_cellbased_binary_prediction["Metadata_Label"].isin(significant_labels)
    ].copy()

    _df_axiom_cellfree_binary_prediction_filtered = df_axiom_cellfree_binary_prediction[
        df_axiom_cellfree_binary_prediction["Metadata_Label"].isin(significant_labels)
    ].copy()

    _df_axiom_cytotox_binary_prediction = df_axiom_cytotox_binary_prediction[
        df_axiom_cytotox_binary_prediction["Metadata_Label"].isin(significant_labels)
    ].copy()

    # Get unique compounds present in at least one filtered prediction dataframe
    all_compounds_train = (
        set(df_axiom_cellbased_binary_prediction["Metadata_compound"].dropna())
        | set(df_axiom_cellfree_binary_prediction["Metadata_compound"].dropna())
        | set(df_axiom_cytotox_binary_prediction["Metadata_compound"].dropna())
    )

    print(f"Number of compounds present in at least one dataset: {len(all_compounds_train)}")
    return (all_compounds_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prepare Assay Annotations""")
    return


@app.cell
def _(np, pd):
    def assign_assay_groups(columns: pd.Index) -> pd.DataFrame:
        """Classify prediction columns into source groups."""
        _cols = pd.Series(columns, dtype="string")

        _mask_cell_free = _cols.str.contains(
            "NVS|CCTE_GLTED_hIYD",
            regex=True,
            na=False,
        )
        _mask_hit_calls = _cols.str.contains(
            "MT|LDH|cell_count",
            regex=True,
            na=False,
        )
        _mask_cytotoxic = (
            _cols.str.contains("cell|tissue", regex=True, na=False)
            & ~_mask_hit_calls
        )
        _mask_cell_based = (
            ~_mask_cell_free & ~_mask_hit_calls & ~_mask_cytotoxic
        )

        _groups = np.select(
            [
                _mask_cell_free,
                _mask_cell_based,
                _mask_cytotoxic,
                _mask_hit_calls,
            ],
            ["Cell-free", "Cell-based", "Cytotoxic", "Hit calls"],
            default="Other",
        )

        return pd.DataFrame({"assay": columns, "group": _groups})
    return (assign_assay_groups,)


@app.cell
def _(assign_assay_groups, np, pd):
    def build_assay_annotations(
        assays: pd.Index,
        performance_df: pd.DataFrame,
        category_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add group, color, category, and performance metadata for assays."""
        _source_colors = {
            "Hits": "#4C72B0",
            "Cell-based": "#DD8452",
            "Cell-free": "#55A868",
            "Cytotox": "#C44E52",
        }
        _group_colors = {
            "Cell-free": _source_colors["Cell-free"],
            "Cell-based": _source_colors["Cell-based"],
            "Cytotoxic": _source_colors["Cytotox"],
            "Hit calls": _source_colors["Hits"],
            "Other": "#888888",
        }
        _group_display = {
            "Cell-free": "ToxCast cell-free",
            "Cell-based": "ToxCast cell-based",
            "Cytotoxic": "ToxCast cytotox",
            "Hit calls": "Cytotox",
            "Other": "Other",
        }

        _annotation_table = assign_assay_groups(assays)

        _category_color_col = next(
            (
                _column
                for _column in [
                    "chagpt_color",
                    "chatgpt_color",
                    "claudeai_color",
                ]
                if _column in category_df.columns
            ),
            None,
        )
        _category_label_col = next(
            (
                _column
                for _column in [
                    "chagpt_label",
                    "chatgpt_label",
                    "claudeai_label",
                ]
                if _column in category_df.columns
            ),
            None,
        )

        if (
            "assay_component_endpoint_name" in category_df.columns
            and _category_color_col is not None
            and _category_label_col is not None
        ):
            _cat_lookup = (
                category_df
                .drop_duplicates("assay_component_endpoint_name")
                .set_index("assay_component_endpoint_name")[
                    [_category_color_col, _category_label_col]
                ]
                .to_dict(orient="index")
            )
        else:
            _cat_lookup = {}

        _perf_lookup = (
            performance_df
            .drop_duplicates("Metadata_Label")
            .set_index("Metadata_Label")
        )

        _annotation_table["group_color"] = (
            _annotation_table["group"].map(_group_colors)
        )
        _annotation_table["group_display"] = (
            _annotation_table["group"].map(_group_display)
        )
        _annotation_table["category_color"] = [
            _cat_lookup.get(_assay, {}).get(_category_color_col, "#dddddd")
            for _assay in _annotation_table["assay"]
        ]
        _annotation_table["category_label"] = [
            _cat_lookup.get(_assay, {}).get(_category_label_col, None)
            for _assay in _annotation_table["assay"]
        ]
        _annotation_table["AUROC_mean_Actual"] = [
            _perf_lookup["AUROC_mean_Actual"].get(_assay, np.nan)
            if "AUROC_mean_Actual" in _perf_lookup.columns
            else np.nan
            for _assay in _annotation_table["assay"]
        ]
        _annotation_table["AUPRG_mean_Actual"] = [
            _perf_lookup["AUPRG_mean_Actual"].get(_assay, np.nan)
            if "AUPRG_mean_Actual" in _perf_lookup.columns
            else np.nan
            for _assay in _annotation_table["assay"]
        ]

        return _annotation_table
    return (build_assay_annotations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Correlate Significant Assays""")
    return


@app.cell
def _(
    all_compounds_train,
    pred_matrix_class_filtered,
    pred_matrix_probability_filtered,
):
    # Filter compounds not in training set

    pred_matrix_probability_test_compounds = pred_matrix_probability_filtered[
        ~pred_matrix_probability_filtered["Metadata_compound"].isin(all_compounds_train)
    ].copy()

    pred_matrix_class_test_compounds = pred_matrix_class_filtered[
        ~pred_matrix_class_filtered["Metadata_compound"].isin(all_compounds_train)
    ].copy()

    print(f"Probability matrix - other compounds: {pred_matrix_probability_test_compounds.shape}")
    print(f"Class matrix - other compounds: {pred_matrix_class_test_compounds.shape}")
    return (
        pred_matrix_class_test_compounds,
        pred_matrix_probability_test_compounds,
    )


@app.cell
def _(
    build_assay_annotations,
    correlation_method,
    df_toxcast_cellbased_assay_info_significant,
    df_ttest_results_significant,
    pred_matrix_class_test_compounds,
    pred_matrix_probability_test_compounds,
):
    df_probability_for_correlation = (
        pred_matrix_probability_test_compounds
        .set_index("Metadata_compound")
        .astype(float)
    )
    df_binary_for_correlation = (
        pred_matrix_class_test_compounds
        .set_index("Metadata_compound")
        .astype(float)
    )

    _non_constant_cols = [
        _column
        for _column in df_probability_for_correlation.columns
        if df_probability_for_correlation[_column].nunique(dropna=True) > 1
    ]
    df_probability_for_correlation = df_probability_for_correlation[
        _non_constant_cols
    ]
    df_binary_for_correlation = df_binary_for_correlation[
        _non_constant_cols
    ]

    correlation_matrix = df_probability_for_correlation.corr(
        method=correlation_method
    )
    binary_correlation_matrix = df_binary_for_correlation.corr(
        method=correlation_method
    )
    df_correlation_assay_annotations = build_assay_annotations(
        correlation_matrix.columns,
        df_ttest_results_significant,
        df_toxcast_cellbased_assay_info_significant,
    )

    print(
        "Compounds used for assay correlation:",
        df_probability_for_correlation.shape[0],
    )
    print(
        "Non-constant significant assays correlated:",
        df_probability_for_correlation.shape[1],
    )
    print(
        "Assay groups:",
        df_correlation_assay_annotations["group"].value_counts().to_dict(),
    )
    return (
        binary_correlation_matrix,
        correlation_matrix,
        df_correlation_assay_annotations,
        df_probability_for_correlation,
    )


@app.cell
def _(correlation_matrix, np):
    _upper_triangle = np.triu(
        np.ones(correlation_matrix.shape, dtype=bool),
        k=1,
    )
    df_top_correlated_assay_pairs = (
        correlation_matrix
        .where(_upper_triangle)
        .stack()
        .rename("correlation")
        .reset_index()
        .rename(columns={"level_0": "assay_1", "level_1": "assay_2"})
        .assign(abs_correlation=lambda _df: _df["correlation"].abs())
        .sort_values("abs_correlation", ascending=False)
        .reset_index(drop=True)
    )

    print("Top positively or negatively correlated assay pairs:")
    print(df_top_correlated_assay_pairs.head(20).to_string(index=False))
    return (df_top_correlated_assay_pairs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Correlation Heatmap""")
    return


@app.cell
def _(df_correlation_assay_annotations):
    df_correlation_assay_annotations
    return


@app.cell
def _(
    correlation_matrix,
    df_correlation_assay_annotations,
    downloads_dir,
    figure_output_dir,
    mpatches,
    plt,
    sns,
):
    _figure_path = (
        figure_output_dir
        / "Figure_S8_B.png"
    )
    _download_path = (
        downloads_dir
        / "correlation_heatmap_axiom_prediction_significant_assays.png"
    )

    figure_output_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    _plot_matrix = correlation_matrix.fillna(0)
    _annotation_lookup = df_correlation_assay_annotations.set_index("assay")
    _color_table = _annotation_lookup.reindex(_plot_matrix.index)[
        ["group_color", "category_color"]
    ]

    if _plot_matrix.empty:
        _fig, _ax = plt.subplots(figsize=(6, 5))
        _ax.text(
            0.5,
            0.5,
            "No non-constant assays to correlate",
            ha="center",
            va="center",
            transform=_ax.transAxes,
        )
        _ax.set_axis_off()
        _fig.savefig(_figure_path, bbox_inches="tight", dpi=300)
        _fig.savefig(_download_path, bbox_inches="tight", dpi=300)
    elif len(_plot_matrix) > 1:
        _fig_size = min(max(10, len(_plot_matrix) * 0.24), 34)
        _label_fontsize = max(10, min(9, 180 / len(_plot_matrix)))
        _clustermap = sns.clustermap(
            _plot_matrix,
            method="average",
            metric="euclidean",
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            row_colors=_color_table,
            col_colors=_color_table,
            figsize=(_fig_size, _fig_size),
            linewidths=0.05,
            dendrogram_ratio=(0.08, 0.08),
            cbar_pos=(0.02, 0.82, 0.02, 0.12),
            xticklabels=True,
            yticklabels=True,
        )

        plt.setp(
            _clustermap.ax_heatmap.get_xticklabels(),
            rotation=90,
            fontsize=_label_fontsize,
            ha="center",
        )
        plt.setp(
            _clustermap.ax_heatmap.get_yticklabels(),
            rotation=0,
            fontsize=_label_fontsize,
            va="center",
        )

        _legend_group_order = [
            "Cell-free",
            "Cell-based",
            "Cytotoxic",
            "Hit calls",
            "Other",
        ]

        _group_legend_patches = []
        for _group in _legend_group_order:
            _group_rows = df_correlation_assay_annotations[
                df_correlation_assay_annotations["group"] == _group
            ]
            if _group_rows.empty:
                continue

            _group_legend_patches.append(
                mpatches.Patch(
                    facecolor=_group_rows["group_color"].iloc[0],
                    label=_group_rows["group_display"].iloc[0],
                )
            )

        _group_legend = _clustermap.ax_heatmap.legend(
            handles=_group_legend_patches,
            loc="upper left",
            bbox_to_anchor=(0.88, 1.21),
            frameon=True,
            facecolor="white",
            edgecolor="lightgrey",
            framealpha=0.95,
            fontsize=9,
            borderpad=0.6,
            title="Assay groups",
            title_fontsize=10,
        )
        _group_legend.get_title().set_fontweight("bold")
        _clustermap.ax_heatmap.add_artist(_group_legend)

        _category_legend_patches = []
        for _category, _category_rows in (
            df_correlation_assay_annotations
            .dropna(subset=["category_color"])
            .groupby("category_label", sort=True)
        ):
            _category_legend_patches.append(
                mpatches.Patch(
                    facecolor=_category_rows["category_color"].iloc[0],
                    label=(
                        _category_rows["category_display"].iloc[0]
                        if "category_display" in _category_rows.columns
                        else _category
                    ),
                )
            )

        _category_legend = _clustermap.ax_heatmap.legend(
            handles=_category_legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.21),
            frameon=True,
            facecolor="white",
            edgecolor="lightgrey",
            framealpha=0.95,
            fontsize=9,
            borderpad=0.6,
            title="Assay categories",
            title_fontsize=10,
        )
        _category_legend.get_title().set_fontweight("bold")

        _clustermap.fig.suptitle(
            "Predicted Activity Correlation Across Significant Assays",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        _clustermap.fig.savefig(_figure_path, bbox_inches="tight", dpi=300)
        _clustermap.fig.savefig(_download_path, bbox_inches="tight", dpi=300)
        _fig = _clustermap.fig
    else:
        _fig, _ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            _plot_matrix,
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            annot=True,
            ax=_ax,
        )
        _ax.set_title(
            "Predicted Activity Correlation Across Significant Assays"
        )
        _fig.savefig(_figure_path, bbox_inches="tight", dpi=300)
        _fig.savefig(_download_path, bbox_inches="tight", dpi=300)

    print(f"Saved manuscript heatmap: {_figure_path}")
    print(f"Saved Downloads heatmap: {_download_path}")

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Active Predictions Compared With Training Set""")
    return


@app.cell
def _(df_ttest_results_significant, pd, pred_matrix_class_test_compounds):
    _training_counts = (
        df_ttest_results_significant[
            [
                "Metadata_Label",
                "Metadata_Count_0",
                "Metadata_Count_1",
            ]
        ]
        .drop_duplicates()
        .groupby("Metadata_Label", as_index=False)
        .sum()
    )
    _training_counts["training_total_count"] = (
        _training_counts["Metadata_Count_0"]
        + _training_counts["Metadata_Count_1"]
    )
    _training_counts["training_pct_active"] = (
        _training_counts["Metadata_Count_1"]
        / _training_counts["training_total_count"]
        * 100
    )

    _prediction_cols = [
        _column
        for _column in pred_matrix_class_test_compounds.columns
        if _column != "Metadata_compound"
    ]
    _prediction_values = pred_matrix_class_test_compounds[_prediction_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    _predicted_active_rate = pd.DataFrame(
        {
            "Metadata_Label": _prediction_cols,
            "predicted_active_count": (
                _prediction_values.eq(1).sum(axis=0).to_numpy()
            ),
            "predicted_total_count": (
                _prediction_values.notna().sum(axis=0).to_numpy()
            ),
            "predicted_pct_active": (
                _prediction_values.mean(axis=0).mul(100).to_numpy()
            ),
        }
    )

    df_axiom_active_rate_comparison = (
        _predicted_active_rate
        .merge(
            _training_counts[
                [
                    "Metadata_Label",
                    "Metadata_Count_0",
                    "Metadata_Count_1",
                    "training_total_count",
                    "training_pct_active",
                ]
            ],
            on="Metadata_Label",
            how="inner",
        )
        .assign(
            difference_pct_points=lambda _df: (
                _df["predicted_pct_active"] - _df["training_pct_active"]
            )
        )
        .sort_values("difference_pct_points", ascending=False)
        .reset_index(drop=True)
    )

    print(
        "Axiom non-training assays compared to training active rate:",
        len(df_axiom_active_rate_comparison),
    )
    df_axiom_active_rate_comparison
    return (df_axiom_active_rate_comparison,)


@app.cell
def _(
    df_axiom_active_rate_comparison,
    downloads_dir,
    figure_output_dir,
    np,
    plt,
):
    _figure_path = (
        figure_output_dir / "axiom_prediction_active_rate_vs_training.png"
    )
    _download_path = (
        downloads_dir / "axiom_prediction_active_rate_vs_training.png"
    )

    figure_output_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    _plot_df = df_axiom_active_rate_comparison.dropna(
        subset=["training_pct_active", "predicted_pct_active"]
    ).copy()

    _fig, _ax = plt.subplots(figsize=(7, 6))
    if _plot_df.empty:
        _ax.text(
            0.5,
            0.5,
            "No overlapping assays to compare",
            ha="center",
            va="center",
            transform=_ax.transAxes,
        )
        _ax.set_axis_off()
    else:
        _max_value = np.nanmax(
            _plot_df[["training_pct_active", "predicted_pct_active"]].to_numpy()
        )
        _axis_limit = min(100, max(5, _max_value * 1.08))

        _ax.scatter(
            _plot_df["training_pct_active"],
            _plot_df["predicted_pct_active"],
            s=48,
            alpha=0.78,
            color="#55A868",
            edgecolor="white",
            linewidth=0.5,
        )
        _ax.plot(
            [0, _axis_limit],
            [0, _axis_limit],
            color="#555555",
            linestyle="--",
            linewidth=1,
            label="Equal active rate",
        )

        _label_df = _plot_df.reindex(
            _plot_df["difference_pct_points"].abs()
            .sort_values(ascending=False)
            .head(8)
            .index
        )
        for _, _row in _label_df.iterrows():
            _ax.annotate(
                _row["Metadata_Label"],
                (
                    _row["training_pct_active"],
                    _row["predicted_pct_active"],
                ),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                alpha=0.85,
            )

        if len(_plot_df) > 1:
            _pearson_r = _plot_df["training_pct_active"].corr(
                _plot_df["predicted_pct_active"],
                method="pearson",
            )
            _spearman_r = _plot_df["training_pct_active"].corr(
                _plot_df["predicted_pct_active"],
                method="spearman",
            )
            _ax.text(
                0.03,
                0.97,
                (
                    f"Pearson r = {_pearson_r:.2f}\n"
                    f"Spearman rho = {_spearman_r:.2f}\n"
                    f"n assays = {len(_plot_df)}"
                ),
                transform=_ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "#cccccc",
                    "alpha": 0.9,
                },
            )

        _ax.set_xlim(0, _axis_limit)
        _ax.set_ylim(0, _axis_limit)
        _ax.set_xlabel("Training set active compounds (%)")
        _ax.set_ylabel("OASIS external test set predicted active compounds (%)")
        _ax.set_title(
            "Axiom External Test Set Predicted Active Rate vs Training Set"
        )
        _ax.grid(True, color="#e6e6e6", linewidth=0.7)
        _ax.legend(loc="lower right", frameon=False)

    _fig.tight_layout()
    _fig.savefig(_figure_path, bbox_inches="tight", dpi=300)
    _fig.savefig(_download_path, bbox_inches="tight", dpi=300)

    print(f"Saved Axiom active-rate plot: {_figure_path}")
    print(f"Saved Downloads active-rate plot: {_download_path}")

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Save Tables""")
    return


@app.cell
def _(
    binary_correlation_matrix,
    correlation_matrix,
    df_correlation_assay_annotations,
    df_model_feature_audit,
    df_probability_for_correlation,
    df_top_correlated_assay_pairs,
    pred_matrix_class_filtered,
    pred_matrix_probability_filtered,
    table_output_dir,
):
    table_output_dir.mkdir(parents=True, exist_ok=True)

    _binary_prediction_path = (
        table_output_dir / "axiom_prediction_activity_binary_significant.csv"
    )
    _probability_prediction_path = (
        table_output_dir
        / "axiom_prediction_activity_probability_significant.csv"
    )
    _probability_correlation_path = (
        table_output_dir
        / "axiom_prediction_assay_probability_correlation.csv"
    )
    _binary_correlation_path = (
        table_output_dir / "axiom_prediction_assay_binary_correlation.csv"
    )
    _assay_annotations_path = (
        table_output_dir / "axiom_prediction_assay_annotations.csv"
    )
    _top_pairs_path = (
        table_output_dir / "axiom_prediction_assay_top_correlations.csv"
    )
    _model_feature_audit_path = (
        table_output_dir / "axiom_prediction_model_feature_audit.csv"
    )

    pred_matrix_class_filtered.to_csv(_binary_prediction_path, index=False)
    pred_matrix_probability_filtered.to_csv(
        _probability_prediction_path,
        index=False,
    )
    correlation_matrix.to_csv(_probability_correlation_path)
    binary_correlation_matrix.to_csv(_binary_correlation_path)
    df_correlation_assay_annotations.to_csv(
        _assay_annotations_path,
        index=False,
    )
    df_top_correlated_assay_pairs.to_csv(_top_pairs_path, index=False)
    df_model_feature_audit.to_csv(_model_feature_audit_path, index=False)

    print("Saved filtered binary predictions:", _binary_prediction_path)
    print(
        "Saved filtered probability predictions:",
        _probability_prediction_path,
    )
    print("Saved probability correlation matrix:", _probability_correlation_path)
    print("Saved binary correlation matrix:", _binary_correlation_path)
    print("Saved assay annotations:", _assay_annotations_path)
    print("Saved top assay pairs:", _top_pairs_path)
    print("Saved feature audit:", _model_feature_audit_path)

    df_probability_for_correlation.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Activity Summary""")
    return


@app.cell
def _(df_correlation_assay_annotations, pred_matrix_class_filtered):
    df_active_by_assay = (
        pred_matrix_class_filtered
        .set_index("Metadata_compound")
        .reset_index()
        .melt(
            id_vars="Metadata_compound",
            var_name="assay",
            value_name="active",
        )
        .groupby("assay")["active"]
        .agg(
            active_count="sum",
            total_count="count",
        )
        .assign(
            percent_active=lambda _df: (
                _df["active_count"] / _df["total_count"] * 100
            ).round(1)
        )
        .reset_index()
        .merge(df_correlation_assay_annotations, on="assay", how="left")
        .sort_values("percent_active", ascending=False)
        .reset_index(drop=True)
    )

    print(f"Assay activity summary rows: {len(df_active_by_assay)}")
    df_active_by_assay
    return


if __name__ == "__main__":
    app.run()
