import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import glob
    import os
    import pickle
    from pathlib import Path

    import marimo as mo
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    from scipy.cluster.hierarchy import leaves_list, linkage
    from tqdm import tqdm
    return (
        ListedColormap,
        Path,
        leaves_list,
        linkage,
        mo,
        mpatches,
        np,
        pd,
        pickle,
        plt,
        sns,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # BSC Prediction With Axiom ToxCast Models

    Clean manuscript copy of the section after
    `## Classification BSC with Axiom (Toxcast models)`.
    """
    )
    return


@app.cell
def _(Path):
    project_root = Path(".")
    classification_output_dir = project_root / "modules/06_classification/outputs"
    model_dir = classification_output_dir / "models"
    figure_output_dir = project_root / "modules/07_downstream_analysis/figures"
    downloads_dir = Path.home() / "Downloads"

    level6_bsc_path = (
        project_root
        / "modules/03_postprocessing/outputs/level6_bsc_selected_features_filtered.parquet"
    )
    bsc_annotation_path = (
        project_root
        / "modules/01_preprocessing/outputs/cellpainting_bsc_annotations_merged.tsv"
    )
    performance_path = classification_output_dir / "results_performance.csv"
    toxcast_category_path = ("modules/07_downstream_analysis/outputs/df_toxcast_cellbased_assay_info_significant_llm_categories.csv"
    )

    n_active_filter = 20
    return (
        bsc_annotation_path,
        downloads_dir,
        figure_output_dir,
        level6_bsc_path,
        model_dir,
        n_active_filter,
        performance_path,
        toxcast_category_path,
    )


@app.cell
def _(bsc_annotation_path, level6_bsc_path, pd):
    # Load the BSC level-6 feature matrix and botanical annotations.
    df_level6_bsc_all = pd.read_parquet(level6_bsc_path)
    df_bsc_annotations = pd.read_csv(bsc_annotation_path, sep="\t")
    df_bsc_constituent_botanical_pairs = (
        df_bsc_annotations[["Metadata_compound", "Plant scode"]]
        .dropna()
        .drop_duplicates()
    )
    df_bsc_unique_constituents = (
        df_bsc_annotations
        .dropna(subset=["Metadata_compound"])
        .drop_duplicates("Metadata_compound")
    )
    _shared_constituent_counts = (
        df_bsc_constituent_botanical_pairs
        .groupby("Metadata_compound")["Plant scode"]
        .nunique()
    )

    list_botanicals = (
        df_bsc_annotations["Plant scode"]
        .dropna()
        .drop_duplicates()
        .to_numpy()
    )

    dic_botanicals_scode_name = (
        df_bsc_annotations[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )

    print(f"BSC feature matrix: {df_level6_bsc_all.shape}")
    print(f"BSC annotation table: {df_bsc_annotations.shape}")
    print(f"Unique BSC constituents: {len(df_bsc_unique_constituents)}")
    print(f"Botanicals in annotation table: {len(list_botanicals)}")
    print(
        "Constituent-botanical assignments:",
        len(df_bsc_constituent_botanical_pairs),
    )
    print(
        "Constituents assigned to multiple botanicals:",
        int(_shared_constituent_counts.gt(1).sum()),
    )
    return (
        df_bsc_annotations,
        df_level6_bsc_all,
        dic_botanicals_scode_name,
        list_botanicals,
    )


@app.cell
def _(ListedColormap, dic_botanicals_scode_name):
    # Binary colormap used by the prediction heatmaps.
    binary_cmap = ListedColormap(["lightgray", "red"])
    compound_to_plant_name = dic_botanicals_scode_name.copy()
    return (binary_cmap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Axiom ToxCast Models""")
    return


@app.cell
def _(model_dir, pickle):
    # Models are produced by the classification pipeline.
    _model_files = sorted(model_dir.glob("*_actual_model.pkl"))

    loaded_models = {}
    for _model_file in _model_files:
        _model_name = _model_file.name.replace("_actual_model.pkl", "")
        with open(_model_file, "rb") as _handle:
            loaded_models[_model_name] = pickle.load(_handle)

    print(f"Loaded models: {len(loaded_models)}")
    if loaded_models:
        print("First models:", ", ".join(list(loaded_models)[:10]))
    return (loaded_models,)


@app.cell
def _(pd, tqdm):
    def get_model_feature_names(model):
        """Return feature names saved with a model, if available."""
        _feature_names = getattr(model, "feature_names_in_", None)
        if _feature_names is not None and len(_feature_names) > 0:
            return list(_feature_names)

        _booster = model.get_booster()
        _feature_names = _booster.feature_names
        if _feature_names:
            return list(_feature_names)

        return None


    def build_prediction_features(
        feature_df: pd.DataFrame,
        model,
        model_name: str,
    ) -> pd.DataFrame:
        """Return a numeric feature matrix aligned to one trained model."""
        _trained_features = get_model_feature_names(model)
        if _trained_features:
            _missing_features = [
                _feature
                for _feature in _trained_features
                if _feature not in feature_df.columns
            ]
            if _missing_features:
                raise ValueError(
                    f"{model_name}: prediction data is missing "
                    f"{len(_missing_features)} trained feature(s), including "
                    f"{_missing_features[:5]}."
                )

            _candidate_features = _trained_features
        else:
            _candidate_features = [
                _column
                for _column in feature_df.columns
                if _column.startswith("Feature_")
            ]
            if not _candidate_features:
                _candidate_features = [
                    _column
                    for _column in feature_df.columns
                    if (
                        not _column.startswith("Metadata_")
                        and _column != "Cell_Count"
                        and pd.api.types.is_numeric_dtype(feature_df[_column])
                    )
                ]

        _expected_features = getattr(model, "n_features_in_", None)
        if (
            _expected_features is not None
            and len(_candidate_features) != _expected_features
        ):
            raise ValueError(
                f"{model_name}: inferred {len(_candidate_features)} "
                f"prediction feature(s), but the model expects "
                f"{_expected_features}. Check that the BSC level-6 matrix "
                "uses the same selected features as model training."
            )

        if not _candidate_features:
            raise ValueError(
                f"{model_name}: could not infer prediction feature columns."
            )

        _X_pred = feature_df.loc[:, _candidate_features]
        _invalid_columns = _X_pred.select_dtypes(
            exclude=["number", "bool"],
        ).columns.tolist()
        if _invalid_columns:
            raise ValueError(
                f"{model_name}: prediction features must be numeric or bool; "
                f"invalid columns include {_invalid_columns[:5]}."
            )

        return _X_pred


    def predict_with_models(
        feature_df: pd.DataFrame,
        models: dict,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predict class labels and activity probabilities for every model."""
        _class_matrix = pd.DataFrame(
            {"Metadata_compound": feature_df["Metadata_compound"].values},
            index=feature_df.index,
        )
        _probability_matrix = pd.DataFrame(
            {"Metadata_compound": feature_df["Metadata_compound"].values},
            index=feature_df.index,
        )

        for _model_name, _model in tqdm(
            models.items(),
            total=len(models),
            desc="Predicting",
        ):
            _X_pred = build_prediction_features(
                feature_df,
                _model,
                _model_name,
            )
            _class_predictions = _model.predict(_X_pred)
            _class_matrix[_model_name] = _class_predictions

            if hasattr(_model, "predict_proba"):
                _probabilities = _model.predict_proba(_X_pred)[:, 1]
            else:
                _probabilities = (
                    pd.Series(_class_predictions)
                    .astype(float)
                    .to_numpy()
                )
            _probability_matrix[_model_name] = _probabilities

        return (
            _class_matrix.reset_index(drop=True),
            _probability_matrix.reset_index(drop=True),
        )
    return (predict_with_models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Predict BSC Botanicals""")
    return


@app.cell
def _(df_level6_bsc_all, list_botanicals, loaded_models, predict_with_models):
    # Botanical-only prediction matrix for quick visual checks.
    df_level6_bsc_botanicals = df_level6_bsc_all[
        df_level6_bsc_all["Metadata_compound"].isin(list_botanicals)
    ].copy()

    pred_matrix, pred_matrix_probability = predict_with_models(
        df_level6_bsc_botanicals,
        loaded_models,
    )

    print(f"Botanical rows predicted: {len(pred_matrix)}")
    print(f"Prediction assays: {pred_matrix.shape[1] - 1}")
    print(
        "Probability prediction assays:",
        pred_matrix_probability.shape[1] - 1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Predict BSC Botanicals And Constituents""")
    return


@app.cell
def _(df_level6_bsc_all, loaded_models, predict_with_models):
    # Full matrix includes botanical mixtures and their annotated constituents.
    pred_matrix_all, pred_matrix_probability_all = predict_with_models(
        df_level6_bsc_all.copy(),
        loaded_models,
    )

    print(f"All BSC rows predicted: {len(pred_matrix_all)}")
    print(f"All class prediction assays: {pred_matrix_all.shape[1] - 1}")
    print(
        "All probability prediction assays:",
        pred_matrix_probability_all.shape[1] - 1,
    )
    return pred_matrix_all, pred_matrix_probability_all


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Select Significant Prediction Assays""")
    return


@app.cell
def _(pd, performance_path, toxcast_category_path):
    df_ttest_results = pd.read_csv(performance_path)
    df_ttest_results_filtered = df_ttest_results

    df_toxcast_cellbased_assay_info_significant = pd.read_csv(
        toxcast_category_path
    )

    print(f"Performance rows before filter: {len(df_ttest_results)}")
    print(f"Performance rows after filter: {len(df_ttest_results_filtered)}")
    print(
        "Cell-based category rows:",
        len(df_toxcast_cellbased_assay_info_significant),
    )
    return (
        df_toxcast_cellbased_assay_info_significant,
        df_ttest_results_filtered,
    )


@app.cell
def _(df_toxcast_cellbased_assay_info_significant):
    df_toxcast_cellbased_assay_info_significant["chagpt_label"].value_counts()
    return


@app.cell
def _(
    df_ttest_results_filtered,
    n_active_filter,
    pred_matrix_all,
    pred_matrix_probability_all,
):
    # Keep only assays retained by performance filtering and active enough.
    _valid_labels = set(df_ttest_results_filtered["Metadata_Label"].values)
    _pred_matrix_all = pred_matrix_all.rename(columns={"MTT": "MT"})
    _pred_matrix_probability_all = pred_matrix_probability_all.rename(
        columns={"MTT": "MT"}
    )

    _prediction_cols = [
        _column
        for _column in _pred_matrix_all.columns
        if _column != "Metadata_compound" and _column in _valid_labels
    ]

    _active_lookup = df_ttest_results_filtered.set_index("Metadata_Label")[
        "Metadata_Count_1"
    ]
    _prediction_cols = [
        _column
        for _column in _prediction_cols
        if _active_lookup.get(_column, 0) > n_active_filter
    ]

    pred_matrix_filtered = _pred_matrix_all[
        ["Metadata_compound"] + _prediction_cols
    ].copy()
    pred_matrix_probability_filtered = _pred_matrix_probability_all[
        ["Metadata_compound"] + _prediction_cols
    ].copy()

    print(f"Filtered prediction assays: {len(_prediction_cols)}")
    print(f"Filtered class prediction matrix: {pred_matrix_filtered.shape}")
    print(
        "Filtered probability prediction matrix:",
        pred_matrix_probability_filtered.shape,
    )
    if _prediction_cols:
        print("Assays retained:", ", ".join(_prediction_cols))
    return pred_matrix_filtered, pred_matrix_probability_filtered


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prepare functions for prediction Manuscript Heatmap Data""")
    return


@app.cell
def _(np, pd):
    def assign_assay_groups(columns: pd.Index) -> pd.DataFrame:
        """Classify prediction columns into the assay groups used in the figure."""
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
def _(leaves_list, linkage, np, pd):
    def cluster_row_indices(
        data_matrix: np.ndarray,
        indices: list[int],
    ) -> list[int]:
        """Return row indices clustered by Jaccard distance."""
        if len(indices) < 2:
            return indices

        _sub_matrix = data_matrix[indices, :]
        _valid_rows = [
            _i
            for _i, _idx in enumerate(indices)
            if not np.all(np.isnan(_sub_matrix[_i, :]))
        ]

        if len(_valid_rows) < 2:
            return indices

        _sub_valid = np.nan_to_num(_sub_matrix[_valid_rows, :], nan=0.0)
        _linkage = linkage(_sub_valid, method="average", metric="jaccard")
        _order = leaves_list(_linkage)

        return [indices[_valid_rows[_i]] for _i in _order]


    def order_bsc_rows(
        pred_matrix_filtered: pd.DataFrame,
        df_bsc_annotations: pd.DataFrame,
        dic_botanicals_scode_name: dict,
        list_botanicals,
    ) -> pd.DataFrame:
        """Order botanical rows and cluster constituents within each botanical."""
        _matrix_by_compound = pred_matrix_filtered.set_index("Metadata_compound")
        _all_scodes = list(_matrix_by_compound.index)
        _scode_to_rowidx = {
            _scode: _idx for _idx, _scode in enumerate(_all_scodes)
        }
        _matrix_values = _matrix_by_compound.values.astype(float)

        _rows_ordered = []

        for _scode in list_botanicals:
            _plant_name = dic_botanicals_scode_name.get(_scode, _scode)
            _constituents = (
                df_bsc_annotations[
                    df_bsc_annotations["Plant scode"] == _scode
                ][["Metadata_compound", "Constituent"]]
                .drop_duplicates()
            )

            _const_rows = []
            for _, _row in _constituents.iterrows():
                if _row["Metadata_compound"] in _matrix_by_compound.index:
                    _const_rows.append(
                        (
                            _row["Metadata_compound"],
                            f"  {_row['Constituent']}",
                        )
                    )

            _has_botanical = _scode in _matrix_by_compound.index
            if not _has_botanical and not _const_rows:
                continue

            if _has_botanical:
                _rows_ordered.append(
                    {
                        "Metadata_compound": _scode,
                        "display_label": _plant_name,
                        "is_botanical": True,
                    }
                )

            _const_matrix_indices = [
                _scode_to_rowidx[_scode_const]
                for _scode_const, _label in _const_rows
                if _scode_const in _scode_to_rowidx
            ]
            _clustered_indices = cluster_row_indices(
                _matrix_values,
                _const_matrix_indices,
            )
            _idx_to_row = {
                _scode_to_rowidx[_scode_const]: (_scode_const, _label)
                for _scode_const, _label in _const_rows
                if _scode_const in _scode_to_rowidx
            }

            for _rowidx in _clustered_indices:
                _scode_const, _label = _idx_to_row[_rowidx]
                _rows_ordered.append(
                    {
                        "Metadata_compound": _scode_const,
                        "display_label": _label,
                        "is_botanical": False,
                    }
                )

        return pd.DataFrame(_rows_ordered)
    return (order_bsc_rows,)


@app.cell
def _(assign_assay_groups, np, pd):
    def build_column_annotations(
        heatmap_columns: pd.Index,
        df_ttest_results_filtered: pd.DataFrame,
        df_toxcast_cellbased_assay_info_significant: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add group, color, category, and AUROC metadata for figure columns."""
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

        _group_table = assign_assay_groups(heatmap_columns)

        _category_color_col = next(
            (
                _column
                for _column in [
                    "chagpt_color",
                    "chatgpt_color",
                    "claudeai_color",
                ]
                if (
                    _column
                    in df_toxcast_cellbased_assay_info_significant.columns
                )
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
                if (
                    _column
                    in df_toxcast_cellbased_assay_info_significant.columns
                )
            ),
            None,
        )
        if (
            "assay_component_endpoint_name"
            in df_toxcast_cellbased_assay_info_significant.columns
            and _category_color_col is not None
            and _category_label_col is not None
        ):
            _cat_lookup = (
                df_toxcast_cellbased_assay_info_significant
                .drop_duplicates("assay_component_endpoint_name")
                .set_index("assay_component_endpoint_name")[
                    [_category_color_col, _category_label_col]
                ]
                .to_dict(orient="index")
            )
        else:
            _cat_lookup = {}
        _auroc_lookup = df_ttest_results_filtered.set_index(
            "Metadata_Label"
        )["AUROC_mean_Actual"]

        _group_table["group_color"] = _group_table["group"].map(_group_colors)
        _group_table["group_display"] = _group_table["group"].map(
            _group_display
        )
        _group_table["category_color"] = [
            _cat_lookup.get(_assay, {}).get(_category_color_col, "#dddddd")
            for _assay in _group_table["assay"]
        ]
        _group_table["category_label"] = [
            _cat_lookup.get(_assay, {}).get(_category_label_col, None)
            for _assay in _group_table["assay"]
        ]
        _group_table["AUROC_mean_Actual"] = [
            _auroc_lookup.get(_assay, np.nan)
            for _assay in _group_table["assay"]
        ]

        return _group_table
    return (build_column_annotations,)


@app.cell
def _(
    assign_assay_groups,
    build_column_annotations,
    df_bsc_annotations,
    df_toxcast_cellbased_assay_info_significant,
    df_ttest_results_filtered,
    dic_botanicals_scode_name,
    list_botanicals,
    order_bsc_rows,
    pred_matrix_filtered,
):
    # Build row order first, then reorder columns by group and cell-based category.
    df_heatmap_rows = order_bsc_rows(
        pred_matrix_filtered,
        df_bsc_annotations,
        dic_botanicals_scode_name,
        list_botanicals,
    )

    _heatmap_data = pred_matrix_filtered.set_index("Metadata_compound").reindex(
        df_heatmap_rows["Metadata_compound"]
    )
    _heatmap_data.index = df_heatmap_rows["display_label"]

    _group_order = ["Cell-free", "Cell-based", "Cytotoxic", "Hit calls"]
    _group_table = assign_assay_groups(_heatmap_data.columns)
    _category_label_col = next(
        (
            _column
            for _column in [
                "chagpt_label",
                "chatgpt_label",
                "claudeai_label",
            ]
            if _column in df_toxcast_cellbased_assay_info_significant.columns
        ),
        None,
    )
    if _category_label_col is None:
        _category_order = {}
    else:
        _category_order = (
            df_toxcast_cellbased_assay_info_significant
            .drop_duplicates("assay_component_endpoint_name")
            .set_index("assay_component_endpoint_name")[_category_label_col]
        )

    _ordered_columns = []
    for _group in _group_order:
        _assays = _group_table.loc[
            _group_table["group"] == _group,
            "assay",
        ].tolist()

        if _group == "Cell-based":
            _assays = sorted(
                _assays,
                key=lambda _assay: _category_order.get(_assay, "zzz"),
            )

        _ordered_columns.extend(_assays)

    # Include any unexpected columns at the end instead of silently dropping them.
    _ordered_columns.extend(
        [
            _assay
            for _assay in _heatmap_data.columns
            if _assay not in _ordered_columns
        ]
    )

    df_heatmap_data = _heatmap_data.loc[:, _ordered_columns]
    df_heatmap_columns = build_column_annotations(
        df_heatmap_data.columns,
        df_ttest_results_filtered,
        df_toxcast_cellbased_assay_info_significant,
    )

    print(f"Heatmap rows: {df_heatmap_data.shape[0]}")
    print(f"Heatmap assays: {df_heatmap_data.shape[1]}")
    print(
        "Heatmap assay groups:",
        df_heatmap_columns["group"].value_counts().to_dict(),
    )
    return df_heatmap_columns, df_heatmap_data, df_heatmap_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Manuscript Heatmap""")
    return


@app.cell
def _(np):
    def build_row_positions(df_heatmap_rows):
        """Add small gaps before each botanical row to match the original figure."""
        _y_positions = []
        _gap = 0.4
        _y = 0.0

        for _idx, _is_botanical in enumerate(df_heatmap_rows["is_botanical"]):
            if _is_botanical and _idx > 0:
                _y += _gap
            _y_positions.append(_y)
            _y += 1.0

        return np.array(_y_positions), _y_positions[-1] + 1.0
    return (build_row_positions,)


@app.cell
def _():
    from scipy.spatial.distance import pdist
    return (pdist,)


@app.cell
def _(
    ListedColormap,
    binary_cmap,
    build_row_positions,
    df_heatmap_columns,
    df_heatmap_data,
    df_heatmap_rows,
    downloads_dir,
    figure_output_dir,
    leaves_list,
    linkage,
    mpatches,
    np,
    pdist,
    plt,
    re,
):
    _figure_path = figure_output_dir / "Figure_4_C.png"
    _download_path = (
        downloads_dir
        / "prediction_heatmap_botanical_constituents_batch_groups_chatgpt.png"
    )

    figure_output_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    _rename_map = {
        "cell_type_HepG2": "ToxCast_cytotox_HepG2",
    }

    def _clean_assay_name(name):
        # Apply explicit mapping if matched
        if name in _rename_map:
            name = _rename_map[name]
        # Strip prefixes 'BSK_LPS_' and 'ATG_'
        name = re.sub(r"^(BSK_LPS_|BSK_|ATG_|TOX21_|LTEA_)", "", name)
        # Strip suffix '_R1881'
        name = re.sub(r"_R1881$", "", name)
        name = re.sub(r"_ratio$", "", name)
        name = re.sub(r"_HepaRG", "HepaRG", name)
        name = re.sub(r"cell_type__HepG2", "ToxCast_cytotox_HepG2", name)
        return name

    # Apply cleaning to data columns and heatmap metadata (local copy — don't
    # mutate the shared df_heatmap_data defined in another cell)
    _df_heatmap_data = df_heatmap_data.copy()
    _df_heatmap_data.columns = [_clean_assay_name(col) for col in _df_heatmap_data.columns]
    _df_heatmap_columns = df_heatmap_columns.copy()

    # --- Cluster assay columns on their binary activity pattern across compounds ---
    # Jaccard distance treats shared "active" calls as similarity and ignores
    # shared "inactive" calls (which dominate sparse activity matrices and would
    # otherwise swamp a Euclidean/Hamming distance). Average linkage doesn't
    # assume Euclidean geometry, unlike Ward.
    if False:
        _col_dist = pdist(_df_heatmap_data.values.T, metric="jaccard")
        _col_dist = np.nan_to_num(_col_dist, nan=1.0)  # all-zero columns -> treat as maximally dissimilar
        _col_linkage = linkage(_col_dist, method="average")
        _col_order = leaves_list(_col_linkage)
    
        _df_heatmap_data = _df_heatmap_data.iloc[:, _col_order]
        _df_heatmap_columns = _df_heatmap_columns.iloc[_col_order].reset_index(drop=True)
    # --------------------------------------------------------------------------

    _n_rows = len(_df_heatmap_data)
    _n_cols = len(_df_heatmap_data.columns)
    _y_positions, _total_height = build_row_positions(df_heatmap_rows)

    _fig = plt.figure(
        figsize=(
            max(14, _n_cols * 0.2),
            max(10, _total_height * 0.14) + 2,
        )
    )
    _gs = _fig.add_gridspec(
        2,
        1,
        height_ratios=[0.01, 1],
        hspace=0.005,
    )
    _ax_cat_band = _fig.add_subplot(_gs[0])
    _ax = _fig.add_subplot(_gs[1])

    # Cell-based category color band.
    _category_colors = _df_heatmap_columns["category_color"].tolist()
    _ax_cat_band.imshow(
        [list(range(_n_cols))],
        aspect="auto",
        cmap=ListedColormap(_category_colors),
        interpolation="none",
    )
    _ax_cat_band.set_xlim(-0.5, _n_cols - 0.5)
    _ax_cat_band.set_yticks([])
    _ax_cat_band.set_xticks([])
    _ax_cat_band.spines[["left", "right", "bottom", "top"]].set_visible(False)

    # NOTE: with clustered columns, "category" no longer forms contiguous
    # blocks, so this will draw a divider almost everywhere category changes
    # between adjacent columns. Consider removing this loop if it gets too busy —
    # the color band alone still communicates category membership.
    _prev_category = _df_heatmap_columns["category_label"].iloc[0]
    for _column_index, _category in enumerate(
        _df_heatmap_columns["category_label"]
    ):
        if _category != _prev_category:
            _ax_cat_band.axvline(
                _column_index - 0.5,
                color="black",
                linewidth=1.0,
                zorder=5,
            )
            _prev_category = _category

    # Heatmap rows.
    for _row_index, (_y, _is_botanical) in enumerate(
        zip(_y_positions, df_heatmap_rows["is_botanical"])
    ):
        _row_values = _df_heatmap_data.values[_row_index : _row_index + 1, :]
        _ax.pcolormesh(
            np.arange(_n_cols + 1),
            np.array([_y, _y + 1.0]),
            _row_values,
            cmap=binary_cmap,
            edgecolors="white",
            linewidth=0.1,
            vmin=0,
            vmax=1,
        )
        if _is_botanical:
            _ax.axhspan(_y, _y + 1.0, color="lightgrey", alpha=0.35, zorder=0)

    _ax.set_yticks(_y_positions + 0.25)
    _ax.set_yticklabels(df_heatmap_rows["display_label"], fontsize=9)
    for _tick, _is_botanical in zip(
        _ax.get_yticklabels(),
        df_heatmap_rows["is_botanical"],
    ):
        if _is_botanical:
            _tick.set_fontweight("bold")
            _tick.set_fontsize(11)
            _tick.set_color("black")
        else:
            _tick.set_fontsize(9)
            _tick.set_color("#444444")

    _ax.set_ylim(0, _total_height)
    _ax.invert_yaxis()

    _ax.set_xticks(np.arange(_n_cols) + 0.5)
    _ax.set_xticklabels(_df_heatmap_data.columns, rotation=90, fontsize=8)
    _ax.xaxis.set_label_position("top")
    _ax.xaxis.tick_top()
    _ax.tick_params(axis="x", length=0, pad=15)

    # NOTE: same caveat as the category band above — "group" blocks are no
    # longer contiguous after clustering, so this will draw many more lines
    # than the original 3-4 group boundaries.
    _prev_group = _df_heatmap_columns["group"].iloc[0]
    for _column_index, _group in enumerate(_df_heatmap_columns["group"]):
        if _group != _prev_group:
            _ax.axvline(
                _column_index,
                color="black",
                linewidth=1.5,
                zorder=5,
            )
            _prev_group = _group

    # Legends use the same visual labels as the original figure.
    _legend_group_order = ["Cell-free", "Cell-based", "Cytotoxic", "Hit calls"]
    _legend_patches = []
    for _group in _legend_group_order:
        _group_rows = _df_heatmap_columns[_df_heatmap_columns["group"] == _group]
        if _group_rows.empty:
            continue
        _legend_patches.append(
            mpatches.Patch(
                facecolor=_group_rows["group_color"].iloc[0],
                label=_group_rows["group_display"].iloc[0],
            )
        )

    _category_legend = (
        _df_heatmap_columns[["category_label", "category_color"]]
        .dropna()
        .drop_duplicates()
    )
    _category_patches = [
        mpatches.Patch(
            facecolor=_row["category_color"],
            label=_row["category_label"],
        )
        for _, _row in _category_legend.iterrows()
    ]

    _fig.savefig(_figure_path, bbox_inches="tight", dpi=300)
    _fig.savefig(_download_path, bbox_inches="tight", dpi=300)

    print(f"Saved manuscript heatmap: {_figure_path}")
    print(f"Saved Downloads heatmap: {_download_path}")

    _fig
    return


@app.cell
def _():
    import re
    return (re,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prediction Summary Tables""")
    return


@app.cell
def _(pd):
    def build_prediction_summary_tables(
        pred_matrix_filtered: pd.DataFrame,
        list_botanicals,
        dic_botanicals_scode_name: dict,
        df_heatmap_columns: pd.DataFrame,
    ):
        """Summarize active predictions for botanical rows only."""
        def _add_active_summary(active_df: pd.DataFrame) -> pd.DataFrame:
            return active_df.assign(
                active_predictions=lambda _df: (
                    _df["n_active"].astype(int).astype(str)
                    + "/"
                    + _df["n_tested"].astype(int).astype(str)
                ),
                active_prediction_summary=lambda _df: _df.apply(
                    lambda _row: (
                        f"{int(_row['n_active'])}/"
                        f"{int(_row['n_tested'])} "
                        f"({_row['pct_active']:.1f}%)"
                    ),
                    axis=1,
                ),
            )

        _botanical_ids = pd.Index(
            pd.Series(list_botanicals)
            .dropna()
            .astype(str)
            .drop_duplicates(),
            name="Metadata_compound",
        )
        _botanical_id_set = set(_botanical_ids)

        _pred_unique = (
            pred_matrix_filtered
            .drop_duplicates("Metadata_compound")
            .copy()
        )
        _pred_unique["Metadata_compound"] = (
            _pred_unique["Metadata_compound"].astype(str)
        )
        _non_botanical_count = (
            ~_pred_unique["Metadata_compound"].isin(_botanical_ids)
        ).sum()
        _missing_botanicals = sorted(
            _botanical_id_set - set(_pred_unique["Metadata_compound"])
        )
        _pred_botanical = (
            _pred_unique[
                _pred_unique["Metadata_compound"].isin(_botanical_ids)
            ]
            .set_index("Metadata_compound")
            .reindex(_botanical_ids)
            .rename_axis("Metadata_compound")
            .reset_index()
        )
        _pred_long = (
            _pred_botanical
            .melt(
                id_vars="Metadata_compound",
                var_name="assay",
                value_name="active",
            )
            .rename(columns={"Metadata_compound": "botanical_scode"})
        )
        _pred_long["botanical"] = (
            _pred_long["botanical_scode"]
            .astype(str)
            .map(dic_botanicals_scode_name)
        )
        df_pred_long_botanical = _pred_long.dropna(
            subset=["botanical"]
        ).copy()

        df_pct_active_by_botanical = (
            df_pred_long_botanical
            .groupby("botanical")["active"]
            .agg(
                n_active=lambda _x: _x.eq(1).sum(),
                n_tested=lambda _x: _x.notna().sum(),
            )
            .assign(
                pct_active=lambda _x: (
                    _x["n_active"] / _x["n_tested"] * 100
                ).round(1)
            )
            .sort_values("pct_active", ascending=False)
            .reset_index()
            .pipe(_add_active_summary)
        )

        _group_series = df_heatmap_columns.set_index("assay")["group"]
        df_pred_long_botanical["group"] = df_pred_long_botanical["assay"].map(
            _group_series
        )

        df_active_by_botanical_group_long = (
            df_pred_long_botanical.dropna(subset=["group"])
            .groupby(["botanical", "group"])["active"]
            .agg(
                n_active=lambda _x: _x.eq(1).sum(),
                n_tested=lambda _x: _x.notna().sum(),
            )
            .assign(
                pct_active=lambda _x: (
                    _x["n_active"] / _x["n_tested"] * 100
                ).round(1)
            )
            .reset_index()
            .pipe(_add_active_summary)
        )
        df_pct_active_by_botanical_group = (
            df_active_by_botanical_group_long
            .pivot(
                index="botanical",
                columns="group",
                values="active_prediction_summary",
            )
            .rename_axis(columns=None)
            .reset_index()
        )
        if "Cell-based" in df_pct_active_by_botanical_group.columns:
            _group_sort_order = (
                df_active_by_botanical_group_long[
                    df_active_by_botanical_group_long["group"] == "Cell-based"
                ]
                .sort_values("pct_active", ascending=False)["botanical"]
                .tolist()
            )
            _remaining_botanicals = [
                _botanical
                for _botanical in df_pct_active_by_botanical_group["botanical"]
                if _botanical not in _group_sort_order
            ]
            df_pct_active_by_botanical_group = (
                df_pct_active_by_botanical_group
                .set_index("botanical")
                .reindex(_group_sort_order + _remaining_botanicals)
                .reset_index()
            )

        _category_series = df_heatmap_columns.set_index("assay")[
            "category_label"
        ]
        df_pred_long_botanical["category_label"] = (
            df_pred_long_botanical["assay"].map(_category_series)
        )

        df_active_by_botanical_category_long = (
            df_pred_long_botanical.dropna(subset=["category_label"])
            .groupby(["botanical", "category_label"])["active"]
            .agg(
                n_active=lambda _x: _x.eq(1).sum(),
                n_tested=lambda _x: _x.notna().sum(),
            )
            .assign(
                pct_active=lambda _x: (
                    _x["n_active"] / _x["n_tested"] * 100
                ).round(1)
            )
            .reset_index()
            .pipe(_add_active_summary)
        )
        df_pct_active_by_botanical_category = (
            df_active_by_botanical_category_long
            .pivot(
                index="botanical",
                columns="category_label",
                values="active_prediction_summary",
            )
            .rename_axis(columns=None)
            .reset_index()
        )

        _category_sort_cols = [
            _column
            for _column in df_pct_active_by_botanical_category.columns
            if _column != "botanical"
        ]
        if _category_sort_cols:
            _category_sort_order = (
                df_active_by_botanical_category_long[
                    df_active_by_botanical_category_long["category_label"]
                    == _category_sort_cols[0]
                ]
                .sort_values("pct_active", ascending=False)["botanical"]
                .tolist()
            )
            _remaining_botanicals = [
                _botanical
                for _botanical in df_pct_active_by_botanical_category[
                    "botanical"
                ]
                if _botanical not in _category_sort_order
            ]
            df_pct_active_by_botanical_category = (
                df_pct_active_by_botanical_category
                .set_index("botanical")
                .reindex(_category_sort_order + _remaining_botanicals)
                .reset_index()
            )

        return (
            df_pct_active_by_botanical,
            df_pct_active_by_botanical_group,
            df_pct_active_by_botanical_category,
            df_pred_long_botanical,
            int(_non_botanical_count),
            _missing_botanicals,
        )
    return (build_prediction_summary_tables,)


@app.cell
def _(
    build_prediction_summary_tables,
    df_heatmap_columns,
    dic_botanicals_scode_name,
    list_botanicals,
    pred_matrix_filtered,
):
    (
        df_pct_active_by_botanical,
        df_pct_active_by_botanical_group,
        df_pct_active_by_botanical_category,
        df_pred_long_botanical,
        non_botanical_rows_excluded,
        missing_botanicals_from_stats,
    ) = build_prediction_summary_tables(
        pred_matrix_filtered,
        list_botanicals,
        dic_botanicals_scode_name,
        df_heatmap_columns,
    )

    print(
        "Botanicals in stats:",
        df_pred_long_botanical["botanical_scode"].nunique(),
    )
    print(
        "Non-botanical rows excluded from stats:",
        non_botanical_rows_excluded,
    )
    print(
        "Missing botanicals from stats:",
        ", ".join(missing_botanicals_from_stats)
        if missing_botanicals_from_stats
        else "none",
    )

    print("Active predictions by botanical:")
    print(df_pct_active_by_botanical.to_string(index=False))

    print("\nActive predictions by botanical and assay group:")
    print(df_pct_active_by_botanical_group.to_string(index=False))

    print("\nActive predictions by botanical and cell-based category:")
    print(df_pct_active_by_botanical_category.to_string(index=False))
    return df_pct_active_by_botanical_category, df_pred_long_botanical


@app.cell
def _(df_pct_active_by_botanical_category):
    df_pct_active_by_botanical_category
    return


@app.cell
def _(df_pred_long_botanical):
    df_pred_long_botanical
    return


@app.cell
def _(df_heatmap_data, df_pred_long_botanical):
    df_heatmap_data[df_pred_long_botanical[df_pred_long_botanical["category_label"] == "Endocrine Disruption — Sex Hormones"]["assay"].unique()].sum(axis=1)
    return


@app.cell
def _(df_heatmap_data, df_pred_long_botanical):
    df_heatmap_data[df_pred_long_botanical[df_pred_long_botanical["category_label"] == "Genotoxicity / DNA Damage"]["assay"].unique()].sum(axis=1)
    return


@app.cell
def _(df_heatmap_data, df_pred_long_botanical):
    df_heatmap_data[df_pred_long_botanical[df_pred_long_botanical["category_label"] == "Endocrine Disruption — Sex Hormones"]["assay"].unique()].sum(axis=1)
    return


@app.cell
def _(df_heatmap_data, df_pred_long_botanical):
    df_heatmap_data[df_pred_long_botanical[df_pred_long_botanical["assay"] == "cell_type__HepG2"]["assay"].unique()].sum(axis=1)
    return


@app.cell
def _(df_heatmap_data, df_pred_long_botanical):
    df_heatmap_data[df_heatmap_data.index == '  Aristolochic acid I'][df_pred_long_botanical[df_pred_long_botanical["category_label"] == "Genotoxicity / DNA Damage"]["assay"].unique()]
    return


@app.cell
def _(df_pred_long_botanical):
    df_active_by_assay = (
        df_pred_long_botanical
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
        .assign(
            active_prediction_summary=lambda _df: _df.apply(
                lambda _row: (
                    f"{int(_row['active_count'])}/"
                    f"{int(_row['total_count'])} "
                    f"({_row['percent_active']:.1f}%)"
                ),
                axis=1,
            )
        )
        .sort_values("percent_active", ascending=False)
        .reset_index()
    )

    print(f"Assay activity summary rows: {len(df_active_by_assay)}")
    df_active_by_assay
    return


@app.cell
def _(df_pred_long_botanical):
    df_pred_long_botanical[
        df_pred_long_botanical["category_label"].str.startswith("Hepa", na=False)
    ][["assay", "category_label"]].drop_duplicates()
    return


@app.cell
def _(df_pred_long_botanical):
    df_pred_long_botanical[
        df_pred_long_botanical["category_label"].str.startswith("Mi", na=False)
    ][["assay", "category_label"]].drop_duplicates()
    return


@app.cell
def _(df_pred_long_botanical):
    df_pred_long_botanical[["assay", "category_label"]].drop_duplicates()["category_label"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Correlation Significant Assays""")
    return


@app.cell
def _(pred_matrix_probability_filtered):
    pred_matrix_probability_filtered
    return


@app.cell
def _(
    build_column_annotations,
    df_toxcast_cellbased_assay_info_significant,
    df_ttest_results_filtered,
    pred_matrix_probability_filtered,
):
    df_probability_for_correlation = (
        pred_matrix_probability_filtered
        .set_index("Metadata_compound")
        .astype(float)
    )

    _non_constant_cols = [
        _column
        for _column in df_probability_for_correlation.columns
        if df_probability_for_correlation[_column].nunique(dropna=True) > 1
    ]
    df_probability_for_correlation = (
        df_probability_for_correlation[_non_constant_cols]
        .dropna(axis=1, how="any")
    )

    correlation_matrix = df_probability_for_correlation.corr(method="pearson")
    df_correlation_assay_annotations = build_column_annotations(
        correlation_matrix.columns,
        df_ttest_results_filtered,
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
    return correlation_matrix, df_correlation_assay_annotations


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
    return


@app.cell
def _(mo):
    mo.md(r"""## Correlation Heatmap""")
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
        / "Figure_S8_A.png"
    )
    _download_path = (
        downloads_dir
        / "correlation_heatmap_bsc_prediction_significant_assays.png"
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
            "Predicted Activity Probability Correlation Across Significant Assays",
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
            "Predicted Activity Probability Correlation Across Significant Assays"
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
def _(df_ttest_results_filtered, pd, pred_matrix_probability_filtered):
    _training_counts = (
        df_ttest_results_filtered[
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
        for _column in pred_matrix_probability_filtered.columns
        if _column != "Metadata_compound"
    ]
    _prediction_values = pred_matrix_probability_filtered[_prediction_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    _predicted_active_rate = pd.DataFrame(
        {
            "Metadata_Label": _prediction_cols,
            "predicted_expected_active_count": (
                _prediction_values.sum(axis=0).to_numpy()
            ),
            "predicted_total_count": (
                _prediction_values.notna().sum(axis=0).to_numpy()
            ),
            "predicted_pct_active": (
                _prediction_values.mean(axis=0).mul(100).to_numpy()
            ),
        }
    )

    df_bsc_active_rate_comparison = (
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
        "BSC probability assays compared to training active rate:",
        len(df_bsc_active_rate_comparison),
    )
    df_bsc_active_rate_comparison
    return (df_bsc_active_rate_comparison,)


@app.cell
def _(
    df_bsc_active_rate_comparison,
    downloads_dir,
    figure_output_dir,
    np,
    plt,
):
    _figure_path = (
        figure_output_dir / "bsc_prediction_active_rate_vs_training.png"
    )
    _download_path = (
        downloads_dir / "bsc_prediction_active_rate_vs_training.png"
    )

    figure_output_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    _plot_df = df_bsc_active_rate_comparison.dropna(
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
            color="#4C72B0",
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
        _ax.set_ylabel("Predicted active probability (%)")
        _ax.set_title("BSC Predicted Activity Probability vs Training Set")
        _ax.grid(True, color="#e6e6e6", linewidth=0.7)
        _ax.legend(loc="lower right", frameon=False)

    _fig.tight_layout()
    _fig.savefig(_figure_path, bbox_inches="tight", dpi=300)
    _fig.savefig(_download_path, bbox_inches="tight", dpi=300)

    print(f"Saved BSC active-rate plot: {_figure_path}")
    print(f"Saved Downloads active-rate plot: {_download_path}")

    _fig
    return


if __name__ == "__main__":
    app.run()
