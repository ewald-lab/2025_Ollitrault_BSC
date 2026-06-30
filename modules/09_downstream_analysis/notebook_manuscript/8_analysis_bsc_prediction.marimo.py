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
    model_dir = classification_output_dir / "models_batch"
    figure_output_dir = project_root / "modules/09_downstream_analysis/figures"
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
    toxcast_category_path = (
        classification_output_dir
        / "df_toxcast_cellbased_assay_info_significant_claudeai_categories.csv"
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
    print(f"Botanicals in annotation table: {len(list_botanicals)}")
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
    def predict_with_models(feature_df: pd.DataFrame, models: dict) -> pd.DataFrame:
        """Predict every row in feature_df with each trained model."""
        _pred_matrix = pd.DataFrame(
            {"Metadata_compound": feature_df["Metadata_compound"].values},
            index=feature_df.index,
        )

        for _model_name, _model in tqdm(
            models.items(),
            total=len(models),
            desc="Predicting",
        ):
            _trained_features = _model.get_booster().feature_names
            _X_pred = feature_df.reindex(columns=_trained_features)
            _pred_matrix[_model_name] = _model.predict(_X_pred)

        return _pred_matrix.reset_index(drop=True)
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

    pred_matrix = predict_with_models(df_level6_bsc_botanicals, loaded_models)

    print(f"Botanical rows predicted: {len(pred_matrix)}")
    print(f"Prediction assays: {pred_matrix.shape[1] - 1}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Predict BSC Botanicals And Constituents""")
    return


@app.cell
def _(df_level6_bsc_all, loaded_models, predict_with_models):
    # Full matrix includes botanical mixtures and their annotated constituents.
    pred_matrix_all = predict_with_models(df_level6_bsc_all.copy(), loaded_models)

    print(f"All BSC rows predicted: {len(pred_matrix_all)}")
    print(f"All prediction assays: {pred_matrix_all.shape[1] - 1}")
    return (pred_matrix_all,)


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
def _(df_ttest_results_filtered, n_active_filter, pred_matrix_all):
    # Keep only assays retained by performance filtering and active enough.
    _valid_labels = set(df_ttest_results_filtered["Metadata_Label"].values)
    _pred_matrix_all = pred_matrix_all.rename(columns={"MTT": "MT"})

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

    print(f"Filtered prediction assays: {len(_prediction_cols)}")
    print(f"Filtered prediction matrix: {pred_matrix_filtered.shape}")
    if _prediction_cols:
        print("Assays retained:", ", ".join(_prediction_cols))
    return (pred_matrix_filtered,)


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

        _cat_lookup = (
            df_toxcast_cellbased_assay_info_significant
            .drop_duplicates("assay_component_endpoint_name")
            .set_index("assay_component_endpoint_name")[
                ["claudeai_color", "claudeai_label"]
            ]
            .to_dict(orient="index")
        )
        _auroc_lookup = df_ttest_results_filtered.set_index(
            "Metadata_Label"
        )["AUROC_mean_Actual"]

        _group_table["group_color"] = _group_table["group"].map(_group_colors)
        _group_table["group_display"] = _group_table["group"].map(
            _group_display
        )
        _group_table["category_color"] = [
            _cat_lookup.get(_assay, {}).get("claudeai_color", "#dddddd")
            for _assay in _group_table["assay"]
        ]
        _group_table["category_label"] = [
            _cat_lookup.get(_assay, {}).get("claudeai_label", None)
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

    _category_order = (
        df_toxcast_cellbased_assay_info_significant
        .drop_duplicates("assay_component_endpoint_name")
        .set_index("assay_component_endpoint_name")["claudeai_label"]
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
def _(
    ListedColormap,
    binary_cmap,
    build_row_positions,
    df_heatmap_columns,
    df_heatmap_data,
    df_heatmap_rows,
    downloads_dir,
    figure_output_dir,
    mpatches,
    np,
    plt,
):
    _figure_path = figure_output_dir / "Figure_8_bsc_prediction_heatmap.png"
    _download_path = (
        downloads_dir
        / "prediction_heatmap_botanical_constituents_batch_groups_claudeai.png"
    )

    figure_output_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    _n_rows = len(df_heatmap_data)
    _n_cols = len(df_heatmap_data.columns)
    _y_positions, _total_height = build_row_positions(df_heatmap_rows)

    _fig = plt.figure(
        figsize=(
            max(14, _n_cols * 0.1),
            max(10, _total_height * 0.16) + 2,
        )
    )
    _gs = _fig.add_gridspec(
        3,
        1,
        height_ratios=[0.01, 0.01, 1],
        hspace=0.005,
    )
    _ax_cat_band = _fig.add_subplot(_gs[0])
    _ax_band = _fig.add_subplot(_gs[1])
    _ax = _fig.add_subplot(_gs[2])

    # Assay group color band.
    _group_colors = df_heatmap_columns["group_color"].tolist()
    _ax_band.imshow(
        [list(range(_n_cols))],
        aspect="auto",
        cmap=ListedColormap(_group_colors),
        interpolation="none",
    )
    _ax_band.set_xlim(-0.5, _n_cols - 0.5)
    _ax_band.set_yticks([])
    _ax_band.set_xticks([])
    _ax_band.spines[["left", "right", "bottom", "top"]].set_visible(False)

    _prev_group = df_heatmap_columns["group"].iloc[0]
    for _column_index, _group in enumerate(df_heatmap_columns["group"]):
        if _group != _prev_group:
            _ax_band.axvline(
                _column_index - 0.5,
                color="black",
                linewidth=1.5,
                zorder=5,
            )
            _prev_group = _group

    # Cell-based category color band.
    _category_colors = df_heatmap_columns["category_color"].tolist()
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

    _prev_category = df_heatmap_columns["category_label"].iloc[0]
    for _column_index, _category in enumerate(
        df_heatmap_columns["category_label"]
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
        _row_values = df_heatmap_data.values[_row_index : _row_index + 1, :]
        _ax.pcolormesh(
            np.arange(_n_cols + 1),
            np.array([_y, _y + 1.0]),
            _row_values,
            cmap=binary_cmap,
            edgecolors="white",
            linewidth=0.3,
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
    _ax.set_xticklabels(df_heatmap_data.columns, rotation=90, fontsize=8)
    _ax.xaxis.set_label_position("top")
    _ax.xaxis.tick_top()
    _ax.tick_params(axis="x", length=0, pad=30)

    _prev_group = df_heatmap_columns["group"].iloc[0]
    for _column_index, _group in enumerate(df_heatmap_columns["group"]):
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
        _group_rows = df_heatmap_columns[df_heatmap_columns["group"] == _group]
        if _group_rows.empty:
            continue
        _legend_patches.append(
            mpatches.Patch(
                facecolor=_group_rows["group_color"].iloc[0],
                label=_group_rows["group_display"].iloc[0],
            )
        )

    _ax.legend(
        handles=_legend_patches,
        loc="upper left",
        bbox_to_anchor=(-0.005, 0),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=9,
        borderpad=0.6,
        title="Assays",
        title_fontsize=10,
    ).get_title().set_fontweight("bold")

    _category_legend = (
        df_heatmap_columns[["category_label", "category_color"]]
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
    if _category_patches:
        _ax_cat_band.legend(
            handles=_category_patches,
            loc="upper left",
            bbox_to_anchor=(0.3, -101.5),
            ncol=2,
            frameon=True,
            facecolor="white",
            edgecolor="lightgrey",
            framealpha=0.95,
            fontsize=8,
            borderpad=0.6,
            title="Cell-based categories",
            title_fontsize=9,
        ).get_title().set_fontweight("bold")

    _fig.savefig(_figure_path, bbox_inches="tight", dpi=300)
    _fig.savefig(_download_path, bbox_inches="tight", dpi=300)

    print(f"Saved manuscript heatmap: {_figure_path}")
    print(f"Saved Downloads heatmap: {_download_path}")

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prediction Summary Tables""")
    return


@app.cell
def _(pd):
    def build_prediction_summary_tables(
        pred_matrix_filtered: pd.DataFrame,
        df_bsc_annotations: pd.DataFrame,
        dic_botanicals_scode_name: dict,
        df_heatmap_columns: pd.DataFrame,
    ):
        """Summarize active predictions by botanical, assay group, and category."""
        _pred_long = (
            pred_matrix_filtered
            .set_index("Metadata_compound")
            .reset_index()
            .melt(
                id_vars="Metadata_compound",
                var_name="assay",
                value_name="active",
            )
        )

        _const_to_botanical = (
            df_bsc_annotations[["Metadata_compound", "Plant scode"]]
            .drop_duplicates()
            .set_index("Metadata_compound")["Plant scode"]
            .to_dict()
        )
        _pred_long["botanical_scode"] = _pred_long["Metadata_compound"].map(
            _const_to_botanical
        )
        _pred_long["botanical"] = _pred_long["botanical_scode"].map(
            dic_botanicals_scode_name
        )

        df_pred_long_all = _pred_long.copy()
        df_pred_long_botanical = _pred_long.dropna(subset=["botanical"]).copy()

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
        )

        _group_series = df_heatmap_columns.set_index("assay")["group"]
        df_pred_long_botanical["group"] = df_pred_long_botanical["assay"].map(
            _group_series
        )

        df_pct_active_by_botanical_group = (
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
            .pivot(index="botanical", columns="group", values="pct_active")
            .reset_index()
        )
        if "Cell-based" in df_pct_active_by_botanical_group.columns:
            df_pct_active_by_botanical_group = (
                df_pct_active_by_botanical_group.sort_values(
                    "Cell-based",
                    ascending=False,
                )
            )

        _category_series = df_heatmap_columns.set_index("assay")[
            "category_label"
        ]
        df_pred_long_botanical["claudeai_label"] = (
            df_pred_long_botanical["assay"].map(_category_series)
        )

        df_pct_active_by_botanical_category = (
            df_pred_long_botanical.dropna(subset=["claudeai_label"])
            .groupby(["botanical", "claudeai_label"])["active"]
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
            .pivot(
                index="botanical",
                columns="claudeai_label",
                values="pct_active",
            )
            .reset_index()
        )

        _category_sort_cols = [
            _column
            for _column in df_pct_active_by_botanical_category.columns
            if _column != "botanical"
        ]
        if _category_sort_cols:
            df_pct_active_by_botanical_category = (
                df_pct_active_by_botanical_category.sort_values(
                    _category_sort_cols[0],
                    ascending=False,
                )
            )

        return (
            df_pct_active_by_botanical,
            df_pct_active_by_botanical_group,
            df_pct_active_by_botanical_category,
            df_pred_long_all,
            df_pred_long_botanical,
        )
    return (build_prediction_summary_tables,)


@app.cell
def _(
    build_prediction_summary_tables,
    df_bsc_annotations,
    df_heatmap_columns,
    dic_botanicals_scode_name,
    pred_matrix_filtered,
):
    (
        df_pct_active_by_botanical,
        df_pct_active_by_botanical_group,
        df_pct_active_by_botanical_category,
        df_pred_long_all,
        df_pred_long_botanical,
    ) = build_prediction_summary_tables(
        pred_matrix_filtered,
        df_bsc_annotations,
        dic_botanicals_scode_name,
        df_heatmap_columns,
    )

    print("Active predictions by botanical:")
    print(df_pct_active_by_botanical.to_string(index=False))

    print("\nActive predictions by botanical and assay group:")
    print(df_pct_active_by_botanical_group.to_string(index=False))

    print("\nActive predictions by botanical and cell-based category:")
    print(df_pct_active_by_botanical_category.to_string(index=False))
    return (df_pred_long_all,)


@app.cell
def _(df_pred_long_all):
    df_active_by_assay = (
        df_pred_long_all
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
        .sort_values("percent_active", ascending=False)
        .reset_index()
    )

    print(f"Assay activity summary rows: {len(df_active_by_assay)}")
    df_active_by_assay
    return


if __name__ == "__main__":
    app.run()
