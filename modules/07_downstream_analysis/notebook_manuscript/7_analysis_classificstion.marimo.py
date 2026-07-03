import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    from scipy import stats
    import warnings
    import plotnine as pn
    from plotnine import (
        aes,
        element_rect,
        element_text,
        facet_wrap,
        geom_boxplot,
        geom_jitter,
        geom_point,
        geom_smooth,
        ggplot,
        labs,
        scale_x_discrete,
        theme,
        theme_bw,
        theme_classic,
        ylim,
    )
    from matplotlib.lines import Line2D
    import seaborn as sns
    from pathlib import Path
    from tqdm.auto import tqdm
    return (
        Line2D,
        Path,
        auc,
        mo,
        np,
        pd,
        precision_recall_curve,
        roc_auc_score,
        sns,
        stats,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Classification Performance Analysis

    This notebook summarizes Axiom classification outputs, computes
    fold-level performance metrics, identifies labels where morphology
    outperforms the random baseline, and generates manuscript-oriented
    performance plots.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Input predictions""")
    return


@app.cell
def _(Path):
    classification_output_dir = Path("modules/06_classification/outputs")
    downloads_dir = Path.home() / "Downloads"
    toxcast_cellbased_assay_info_path = Path(
        "/Users/ollitrault/Documents/projects/databases/proc_toxcast_annotations/"
        "toxcast_cellbased_assay_info.csv"
    )
    return classification_output_dir, downloads_dir


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
def _(
    df_axiom_cellbased_binary_prediction,
    df_axiom_cellfree_binary_prediction,
    df_axiom_cytotox_binary_prediction,
    df_axiom_hits_prediction,
):
    # Harmonize the morphology endpoint naming used in downstream displays.
    for _df in [
        df_axiom_hits_prediction,
        df_axiom_cellbased_binary_prediction,
        df_axiom_cellfree_binary_prediction,
        df_axiom_cytotox_binary_prediction,
    ]:
        _df["Metadata_Label"] = _df["Metadata_Label"].str.replace("MTT", "MT", regex=False)
    return


@app.cell
def _(mo):
    mo.md(r"""## Compute performance metrics""")
    return


@app.cell
def _(auc, np, pd, precision_recall_curve, roc_auc_score, tqdm):
    def precision_recall_gain_curve(y_true, y_scores):
        """Compute precision/recall gain values from sorted prediction scores."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_scores = np.asarray(y_scores, dtype=np.float64)
        positives = y_true.sum()
        if y_true.size == 0 or positives == 0:
            empty = np.full(y_true.size, np.nan)
            return empty, empty, empty, empty

        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]
        true_positives = np.cumsum(y_true_sorted)
        thresholds = np.arange(1, y_true_sorted.size + 1, dtype=np.float64)
        baseline = positives / y_true_sorted.size

        precisions = true_positives / thresholds
        recalls = true_positives / positives
        precision_gains = precisions - baseline
        recall_gains = recalls - baseline

        return precisions, recalls, precision_gains, recall_gains


    def prg_auc(y_true, y_scores):
        """Area under the precision-recall gain curve."""
        _, _, precision_gains, recall_gains = precision_recall_gain_curve(y_true, y_scores)
        if np.isnan(precision_gains).all():
            return np.nan
        return float(np.trapezoid(precision_gains, x=recall_gains))

    def compute_metrics(y_actual, y_prob):
        """Return AUROC, PRAUC, and AUPRG for one label/model/fold."""
        y_actual = np.asarray(y_actual)
        y_prob = np.asarray(y_prob, dtype=np.float64)

        try:
            auroc = roc_auc_score(y_actual, y_prob)
        except ValueError:
            auroc = np.nan

        try:
            precision, recall, _ = precision_recall_curve(y_actual, y_prob)
            prauc = auc(recall, precision)
        except ValueError:
            prauc = np.nan

        auprg = prg_auc(y_actual, y_prob)

        return auroc, prauc, auprg

    def process_metrics(pred: pd.DataFrame, *, desc: str = "Fold metrics", keep_fold_predictions: bool = False):
        """Summarize fold-level predictions into per-label model metrics."""
        class_balance = (
            pred[["Metadata_Label", "Metadata_Count_0", "Metadata_Count_1", "Model_type"]]
            .drop_duplicates()
        )

        group_cols = ["Metadata_Label", "Model_type", "repeat", "k_fold"]
        grouped = pred.groupby(group_cols, sort=False, observed=True)
        fold_records = []
        fold_columns = [
            *group_cols,
            "Metadata_Count_0",
            "Metadata_Count_1",
            "AUROC",
            "PRAUC",
            "AUPRG",
        ]
        if keep_fold_predictions:
            fold_columns.extend(["y_pred", "y_actual", "y_prob"])

        for keys, group in tqdm(
            grouped,
            total=grouped.ngroups,
            desc=desc,
            unit="fold",
            dynamic_ncols=True,
        ):
            y_actual = group["y_actual"].to_numpy(copy=False)
            y_prob = group["y_prob"].to_numpy(copy=False)
            auroc, prauc, auprg = compute_metrics(y_actual, y_prob)
            record = [
                *keys,
                group["Metadata_Count_0"].iat[0],
                group["Metadata_Count_1"].iat[0],
                auroc,
                prauc,
                auprg,
            ]
            if keep_fold_predictions:
                record.extend([
                    group["y_pred"].to_list(),
                    group["y_actual"].to_list(),
                    group["y_prob"].to_list(),
                ])
            fold_records.append(record)

        fold_metrics = pd.DataFrame.from_records(fold_records, columns=fold_columns)

        result = fold_metrics.groupby(
            ["Metadata_Label", "Model_type"],
            as_index=False,
            sort=False,
            observed=True,
        ).agg(
            AUROC=("AUROC", "mean"),
            AUROC_std=("AUROC", "std"),
            PRAUC=("PRAUC", "mean"),
            PRAUC_std=("PRAUC", "std"),
            AUPRG=("AUPRG", "mean"),
            AUPRG_std=("AUPRG", "std"),
        )

        result = result.merge(class_balance, on=["Metadata_Label", "Model_type"], how="left")

        return result, fold_metrics
    return (process_metrics,)


@app.cell
def _(
    df_axiom_cellbased_binary_prediction,
    df_axiom_cellfree_binary_prediction,
    df_axiom_cytotox_binary_prediction,
    df_axiom_hits_prediction,
    process_metrics,
):
    df_axiom_hits_metrics, df_axiom_hits_metrics_cv = process_metrics(
        df_axiom_hits_prediction,
        desc="Axiom cytotoxicity CV metrics",
    )
    df_axiom_cellbased_binary_metrics, df_axiom_cellbased_binary_metrics_cv = process_metrics(
        df_axiom_cellbased_binary_prediction,
        desc="ToxCast cell-based CV metrics",
    )
    df_axiom_cellfree_binary_metrics, df_axiom_cellfree_binary_metrics_cv = process_metrics(
        df_axiom_cellfree_binary_prediction,
        desc="ToxCast cell-free CV metrics",
    )
    df_axiom_cytotox_binary_metrics, df_axiom_cytotox_binary_metrics_cv = process_metrics(
        df_axiom_cytotox_binary_prediction,
        desc="ToxCast cytotox CV metrics",
    )
    print(f"Axiom cytotoxicity metrics: {df_axiom_hits_metrics.shape}")
    print(f"ToxCast cell-based metrics: {df_axiom_cellbased_binary_metrics.shape}")
    print(f"ToxCast cell-free metrics: {df_axiom_cellfree_binary_metrics.shape}")
    print(f"ToxCast cytotox metrics: {df_axiom_cytotox_binary_metrics.shape}")
    return (
        df_axiom_cellbased_binary_metrics,
        df_axiom_cellbased_binary_metrics_cv,
        df_axiom_cellfree_binary_metrics,
        df_axiom_cellfree_binary_metrics_cv,
        df_axiom_cytotox_binary_metrics,
        df_axiom_cytotox_binary_metrics_cv,
        df_axiom_hits_metrics,
        df_axiom_hits_metrics_cv,
    )


@app.cell
def _(df_axiom_cytotox_binary_metrics_cv):
    df_axiom_cytotox_binary_metrics_cv
    return


@app.cell
def _(mo):
    mo.md(r"""# Annotate Metrics""")
    return


@app.cell
def _(
    df_axiom_cellbased_binary_metrics,
    df_axiom_cellfree_binary_metrics,
    df_axiom_cytotox_binary_metrics,
    df_axiom_hits_metrics,
    pd,
):
    # Map endpoint names to their corresponding metric dataframes
    endpoint_map = {
        "axiom_hits": df_axiom_hits_metrics,
        "toxcast_cellbased_binary": df_axiom_cellbased_binary_metrics,
        "toxcast_cellfree_binary": df_axiom_cellfree_binary_metrics,
        "toxcast_cytotox_binary": df_axiom_cytotox_binary_metrics,
    }

    pred_list = []

    for et, df in endpoint_map.items():
        df = df.copy()

        # Total compounds
        df["Total_Compounds"] = df["Metadata_Count_0"] + df["Metadata_Count_1"]

        # Endpoint type
        df["Endpoint_type"] = et

        pred_list.append(df)

    # Concatenate
    pred_pandas = pd.concat(pred_list, axis=0, ignore_index=True)

    # Rename if needed (mimicking your Polars logic)
    pred_pandas["Endpoint_type"] = pred_pandas["Endpoint_type"].replace({
        "axiom_hits": "axiom_cytotox"
    })
    pred_pandas["Endpoint_type"] = pred_pandas["Endpoint_type"].replace({
        "toxcast_cellbased_binary": "toxcast_cellbased"
    })
    pred_pandas["Endpoint_type"] = pred_pandas["Endpoint_type"].replace({
        "toxcast_cellfree_binary": "toxcast_cellfree"
    })
    pred_pandas["Endpoint_type"] = pred_pandas["Endpoint_type"].replace({
        "toxcast_cytotox_binary": "toxcast_cytotox"
    })

    # Percent positive
    pred_pandas["Percent_Positive"] = (
        pred_pandas["Metadata_Count_1"] / pred_pandas["Total_Compounds"]
    )

    # Ordered categorical
    pred_pandas["Endpoint_type"] = pd.Categorical(
        pred_pandas["Endpoint_type"],
        categories=[
            "axiom_cytotox",
            "toxcast_cytotox",
            "toxcast_cellbased",
            "toxcast_cellfree",
        ],
        ordered=True
    )
    return (pred_pandas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Significant Performance Labels""")
    return


@app.cell
def _(
    df_axiom_cellbased_binary_metrics_cv,
    df_axiom_cellfree_binary_metrics_cv,
    df_axiom_cytotox_binary_metrics_cv,
    df_axiom_hits_metrics_cv,
    pd,
):
    df_metrics_cv = pd.concat([df_axiom_hits_metrics_cv,
                               df_axiom_cellbased_binary_metrics_cv,
                               df_axiom_cellfree_binary_metrics_cv,
                               df_axiom_cytotox_binary_metrics_cv
                              ], axis=0, ignore_index=True)
    df_metrics_cv
    return (df_metrics_cv,)


@app.cell
def _(df_metrics_cv, pd, stats, tqdm):
    # Filter for the two models to compare
    df_comparison = df_metrics_cv[
        df_metrics_cv["Model_type"].isin(["Actual", "Random_baseline"])
    ].copy()

    # Pivot both metrics once to get paired structure
    df_paired = df_comparison.pivot_table(
        index=["Metadata_Label", "repeat", "k_fold"],
        columns="Model_type",
        values=["AUROC", "AUPRG"],
        aggfunc="first"
    ).reset_index()
    df_paired.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in df_paired.columns
    ]

    # Dataset-level class counts (constant per label)
    df_counts = (
        df_comparison[df_comparison["Model_type"] == "Actual"]
        .groupby("Metadata_Label")[["Metadata_Count_0", "Metadata_Count_1"]]
        .first()
        .reset_index()
    )
    _counts_lookup = df_counts.set_index("Metadata_Label")

    def _paired_ttest(actual, baseline):
        _valid = actual.notna() & baseline.notna()
        if _valid.sum() < 2:
            return float("nan"), float("nan")
        return stats.ttest_rel(actual[_valid], baseline[_valid])

    # Perform paired t-test for each Metadata_Label
    results = []
    label_groups = df_paired.groupby("Metadata_Label", sort=False, observed=True)

    for label, label_data in tqdm(
        label_groups,
        total=label_groups.ngroups,
        desc="Paired t-tests",
        unit="label",
        dynamic_ncols=True,
    ):
        t_stat_auroc, p_val_auroc = _paired_ttest(
            label_data["AUROC_Actual"],
            label_data["AUROC_Random_baseline"],
        )
        t_stat_auprg, p_val_auprg = _paired_ttest(
            label_data["AUPRG_Actual"],
            label_data["AUPRG_Random_baseline"],
        )

        mean_auroc_actual = label_data["AUROC_Actual"].mean()
        mean_auroc_random = label_data["AUROC_Random_baseline"].mean()
        mean_auprg_actual = label_data["AUPRG_Actual"].mean()
        mean_auprg_random = label_data["AUPRG_Random_baseline"].mean()

        results.append({
            "Metadata_Label":        label,
            "Metadata_Count_0":      _counts_lookup.loc[label, "Metadata_Count_0"],
            "Metadata_Count_1":      _counts_lookup.loc[label, "Metadata_Count_1"],
            "AUROC_mean_Actual":     mean_auroc_actual,
            "AUROC_mean_Random_baseline":  mean_auroc_random,
            "AUROC_diff":            mean_auroc_actual - mean_auroc_random,
            "AUROC_t_stat":          t_stat_auroc,
            "AUROC_p_value":         p_val_auroc,
            "AUPRG_mean_Actual":     mean_auprg_actual,
            "AUPRG_mean_Random_baseline":  mean_auprg_random,
            "AUPRG_diff":            mean_auprg_actual - mean_auprg_random,
            "AUPRG_t_stat":          t_stat_auprg,
            "AUPRG_p_value":         p_val_auprg,
            "n_folds":               len(label_data)
        })

    # Create results dataframe
    df_ttest_results = pd.DataFrame(results)

    # Add significance indicators
    df_ttest_results["AUROC_significant"] = df_ttest_results["AUROC_p_value"] < 0.05
    df_ttest_results["AUPRG_significant"] = df_ttest_results["AUPRG_p_value"] < 0.05

    # Sort by AUROC difference
    df_ttest_results = df_ttest_results.sort_values("AUROC_diff", ascending=False)

    if not df_ttest_results.empty:
        print(f"\nPaired t-test results (n={df_ttest_results['n_folds'].iloc[0]} folds per label)")
    print("Comparing: Actual (Morphology) vs Random_baseline\n")
    print(df_ttest_results)
    return (df_ttest_results,)


@app.cell
def _(df_ttest_results):
    df_ttest_results
    return


@app.cell
def _(df_ttest_results):
    df_ttest_results[df_ttest_results["AUPRG_significant"]][
        ["Metadata_Label", "AUROC_mean_Actual", "AUROC_mean_Random_baseline", "AUROC_p_value", 
         "AUPRG_mean_Actual", "AUPRG_mean_Random_baseline", "AUPRG_p_value", "Metadata_Count_0", "Metadata_Count_1"]
    ].sort_values("AUPRG_p_value", ascending=True)
    return


@app.cell
def _(df_axiom_cellbased_binary_metrics_cv, df_ttest_results, pd):
    excluded_labels = [
        "NVS_TR_hDAT",
        "tissue__liver",
        "tissue__prostate"
    ]

    # 1. Main significant performance labels
    significant_labels = df_ttest_results[
        (df_ttest_results["AUPRG_significant"]) &
        (df_ttest_results["AUROC_significant"]) &
        (df_ttest_results["AUROC_mean_Actual"] > df_ttest_results["AUROC_mean_Random_baseline"]) &
        (df_ttest_results["AUPRG_mean_Actual"] > df_ttest_results["AUPRG_mean_Random_baseline"]) &
        (df_ttest_results["Metadata_Count_1"] > 20) &
        (df_ttest_results["Metadata_Label"].isin(df_axiom_cellbased_binary_metrics_cv["Metadata_Label"].unique())) &
        (~df_ttest_results["Metadata_Label"].isin(excluded_labels))
    ]["Metadata_Label"]

    # 2. Extra labels to add separately: LDH, MT, or cell_type__HepG2
    extra_labels = df_ttest_results[
        (
            df_ttest_results["Metadata_Label"].str.contains("LDH", na=False) |
            df_ttest_results["Metadata_Label"].str.contains("MT", na=False) |
            df_ttest_results["Metadata_Label"].str.contains("cell_type__HepG2", na=False)
        )
    ]["Metadata_Label"]

    # Combine and remove duplicates
    list_significative_performance = pd.Series(
        pd.concat([significant_labels, extra_labels]).unique()
    )

    print(f"Significant performance labels retained: {len(significant_labels)}")
    print(f"Extra LDH / MT / HepG2 labels added: {len(set(extra_labels) - set(significant_labels))}")
    print(f"Total labels retained: {len(list_significative_performance)}")
    return (list_significative_performance,)


@app.cell
def _(
    classification_output_dir,
    df_ttest_results,
    list_significative_performance,
):
    _mask_assays = (
        (df_ttest_results["Metadata_Label"].isin(list_significative_performance))
    )

    df_ttest_results[_mask_assays].to_csv(
        classification_output_dir / "results_performance.csv",
        index=False
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Correlation Between Significant Actual Models [outdated]""")
    return


@app.cell
def _(
    df_axiom_cellbased_binary_prediction,
    df_axiom_cellfree_binary_prediction,
    df_axiom_cytotox_binary_prediction,
    df_axiom_hits_prediction,
    list_significative_performance,
    pd,
    tqdm,
):
    # ── Step 1: Filter function ────────────────────────────────────────────────
    def _filter_df(_df):
        return _df[
            (_df["Model_type"] == "Actual") &
            (_df["Metadata_Label"].isin(list_significative_performance))
        ][["Metadata_compound", "repeat", "Metadata_Label", "y_prob"]].copy()


    # ── Step 2: Apply filtering ────────────────────────────────────────────────
    _prediction_sources = [
        ("Hits", df_axiom_hits_prediction),
        ("Cell-based", df_axiom_cellbased_binary_prediction),
        ("Cell-free", df_axiom_cellfree_binary_prediction),
        ("Cytotox", df_axiom_cytotox_binary_prediction),
    ]
    _filtered_predictions = []
    for _name, _df in tqdm(
        _prediction_sources,
        desc="Filter prediction matrices",
        unit="dataset",
        dynamic_ncols=True,
    ):
        _filtered = _filter_df(_df)
        _filtered["y_prob"] = pd.to_numeric(_filtered["y_prob"], downcast="float")
        _filtered_predictions.append(_filtered)
        tqdm.write(f"{_name}: {_filtered.shape[0]:,} rows retained for correlation")

    # ── Step 3: Concatenate and pivot all datasets ─────────────────────────────
    _df_all = pd.concat(_filtered_predictions, ignore_index=True)

    # ── Pivot ────────────────────────────────────────────────────────────────
    df_pivot_correlation = _df_all.pivot_table(
        index=["Metadata_compound", "repeat"],
        columns="Metadata_Label",
        values="y_prob",
        observed=True,
    )
    return (df_pivot_correlation,)


@app.cell
def _(df_pivot_correlation, downloads_dir, plt, sns, tqdm):
    # ── Step 5: Compute correlation matrix ─────────────────────────────────────
    with tqdm(
        total=3,
        desc="Correlation heatmap",
        unit="step",
        dynamic_ncols=True,
    ) as _progress:
        _corr = df_pivot_correlation.corr(method="pearson").astype("float32")
        _progress.update()

        # ── Step 6: Clustered heatmap ──────────────────────────────────────────────
        fig_size  = 28 # canvas grows dynamically with label count

        _clustermap = sns.clustermap(
            _corr,
            method="average",
            metric="euclidean",
            cmap="coolwarm",
            vmin=-1, vmax=1,
            figsize=(fig_size, fig_size),  # ← dynamic, not hardcoded (20,20)
            linewidths=0.1,
            dendrogram_ratio=(0.10, 0.10),
            cbar_pos=(0.02, 0.82, 0.02, 0.12),
            xticklabels=True,              # ← forces ALL x labels (key fix)
            yticklabels=True,              # ← forces ALL y labels (key fix)
        )
        _progress.update()

        # ── Label readability ──────────────────────────────────────────────────────

        plt.setp(_clustermap.ax_heatmap.get_xticklabels(),
                 rotation=90, fontsize=16, ha="center")
        plt.setp(_clustermap.ax_heatmap.get_yticklabels(),
                 rotation=0,  fontsize=16, va="center")

        _clustermap.fig.suptitle(
            "Clustered Correlation of Metadata_Labels (y_prob)",
            fontsize=14, fontweight="bold", y=1.01  # y>1 avoids overlap with dendro
        )
        plt.savefig(downloads_dir / "correlation_heatmap_axiom_batch.png",
                     bbox_inches="tight", dpi=300)
        _progress.update()

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Performance by Endpoint Category: Significant Labels""")
    return


@app.cell
def _():
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    return mpatches, plt


@app.cell
def _(
    df_axiom_cellbased_binary_metrics_cv,
    df_axiom_cellfree_binary_metrics_cv,
    df_axiom_cytotox_binary_metrics_cv,
    df_axiom_hits_metrics_cv,
    df_metrics_cv,
    list_significative_performance,
):
    # ── 1. Build source-lookup ────────────────────────────────────────────────────
    _source_map = {}
    _source_dfs = {
        "Hits":       df_axiom_hits_metrics_cv,
        "Cell-based": df_axiom_cellbased_binary_metrics_cv,
        "Cell-free":  df_axiom_cellfree_binary_metrics_cv,
        "Cytotox":    df_axiom_cytotox_binary_metrics_cv,
    }
    for _name, _df in _source_dfs.items():
        for _label in _df["Metadata_Label"].unique():
            _source_map[_label] = _name

    # ── 2. Significant labels (BOTH AUROC AND PRAUC p < 0.05) ────────────────────
    _sig_labels = list_significative_performance.to_list()

    # ── 3. Filter metrics ─────────────────────────────────────────────────────────
    _MODELS  = ["Actual", "Cellcount_baseline", "Random_baseline"]
    _df_plot = df_metrics_cv[
        df_metrics_cv["Model_type"].isin(_MODELS) &
        df_metrics_cv["Metadata_Label"].isin(_sig_labels)
    ].copy()
    _df_plot["Source"] = _df_plot["Metadata_Label"].map(_source_map)

    # ── 4. Compute mean ± std ─────────────────────────────────────────────────────
    _df_agg = (
        _df_plot
        .groupby(["Source","Metadata_Label", "Model_type"])["AUROC"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    _df_agg[["Source", "Metadata_Label"]].drop_duplicates()["Source"].value_counts()
    return


@app.cell
def _():
    import math
    return


@app.cell
def _(mo):
    mo.md(r"""# Plot comparison AUPRG AUC""")
    return


@app.cell
def _():
    from adjustText import adjust_text
    return (adjust_text,)


@app.cell
def _(
    Line2D,
    adjust_text,
    df_axiom_cellbased_binary_metrics_cv,
    df_axiom_cytotox_binary_metrics_cv,
    df_axiom_hits_metrics_cv,
    df_metrics_cv,
    list_significative_performance,
    mpatches,
    plt,
):
    # ── 1. Build source-lookup ────────────────────────────────────────────────────
    _source_map = {}

    _source_dfs = {
        "Hits":       df_axiom_hits_metrics_cv,
        "Cell-based": df_axiom_cellbased_binary_metrics_cv,
        # "Cell-free":  df_axiom_cellfree_binary_metrics_cv,
        "Cytotox":    df_axiom_cytotox_binary_metrics_cv,
    }

    for _name, _df in _source_dfs.items():
        for _label in _df["Metadata_Label"].unique():
            _source_map[_label] = _name

    # ── 2. Significant assays only ────────────────────────────────────────────────
    _sig_labels = list_significative_performance.to_list()

    # ── 3. Filter metrics to significant assays only ──────────────────────────────
    _models = ["Actual", "Cellcount_baseline", "Random_baseline"]

    _df_plot = df_metrics_cv[
        (df_metrics_cv["Model_type"].isin(_models)) &
        (df_metrics_cv["Metadata_Label"].isin(_sig_labels))
    ].copy()

    _df_plot["Source"] = _df_plot["Metadata_Label"].map(_source_map)
    _df_plot = _df_plot.dropna(subset=["Source"])

    # ── 4. Aggregate AUROC and AUPRG ──────────────────────────────────────────────
    _df_agg_auroc = (
        _df_plot
        .groupby(["Metadata_Label", "Model_type"])["AUROC"]
        .agg(auroc_mean="mean", auroc_std="std")
        .reset_index()
    )

    _df_agg_auprg = (
        _df_plot
        .groupby(["Metadata_Label", "Model_type"])["AUPRG"]
        .agg(auprg_mean="mean", auprg_std="std")
        .reset_index()
    )

    _df_scatter = (
        _df_agg_auroc[_df_agg_auroc["Model_type"] == "Actual"]
        .merge(
            _df_agg_auprg[_df_agg_auprg["Model_type"] == "Actual"],
            on=["Metadata_Label", "Model_type"]
        )
    )

    _df_scatter["Source"] = _df_scatter["Metadata_Label"].map(_source_map)
    _df_scatter = _df_scatter.dropna(subset=["Source"])

    # ── 5. Thresholds ─────────────────────────────────────────────────────────────
    _auroc_t1, _auroc_t2 = 0.5, 0.70
    _auprg_t1, _auprg_t2 = 0.0, 0.3

    # ── 6. Background colors ──────────────────────────────────────────────────────
    _bg = {
        "worse":    "#f7d4d4",
        "moderate": "#fff4cc",
        "well":     "#d4edda",
    }

    # ── 7. Source colors and lowercase labels ────────────────────────────────────
    _source_colors = {
        "Hits":       "#4C72B0",
        "Cell-based": "#DD8452",
        "Cell-free":  "#55A868",
        "Cytotox":    "#C44E52",
    }

    _label_map = {
        "Hits": "cytotox",
        "Cell-based": "toxcast cell-based",
        "Cell-free": "toxcast cell-free",
        "Cytotox": "toxcast cytotox",
    }

    # ── 8. Figure ─────────────────────────────────────────────────────────────────
    _fig, _ax = plt.subplots(figsize=(6, 6))

    _xmin, _xmax = -0.2, 1
    _ymin, _ymax = 0.2, 1

    # ── 9. Background regions ─────────────────────────────────────────────────────
    _ax.fill_between(
        [_xmin, _xmax],
        _ymin,
        _ymax,
        color=_bg["worse"],
        zorder=0
    )

    _ax.fill_between(
        [_auprg_t1, _xmax],
        _auroc_t1,
        _ymax,
        color=_bg["moderate"],
        zorder=0
    )

    _ax.fill_between(
        [_auprg_t2, _xmax],
        _auroc_t2,
        _ymax,
        color=_bg["well"],
        zorder=0
    )

    # ── 10. Scatter points: significant assays only ───────────────────────────────
    texts = []

    for _src, _df_src in _df_scatter.groupby("Source"):
        _color = _source_colors.get(_src, "grey")

        _ax.scatter(
            _df_src["auprg_mean"],
            _df_src["auroc_mean"],
            color=_color,
            s=60,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            marker="o",
            zorder=4,
            label=_src.lower(),
        )

        for _, _row in _df_src.iterrows():
            if _row["auprg_mean"] > _auprg_t2 and _row["auroc_mean"] > _auroc_t2:
                texts.append(
                    _ax.text(
                        _row["auprg_mean"],
                        _row["auroc_mean"],
                        _row["Metadata_Label"],
                        fontsize=12,
                        color="#1a1a1a",
                        fontstyle="italic",
                        zorder=6,
                    )
                )

    # ── 11. Label repulsion ───────────────────────────────────────────────────────
    adjust_text(
        texts,
        x=_df_scatter["auprg_mean"].values,
        y=_df_scatter["auroc_mean"].values,
        ax=_ax,
        max_move=3.0,
        force_points=(0.3, 0.3),
        add_objects=[],
        arrowprops=dict(
            arrowstyle="-",
            color="grey",
            lw=0.4,
            alpha=0.6,
            shrinkA=0,
            shrinkB=3
        ),
    )

    # ── 12. Region counts ─────────────────────────────────────────────────────────
    _df_sig = _df_scatter.copy()

    _n_worse = (
        (_df_sig["auprg_mean"] < _auprg_t1) |
        (_df_sig["auroc_mean"] < _auroc_t1)
    ).sum()

    _n_moderate = (
        ((_df_sig["auprg_mean"] >= _auprg_t1) & (_df_sig["auroc_mean"] >= _auroc_t1)) &
        ~((_df_sig["auprg_mean"] >= _auprg_t2) & (_df_sig["auroc_mean"] >= _auroc_t2))
    ).sum()

    _n_well = (
        (_df_sig["auprg_mean"] >= _auprg_t2) &
        (_df_sig["auroc_mean"] >= _auroc_t2)
    ).sum()

    _ax.text(
        0.55,
        0.40,
        f"worse than random\n(n={_n_worse})",
        fontsize=10,
        color="#c0392b",
        alpha=0.7,
        fontstyle="italic",
        ha="center"
    )

    _ax.text(
        0.65,
        0.53,
        f"moderately predicted\n(n={_n_moderate})",
        fontsize=10,
        color="#b7950b",
        alpha=0.7,
        fontstyle="italic",
        ha="center"
    )

    _ax.text(
        0.8,
        0.73,
        f"well predicted\n(n={_n_well})",
        fontsize=10,
        color="#1e8449",
        alpha=0.7,
        fontstyle="italic",
        ha="center"
    )

    # ── 13. Style ─────────────────────────────────────────────────────────────────
    _ax.set_xlim(_xmin, _xmax)
    _ax.set_ylim(_ymin, _ymax)

    _ax.set_xlabel("auprg (mean)", fontsize=12)
    _ax.set_ylabel("auroc (mean)", fontsize=12)

    _ax.set_facecolor("white")
    _fig.patch.set_facecolor("white")

    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    _ax.spines["left"].set_color("black")
    _ax.spines["left"].set_linewidth(1)

    _ax.spines["bottom"].set_color("black")
    _ax.spines["bottom"].set_linewidth(1)

    _ax.tick_params(
        axis="both",
        which="major",
        colors="black",
        length=4,
        width=1,
        direction="out"
    )

    _ax.grid(False)

    # ── 14. Legend ────────────────────────────────────────────────────────────────
    _src_patches = [
        mpatches.Patch(
            facecolor=_source_colors[_s],
            edgecolor="none",
            label=_label_map.get(_s, _s).lower()
        )
        for _s in _source_dfs
    ]

    _marker_patches = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            linestyle="none",
            markersize=7,
            label="significant assays"
        )
    ]

    _ax.legend(
        handles=_src_patches + _marker_patches,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=8,
        borderpad=0.8,
        title="assays",
        title_fontsize=10,
    ).get_title().set_fontweight("bold")

    plt.tight_layout()

    plt.savefig(
        "modules/07_downstream_analysis/figures/Figure_6.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Focus performance""")
    return


@app.cell
def _(pd):
    # Visualisation function
    def create_styled_metrics_table(
        df: pd.DataFrame,
        sort_by: str = "PRAUC",
        ascending: bool = False,
        model_type_rename: dict = None,
        model_type_order: list = None
    ):
        """
        Create a styled table with AUROC and PRAUC metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with predictions and metrics
        sort_by : str, default "PRAUC"
            Metric to sort by ("PRAUC" or "AUROC")
        ascending : bool, default False
            Sort order (False for descending, True for ascending)
        model_type_rename : dict, optional
            Dictionary to rename Model_type values
            Default: {"Actual": "Morphology", "Cellcount_baseline": "Cell count", 
                      "Random_baseline": "Random baseline"}
        model_type_order : list, optional
            Ordered list of model types for display
            Default: ["Morphology", "Cell count", "Random baseline"]

        Returns
        -------
        pandas.io.formats.style.Styler
            Styled dataframe ready for display
        """
        # Default rename mapping
        if model_type_rename is None:
            model_type_rename = {
                "Actual": "Morphology",
                "Cellcount_baseline": "Cell count",
                "Random_baseline": "Random baseline"
            }

        # Default order
        if model_type_order is None:
            model_type_order = ["Morphology", "Cell count", "Random baseline"]

        # Copy to avoid modifying original
        _df = df.copy()

        # Rename Model_type values
        _df["Model_type"] = _df["Model_type"].replace(model_type_rename)

        # Make Model_type categorical with specified order
        _df["Model_type"] = pd.Categorical(
            _df["Model_type"],
            categories=model_type_order,
            ordered=True
        )

        # Filter Morphology rows for sorting
        _morph_data = _df[_df["Model_type"] == model_type_order[0]]

        # Sort Metadata_Label by specified metric
        _ordered_labels = _morph_data.sort_values(sort_by, ascending=ascending)["Metadata_Label"]

        # Pivot for AUPRG
        _pivot_prauc = _df.pivot_table(
            index='Metadata_Label',
            columns='Model_type',
            values='AUPRG',
            aggfunc='first'
        ).reindex(_ordered_labels)

        # Pivot for AUROC
        _pivot_auroc = _df.pivot_table(
            index='Metadata_Label',
            columns='Model_type',
            values='AUROC',
            aggfunc='first'
        ).reindex(_ordered_labels)

        # Rename columns to indicate metric
        _pivot_prauc.columns = [f'{col} AUPRG' for col in _pivot_prauc.columns]
        _pivot_auroc.columns = [f'{col} AUROC' for col in _pivot_auroc.columns]

        # Add metadata columns
        _metadata = _df[["Metadata_Label", "Metadata_Count_0", "Metadata_Count_1", "Total_Compounds"]].drop_duplicates()
        _metadata = _metadata.set_index("Metadata_Label").reindex(_ordered_labels)

        # Combine all tables: metadata + AUPRG + AUROC
        _final_table = pd.concat([_metadata, _pivot_prauc, _pivot_auroc], axis=1)

        # Reset index
        _final_table = _final_table.reset_index()

        # Rename columns for clarity
        _final_table.columns.name = None
        _final_table = _final_table.rename(columns={
            "Metadata_Label": "Metadata Label",
            "Metadata_Count_0": "Inactive",
            "Metadata_Count_1": "Active",
            "Total_Compounds": "Total"
        })

        # Get all AUPRG and AUROC columns for styling
        _prauc_cols = [col for col in _final_table.columns if 'AUPRG' in col]
        _auroc_cols = [col for col in _final_table.columns if 'AUROC' in col]

        # Style the table
        _styled_table = (_final_table.style
            .format({
                **{col: '{:.3f}' for col in _prauc_cols},
                **{col: '{:.3f}' for col in _auroc_cols},
                'Inactive': '{:.0f}',
                'Active': '{:.0f}',
                'Total': '{:.0f}'
            })
            .background_gradient(
                subset=_prauc_cols + _auroc_cols,
                cmap='RdYlGn',
                vmin=0,
                vmax=1
            )
            .bar(
                subset=['Inactive', 'Active', 'Total'],
                color='#5fba7d',
                vmin=0
            )
            .set_properties(**{
                'text-align': 'center',
                'font-family': 'Arial',
                'font-size': '10pt'
            })
            .set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#2c3e50'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('font-size', '11pt'),
                ]},
                {'selector': 'td', 'props': [
                    ('border', '1px solid #ddd'),
                ]},
                {'selector': '', 'props': [
                    ('border-collapse', 'collapse'),
                    ('font-family', 'Arial, sans-serif'),
                    ('box-shadow', '0 0 20px rgba(0, 0, 0, 0.15)')
                ]}
            ])
        )

        return _styled_table
    return (create_styled_metrics_table,)


@app.cell
def _(
    create_styled_metrics_table,
    list_significative_performance,
    pred_pandas,
):
    # Filtered dataframe
    _df_cell_cytotox = pred_pandas[
        pred_pandas["Metadata_Label"].isin(list_significative_performance)
    ].copy()

    # Create styled table
    styled_table = create_styled_metrics_table(_df_cell_cytotox)

    # Save styled table to Excel
    styled_table.to_excel(
        "modules/07_downstream_analysis/tables/Table_S2_perf_models_filtered.xlsx",
        sheet_name="Cell Cytotox Metrics",
        index=False,
        engine="openpyxl"
    )
    styled_table
    return


@app.cell
def _(mo):
    mo.md(r"""# Check assays modalities cell-based using LLM""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Use LLM to make toxicity categories
    https://chatgpt.com/share/e/6a43ce12-158c-8017-9854-91a53900bb75
    """
    )
    return


@app.cell
def _(pd):
    df_toxcast_cellbased_assay_info = pd.read_csv("/Users/ollitrault/Documents/projects/databases/proc_toxcast_annotations/toxcast_cellbased_assay_info.csv")
    return (df_toxcast_cellbased_assay_info,)


@app.cell
def _(list_significative_performance):
    list(list_significative_performance)
    return


@app.cell
def _():
    dict_categories_significative_cell_based = {
        "categories": [
            {
                "id": "endocrine_sex_hormones",
                "label": "Endocrine Disruption — Sex Hormones",
                "color": "#d62728",
                "description": "ER, AR, PR and aromatase signaling assays.",
                "assays": [
                    "TOX21_AR_LUC_MDAKB2_Antagonist_10nM_R1881",
                    "TOX21_AR_LUC_MDAKB2_Agonist",
                    "TOX21_AR_LUC_MDAKB2_Antagonist_0.5nM_R1881",
                    "TOX21_AR_BLA_Agonist_ratio",
                    "TOX21_AR_BLA_Antagonist_ratio",
                    "TOX21_ERa_LUC_VM7_Agonist",
                    "TOX21_ERa_LUC_VM7_Antagonist_0.1nM_E2",
                    "TOX21_ERa_LUC_VM7_Antagonist_0.5nM_E2",
                    "TOX21_ERa_BLA_Antagonist_ratio",
                    "TOX21_ERb_BLA_Antagonist_ratio",
                    "TOX21_PR_BLA_Agonist_ratio",
                    "TOX21_PR_BLA_Antagonist_ratio",
                    "TOX21_Aromatase_Inhibition",
                    "OT_ER_ERaERb_1440",
                    "OT_ER_ERbERb_1440",
                    "OT_ERa_GFPERaERE_0480",
                    "ATG_ERE_CIS",
                ],
            },
            {
                "id": "endocrine_steroidogenesis",
                "label": "Endocrine Disruption — Steroidogenesis & Adrenal",
                "color": "#ff7f0e",
                "description": "H295R steroidogenesis assays.",
                "assays": [
                    "CEETOX_H295R_11DCORT",
                    "CEETOX_H295R_OHPREG",
                ],
            },
            {
                "id": "endocrine_metabolic_nuclear_receptors",
                "label": "Endocrine Disruption — Metabolic / Orphan Nuclear Receptors",
                "color": "#ffbb00",
                "description": "Metabolic nuclear receptor signaling.",
                "assays": [
                    "TOX21_GR_BLA_Antagonist_ratio",
                    "TOX21_FXR_BLA_Antagonist_ratio",
                    "TOX21_PGC_ERR_Antagonist",
                    "TOX21_ERR_Antagonist",
                    "TOX21_PXR_agonist",
                    "TOX21_CAR_Agonist",
                    "TOX21_PPARg_BLA_Antagonist_ratio",
                    "ATG_DR4_LXR_CIS",
                    "ATG_PPRE_CIS",
                    "ATG_PXRE_CIS",
                    "ATG_VDRE_CIS",
                ],
            },
            {
                "id": "hepatotoxicity_cyp_induction",
                "label": "Hepatotoxicity — Xenobiotic Metabolism",
                "color": "#8c564b",
                "description": "Drug metabolism and xenobiotic response pathways.",
                "assays": [
                    "LTEA_HepaRG_CYP1A1",
                    "LTEA_HepaRG_CYP1A2",
                    "LTEA_HepaRG_CYP2B6",
                    "LTEA_HepaRG_CYP2C8",
                    "LTEA_HepaRG_CYP2C19",
                    "LTEA_HepaRG_CYP2E1",
                    "LTEA_HepaRG_CYP3A7",
                    "LTEA_HepaRG_CYP4A11",
                    "LTEA_HepaRG_CYP4A22",
                    "LTEA_HepaRG_UGT1A1",
                    "ATG_Ahr_CIS",
                    "TOX21_AhR_LUC_Agonist",
                ],
            },
            {
                "id": "hepatotoxicity_lipid_transport",
                "label": "Hepatotoxicity — Lipid Metabolism & Transport",
                "color": "#a07040",
                "description": "Markers of steatosis, lipid handling and transporter activity.",
                "assays": [
                    "LTEA_HepaRG_HMGCS2",
                    "LTEA_HepaRG_ABCB1",
                    "LTEA_HepaRG_ABCB11",
                    "LTEA_HepaRG_ABCG2",
                    "LTEA_HepaRG_THRSP",
                    "LTEA_HepaRG_FABP1",
                ],
            },
            {
                "id": "hepatotoxicity_general",
                "label": "Hepatotoxicity — General Stress Response",
                "color": "#2ca02c",
                "description": "Oxidative stress and adaptive transcriptional responses.",
                "assays": [
                    "LTEA_HepaRG_ALPP",
                    "LTEA_HepaRG_PEG10",
                    "ATG_NRF2_ARE_CIS",
                    "TOX21_ARE_BLA_Agonist_ratio",
                    "ATG_AP_1_CIS",
                    "TOX21_AP1_BLA_Agonist_ratio",
                    "ATG_MRE_CIS",
                    "ATG_Oct_MLP_CIS",
                    "ATG_TCF_b_cat_CIS",
                    "ATG_PBREM_CIS",
                    "ATG_SREBP_CIS",
                ],
            },
            {
                "id": "genotoxicity",
                "label": "Genotoxicity / DNA Damage",
                "color": "#1f77b4",
                "description": "DNA damage response and epigenetic perturbation.",
                "assays": [
                    "TOX21_p53_BLA_p1_ratio",
                    "TOX21_p53_BLA_p2_ratio",
                    "TOX21_p53_BLA_p3_ratio",
                    "TOX21_p53_BLA_p4_ratio",
                    "TOX21_p53_BLA_p5_ratio",
                    "TOX21_HDAC_Inhibition",
                ],
            },
            {
                "id": "cardiovascular_vascular_inflammation",
                "label": "Cardiovascular / Vascular Inflammation",
                "color": "#9467bd",
                "description": "BioMAP vascular and inflammatory endpoints.",
                "assays": [
                    "BSK_LPS_VCAM1",
                    "BSK_LPS_CD40",
                    "BSK_LPS_MCSF",
                    "BSK_LPS_PGE2",
                    "BSK_CASM3C_Thrombomodulin",
                    "BSK_CASM3C_Proliferation",
                    "BSK_3C_Proliferation",
                    "BSK_3C_SRB",
                    "BSK_3C_uPAR",
                    "BSK_4H_VCAM1",
                    "BSK_4H_MCP1",
                    "BSK_4H_SRB",
                    "BSK_4H_uPAR",
                    "BSK_SAg_CD40",
                    "BSK_SAg_Eselectin",
                    "BSK_SAg_Proliferation",
                ],
            },
            {
                "id": "mitochondrial_toxicity",
                "label": "Mitochondrial Toxicity",
                "color": "#8c6bb1",
                "description": "Loss of mitochondrial membrane potential.",
                "assays": [
                    "TOX21_MMP_ratio",
                ],
            },
            {
                "id": "general_cytotoxicity_viability",
                "label": "General Cytotoxicity / Cell Viability",
                "color": "#7f7f7f",
                "description": "Generic viability and membrane integrity measurements.",
                "assays": [
                    "LDH",
                    "MT",
                ],
            },
            {
                "id": "assay_context_metadata",
                "label": "Assay Context",
                "color": "#17becf",
                "description": "Metadata describing the biological assay system.",
                "assays": [
                    "cell_type__HepG2",
                ],
            },
        ]
    }
    return (dict_categories_significative_cell_based,)


@app.cell
def _(dict_categories_significative_cell_based):
    list_claudeai_assays = []
    for i in dict_categories_significative_cell_based["categories"]:
        print(i["assays"])
        list_claudeai_assays += i["assays"]
    print(len(list_claudeai_assays))
    return (list_claudeai_assays,)


@app.cell
def _(
    df_toxcast_cellbased_assay_info,
    list_claudeai_assays,
    list_significative_performance,
):
    _list_claudeai = set(list_claudeai_assays)
    _list_df = set(
        df_toxcast_cellbased_assay_info[
            df_toxcast_cellbased_assay_info["assay_component_endpoint_name"].isin(list_significative_performance)
        ]["assay_component_endpoint_name"]
    )

    print("In chagpt AI list but NOT in dataframe:", _list_claudeai - _list_df)
    print("In dataframe but NOT in chagpt AI list:", _list_df - _list_claudeai)
    return


@app.cell
def _(
    df_toxcast_cellbased_assay_info,
    dict_categories_significative_cell_based,
    list_significative_performance,
    pd,
):
    _cat_lookup = {
        _assay: {
            "chagpt_id":          _cat["id"],
            "chagpt_label":       _cat["label"],
            "chagpt_color":       _cat["color"],
            "chagpt_description": _cat["description"],
        }
        for _cat in dict_categories_significative_cell_based["categories"]
        for _assay in _cat["assays"]
    }

    df_toxcast_cellbased_assay_info_significant = (
        df_toxcast_cellbased_assay_info[
            df_toxcast_cellbased_assay_info["assay_component_endpoint_name"].isin(list_significative_performance)
        ]
        .copy()
        .join(
            pd.DataFrame.from_dict(_cat_lookup, orient="index"),
            on="assay_component_endpoint_name"
        )
    )
    df_toxcast_cellbased_assay_info_significant
    return (df_toxcast_cellbased_assay_info_significant,)


@app.cell
def _(df_toxcast_cellbased_assay_info_significant):
    # save it 
    df_toxcast_cellbased_assay_info_significant.to_csv("modules/07_downstream_analysis/outputs/df_toxcast_cellbased_assay_info_significant_llm_categories.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""## Add performances and other models""")
    return


@app.cell
def _(df_toxcast_cellbased_assay_info_significant):
    df_toxcast_cellbased_assay_info_significant
    return


@app.cell
def _(list_significative_performance, pd, pred_pandas):
    _cols = pd.Series(pred_pandas["Metadata_Label"].unique())

    _mask_cell_free  = _cols.str.contains("NVS|CCTE_GLTED_hIYD", regex=True)
    _mask_hit_calls  = _cols.str.contains("MT|LDH|cell_count", regex=True)
    _mask_cytotoxic  = _cols.str.contains("cell|tissue", regex=True) & ~_mask_hit_calls
    _mask_cell_based = ~_mask_cell_free & ~_mask_hit_calls & ~_mask_cytotoxic

    _valid_labels = set(_cols[_mask_cell_free | _mask_hit_calls | _mask_cytotoxic | _mask_cell_based])

    pred_pandas_filtered = pred_pandas[
        (pred_pandas["Model_type"] == "Actual") &
        (pred_pandas["Metadata_Label"].isin(list_significative_performance)) &
        (pred_pandas["Metadata_Label"].isin(_valid_labels)) &
        (pred_pandas["Metadata_Label"] != "cell_count") &
        (~pred_pandas["Metadata_Label"].str.startswith("NVS_", na=False)) &
        (
            ~(
                pred_pandas["Metadata_Label"].str.contains("tissue", case=False, na=False) |
                pred_pandas["Metadata_Label"].str.contains("cell_type", case=False, na=False)
            ) |
            (pred_pandas["Metadata_Label"] == "cell_type__HepG2")
        )
    ]
    pred_pandas_filtered
    return (pred_pandas_filtered,)


@app.cell
def _(df_toxcast_cellbased_assay_info_significant, pred_pandas_filtered):
    df_toxcast_assay_info_perf_significant = pred_pandas_filtered.merge(
        df_toxcast_cellbased_assay_info_significant,
        left_on="Metadata_Label",
        right_on="assay_component_endpoint_name",
        how="left"
    )
    df_toxcast_assay_info_perf_significant.to_csv("modules/07_downstream_analysis/tables/Table_S2.csv")
    return


if __name__ == "__main__":
    app.run()
