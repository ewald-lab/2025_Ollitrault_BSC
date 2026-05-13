import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import polars as pl 
    import pandas as pd
    import numpy as np
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import MultiComparison
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    from scipy import stats
    import warnings
    import plotnine as pn
    from plotnine import geom_point, ggplot, aes, labs, theme_bw, facet_wrap, geom_smooth, labeller
    from plotnine import (
        ggplot, aes, geom_sina, geom_boxplot, labs, theme_classic, theme_minimal,
        facet_wrap, theme, element_text, scale_x_discrete, element_rect, geom_jitter, ylim, geom_text, geom_tile, scale_fill_gradient
    )
    from matplotlib.lines import Line2D
    import seaborn as sns
    from pathlib import Path
    return (
        Line2D,
        Path,
        aes,
        auc,
        element_rect,
        element_text,
        facet_wrap,
        geom_boxplot,
        geom_jitter,
        geom_point,
        geom_smooth,
        ggplot,
        labs,
        mo,
        np,
        os,
        pd,
        pn,
        precision_recall_curve,
        roc_auc_score,
        scale_x_discrete,
        sns,
        stats,
        theme,
        theme_bw,
        theme_classic,
        warnings,
        ylim,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Notebook to explore classificxation results""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Input""")
    return


@app.cell
def _(pd):
    df_axiom_hits_prediction = pd.read_parquet("modules/07_classification/outputs/axiom_axiom_hits_predictions_batch.parquet")
    df_axiom_cellbased_binary_prediction = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cellbased_binary_predictions_batch.parquet")
    df_axiom_cellfree_binary_prediction = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cellfree_binary_predictions_batch.parquet")
    df_axiom_cytotox_binary_prediction = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cytotox_binary_predictions_batch.parquet")
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
    mo.md(r"""## Compute metrics""")
    return


@app.cell
def _(auc, np, pd, precision_recall_curve, roc_auc_score):
    def precision_recall_gain_curve(y_true, y_scores):
        # Sort predictions and true labels by descending score
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]

        # Calculate baseline precision and recall
        baseline_precision = np.mean(y_true)
        baseline_recall = np.sum(y_true) / len(y_true)

        precisions = []
        recalls = []
        precision_gains = []
        recall_gains = []

        for threshold in range(1, len(y_true) + 1):
            # Predictions above threshold considered positive
            predicted_positive = y_true_sorted[:threshold]

            # Calculate precision and recall
            true_positives = np.sum(predicted_positive)
            precision = true_positives / threshold
            recall = true_positives / np.sum(y_true)

            # Calculate precision gain and recall gain
            precision_gain = precision - baseline_precision
            recall_gain = recall - baseline_recall

            precisions.append(precision)
            recalls.append(recall)
            precision_gains.append(precision_gain)
            recall_gains.append(recall_gain)

        return precisions, recalls, precision_gains, recall_gains


    def prg_auc(y_true, y_scores):
        # Calculate PRG curve
        _, _, precision_gains, recall_gains = precision_recall_gain_curve(y_true, y_scores)

        # Calculate area under PRG curve using trapezoidal rule
        prg_auc = np.trapz(precision_gains, x=recall_gains)

        return prg_auc
    def compute_metrics(y_pred, y_actual, y_prob):
        # AUROC
        try:
            auroc = roc_auc_score(y_actual, y_prob)
        except ValueError:
            auroc = None

        # PRAUC
        try:
            precision, recall, _ = precision_recall_curve(y_actual, y_prob)
            prauc = auc(recall, precision)
        except ValueError:
            prauc = None

        # AUPRG
        auprg = prg_auc(np.array(y_actual), np.array(y_prob))

        return auroc, prauc, auprg
    def process_metrics(pred: pd.DataFrame):
        class_balance = (
            pred[["Metadata_Label", "Metadata_Count_0", "Metadata_Count_1", "Model_type"]]
            .drop_duplicates()
        )

        fold_metrics = (
            pred
            .groupby(["Metadata_Label", "Model_type", "k_fold"], as_index=False)
            .agg({
                "y_pred": list,
                "y_actual": list,
                "y_prob": list,
                "Metadata_Count_0": "first",
                "Metadata_Count_1": "first",
            })
        )

        fold_metrics[["AUROC", "PRAUC", "AUPRG"]] = fold_metrics.apply(
            lambda row: pd.Series(
                compute_metrics(row["y_pred"], row["y_actual"], row["y_prob"])
            ),
            axis=1
        )

        result = (
            fold_metrics
            .groupby(["Metadata_Label", "Model_type"], as_index=False)
            .agg({
                "AUROC":  ["mean", "std"],
                "PRAUC":  ["mean", "std"],
                "AUPRG":  ["mean", "std"],
            })
        )

        result.columns = [
            "Metadata_Label", "Model_type",
            "AUROC",  "AUROC_std",
            "PRAUC",  "PRAUC_std",
            "AUPRG",  "AUPRG_std",
        ]

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
    df_axiom_hits_metrics, df_axiom_hits_metrics_cv = process_metrics(df_axiom_hits_prediction)
    df_axiom_cellbased_binary_metrics, df_axiom_cellbased_binary_metrics_cv = process_metrics(df_axiom_cellbased_binary_prediction)
    df_axiom_cellfree_binary_metrics, df_axiom_cellfree_binary_metrics_cv = process_metrics(df_axiom_cellfree_binary_prediction)
    df_axiom_cytotox_binary_metrics, df_axiom_cytotox_binary_metrics_cv = process_metrics(df_axiom_cytotox_binary_prediction)
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
    mo.md(r"""# Annotate metrics""")
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


@app.cell
def _(mo):
    mo.md(r"""# Plot performance by dataset size""")
    return


@app.cell
def _(aes, geom_point, geom_smooth, ggplot, labs, pn, pred_pandas, theme_bw):
    pn.options.figure_size = (5, 5)

    _df_actual = pred_pandas[pred_pandas["Model_type"] == "Actual"]

    _plot_auroc_total = (
        ggplot(_df_actual, aes(x='Total_Compounds', y='AUROC')) 
        + geom_point(alpha=0.2) 
        + geom_smooth(method='lm', color='red', se=False)
        + labs(title='AUROC Scores', x='Total compound number', y='AUROC') 
        + theme_bw()
    )

    _plot_prauc_total = (
        ggplot(_df_actual, aes(x='Total_Compounds', y='PRAUC')) 
        + geom_point(alpha=0.2) 
        + geom_smooth(method='lm', color='red', se=False)
        + labs(title='PRAUC Scores', x='Total compound number', y='PRAUC') 
        + theme_bw()
    )

    _plot_auroc_percent = (
        ggplot(_df_actual, aes(x='Percent_Positive', y='AUROC')) 
        + geom_point(alpha=0.2) 
        + geom_smooth(method='lm', color='red', se=False)
        + labs(title='AUROC Scores', x='Percent positive', y='AUROC') 
        + theme_bw()
    )

    _plot_prauc_percent = (
        ggplot(_df_actual, aes(x='Percent_Positive', y='PRAUC')) 
        + geom_point(alpha=0.2) 
        + geom_smooth(method='lm', color='red', se=False)
        + labs(title='PRAUC Scores', x='Percent positive', y='PRAUC') 
        + theme_bw()
    )

    _plot_auroc_total.show()
    _plot_prauc_total.show()
    _plot_auroc_percent.show()
    _plot_prauc_percent.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Check significative performance""")
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
def _(df_metrics_cv, pd, stats):
    # Filter for the two models to compare
    df_comparison = df_metrics_cv[
        df_metrics_cv["Model_type"].isin(["Actual", "Random_baseline"])
    ].copy()

    # Pivot to get paired structure
    df_pivot_auroc = df_comparison.pivot_table(
        index=["Metadata_Label", "k_fold"],
        columns="Model_type",
        values="AUROC",
        aggfunc="first"
    ).reset_index()

    df_pivot_auprg = df_comparison.pivot_table(
        index=["Metadata_Label", "k_fold"],
        columns="Model_type",
        values="AUPRG",
        aggfunc="first"
    ).reset_index()

    # Dataset-level class counts (constant per label)
    df_counts = (
        df_comparison[df_comparison["Model_type"] == "Actual"]
        .groupby("Metadata_Label")[["Metadata_Count_0", "Metadata_Count_1"]]
        .first()
        .reset_index()
    )
    _counts_lookup = df_counts.set_index("Metadata_Label")

    # Perform paired t-test for each Metadata_Label
    results = []

    for label in df_pivot_auroc["Metadata_Label"].unique():
        auroc_data = df_pivot_auroc[df_pivot_auroc["Metadata_Label"] == label]
        auprg_data = df_pivot_auprg[df_pivot_auprg["Metadata_Label"] == label]

        t_stat_auroc, p_val_auroc = stats.ttest_rel(
            auroc_data["Actual"], auroc_data["Random_baseline"]
        )
        t_stat_auprg, p_val_auprg = stats.ttest_rel(
            auprg_data["Actual"], auprg_data["Random_baseline"]
        )

        mean_diff_auroc = auroc_data["Actual"].mean() - auroc_data["Random_baseline"].mean()
        mean_diff_auprg = auprg_data["Actual"].mean() - auprg_data["Random_baseline"].mean()

        results.append({
            "Metadata_Label":        label,
            "Metadata_Count_0":      _counts_lookup.loc[label, "Metadata_Count_0"],
            "Metadata_Count_1":      _counts_lookup.loc[label, "Metadata_Count_1"],
            "AUROC_mean_Actual":     auroc_data["Actual"].mean(),
            "AUROC_mean_Random_baseline":  auroc_data["Random_baseline"].mean(),
            "AUROC_diff":            mean_diff_auroc,
            "AUROC_t_stat":          t_stat_auroc,
            "AUROC_p_value":         p_val_auroc,
            "AUPRG_mean_Actual":     auprg_data["Actual"].mean(),
            "AUPRG_mean_Random_baseline":  auprg_data["Random_baseline"].mean(),
            "AUPRG_diff":            mean_diff_auprg,
            "AUPRG_t_stat":          t_stat_auprg,
            "AUPRG_p_value":         p_val_auprg,
            "n_folds":               len(auroc_data)
        })

    # Create results dataframe
    df_ttest_results = pd.DataFrame(results)

    # Add significance indicators
    df_ttest_results["AUROC_significant"] = df_ttest_results["AUROC_p_value"] < 0.05
    df_ttest_results["AUPRG_significant"] = df_ttest_results["AUPRG_p_value"] < 0.05

    # Sort by AUROC difference
    df_ttest_results = df_ttest_results.sort_values("AUROC_diff", ascending=False)

    print(f"\nPaired t-test results (n={df_ttest_results['n_folds'].iloc[0]} folds per label)")
    print(f"Comparing: Actual (Morphology) vs Random_baseline\n")
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
def _(df_ttest_results):
    df_ttest_results
    return


@app.cell
def _(df_ttest_results):
    list_significative_performance = df_ttest_results[
        (df_ttest_results["AUPRG_significant"]) &
        (df_ttest_results["AUROC_significant"]) &
        (df_ttest_results["AUROC_mean_Actual"] > df_ttest_results["AUROC_mean_Random_baseline"]) &
        (df_ttest_results["AUPRG_mean_Actual"] > df_ttest_results["AUPRG_mean_Random_baseline"]) &
        (df_ttest_results["Metadata_Count_1"] > 20) &
        (~df_ttest_results["Metadata_Label"].isin(["NVS_TR_hDAT"])) &
        (~df_ttest_results["Metadata_Label"].isin(["tissue__liver", "tissue__prostate"]))
    ]["Metadata_Label"]
    list_significative_performance
    return (list_significative_performance,)


@app.cell
def _(df_ttest_results):
    df_ttest_results[
        (df_ttest_results["AUPRG_significant"]) & 
        (df_ttest_results["AUROC_significant"]) &
        (df_ttest_results["AUROC_mean_Actual"] > df_ttest_results["AUROC_mean_Random_baseline"]) &
        (df_ttest_results["AUPRG_mean_Actual"] > df_ttest_results["AUPRG_mean_Random_baseline"]) &
        (df_ttest_results["Metadata_Count_1"] > 20) &
        (~df_ttest_results["Metadata_Label"].isin(["tissue__liver", "tissue__prostate"]))
    ].to_csv("modules/07_classification/outputs/results_performance.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""# Plot correlation models (Actual) significative""")
    return


@app.cell
def _(
    df_axiom_cellbased_binary_prediction,
    df_axiom_cellfree_binary_prediction,
    df_axiom_cytotox_binary_prediction,
    df_axiom_hits_prediction,
):
    df_axiom_hits_prediction 
    df_axiom_cellbased_binary_prediction
    df_axiom_cellfree_binary_prediction
    df_axiom_cytotox_binary_prediction
    return


@app.cell
def _(df_axiom_cytotox_binary_prediction):
    df_axiom_cytotox_binary_prediction
    return


@app.cell
def _():
    return


@app.cell
def _(
    df_axiom_cellbased_binary_prediction,
    df_axiom_cellfree_binary_prediction,
    df_axiom_cytotox_binary_prediction,
    df_axiom_hits_prediction,
    list_significative_performance,
    pd,
):
    # ── Step 1: Filter function ────────────────────────────────────────────────
    def _filter_df(_df):
        return _df[
            (_df["Model_type"] == "Actual") &
            (_df["Metadata_Label"].isin(list_significative_performance))
        ][["Metadata_compound", "repeat", "Metadata_Label", "y_prob"]].copy()


    # ── Step 2: Apply filtering ────────────────────────────────────────────────
    _df_hits      = _filter_df(df_axiom_hits_prediction)
    _df_cellbased = _filter_df(df_axiom_cellbased_binary_prediction)
    _df_cellfree  = _filter_df(df_axiom_cellfree_binary_prediction)
    _df_cytotox   = _filter_df(df_axiom_cytotox_binary_prediction)


    # ── Step 3: Concatenate all datasets ───────────────────────────────────────
    _df_all = pd.concat(
        [_df_hits, _df_cellbased, _df_cellfree, _df_cytotox],
        ignore_index=True
    )


    # ── Create combined index ────────────────────────────────────────────────
    _df_all["_compound_repeat"] = (
        _df_all["Metadata_compound"].astype(str) + "_r" + _df_all["repeat"].astype(str)
    )

    # ── Pivot ────────────────────────────────────────────────────────────────
    df_pivot_correlation = _df_all.pivot_table(
        index="_compound_repeat",
        columns="Metadata_Label",
        values="y_prob",
    )
    return (df_pivot_correlation,)


@app.cell
def _(df_pivot_correlation, os, plt, sns):
    # ── Step 5: Compute correlation matrix ─────────────────────────────────────
    _corr = df_pivot_correlation.corr(method="pearson")


    # ── Step 6: Clustered heatmap ──────────────────────────────────────────────
    n = len(_corr)
    cell_size = 0.4           # inches per cell — increase if labels still overlap
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

    # ── Label readability ──────────────────────────────────────────────────────

    plt.setp(_clustermap.ax_heatmap.get_xticklabels(),
             rotation=90, fontsize=16, ha="center")
    plt.setp(_clustermap.ax_heatmap.get_yticklabels(),
             rotation=0,  fontsize=16, va="center")

    _clustermap.fig.suptitle(
        "Clustered Correlation of Metadata_Labels (y_prob)",
        fontsize=14, fontweight="bold", y=1.01  # y>1 avoids overlap with dendro
    )
    plt.savefig(os.path.expanduser("~/Downloads/correlation_heatmap_axiom_batch.png"),
                 bbox_inches="tight", dpi=300)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Plot performance by Endpoint categories""")
    return


@app.cell
def _(
    aes,
    element_rect,
    element_text,
    facet_wrap,
    geom_boxplot,
    geom_jitter,
    ggplot,
    labs,
    pd,
    pn,
    pred_pandas,
    scale_x_discrete,
    theme,
    theme_classic,
    ylim,
):
    pn.options.figure_size = (10, 5)

    _custom_labels = {
        "axiom_cytotox": "Cytotox (LDH & MT & Cell Count)",
        "toxcast_cytotox": "ToxCast cytotox",
        "toxcast_cellbased": "ToxCast cell-based",
        "toxcast_cellfree": "ToxCast cell-free",
    }

    def _label_func(label_dict):
        return lambda x: label_dict.get(x, x)

    _df_filtered = pred_pandas.copy()

    _df_filtered["Endpoint_type"] = pd.Categorical(
        _df_filtered["Endpoint_type"],
        categories=[
            "axiom_cytotox",
            "toxcast_cytotox",
            "toxcast_cellbased",
            "toxcast_cellfree"
        ],
        ordered=True
    )
    # Apply custom labels using cat.rename_categories
    _df_filtered["Endpoint_type"] = _df_filtered["Endpoint_type"].cat.rename_categories(_custom_labels)

    _plot_auroc = (
        ggplot(_df_filtered, aes(x='Model_type', y='AUROC'))
        #+ geom_sina(color="grey")
        + geom_jitter(color="grey", alpha=0.6, width=0.2)
        + geom_boxplot(color="red", alpha=0.25, size=1)
        + facet_wrap("Endpoint_type", ncol=4)
        + labs(x='Model Type', y='AUROC')
        + theme_classic()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, color='black', size=14, family='Arial'),
            axis_text_y=element_text(color='black', size=14, family='Arial'),
            axis_title_x=element_text(size=14, color='black', weight='bold', family='Arial'),
            axis_title_y=element_text(size=14, color='black', weight='bold', family='Arial'),
            strip_text_x=element_text(size=14, color='black', family='Arial'),
            strip_background=element_rect(fill=None, color='white')
        )
        + scale_x_discrete(labels={
            "Actual": "Morphology",
            "Cellcount_baseline": "Cell count",
            "Random_baseline": "Random baseline"
        })
        + ylim(0, 1) 
    )

    _plot_prauc = (
        ggplot(_df_filtered, aes(x='Model_type', y='PRAUC'))
        #+ geom_sina(color="grey")
        + geom_jitter(color="grey", alpha=0.6, width=0.2)
        + geom_boxplot(color="red", alpha=0.25, size=1)
        + facet_wrap("Endpoint_type", ncol=4)
        + labs(x='Model Type', y='PRAUC')
        + theme_classic()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, color='black', size=14, family='Arial'),
            axis_text_y=element_text(color='black', size=14, family='Arial'),
            axis_title_x=element_text(size=14, color='black', weight='bold', family='Arial'),
            axis_title_y=element_text(size=14, color='black', weight='bold', family='Arial'),
            strip_text_x=element_text(size=14, color='black', family='Arial'),
            strip_background=element_rect(fill=None, color='white')
        )
        + scale_x_discrete(labels={
            "Actual": "Morphology",
            "Cellcount_baseline": "Cell count",
            "Random_baseline": "Random baseline"
        })
        + ylim(0, 1) 
    )

    _plot_auroc.show()
    _plot_prauc.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Plot performance by Endpoint categories significants""")
    return


@app.cell
def _(
    aes,
    element_rect,
    element_text,
    facet_wrap,
    geom_boxplot,
    geom_jitter,
    ggplot,
    labs,
    list_significative_performance,
    pd,
    pn,
    pred_pandas,
    scale_x_discrete,
    theme,
    theme_classic,
    ylim,
):
    pn.options.figure_size = (10, 5)

    _custom_labels = {
        "axiom_cytotox": "Cytotox (LDH & MT & Cell Count)",
        "toxcast_cytotox": "ToxCast cytotox",
        "toxcast_cellbased": "ToxCast cell-based",
        "toxcast_cellfree": "ToxCast cell-free",
    }

    # ── Significant labels ────────────────────────────────────────────────────────
    _sig_labels = list_significative_performance.to_list()

    # ── Filter pred_pandas to significant labels only ─────────────────────────────
    _df_filtered = pred_pandas[
        pred_pandas["Metadata_Label"].isin(_sig_labels)
    ].copy()

    _df_filtered["Endpoint_type"] = pd.Categorical(
        _df_filtered["Endpoint_type"],
        categories=["axiom_cytotox", "toxcast_cytotox", "toxcast_cellbased", "toxcast_cellfree"],
        ordered=True
    )
    _df_filtered["Endpoint_type"] = _df_filtered["Endpoint_type"].cat.rename_categories(_custom_labels)

    # ── Shared elements as a LIST ─────────────────────────────────────────────────
    _shared_layers = [
        geom_jitter(color="grey", alpha=0.6, width=0.2),
        geom_boxplot(color="red", alpha=0.25, size=1),
        facet_wrap("Endpoint_type", ncol=4),
        scale_x_discrete(labels={
            "Actual": "Morphology",
            "Cellcount_baseline": "Cell count",
            "Random_baseline": "Random baseline"
        }),
        ylim(0, 1),
        theme_classic(),
        theme(
            axis_text_x=element_text(angle=45, hjust=1, color='black', size=14, family='Arial'),
            axis_text_y=element_text(color='black', size=14, family='Arial'),
            axis_title_x=element_text(size=14, color='black', weight='bold', family='Arial'),
            axis_title_y=element_text(size=14, color='black', weight='bold', family='Arial'),
            strip_text_x=element_text(size=14, color='black', family='Arial'),
            strip_background=element_rect(fill=None, color='white')
        ),
    ]

    # ── AUROC plot ────────────────────────────────────────────────────────────────
    _plot_auroc = (
        ggplot(_df_filtered, aes(x='Model_type', y='AUROC'))
        + _shared_layers
        + labs(x='Model Type', y='AUROC')
    )

    # ── PRAUC plot ────────────────────────────────────────────────────────────────
    _plot_prauc = (
        ggplot(_df_filtered, aes(x='Model_type', y='PRAUC'))
        + _shared_layers
        + labs(x='Model Type', y='PRAUC')
    )
    # ── AUPRG plot ────────────────────────────────────────────────────────────────
    _plot_auprg = (
        ggplot(_df_filtered, aes(x='Model_type', y='AUPRG'))
        + _shared_layers
        + ylim(-0.2, 0.75)
        + labs(x='Model Type', y='AUPRG')
    )

    _plot_auroc.show()
    _plot_auprg.show()
    _plot_prauc.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Plot performance by Endpoint categories (Filter on compound number > 100)""")
    return


@app.cell
def _(
    aes,
    element_rect,
    element_text,
    facet_wrap,
    geom_boxplot,
    geom_jitter,
    ggplot,
    labs,
    pd,
    pn,
    pred_pandas,
    scale_x_discrete,
    theme,
    theme_classic,
    ylim,
):
    pn.options.figure_size = (10, 5)

    _custom_labels = {
        "axiom_cytotox": "Cytotox (LDH & MT & Cell Count)",
        "toxcast_cytotox": "ToxCast cytotox",
        "toxcast_cellbased": "ToxCast cell-based",
        "toxcast_cellfree": "ToxCast cell-free",
    }

    def _label_func(label_dict):
        return lambda x: label_dict.get(x, x)

    _df_filtered = pred_pandas[pred_pandas["Total_Compounds"] > 100].copy()

    _df_filtered["Endpoint_type"] = pd.Categorical(
        _df_filtered["Endpoint_type"],
        categories=[
            "axiom_cytotox",
            "toxcast_cytotox",
            "toxcast_cellbased",
            "toxcast_cellfree"
        ],
        ordered=True
    )
    # Apply custom labels using cat.rename_categories
    _df_filtered["Endpoint_type"] = _df_filtered["Endpoint_type"].cat.rename_categories(_custom_labels)

    _plot_auroc = (
        ggplot(_df_filtered, aes(x='Model_type', y='AUROC'))
        #+ geom_sina(color="grey")
        + geom_jitter(color="grey", alpha=0.6, width=0.2)
        + geom_boxplot(color="red", alpha=0.25, size=1)
        + facet_wrap("Endpoint_type", ncol=4)
        + labs(x='Model Type', y='AUROC')
        + theme_classic()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, color='black', size=14, family='Arial'),
            axis_text_y=element_text(color='black', size=14, family='Arial'),
            axis_title_x=element_text(size=14, color='black', weight='bold', family='Arial'),
            axis_title_y=element_text(size=14, color='black', weight='bold', family='Arial'),
            strip_text_x=element_text(size=14, color='black', family='Arial'),
            strip_background=element_rect(fill=None, color='white')
        )
        + scale_x_discrete(labels={
            "Actual": "Morphology",
            "Cellcount_baseline": "Cell count",
            "Random_baseline": "Random baseline"
        })
        + ylim(0, 1) 
    )

    _plot_prauc = (
        ggplot(_df_filtered, aes(x='Model_type', y='PRAUC'))
        #+ geom_sina(color="grey")
        + geom_jitter(color="grey", alpha=0.6, width=0.2)
        + geom_boxplot(color="red", alpha=0.25, size=1)
        + facet_wrap("Endpoint_type", ncol=4)
        + labs(x='Model Type', y='PRAUC')
        + theme_classic()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, color='black', size=14, family='Arial'),
            axis_text_y=element_text(color='black', size=14, family='Arial'),
            axis_title_x=element_text(size=14, color='black', weight='bold', family='Arial'),
            axis_title_y=element_text(size=14, color='black', weight='bold', family='Arial'),
            strip_text_x=element_text(size=14, color='black', family='Arial'),
            strip_background=element_rect(fill=None, color='white')
        )
        + scale_x_discrete(labels={
            "Actual": "Morphology",
            "Cellcount_baseline": "Cell count",
            "Random_baseline": "Random baseline"
        })
        + ylim(0, 1) 
    )

    _plot_prauc = (
        ggplot(_df_filtered, aes(x='Model_type', y='PRAUC'))
        #+ geom_sina(color="grey")
        + geom_jitter(color="grey", alpha=0.6, width=0.2)
        + geom_boxplot(color="red", alpha=0.25, size=1)
        + facet_wrap("Endpoint_type", ncol=4)
        + labs(x='Model Type', y='PRAUC')
        + theme_classic()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, color='black', size=14, family='Arial'),
            axis_text_y=element_text(color='black', size=14, family='Arial'),
            axis_title_x=element_text(size=14, color='black', weight='bold', family='Arial'),
            axis_title_y=element_text(size=14, color='black', weight='bold', family='Arial'),
            strip_text_x=element_text(size=14, color='black', family='Arial'),
            strip_background=element_rect(fill=None, color='white')
        )
        + scale_x_discrete(labels={
            "Actual": "Morphology",
            "Cellcount_baseline": "Cell count",
            "Random_baseline": "Random baseline"
        })
        + ylim(0, 1) 
    )

    _plot_auroc.show()
    _plot_prauc.show()
    return


@app.cell
def _(pred_pandas):
    pred_pandas
    return


@app.cell
def _(mo):
    mo.md(r"""# Barplot performance significative""")
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
def _(
    df_axiom_cellbased_binary_metrics_cv,
    df_axiom_cellfree_binary_metrics_cv,
    df_axiom_cytotox_binary_metrics_cv,
    df_axiom_hits_metrics_cv,
    df_metrics_cv,
    df_ttest_results,
    list_significative_performance,
    mpatches,
    np,
    os,
    plt,
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
        .groupby(["Metadata_Label", "Model_type"])["AUROC"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    # ── 5. Style config ───────────────────────────────────────────────────────────
    _SOURCE_COLORS = {
        "Hits":       "#4C72B0",
        "Cell-based": "#DD8452",
        "Cell-free":  "#55A868",
        "Cytotox":    "#C44E52",
    }
    _MODEL_HATCHES = {
        "Actual":             "",
        "Cellcount_baseline": "///",
        "Random_baseline":    "xxx",
    }
    _MODEL_ALPHA = {
        "Actual":             1.0,
        "Cellcount_baseline": 0.65,
        "Random_baseline":    0.40,
    }
    _sig_lookup = df_ttest_results.set_index("Metadata_Label")

    # ── 6. Build per-source label lists sorted by Actual AUROC ───────────────────
    _sources_present = [_s for _s in _source_dfs if any(
        _source_map.get(_l) == _s for _l in _sig_labels
    )]
    _n_sources = len(_sources_present)

    _source_labels = {}
    for _src in _sources_present:
        _lbls = [_l for _l in _sig_labels if _source_map.get(_l) == _src]
        _actual = _df_agg[
            (_df_agg["Model_type"] == "Actual") &
            (_df_agg["Metadata_Label"].isin(_lbls))
        ].sort_values("mean", ascending=True)
        _source_labels[_src] = _actual["Metadata_Label"].tolist()

    # ── 7. Figure layout — width shared, height per subplot ∝ label count ────────
    # ── 7. Figure layout — paired axes (bar | counts) per source ─────────────────
    _bar_w    = 0.25
    _row_h    = 0.45
    _fig_w    = 13        # wider to accommodate counts panel

    _heights  = [max(2, len(_source_labels[_s]) * _row_h) for _s in _sources_present]

    _fig = plt.figure(figsize=(_fig_w, sum(_heights)))
    _outer_gs = _fig.add_gridspec(
        _n_sources, 1,
        height_ratios=_heights,
        hspace=0.35
    )

    _counts_lookup = df_ttest_results.set_index("Metadata_Label")

    _axes_bar    = []
    _axes_counts = []

    for _si, _src in enumerate(_sources_present):
        _inner_gs = _outer_gs[_si].subgridspec(
            1, 2,
            width_ratios=[3, 1],   # AUROC bar | counts
            wspace=0.05
        )
        _ax_bar    = _fig.add_subplot(_inner_gs[0])
        _ax_counts = _fig.add_subplot(_inner_gs[1])
        _axes_bar.append(_ax_bar)
        _axes_counts.append(_ax_counts)

    # ── 8. Draw each subplot ──────────────────────────────────────────────────────
    for _si, (_ax, _ax_c, _src) in enumerate(zip(_axes_bar, _axes_counts, _sources_present)):
        _label_order = _source_labels[_src]
        _n           = len(_label_order)
        _y           = np.arange(_n)
        _color       = _SOURCE_COLORS[_src]

        # ── AUROC bars ────────────────────────────────────────────────────────────
        for _i, _model in enumerate(_MODELS):
            _subset = _df_agg[_df_agg["Model_type"] == _model].set_index("Metadata_Label")
            _means  = [_subset.loc[_l, "mean"] if _l in _subset.index else np.nan for _l in _label_order]
            _stds   = [_subset.loc[_l, "std"]  if _l in _subset.index else 0      for _l in _label_order]

            _ax.barh(
                _y + (_i - 1) * _bar_w, _means,
                height=_bar_w, color=_color,
                alpha=_MODEL_ALPHA[_model],
                hatch=_MODEL_HATCHES[_model],
                edgecolor="white", linewidth=0.5,
                xerr=_stds, capsize=2,
                error_kw=dict(elinewidth=0.7, ecolor="black", capthick=0.7),
                zorder=3,
            )

        # Significance stars
        for _yi, _label in enumerate(_label_order):
            if _label not in _sig_lookup.index:
                continue
            _p_val = _sig_lookup.loc[_label, "AUROC_p_value"]
            _stars = "***" if _p_val < 0.001 else "**" if _p_val < 0.01 else "*"
            _x_right = (
                _df_agg[_df_agg["Metadata_Label"] == _label]["mean"].max() +
                _df_agg[_df_agg["Metadata_Label"] == _label]["std"].max() + 0.02
            )
            _ax.text(_x_right, _yi, _stars, ha="left", va="center",
                     fontsize=9, fontweight="bold", color="black")

        _ax.set_yticks(_y)
        _ax.set_yticklabels(_label_order, fontsize=8)
        _ax.set_title(_src, fontsize=11, fontweight="bold",
                      color=_color, loc="left", pad=4)
        _ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, zorder=1)
        _ax.set_xlim(0, 1.12)
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

        if _src == _sources_present[-1]:
            _ax.set_xlabel("AUROC (mean ± std across folds)", fontsize=10)
        else:
            _ax.set_xticklabels([])

        # ── Counts subplot ────────────────────────────────────────────────────────
        for _yi, _label in enumerate(_label_order):
            if _label not in _counts_lookup.index:
                continue
            _n0 = int(_counts_lookup.loc[_label, "Metadata_Count_0"])   # inactive
            _n1 = int(_counts_lookup.loc[_label, "Metadata_Count_1"])   # active
            _tot = _n0 + _n1

            # Stacked bar: inactive (light) | active (solid)
            _ax_c.barh(_yi, _n0, height=0.55, color=_color, alpha=0.25,
                       edgecolor="none", zorder=3)
            _ax_c.barh(_yi, _n1, height=0.55, left=_n0, color=_color, alpha=0.85,
                       edgecolor="none", zorder=3)

            # Annotate: "act / inact (tot)"
            _ax_c.text(
                _tot + _tot * 0.03, _yi,
                f"{_n1} / {_n0}",
                va="center", ha="left", fontsize=7, color="#333333"
            )

        _ax_c.set_yticks(_y)
        _ax_c.set_yticklabels([])                    # shared y with bar plot
        _ax_c.set_xlim(0, _ax_c.get_xlim()[1] * 1.5)
        _ax_c.spines[["top", "right", "left"]].set_visible(False)
        _ax_c.tick_params(axis="y", length=0)
        _ax_c.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)

        if _src == _sources_present[-1]:
            _ax_c.set_xlabel("N compounds", fontsize=9)
        else:
            _ax_c.set_xticklabels([])

        # Column header on first panel only
        if _si == 0:
            _ax_c.set_title("active / inactive", fontsize=8,
                            color="grey", loc="center", pad=4)

    # ── 9. Shared legend ──────────────────────────────────────────────────────────
    _model_patches = [
        mpatches.Patch(facecolor="grey", hatch=_MODEL_HATCHES[_m],
                       alpha=_MODEL_ALPHA[_m], edgecolor="black", label=_m)
        for _m in _MODELS
    ]
    # Add count legend
    _count_patches = [
        mpatches.Patch(facecolor="grey", alpha=0.85, edgecolor="none", label="Active"),
        mpatches.Patch(facecolor="grey", alpha=0.25, edgecolor="none", label="Inactive"),
    ]
    _fig.legend(
        handles=_model_patches + _count_patches,
        title="Model / Class",
        loc="upper right", fontsize=8, title_fontsize=9,
        bbox_to_anchor=(1.02, 1.0)
    )

    _fig.suptitle(
        f"AUROC — significant labels (both AUROC & PRAUC, n={len(_sig_labels)})\n"
        "(* p<0.05, ** p<0.01, *** p<0.001  |  Actual vs Cellcount_baseline)",
        fontsize=11, y=1.01
    )

    plt.savefig(os.path.expanduser('~/Downloads/auroc_by_source.png'), dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _():
    import math
    return (math,)


@app.cell
def _(
    df_axiom_cellbased_binary_metrics_cv,
    df_axiom_cellfree_binary_metrics_cv,
    df_axiom_cytotox_binary_metrics_cv,
    df_axiom_hits_metrics_cv,
    df_metrics_cv,
    df_ttest_results,
    list_significative_performance,
    math,
    mpatches,
    np,
    os,
    plt,
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
        .groupby(["Metadata_Label", "Model_type"])["AUROC"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    # ── 5. Style config ───────────────────────────────────────────────────────────
    _SOURCE_COLORS = {
        "Hits":       "#4C72B0",
        "Cell-based": "#DD8452",
        "Cell-free":  "#55A868",
        "Cytotox":    "#C44E52",
    }
    _sig_lookup    = df_ttest_results.set_index("Metadata_Label")
    _counts_lookup = df_ttest_results.set_index("Metadata_Label")

    # ── 6. Single label list sorted by Actual AUROC, grouped by source ────────────
    _sig_labels_ordered = []
    for _src in _source_dfs:
        _lbls = [_l for _l in _sig_labels if _source_map.get(_l) == _src]
        _actual = (
            _df_agg[
                (_df_agg["Model_type"] == "Actual") &
                (_df_agg["Metadata_Label"].isin(_lbls))
            ]
            .sort_values("mean", ascending=True)
        )
        _sig_labels_ordered.extend(_actual["Metadata_Label"].tolist())

    # ── 7. Split into two panels (left / right) ───────────────────────────────────
    _n_lbls  = len(_sig_labels_ordered)
    _half    = math.ceil(_n_lbls / 2)
    _panels  = [_sig_labels_ordered[:_half], _sig_labels_ordered[_half:]]

    # ── 8. Figure: 2 columns, each column = AUROC bar + counts ───────────────────
    _row_h = 0.38
    _fig_w = 22
    _fig_h = max(4, _half * _row_h)

    _fig = plt.figure(figsize=(_fig_w, _fig_h))
    _outer_gs = _fig.add_gridspec(1, 2, wspace=0.35)   # two main columns

    _all_axes = []   # (ax_bar, ax_counts) per panel
    for _pi in range(2):
        _inner_gs = _outer_gs[_pi].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
        _all_axes.append((
            _fig.add_subplot(_inner_gs[0]),
            _fig.add_subplot(_inner_gs[1]),
        ))

    # ── 9. Draw each panel ────────────────────────────────────────────────────────
    _subset_actual = _df_agg[_df_agg["Model_type"] == "Actual"].set_index("Metadata_Label")

    for _pi, ((_ax, _ax_c), _label_list) in enumerate(zip(_all_axes, _panels)):
        _n   = len(_label_list)
        _y   = np.arange(_n)

        # ── AUROC bars ────────────────────────────────────────────────────────────
        for _yi, _label in enumerate(_label_list):
            _src   = _source_map.get(_label, "Hits")
            _color = _SOURCE_COLORS[_src]
            _mean  = _subset_actual.loc[_label, "mean"] if _label in _subset_actual.index else np.nan
            _std   = _subset_actual.loc[_label, "std"]  if _label in _subset_actual.index else 0

            _ax.barh(
                _yi, _mean,
                height=0.65, color=_color,
                edgecolor="white", linewidth=0.4,
                xerr=_std, capsize=2,
                error_kw=dict(elinewidth=0.7, ecolor="#555555", capthick=0.7),
                zorder=3,
            )

            if _label in _sig_lookup.index:
                _p_val = _sig_lookup.loc[_label, "AUROC_p_value"]
                _stars = "***" if _p_val < 0.001 else "**" if _p_val < 0.01 else "*"
                _ax.text(
                    _mean + _std + 0.02, _yi, _stars,
                    ha="left", va="center", fontsize=8, fontweight="bold", color="black"
                )

        # ── Source group bands + separators ──────────────────────────────────────
        _src_first = {}
        _src_last  = {}
        for _yi, _label in enumerate(_label_list):
            _src = _source_map.get(_label)
            if _src not in _src_first:
                _src_first[_src] = _yi
            _src_last[_src] = _yi

        for _src, _first in _src_first.items():
            _last  = _src_last[_src]
            _color = _SOURCE_COLORS[_src]
            _ax.axhspan(_first - 0.5, _last + 0.5, color=_color, alpha=0.06, zorder=0)
            _ax_c.axhspan(_first - 0.5, _last + 0.5, color=_color, alpha=0.06, zorder=0)
            _ax.text(
                1.13, (_first + _last) / 2, _src,
                va="center", ha="left", fontsize=8, fontweight="bold",
                color=_color, transform=_ax.get_yaxis_transform()
            )
            if _first > 0:
                _ax.axhline(_first - 0.5, color="grey", linewidth=0.6, linestyle="--", zorder=1)
                _ax_c.axhline(_first - 0.5, color="grey", linewidth=0.6, linestyle="--", zorder=1)

        # ── AUROC axis formatting ─────────────────────────────────────────────────
        _ax.set_yticks(_y)
        _ax.set_yticklabels(_label_list, fontsize=7.5)
        _ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, zorder=1)
        _ax.set_xlim(0, 1.15)
        _ax.set_ylim(-0.5, _n - 0.5)
        _ax.invert_yaxis()
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        _ax.set_xlabel("AUROC (mean ± std across folds)", fontsize=9)
        _ax.set_title(f"Actual model — panel {_pi + 1}", fontsize=11,
                      fontweight="bold", pad=6)

        # ── Counts subplot ────────────────────────────────────────────────────────
        for _yi, _label in enumerate(_label_list):
            if _label not in _counts_lookup.index:
                continue
            _src   = _source_map.get(_label, "Hits")
            _color = _SOURCE_COLORS[_src]
            _n0    = int(_counts_lookup.loc[_label, "Metadata_Count_0"])
            _n1    = int(_counts_lookup.loc[_label, "Metadata_Count_1"])
            _tot   = _n0 + _n1

            _ax_c.barh(_yi, _n0, height=0.55, color=_color, alpha=0.25, edgecolor="none", zorder=3)
            _ax_c.barh(_yi, _n1, height=0.55, left=_n0, color=_color, alpha=0.85, edgecolor="none", zorder=3)
            _ax_c.text(
                _tot * 1.03, _yi, f"{_n1} / {_n0}",
                va="center", ha="left", fontsize=7, color="#333333"
            )

        _ax_c.set_yticks(_y)
        _ax_c.set_yticklabels([])
        _ax_c.set_xlim(0, _ax_c.get_xlim()[1] * 1.6)
        _ax_c.set_ylim(-0.5, _n - 0.5)
        _ax_c.invert_yaxis()
        _ax_c.spines[["top", "right", "left"]].set_visible(False)
        _ax_c.tick_params(axis="y", length=0)
        _ax_c.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
        _ax_c.set_xlabel("N compounds", fontsize=9)
        _ax_c.set_title("active / inactive", fontsize=9, color="grey", pad=6)

    # ── 10. Shared legend ─────────────────────────────────────────────────────────
    _src_patches = [
        mpatches.Patch(facecolor=_SOURCE_COLORS[_s], edgecolor="none", label=_s)
        for _s in _source_dfs
    ]
    _count_patches = [
        mpatches.Patch(facecolor="grey", alpha=0.85, edgecolor="none", label="Active"),
        mpatches.Patch(facecolor="grey", alpha=0.25, edgecolor="none", label="Inactive"),
    ]
    _fig.legend(
        handles=_src_patches + _count_patches,
        title="Source / Class", loc="upper right",
        fontsize=8, title_fontsize=9,
        bbox_to_anchor=(1.01, 1.0)
    )

    _fig.suptitle(
        f"AUROC — Actual model, significant labels (n={len(_sig_labels_ordered)})\n"
        "(* p<0.05, ** p<0.01, *** p<0.001  |  AUROC & PRAUC significant, n_active > 20)",
        fontsize=11, y=1.01
    )

    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/Downloads/classification_auroc_by_source.png"), dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(
    df_axiom_cellbased_binary_metrics_cv,
    df_axiom_cellfree_binary_metrics_cv,
    df_axiom_cytotox_binary_metrics_cv,
    df_axiom_hits_metrics_cv,
    df_metrics_cv,
    df_ttest_results,
    dict_categores_significative_cell_based_claudeai,
    list_significative_performance,
    math,
    mpatches,
    np,
    os,
    plt,
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

    # ── Build cell-based category lookup ─────────────────────────────────────────
    _cellbased_cat_map   = {}  # label -> category label
    _cellbased_cat_color = {}  # label -> category color
    for _cat in dict_categores_significative_cell_based_claudeai["categories"]:
        for _assay in _cat["assays"]:
            _cellbased_cat_map[_assay]   = _cat["label"]
            _cellbased_cat_color[_assay] = _cat["color"]

    # ── 2. Significant labels ─────────────────────────────────────────────────────
    _sig_labels = list_significative_performance.to_list()

    # ── 3. Filter metrics ─────────────────────────────────────────────────────────
    _MODELS  = ["Actual", "Cellcount_baseline", "Random_baseline"]
    _df_plot = df_metrics_cv[
        df_metrics_cv["Model_type"].isin(_MODELS) &
        df_metrics_cv["Metadata_Label"].isin(_sig_labels)
    ].copy()
    _df_plot["Source"] = _df_plot["Metadata_Label"].map(_source_map)

    # ── 4. Compute mean ± std for both metrics ────────────────────────────────────
    _df_agg = (
        _df_plot
        .groupby(["Metadata_Label", "Model_type"])[["AUROC", "AUPRG"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    _df_agg.columns = ["Metadata_Label", "Model_type",
                       "AUROC_mean", "AUROC_std",
                       "AUPRG_mean", "AUPRG_std"]

    # ── 5. Style config ───────────────────────────────────────────────────────────
    _SOURCE_COLORS = {
        "Hits":       "#4C72B0",
        "Cell-based": "#DD8452",
        "Cell-free":  "#55A868",
        "Cytotox":    "#C44E52",
    }
    _sig_lookup    = df_ttest_results.set_index("Metadata_Label")
    _counts_lookup = df_ttest_results.set_index("Metadata_Label")

    # ── 6. Build ordered label list ───────────────────────────────────────────────
    # Cell-based: group by claudeai category, sorted by AUROC within each category
    # Other sources: sorted by AUROC within source
    _subset_actual = _df_agg[_df_agg["Model_type"] == "Actual"].set_index("Metadata_Label")

    _sig_labels_ordered = []
    for _src in _source_dfs:
        _lbls = [_l for _l in _sig_labels if _source_map.get(_l) == _src]
        if _src == "Cell-based":
            # Group by claudeai category
            _cat_order = [_cat["label"] for _cat in dict_categores_significative_cell_based_claudeai["categories"]]
            for _cat_label in _cat_order:
                _cat_lbls = [_l for _l in _lbls if _cellbased_cat_map.get(_l) == _cat_label]
                if not _cat_lbls:
                    continue
                _actual = (
                    _subset_actual.loc[_subset_actual.index.isin(_cat_lbls)]
                    .sort_values("AUROC_mean", ascending=True)
                )
                _sig_labels_ordered.extend(_actual.index.tolist())
            # Append any cell-based not in any category
            _uncategorized = [_l for _l in _lbls if _l not in _cellbased_cat_map]
            if _uncategorized:
                _actual = (
                    _subset_actual.loc[_subset_actual.index.isin(_uncategorized)]
                    .sort_values("AUROC_mean", ascending=True)
                )
                _sig_labels_ordered.extend(_actual.index.tolist())
        else:
            _actual = (
                _subset_actual.loc[_subset_actual.index.isin(_lbls)]
                .sort_values("AUROC_mean", ascending=True)
            )
            _sig_labels_ordered.extend(_actual.index.tolist())

    # ── 7. Split into two panels ──────────────────────────────────────────────────
    _n_lbls = len(_sig_labels_ordered)
    _half   = math.ceil(_n_lbls / 2)
    _panels = [_sig_labels_ordered[:_half], _sig_labels_ordered[_half:]]

    # ── 8. Figure: 2 columns, each = AUROC bar + AUPRG bar + counts ──────────────
    _row_h = 0.38
    _fig_w = 28
    _fig_h = max(4, _half * _row_h)

    _fig = plt.figure(figsize=(_fig_w, _fig_h))
    _outer_gs = _fig.add_gridspec(1, 2, wspace=0.4)

    _all_axes = []
    for _pi in range(2):
        _inner_gs = _outer_gs[_pi].subgridspec(1, 3, width_ratios=[3, 3, 1], wspace=0.05)
        _all_axes.append((
            _fig.add_subplot(_inner_gs[0]),  # AUROC
            _fig.add_subplot(_inner_gs[1]),  # AUPRG
            _fig.add_subplot(_inner_gs[2]),  # counts
        ))

    # ── 9. Draw each panel ────────────────────────────────────────────────────────
    def _get_stars(_label, _col):
        if _label not in _sig_lookup.index:
            return ""
        _p = _sig_lookup.loc[_label, _col]
        return "***" if _p < 0.001 else "**" if _p < 0.01 else "*"

    for _pi, ((_ax_auroc, _ax_auprg, _ax_c), _label_list) in enumerate(zip(_all_axes, _panels)):
        _n = len(_label_list)
        _y = np.arange(_n)

        for _yi, _label in enumerate(_label_list):
            _src = _source_map.get(_label, "Hits")

            # Bar color: claudeai category color for cell-based, source color otherwise
            if _src == "Cell-based" and _label in _cellbased_cat_color:
                _bar_color = _cellbased_cat_color[_label]
            else:
                _bar_color = _SOURCE_COLORS[_src]

            if _label not in _subset_actual.index:
                continue
            _row = _subset_actual.loc[_label]

            # ── AUROC bar ─────────────────────────────────────────────────────────
            _ax_auroc.barh(
                _yi, _row["AUROC_mean"], height=0.65,
                color=_bar_color, edgecolor="white", linewidth=0.4,
                xerr=_row["AUROC_std"], capsize=2,
                error_kw=dict(elinewidth=0.7, ecolor="#555555", capthick=0.7),
                zorder=3,
            )
            _stars = _get_stars(_label, "AUROC_p_value")
            if _stars:
                _ax_auroc.text(
                    _row["AUROC_mean"] + _row["AUROC_std"] + 0.02, _yi, _stars,
                    ha="left", va="center", fontsize=8, fontweight="bold"
                )

            # ── AUPRG bar ─────────────────────────────────────────────────────────
            _ax_auprg.barh(
                _yi, _row["AUPRG_mean"], height=0.65,
                color=_bar_color, edgecolor="white", linewidth=0.4,
                xerr=_row["AUPRG_std"], capsize=2,
                error_kw=dict(elinewidth=0.7, ecolor="#555555", capthick=0.7),
                zorder=3,
            )
            #_stars_p = _get_stars(_label, "PRAUC_p_value")
            #if _stars_p:
            #    _ax_auprg.text(
            #        _row["AUPRG_mean"] + _row["AUPRG_std"] + 0.02, _yi, _stars_p,
            #        ha="left", va="center", fontsize=8, fontweight="bold"
            #    )

        # ── Source/category group bands + separators ──────────────────────────────
        # Track group boundaries: for cell-based use claudeai category, else source
        _group_first = {}
        _group_last  = {}
        _group_color = {}
        _group_name  = {}
        for _yi, _label in enumerate(_label_list):
            _src = _source_map.get(_label)
            if _src == "Cell-based":
                _grp = _cellbased_cat_map.get(_label, "Cell-based (uncategorized)")
                _gc  = _cellbased_cat_color.get(_label, _SOURCE_COLORS["Cell-based"])
            else:
                _grp = _src
                _gc  = _SOURCE_COLORS.get(_src, "#aaaaaa")
            if _grp not in _group_first:
                _group_first[_grp] = _yi
                _group_color[_grp] = _gc
                _group_name[_grp]  = _grp
            _group_last[_grp] = _yi

        for _grp, _first in _group_first.items():
            _last  = _group_last[_grp]
            _gc    = _group_color[_grp]
            for _axx in [_ax_auroc, _ax_auprg, _ax_c]:
                _axx.axhspan(_first - 0.5, _last + 0.5, color=_gc, alpha=0.07, zorder=0)
                if _first > 0:
                    _axx.axhline(_first - 0.5, color="grey", linewidth=0.6, linestyle="--", zorder=1)
            # Label on right of AUPRG
            _ax_auprg.text(
                1.13, (_first + _last) / 2,
                _grp.replace("Endocrine Disruption — ", "ED — ")
                    .replace("Hepatotoxicity — ", "Hepato — ")
                    .replace("Cardiovascular / ", "CV / "),
                va="center", ha="left", fontsize=7, fontweight="bold",
                color=_gc, transform=_ax_auprg.get_yaxis_transform()
            )

        # ── AUROC axis ────────────────────────────────────────────────────────────
        _ax_auroc.set_yticks(_y)
        _ax_auroc.set_yticklabels(_label_list, fontsize=7.5)
        _ax_auroc.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, zorder=1)
        _ax_auroc.set_xlim(0, 1.15)
        _ax_auroc.set_ylim(-0.5, _n - 0.5)
        _ax_auroc.invert_yaxis()
        _ax_auroc.spines[["top", "right"]].set_visible(False)
        _ax_auroc.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        _ax_auroc.set_xlabel("AUROC (mean ± std)", fontsize=9)
        _ax_auroc.set_title(f"AUROC — panel {_pi + 1}", fontsize=11, fontweight="bold", pad=6)

        # ── AUPRG axis ────────────────────────────────────────────────────────────
        _ax_auprg.set_yticks(_y)
        _ax_auprg.set_yticklabels([])
        _ax_auprg.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, zorder=1)
        _ax_auprg.set_xlim(0, 1.15)
        _ax_auprg.set_ylim(-0.5, _n - 0.5)
        _ax_auprg.invert_yaxis()
        _ax_auprg.spines[["top", "right"]].set_visible(False)
        _ax_auprg.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        _ax_auprg.set_xlabel("AUPRG (mean ± std)", fontsize=9)
        _ax_auprg.set_title(f"AUPRG — panel {_pi + 1}", fontsize=11, fontweight="bold", pad=6)

        # ── Counts subplot ────────────────────────────────────────────────────────
        for _yi, _label in enumerate(_label_list):
            if _label not in _counts_lookup.index:
                continue
            _src   = _source_map.get(_label, "Hits")
            _color = _SOURCE_COLORS[_src]
            _n0    = int(_counts_lookup.loc[_label, "Metadata_Count_0"])
            _n1    = int(_counts_lookup.loc[_label, "Metadata_Count_1"])
            _tot   = _n0 + _n1
            _ax_c.barh(_yi, _n0, height=0.55, color=_color, alpha=0.25, edgecolor="none", zorder=3)
            _ax_c.barh(_yi, _n1, height=0.55, left=_n0, color=_color, alpha=0.85, edgecolor="none", zorder=3)
            _ax_c.text(
                _tot * 1.03, _yi, f"{_n1} / {_n0}",
                va="center", ha="left", fontsize=7, color="#333333"
            )

        _ax_c.set_yticks(_y)
        _ax_c.set_yticklabels([])
        _ax_c.set_xlim(0, _ax_c.get_xlim()[1] * 1.6)
        _ax_c.set_ylim(-0.5, _n - 0.5)
        _ax_c.invert_yaxis()
        _ax_c.spines[["top", "right", "left"]].set_visible(False)
        _ax_c.tick_params(axis="y", length=0)
        _ax_c.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
        _ax_c.set_xlabel("N compounds", fontsize=9)
        _ax_c.set_title("active / inactive", fontsize=9, color="grey", pad=6)

    # ── 10. Shared legend ─────────────────────────────────────────────────────────
    _src_patches = [
        mpatches.Patch(facecolor=_SOURCE_COLORS[_s], edgecolor="none", label=_s)
        for _s in ["Hits", "Cell-free", "Cytotox"]
    ]
    _cat_patches = [
        mpatches.Patch(facecolor=_cat["color"], edgecolor="none", label=_cat["label"])
        for _cat in dict_categores_significative_cell_based_claudeai["categories"]
    ]
    _count_patches = [
        mpatches.Patch(facecolor="grey", alpha=0.85, edgecolor="none", label="Active"),
        mpatches.Patch(facecolor="grey", alpha=0.25, edgecolor="none", label="Inactive"),
    ]
    _fig.legend(
        handles=_src_patches + _cat_patches + _count_patches,
        title="Source / Category / Class",
        loc="upper right",
        fontsize=8, title_fontsize=9,
        bbox_to_anchor=(1.01, 1.0),
        ncol=2,
    )

    _fig.suptitle(
        f"AUROC & AUPRG — Actual model, significant labels (n={len(_sig_labels_ordered)})\n"
        "(* p<0.05, ** p<0.01, *** p<0.001  |  AUROC & PRAUC significant, n_active > 20)",
        fontsize=11, y=1.01
    )

    plt.tight_layout()
    plt.savefig(
        os.path.expanduser("~/Downloads/classification_auroc_auprg_by_source_category.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Plot comparison AUPRG AUC""")
    return


@app.cell
def _(list_significative_performance):
    list_significative_performance
    return


@app.cell
def _(
    Line2D,
    Path,
    df_axiom_cellbased_binary_metrics_cv,
    df_axiom_cytotox_binary_metrics_cv,
    df_axiom_hits_metrics_cv,
    df_metrics_cv,
    list_significative_performance,
    mpatches,
    plt,
):
    from adjustText import adjust_text
    # ── 1. Build source-lookup ────────────────────────────────────────────────────
    _source_map = {}
    _source_dfs = {
        "Hits":       df_axiom_hits_metrics_cv,
        "Cell-based": df_axiom_cellbased_binary_metrics_cv,
        #"Cell-free":  df_axiom_cellfree_binary_metrics_cv,
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
        df_metrics_cv["Model_type"].isin(_MODELS) #&
        #df_metrics_cv["Metadata_Label"].isin(_sig_labels)
    ].copy()
    _df_plot["Source"] = _df_plot["Metadata_Label"].map(_source_map)

    # ── 1. Aggregate AUROC and PRAUC ──────────────────────────────────────────────
    _df_agg_auroc = (
        _df_plot
        .groupby(["Metadata_Label", "Model_type"])["AUROC"]
        .agg(auroc_mean="mean", auroc_std="std")
        .reset_index()
    )
    _df_agg_prauc = (
        _df_plot
        .groupby(["Metadata_Label", "Model_type"])["PRAUC"]
        .agg(prauc_mean="mean", prauc_std="std")
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

    # ── 2. Thresholds ─────────────────────────────────────────────────────────────
    _AUROC_T1, _AUROC_T2 = 0.5, 0.70   # y-axis
    _PRAUC_T1, _PRAUC_T2 = 0.0, 0.3   # x-axis — 0 = random for AUPRG

    # ── 3. Background region colors ───────────────────────────────────────────────
    _BG = {
        "worse":    "#f7d4d4",   # red-ish
        "moderate": "#fff4cc",   # yellow-ish
        "well":     "#d4edda",   # green-ish
    }

    # ── 4. Figure ─────────────────────────────────────────────────────────────────
    _SOURCE_COLORS = {
        "Hits":       "#4C72B0",
        "Cell-based": "#DD8452",
        "Cell-free":  "#55A868",
        "Cytotox":    "#C44E52",
    }
    _LABEL_MAP = {
        "Hits": "Cytotox",
        "Cell-based": "ToxCast cell-based",
        "Cell-free": "ToxCast cell-free",
        "Cytotox": "ToxCast cytotox",
    }
    _fig, _ax = plt.subplots(figsize=(6, 6))

    # ── 5. Draw quadrant backgrounds ─────────────────────────────────────────────
    _xmin, _xmax = -0.2, 1
    _ymin, _ymax = 0.2, 1

    # worse than random: both below T1
    # worse zones
    _ax.fill_between([_xmin, _xmax], _ymin, _ymax,
                     color=_BG["worse"], zorder=0)
    # moderate: mixed zone (one axis moderate, other not well)
    _ax.fill_between([_PRAUC_T1, _xmax], _AUROC_T1, _ymax,
                     color=_BG["moderate"], zorder=0)
    # well predicted: both above T2
    _ax.fill_between([_PRAUC_T2, _xmax], _AUROC_T2, _ymax,
                     color=_BG["well"], zorder=0, label="Well predicted")
    # ── 6. Threshold lines ────────────────────────────────────────────────────────
    #for _t in [_AUROC_T1, _AUROC_T2]:
    #    _ax.axhline(_t, color="grey", linestyle="--", linewidth=0.8, zorder=1)
    #for _t in [_PRAUC_T1, _PRAUC_T2]:
    #    _ax.axvline(_t, color="grey", linestyle="--", linewidth=0.8, zorder=1)

    # ── 7. Scatter points ─────────────────────────────────────────────────────────
    texts = []
    for _src, _df_src in _df_scatter.groupby("Source"):
        _color = _SOURCE_COLORS.get(_src, "grey")

        _sig   = _df_src[_df_src["Metadata_Label"].isin(_sig_labels)]
        _nonsig = _df_src[~_df_src["Metadata_Label"].isin(_sig_labels)]

        # Significant — filled circle
        _ax.scatter(
            _sig["auprg_mean"], _sig["auroc_mean"],
            color=_color, s=60, alpha=0.85,
            edgecolors="white", linewidths=0.5,
            marker="o", zorder=4, label=_src,
        )
        # Non-significant — cross
        _ax.scatter(
            _nonsig["auprg_mean"], _nonsig["auroc_mean"],
            color=_color, s=60, alpha=0.45,
            marker="x", linewidths=1.2,
            zorder=4,
        )

        for _, _row in _df_src.iterrows():
            if _row["auprg_mean"] > 0.3 and _row["auroc_mean"] > 0.7:
                texts.append(_ax.text(
                    _row["auprg_mean"], _row["auroc_mean"],
                    _row["Metadata_Label"],
                    fontsize=8, color="#1a1a1a",
                    fontstyle="italic", zorder=5,
                ))

    # ── 8. Label repulsion ────────────────────────────────────────────────────────
    adjust_text(
        texts,
        x=_df_scatter["auprg_mean"].values,
        y=_df_scatter["auroc_mean"].values,
        ax=_ax,
        max_move=3.0,
        #expand=(1.5, 1.5),
        #force_text=(0.5, 0.5),
        force_points=(0.3, 0.3),
        add_objects=[],
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.4, alpha=0.6, shrinkA=0, shrinkB=3),
    )

    # ── 10. Region labels ─────────────────────────────────────────────────────────
    _df_sig = _df_scatter[_df_scatter["Metadata_Label"].isin(_sig_labels)]

    _n_worse    = (((_df_sig["auprg_mean"] < _PRAUC_T1) | (_df_sig["auroc_mean"] < _AUROC_T1))).sum()
    _n_moderate = (
        ((_df_sig["auprg_mean"] >= _PRAUC_T1) & (_df_sig["auroc_mean"] >= _AUROC_T1)) &
        ~((_df_sig["auprg_mean"] >= _PRAUC_T2) & (_df_sig["auroc_mean"] >= _AUROC_T2))
    ).sum()
    _n_well     = ((_df_sig["auprg_mean"] >= _PRAUC_T2) & (_df_sig["auroc_mean"] >= _AUROC_T2)).sum()

    _ax.text(0.55, 0.40, f"Worse than random\n(n={_n_worse})",       fontsize=10, color="#c0392b", alpha=0.7, fontstyle="italic", ha="center")
    _ax.text(0.65, 0.53, f"Moderately predicted\n(n={_n_moderate})", fontsize=10, color="#b7950b", alpha=0.7, fontstyle="italic", ha="center")
    _ax.text(0.8, 0.73, f"Well predicted\n(n={_n_well})",           fontsize=10, color="#1e8449", alpha=0.7, fontstyle="italic", ha="center")
    # ── 11. Style ─────────────────────────────────────────────────────────────────
    _ax.set_xlim(_xmin, _xmax)
    _ax.set_ylim(_ymin, _ymax)
    _ax.set_xlabel("AUPRG (mean)", fontsize=12)
    _ax.set_ylabel("AUROC (mean)", fontsize=12)
    _ax.set_facecolor("white")
    _fig.patch.set_facecolor("white")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["left"].set_color("black")
    _ax.spines["left"].set_linewidth(1)
    _ax.spines["bottom"].set_color("black")
    _ax.spines["bottom"].set_linewidth(1)
    _ax.tick_params(axis="both", which="major", colors="black", length=4, width=1, direction="out")
    _ax.grid(False)

    # ── 12. Legend ────────────────────────────────────────────────────────────────
    _src_patches = [
        mpatches.Patch(
            facecolor=_SOURCE_COLORS[_s],
            edgecolor="none",
            label=_LABEL_MAP.get(_s, _s)  # <- change here
        )
        for _s in _source_dfs
    ]
    _marker_patches = [
        Line2D([0], [0], marker="o", color="grey", linestyle="none", markersize=7, label="Significant"),
        Line2D([0], [0], marker="x", color="grey", linestyle="none", markersize=7, markeredgewidth=1.2, label="Non-significant"),
    ]
    _ax.legend(
        handles=_src_patches+_marker_patches,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=8,
        borderpad=0.8,
        title="Assays",
        title_fontsize=10,
    ).get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(Path.home() / "Downloads" / "auroc_vs_prauc_scatter.png", dpi=300, bbox_inches="tight")
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
def _(mo):
    mo.md(r"""## Focus performance toxcast_cytotox""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Tisue (Total_Compounds > 100 & Metadata_Count_1 > 20) & significative""")
    return


@app.cell
def _(pred_pandas):
    # Tissue
    pred_pandas[
        (pred_pandas["Total_Compounds"] > 100) &
        (pred_pandas["Metadata_Count_1"] > 20) &
        (pred_pandas["Endpoint_type"] == "toxcast_cytotox") &
        (pred_pandas["Metadata_Label"].str.contains("tissue"))
    ]
    return


@app.cell
def _(
    create_styled_metrics_table,
    list_significative_performance,
    pred_pandas,
):
    # Filtered dataframe
    _df_tissue_cytotox = pred_pandas[
        (pred_pandas["Total_Compounds"] > 100) &
        (pred_pandas["Metadata_Count_1"] > 20) &
        (pred_pandas["Endpoint_type"] == "toxcast_cytotox") &
        (pred_pandas["Metadata_Label"].str.contains("tissue", na=False)) &
        (pred_pandas["Metadata_Label"].isin(list_significative_performance))
    ].copy()
    create_styled_metrics_table(_df_tissue_cytotox)
    return


@app.cell
def _(mo):
    mo.md(r"""### Cell (Total_Compounds > 100 & Metadata_Count_1 > 20) & significative""")
    return


@app.cell
def _(
    create_styled_metrics_table,
    list_significative_performance,
    pred_pandas,
):
    # Filtered dataframe
    _df_cell_cytotox = pred_pandas[
        (pred_pandas["Total_Compounds"] > 100) &
        (pred_pandas["Metadata_Count_1"] > 20) &
        (pred_pandas["Endpoint_type"] == "toxcast_cytotox") &
        (pred_pandas["Metadata_Label"].str.contains("cell", na=False)) &
        (pred_pandas["Metadata_Label"].isin(list_significative_performance))
    ].copy()
    create_styled_metrics_table(_df_cell_cytotox)
    return


@app.cell
def _(mo):
    mo.md(r"""### All""")
    return


@app.cell
def _(create_styled_metrics_table, pred_pandas):
    # Filtered dataframe
    _df_all_cytotox = pred_pandas[
        (pred_pandas["Endpoint_type"] == "toxcast_cytotox")
    ].copy()
    create_styled_metrics_table(_df_all_cytotox)
    return


@app.cell
def _(mo):
    mo.md(r"""## Focus performance toxcast_cellbased""")
    return


@app.cell
def _(mo):
    mo.md(r"""### All""")
    return


@app.cell
def _(create_styled_metrics_table, pred_pandas):
    # Filtered dataframe
    _df_all_cellbased = pred_pandas[
        (pred_pandas["Endpoint_type"] == "toxcast_cellbased")
    ].copy()
    create_styled_metrics_table(_df_all_cellbased)
    return


@app.cell
def _(mo):
    mo.md(r"""### Tox21 & significative""")
    return


@app.cell
def _(
    create_styled_metrics_table,
    list_significative_performance,
    pred_pandas,
):
    # Filtered dataframe
    _df_tox21_cellbased = pred_pandas[
        (pred_pandas["Endpoint_type"] == "toxcast_cellbased")  &
        (pred_pandas["Metadata_Label"].str.contains("TOX21", na=False)) &
        (pred_pandas["Metadata_Label"].isin(list_significative_performance))
    ].copy()
    create_styled_metrics_table(_df_tox21_cellbased)
    return


@app.cell
def _(mo):
    mo.md(r"""### All (Total_Compounds > 50 & Metadata_Count_1 > 20) & significative""")
    return


@app.cell
def _(
    create_styled_metrics_table,
    list_significative_performance,
    pred_pandas,
):
    # Filtered dataframe
    _df_filter_cellbased = pred_pandas[
        (pred_pandas["Total_Compounds"] > 50) &
        (pred_pandas["Metadata_Count_1"] > 20) &
        (pred_pandas["Endpoint_type"] == "toxcast_cellbased") &
        (pred_pandas["Metadata_Label"].isin(list_significative_performance))
    ].copy()
    create_styled_metrics_table(_df_filter_cellbased)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Focus performance toxcast_cellfree""")
    return


@app.cell
def _(mo):
    mo.md(r"""### All""")
    return


@app.cell
def _(create_styled_metrics_table, pred_pandas):
    # Filtered dataframe
    _df_all_cellfree = pred_pandas[
        (pred_pandas["Endpoint_type"] == "toxcast_cellfree") 
    ].copy()
    create_styled_metrics_table(_df_all_cellfree)
    return


@app.cell
def _(mo):
    mo.md(r"""### All (Total_Compounds > 50 & Metadata_Count_1 > 20) & significative""")
    return


@app.cell
def _(list_significative_performance, pred_pandas):
    pred_pandas[
        (pred_pandas["Metadata_Label"].isin(list_significative_performance))
    ].copy()
    return


@app.cell
def _(
    create_styled_metrics_table,
    list_significative_performance,
    pred_pandas,
):
    # Filtered dataframe
    _df_all_cellfree = pred_pandas[
        #(pred_pandas["Total_Compounds"] > 50) &
        #(pred_pandas["Metadata_Count_1"] > 20) &
        #(pred_pandas["Endpoint_type"] == "toxcast_cellfree") &
        (pred_pandas["Metadata_Label"].isin(list_significative_performance))
    ].copy()
    create_styled_metrics_table(_df_all_cellfree)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Compare performance with and without batch correction""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Input""")
    return


@app.cell
def _(pd):
    df_axiom_hits_prediction_batch = pd.read_parquet("modules/07_classification/outputs/axiom_axiom_hits_predictions_batch.parquet")
    df_axiom_cellbased_binary_prediction_batch = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cellbased_binary_predictions_batch.parquet")
    df_axiom_cellfree_binary_prediction_batch = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cellfree_binary_predictions_batch.parquet")
    df_axiom_cytotox_binary_prediction_batch = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cytotox_binary_predictions_batch.parquet")
    return (
        df_axiom_cellbased_binary_prediction_batch,
        df_axiom_cellfree_binary_prediction_batch,
        df_axiom_cytotox_binary_prediction_batch,
        df_axiom_hits_prediction_batch,
    )


@app.cell
def _(pd):
    df_axiom_hits_prediction_no_batch = pd.read_parquet("modules/07_classification/outputs/axiom_axiom_hits_predictions_no_batch.parquet")
    df_axiom_cellbased_binary_prediction_no_batch = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cellbased_binary_predictions_no_batch.parquet")
    df_axiom_cellfree_binary_prediction_no_batch = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cellfree_binary_predictions_no_batch.parquet")
    df_axiom_cytotox_binary_prediction_no_batch = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cytotox_binary_predictions_no_batch.parquet")
    return (
        df_axiom_cellbased_binary_prediction_no_batch,
        df_axiom_cellfree_binary_prediction_no_batch,
        df_axiom_cytotox_binary_prediction_no_batch,
        df_axiom_hits_prediction_no_batch,
    )


@app.cell
def _(mo):
    mo.md(r"""## Compute metrics""")
    return


@app.cell
def _(
    df_axiom_cellbased_binary_prediction_batch,
    df_axiom_cellfree_binary_prediction_batch,
    df_axiom_cytotox_binary_prediction_batch,
    df_axiom_hits_prediction_batch,
    process_metrics,
):
    df_axiom_hits_metrics_batch, df_axiom_hits_metrics_batch_cv = process_metrics(df_axiom_hits_prediction_batch)
    df_axiom_cellbased_binary_metrics_batch, df_axiom_cellbased_binary_metrics_batch_cv = process_metrics(df_axiom_cellbased_binary_prediction_batch)
    df_axiom_cellfree_binary_metrics_batch, df_axiom_cellfree_binary_metrics_batch_cv = process_metrics(df_axiom_cellfree_binary_prediction_batch)
    df_axiom_cytotox_binary_metrics_batch, df_axiom_cytotox_binary_metrics_batch_cv = process_metrics(df_axiom_cytotox_binary_prediction_batch)
    return (
        df_axiom_cellbased_binary_metrics_batch,
        df_axiom_cellbased_binary_metrics_batch_cv,
        df_axiom_cellfree_binary_metrics_batch,
        df_axiom_cellfree_binary_metrics_batch_cv,
        df_axiom_cytotox_binary_metrics_batch,
        df_axiom_cytotox_binary_metrics_batch_cv,
        df_axiom_hits_metrics_batch,
        df_axiom_hits_metrics_batch_cv,
    )


@app.cell
def _(
    df_axiom_cellbased_binary_prediction_no_batch,
    df_axiom_cellfree_binary_prediction_no_batch,
    df_axiom_cytotox_binary_prediction_no_batch,
    df_axiom_hits_prediction_no_batch,
    process_metrics,
):
    df_axiom_hits_metrics_no_batch, df_axiom_hits_metrics_no_batch_cv = process_metrics(df_axiom_hits_prediction_no_batch)
    df_axiom_cellbased_binary_metrics_no_batch, df_axiom_cellbased_binary_metrics_no_batch_cv = process_metrics(df_axiom_cellbased_binary_prediction_no_batch)
    df_axiom_cellfree_binary_metrics_no_batch, df_axiom_cellfree_binary_metrics_no_batch_cv = process_metrics(df_axiom_cellfree_binary_prediction_no_batch)
    df_axiom_cytotox_binary_metrics_no_batch, df_axiom_cytotox_binary_metrics_no_batch_cv = process_metrics(df_axiom_cytotox_binary_prediction_no_batch)
    return (
        df_axiom_cellbased_binary_metrics_no_batch,
        df_axiom_cellbased_binary_metrics_no_batch_cv,
        df_axiom_cellfree_binary_metrics_no_batch,
        df_axiom_cellfree_binary_metrics_no_batch_cv,
        df_axiom_cytotox_binary_metrics_no_batch,
        df_axiom_cytotox_binary_metrics_no_batch_cv,
        df_axiom_hits_metrics_no_batch,
        df_axiom_hits_metrics_no_batch_cv,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Annotate metrics""")
    return


@app.cell
def _(
    df_axiom_cellbased_binary_metrics_batch,
    df_axiom_cellfree_binary_metrics_batch,
    df_axiom_cytotox_binary_metrics_batch,
    df_axiom_hits_metrics_batch,
    pd,
):
    # Map endpoint names to their corresponding metric dataframes
    endpoint_map_batch = {
        "axiom_hits": df_axiom_hits_metrics_batch,
        "toxcast_cellbased_binary": df_axiom_cellbased_binary_metrics_batch,
        "toxcast_cellfree_binary": df_axiom_cellfree_binary_metrics_batch,
        "toxcast_cytotox_binary": df_axiom_cytotox_binary_metrics_batch,
    }

    pred_list_batch = []
    for et_batch, df_batch in endpoint_map_batch.items():
        df_batch = df_batch.copy()
        # Total compounds
        df_batch["Total_Compounds"] = df_batch["Metadata_Count_0"] + df_batch["Metadata_Count_1"]
        # Endpoint type
        df_batch["Endpoint_type"] = et_batch
        pred_list_batch.append(df_batch)

    # Concatenate
    pred_pandas_batch = pd.concat(pred_list_batch, axis=0, ignore_index=True)

    # Rename if needed (mimicking your Polars logic)
    pred_pandas_batch["Endpoint_type"] = pred_pandas_batch["Endpoint_type"].replace({
        "axiom_hits": "axiom_cytotox"
    })
    pred_pandas_batch["Endpoint_type"] = pred_pandas_batch["Endpoint_type"].replace({
        "toxcast_cellbased_binary": "toxcast_cellbased"
    })
    pred_pandas_batch["Endpoint_type"] = pred_pandas_batch["Endpoint_type"].replace({
        "toxcast_cellfree_binary": "toxcast_cellfree"
    })
    pred_pandas_batch["Endpoint_type"] = pred_pandas_batch["Endpoint_type"].replace({
        "toxcast_cytotox_binary": "toxcast_cytotox"
    })

    # Percent positive
    pred_pandas_batch["Percent_Positive"] = (
        pred_pandas_batch["Metadata_Count_1"] / pred_pandas_batch["Total_Compounds"]
    )

    # Ordered categorical
    pred_pandas_batch["Endpoint_type"] = pd.Categorical(
        pred_pandas_batch["Endpoint_type"],
        categories=[
            "axiom_cytotox",
            "toxcast_cytotox",
            "toxcast_cellbased",
            "toxcast_cellfree",
        ],
        ordered=True
    )
    return (pred_pandas_batch,)


@app.cell
def _(
    df_axiom_cellbased_binary_metrics_no_batch,
    df_axiom_cellfree_binary_metrics_no_batch,
    df_axiom_cytotox_binary_metrics_no_batch,
    df_axiom_hits_metrics_no_batch,
    pd,
):
    # Map endpoint names to their corresponding metric dataframes
    endpoint_map_no_batch = {
        "axiom_hits": df_axiom_hits_metrics_no_batch,
        "toxcast_cellbased_binary": df_axiom_cellbased_binary_metrics_no_batch,
        "toxcast_cellfree_binary": df_axiom_cellfree_binary_metrics_no_batch,
        "toxcast_cytotox_binary": df_axiom_cytotox_binary_metrics_no_batch,
    }

    pred_list_no_batch = []
    for et_no_batch, df_no_batch in endpoint_map_no_batch.items():
        df_no_batch = df_no_batch.copy()
        # Total compounds
        df_no_batch["Total_Compounds"] = df_no_batch["Metadata_Count_0"] + df_no_batch["Metadata_Count_1"]
        # Endpoint type
        df_no_batch["Endpoint_type"] = et_no_batch
        pred_list_no_batch.append(df_no_batch)

    # Concatenate
    pred_pandas_no_batch = pd.concat(pred_list_no_batch, axis=0, ignore_index=True)

    # Rename if needed (mimicking your Polars logic)
    pred_pandas_no_batch["Endpoint_type"] = pred_pandas_no_batch["Endpoint_type"].replace({
        "axiom_hits": "axiom_cytotox"
    })
    pred_pandas_no_batch["Endpoint_type"] = pred_pandas_no_batch["Endpoint_type"].replace({
        "toxcast_cellbased_binary": "toxcast_cellbased"
    })
    pred_pandas_no_batch["Endpoint_type"] = pred_pandas_no_batch["Endpoint_type"].replace({
        "toxcast_cellfree_binary": "toxcast_cellfree"
    })
    pred_pandas_no_batch["Endpoint_type"] = pred_pandas_no_batch["Endpoint_type"].replace({
        "toxcast_cytotox_binary": "toxcast_cytotox"
    })

    # Percent positive
    pred_pandas_no_batch["Percent_Positive"] = (
        pred_pandas_no_batch["Metadata_Count_1"] / pred_pandas_no_batch["Total_Compounds"]
    )

    # Ordered categorical
    pred_pandas_no_batch["Endpoint_type"] = pd.Categorical(
        pred_pandas_no_batch["Endpoint_type"],
        categories=[
            "axiom_cytotox",
            "toxcast_cytotox",
            "toxcast_cellbased",
            "toxcast_cellfree",
        ],
        ordered=True
    )
    return (pred_pandas_no_batch,)


@app.cell
def _(pred_pandas_no_batch):
    pred_pandas_no_batch
    return


@app.cell
def _(pred_pandas_batch):
    pred_pandas_batch
    return


@app.cell
def _(mo):
    mo.md(r"""## T-test check mean diffference""")
    return


@app.cell
def _(
    df_axiom_cellbased_binary_metrics_batch_cv,
    df_axiom_cellbased_binary_metrics_no_batch_cv,
    df_axiom_cellfree_binary_metrics_batch_cv,
    df_axiom_cellfree_binary_metrics_no_batch_cv,
    df_axiom_cytotox_binary_metrics_batch_cv,
    df_axiom_cytotox_binary_metrics_no_batch_cv,
    df_axiom_hits_metrics_batch_cv,
    df_axiom_hits_metrics_no_batch_cv,
    np,
    pd,
    stats,
):
    # ============================================================================
    # PAIRED T-TEST FOR CROSS-VALIDATION METRICS
    # ============================================================================

    # Map batch and no_batch CV metrics dataframes
    _cv_metrics_pairs = {
        "axiom_hits": (df_axiom_hits_metrics_batch_cv, df_axiom_hits_metrics_no_batch_cv),
        "toxcast_cellbased_binary": (df_axiom_cellbased_binary_metrics_batch_cv, df_axiom_cellbased_binary_metrics_no_batch_cv),
        "toxcast_cellfree_binary": (df_axiom_cellfree_binary_metrics_batch_cv, df_axiom_cellfree_binary_metrics_no_batch_cv),
        "toxcast_cytotox_binary": (df_axiom_cytotox_binary_metrics_batch_cv, df_axiom_cytotox_binary_metrics_no_batch_cv),
    }

    _cv_stats_results = []

    # Process each CV metrics pair
    for _endpoint_name, (_cv_df_batch, _cv_df_no_batch) in _cv_metrics_pairs.items():
        # Filter for Model_type == "Actual" only
        _cv_batch_actual = _cv_df_batch[_cv_df_batch["Model_type"] == "Actual"].copy()
        _cv_no_batch_actual = _cv_df_no_batch[_cv_df_no_batch["Model_type"] == "Actual"].copy()

        if len(_cv_batch_actual) == 0 or len(_cv_no_batch_actual) == 0:
            continue

        # Get unique metadata labels from batch corrected
        _labels = _cv_batch_actual["Metadata_Label"].unique()

        for _label in _labels:
            # Filter for this label from both batch and no_batch
            _batch_label_data = _cv_batch_actual[_cv_batch_actual["Metadata_Label"] == _label]
            _no_batch_label_data = _cv_no_batch_actual[_cv_no_batch_actual["Metadata_Label"] == _label]

            # Get AUC values
            _batch_values = _batch_label_data["AUROC"].values
            _no_batch_values = _no_batch_label_data["AUROC"].values

            if len(_batch_values) > 0 and len(_no_batch_values) > 0:
                # Calculate means
                _no_batch_mean = _no_batch_values.mean()
                _batch_mean = _batch_values.mean()
                _mean_diff = _batch_mean - _no_batch_mean
                _percent_change = (_mean_diff / _no_batch_mean * 100) if _no_batch_mean != 0 else 0

                # Perform paired t-test
                try:
                    # Use minimum length for paired test
                    _ttest_stat, _ttest_pval = stats.wilcoxon(_batch_values, _no_batch_values)

                except:
                    _ttest_stat = np.nan
                    _ttest_pval = np.nan

                # Store results
                _cv_stats_results.append({
                    "Endpoint": _endpoint_name,
                    "Metadata_Label": _label,
                    "N": min(len(_batch_values), len(_no_batch_values)),
                    "No Batch Mean": _no_batch_mean,
                    "Batch Mean": _batch_mean,
                    "Mean Difference": _mean_diff,
                    "Percent Change (%)": _percent_change,
                    "t-statistic": _ttest_stat,
                    "p-value": _ttest_pval,
                    "Significant (p<0.05)": "Yes" if _ttest_pval < 0.05 else "No"
                })

    # Create results dataframe
    cv_stats_results_df = pd.DataFrame(_cv_stats_results)

    cv_stats_results_df
    return (cv_stats_results_df,)


@app.cell
def _(cv_stats_results_df, list_significative_performance):
    cv_stats_results_df[cv_stats_results_df["Metadata_Label"].isin(list_significative_performance)]
    return


@app.cell
def _(cv_stats_results_df, list_significative_performance, pd):
    # Filter for significant labels
    _cv_filtered = cv_stats_results_df[cv_stats_results_df["Metadata_Label"].isin(list_significative_performance)].copy()

    # Count significant
    _sig_yes = (_cv_filtered["Significant (p<0.05)"] == "Yes").sum()
    _sig_no = (_cv_filtered["Significant (p<0.05)"] == "No").sum()

    _sig_yes_tf = (_cv_filtered["Significant (p<0.05)"] == "Yes")
    _sig_no_tf = (_cv_filtered["Significant (p<0.05)"] == "No")

    # Count direction
    _batch_better = (_cv_filtered[_sig_yes_tf]["Batch Mean"] > _cv_filtered[_sig_yes_tf]["No Batch Mean"]).sum()
    _no_batch_better = (_cv_filtered[_sig_yes_tf]["No Batch Mean"] > _cv_filtered[_sig_yes_tf]["Batch Mean"]).sum()

    # Summary
    _summary = pd.DataFrame({
        "Metric": ["Significant", "Not Significant", "Batch Mean > No Batch", "No Batch Mean > Batch"],
        "Count": [_sig_yes, _sig_no, _batch_better, _no_batch_better]
    })
    _summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Boxplot comparison perf""")
    return


@app.cell
def _(pd, plt, pred_pandas_batch, pred_pandas_no_batch, sns):
    # Combine both dataframes with batch indicator
    _pred_pandas_no_batch_copy = pred_pandas_no_batch.copy()
    _pred_pandas_no_batch_copy["Batch_Correction"] = "No Batch Correction"

    _pred_pandas_batch_copy = pred_pandas_batch.copy()
    _pred_pandas_batch_copy["Batch_Correction"] = "Batch Corrected"

    # Concatenate
    _pred_pandas_combined = pd.concat([_pred_pandas_no_batch_copy, _pred_pandas_batch_copy], axis=0, ignore_index=True)

    # Create boxplot
    _fig, _axes = plt.subplots(2, 2, figsize=(16, 12))
    _fig.suptitle("Prediction Comparison: Batch Corrected vs No Batch Correction", fontsize=16, fontweight='bold')

    # Get unique endpoint types
    _endpoints = _pred_pandas_combined["Endpoint_type"].unique()

    # Plot for each endpoint
    for _idx, _endpoint in enumerate(_endpoints):
        _ax = _axes[_idx // 2, _idx % 2]

        # Filter data for this endpoint
        _endpoint_data = _pred_pandas_combined[_pred_pandas_combined["Endpoint_type"] == _endpoint]

        # Create boxplot
        sns.boxplot(
            data=_endpoint_data,
            x="Model_type",
            y="Percent_Positive",
            hue="Batch_Correction",
            ax=_ax,
            palette="Set2"
        )

        # Add jittered points
        sns.stripplot(
            data=_endpoint_data,
            x="Model_type",
            y="Percent_Positive",
            hue="Batch_Correction",
            ax=_ax,
            dodge=True,
            alpha=0.4,
            size=4,
            jitter=True,
            legend=False
        )

        _ax.set_title(f"{_endpoint}", fontsize=12, fontweight='bold')
        _ax.set_xlabel("Model Type", fontsize=11)
        _ax.set_ylabel("Percent Positive", fontsize=11)
        _ax.legend(title="Batch Correction", loc='best')
        _ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    #plt.savefig("boxplot_batch_correction_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(np, pd, plt, pred_pandas_batch, pred_pandas_no_batch, sns, stats):
    # Combine both dataframes with batch indicator
    _pred_pandas_no_batch_copy = pred_pandas_no_batch.copy()
    _pred_pandas_no_batch_copy["Batch_Correction"] = "No Batch Correction"

    _pred_pandas_batch_copy = pred_pandas_batch.copy()
    _pred_pandas_batch_copy["Batch_Correction"] = "Batch Corrected"

    # Concatenate
    _pred_pandas_combined = pd.concat([_pred_pandas_no_batch_copy, _pred_pandas_batch_copy], axis=0, ignore_index=True)


    # Filter for Model_type == "Actual" only
    _pred_pandas_combined_actual = _pred_pandas_combined[_pred_pandas_combined["Model_type"] == "Actual"].copy()

    # ============================================================================
    # STATISTICAL TESTING - PAIRED T-TEST
    # ============================================================================

    # Store test results
    _stats_summary = []

    # Get unique endpoint types
    _endpoints = _pred_pandas_combined_actual["Endpoint_type"].unique()

    for _endpoint in _endpoints:
        # Filter data for this endpoint
        _endpoint_data = _pred_pandas_combined_actual[_pred_pandas_combined_actual["Endpoint_type"] == _endpoint]

        # Get values for each batch correction type
        _no_batch_values = _endpoint_data[_endpoint_data["Batch_Correction"] == "No Batch Correction"]["AUROC"].values
        _batch_values = _endpoint_data[_endpoint_data["Batch_Correction"] == "Batch Corrected"]["AUROC"].values

        # Calculate means
        _no_batch_mean = _no_batch_values.mean()
        _batch_mean = _batch_values.mean()
        _mean_diff = _batch_mean - _no_batch_mean
        _percent_change = (_mean_diff / _no_batch_mean * 100) if _no_batch_mean != 0 else 0

        # Perform paired t-test
        if len(_no_batch_values) > 0 and len(_batch_values) > 0:
        #try:
            _ttest_stat, _ttest_pval = stats.ttest_rel(_batch_values, _no_batch_values)
        #except:
            #_ttest_stat = np.nan
            #_ttest_pval = np.nan
        else:
            _ttest_stat = np.nan
            _ttest_pval = np.nan

        # Store results
        _stats_summary.append({
            "Endpoint": _endpoint,
            "N": len(_no_batch_values),
            "No Batch Mean": _no_batch_mean,
            "Batch Mean": _batch_mean,
            "Mean Difference": _mean_diff,
            "Percent Change (%)": _percent_change,
            "t-statistic": _ttest_stat,
            "p-value": _ttest_pval,
            "Significant (p<0.05)": "Yes" if _ttest_pval < 0.05 else "No"
        })

    # Create statistics summary dataframe
    _stats_df = pd.DataFrame(_stats_summary)
    print(_stats_df)
    # ============================================================================
    # VISUALIZATION - BOXPLOTS
    # ============================================================================

    # Create boxplot
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))
    _fig.suptitle("Percent Positive: Batch Corrected vs No Batch Correction (Actual Model Only)", fontsize=16, fontweight='bold')

    # Plot for each endpoint
    for _idx, _endpoint in enumerate(_endpoints):
        _ax = _axes[_idx // 2, _idx % 2]

        # Filter data for this endpoint
        _endpoint_data = _pred_pandas_combined_actual[_pred_pandas_combined_actual["Endpoint_type"] == _endpoint]

        # Create boxplot
        sns.boxplot(
            data=_endpoint_data,
            x="Batch_Correction",
            y="Percent_Positive",
            ax=_ax,
            palette="Set2"
        )

        # Add jittered points
        sns.stripplot(
            data=_endpoint_data,
            x="Batch_Correction",
            y="Percent_Positive",
            ax=_ax,
            palette="Set2",
            alpha=0.4,
            size=4,
            jitter=True,
            legend=False
        )

        _ax.set_title(f"{_endpoint}", fontsize=11, fontweight='bold')
        _ax.set_xlabel("Batch Correction", fontsize=11)
        _ax.set_ylabel("Percent Positive", fontsize=11)
        _ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    #plt.savefig("boxplot_actual_model_batch_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Check assays modalities cell-based""")
    return


@app.cell
def _(pd):
    df_toxcast_cellbased_assay_info = pd.read_csv("/Users/ollitrault/Documents/projects/databases/proc_toxcast_annotations/toxcast_cellbased_assay_info.csv")
    return (df_toxcast_cellbased_assay_info,)


@app.cell
def _(df_toxcast_cellbased_assay_info, list_significative_performance):
    df_toxcast_cellbased_assay_info[df_toxcast_cellbased_assay_info["assay_component_endpoint_name"].isin(list_significative_performance)]
    return


@app.cell
def _(df_toxcast_cellbased_assay_info, list_significative_performance):
    df_toxcast_cellbased_assay_info[df_toxcast_cellbased_assay_info["assay_component_endpoint_name"].isin(list_significative_performance)].to_csv("~/Downloads/cell-based-significative.csv")
    return


@app.cell
def _(df_toxcast_cellbased_assay_info, list_significative_performance):
    print(
        df_toxcast_cellbased_assay_info[
            df_toxcast_cellbased_assay_info["assay_component_endpoint_name"].isin(list_significative_performance)
        ]
        .groupby("tissue")["assay_component_endpoint_name"]
        .count()
        .sort_values(ascending=False)
        .to_string()
    )
    return


@app.cell
def _(df_toxcast_cellbased_assay_info, list_significative_performance):
    df_toxcast_cellbased_assay_info[
            df_toxcast_cellbased_assay_info["assay_component_endpoint_name"].isin(list_significative_performance)
        ].groupby("tissue")["assay_component_endpoint_name"].count().sort_values(ascending=False)
    return


@app.cell
def _():
    dict_categores_significative_cell_based_claudeai = {
      "categories": [
        {
          "id": "endocrine_sex_hormones",
          "label": "Endocrine Disruption — Sex Hormones",
          "color": "#d62728",
          "description": "Assays targeting ER-alpha, ER-beta, AR, PR — agonist & antagonist modes across breast, kidney, uterus cell lines.",
          "assays": [
            "TOX21_ERa_LUC_VM7_Agonist",
            "TOX21_ERa_LUC_VM7_Antagonist_0.1nM_E2",
            "TOX21_ERa_LUC_VM7_Antagonist_0.5nM_E2",
            "TOX21_ERa_BLA_Antagonist_ratio",
            "TOX21_ERb_BLA_Antagonist_ratio",
            "TOX21_AR_BLA_Agonist_ratio",
            "TOX21_AR_BLA_Antagonist_ratio",
            "TOX21_AR_LUC_MDAKB2_Agonist",
            "TOX21_AR_LUC_MDAKB2_Antagonist_0.5nM_R1881",
            "TOX21_AR_LUC_MDAKB2_Antagonist_10nM_R1881",
            "TOX21_PR_BLA_Agonist_ratio",
            "TOX21_PR_BLA_Antagonist_ratio",
            "TOX21_Aromatase_Inhibition",
            "OT_ER_ERaERb_1440",
            "OT_ER_ERbERb_1440",
            "ATG_ERE_CIS"
          ]
        },
        {
          "id": "endocrine_steroidogenesis",
          "label": "Endocrine Disruption — Steroidogenesis & Adrenal",
          "color": "#ff7f0e",
          "description": "Adrenal H295R cell line measuring steroid hormone levels directly (glucocorticoids, progestagens).",
          "assays": [
            "CEETOX_H295R_OHPREG",
            "CEETOX_H295R_11DCORT"
          ]
        },
        {
          "id": "endocrine_metabolic_nuclear_receptors",
          "label": "Endocrine Disruption — Metabolic / Orphan Nuclear Receptors",
          "color": "#ffbb00",
          "description": "PPARg, LXR, FXR, VDR, PXR, CAR, ERR, PGC/ERR — receptors regulating lipid, bile, glucose, and xenobiotic metabolism.",
          "assays": [
            "TOX21_PPARg_BLA_Antagonist_ratio",
            "TOX21_PXR_agonist",
            "TOX21_CAR_Agonist",
            "TOX21_FXR_BLA_Antagonist_ratio",
            "TOX21_ERR_Antagonist",
            "TOX21_PGC_ERR_Antagonist",
            'TOX21_GR_BLA_Antagonist_ratio',
            "ATG_DR4_LXR_CIS",
            "ATG_PXRE_CIS",
            "ATG_PPRE_CIS",
            "ATG_VDRE_CIS"
          ]
        },
        {
          "id": "hepatotoxicity_cyp_induction",
          "label": "Hepatotoxicity — Xenobiotic Metabolism (CYP induction)",
          "color": "#8c564b",
          "description": "HepaRG and HepG2 liver cell lines measuring CYP enzyme induction — key markers of drug metabolism perturbation and liver stress.",
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
            "TOX21_AhR_LUC_Agonist"
          ]
        },
        {
          "id": "hepatotoxicity_steatosis",
          "label": "Hepatotoxicity — Steatosis & Lipid Transport",
          "color": "#a07040",
          "description": "Markers of fatty liver disease — lipid accumulation (HMGCS2), bile transport (ABCB11), and drug efflux (ABCB1).",
          "assays": [
            "LTEA_HepaRG_HMGCS2",
            "LTEA_HepaRG_ABCB11",
            "LTEA_HepaRG_ABCB1",
            "LTEA_HepaRG_THRSP"
          ]
        },
        {
          "id": "hepatotoxicity_general",
          "label": "Hepatotoxicity — General Liver Stress / Cell Viability",
          "color": "#2ca02c",
          "description": "HepG2/HepaRG markers of cell proliferation, survival, and death; oxidative stress (NRF2/ARE).",
          "assays": [
            "LTEA_HepaRG_ALPP",
            "LTEA_HepaRG_PEG10",
            "ATG_NRF2_ARE_CIS",
            "TOX21_ARE_BLA_Agonist_ratio",
            "ATG_AP_1_CIS",
            "ATG_MRE_CIS",
            "ATG_Oct_MLP_CIS",
            "ATG_TCF_b_cat_CIS"
          ]
        },
        {
          "id": "genotoxicity",
          "label": "Genotoxicity / DNA Damage",
          "color": "#1f77b4",
          "description": "p53 activation (tumor suppressor pathway) and HDAC inhibition — hallmarks of genotoxic stress and epigenetic disruption.",
          "assays": [
            "TOX21_p53_BLA_p1_ratio",
            "TOX21_p53_BLA_p2_ratio",
            "TOX21_p53_BLA_p3_ratio",
            "TOX21_p53_BLA_p4_ratio",
            "TOX21_p53_BLA_p5_ratio",
            "TOX21_HDAC_Inhibition"
          ]
        },
        {
          "id": "cardiovascular_vascular_inflammation",
          "label": "Cardiovascular / Vascular Inflammation",
          "color": "#9467bd",
          "description": "Primary vascular cells (HUVEC, coronary artery smooth muscle) measuring endothelial activation, cell adhesion molecules, cytokines — indicators of vascular inflammation and atherogenesis.",
          "assays": [
            "BSK_4H_VCAM1",
            "BSK_4H_MCP1",
            "BSK_4H_SRB",
            "BSK_4H_uPAR",
            "BSK_3C_SRB",
            "BSK_3C_Proliferation",
            "BSK_3C_uPAR",
            "BSK_CASM3C_Thrombomodulin",
            "BSK_CASM3C_Proliferation",
            "BSK_LPS_VCAM1",
            "BSK_LPS_MCSF",
            "BSK_LPS_CD40",
            "BSK_SAg_CD40",
            "BSK_SAg_Eselectin",
            "BSK_SAg_Proliferation"
          ]
        }
      ]
    }
    return (dict_categores_significative_cell_based_claudeai,)


@app.cell
def _(dict_categores_significative_cell_based_claudeai):
    list_claudeai_assays = []
    for i in dict_categores_significative_cell_based_claudeai["categories"]:
        print(i["assays"])
        list_claudeai_assays += i["assays"]
    print(len(list_claudeai_assays))
    return (list_claudeai_assays,)


@app.cell
def _(list_claudeai_assays):
    list_claudeai_assays
    return


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

    print("In Claude AI list but NOT in dataframe:", _list_claudeai - _list_df)
    print("In dataframe but NOT in Claude AI list:", _list_df - _list_claudeai)
    return


@app.cell
def _(
    df_toxcast_cellbased_assay_info,
    dict_categores_significative_cell_based_claudeai,
    list_significative_performance,
    pd,
):
    _cat_lookup = {
        _assay: {
            "claudeai_id":          _cat["id"],
            "claudeai_label":       _cat["label"],
            "claudeai_color":       _cat["color"],
            "claudeai_description": _cat["description"],
        }
        for _cat in dict_categores_significative_cell_based_claudeai["categories"]
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
    df_toxcast_cellbased_assay_info_significant.to_csv("modules/07_classification/outputs/df_toxcast_cellbased_assay_info_significant_claudeai_categories.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""### Add performances and other models""")
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
    df_toxcast_assay_info_perf_significant.to_csv("modules/07_classification/outputs/df_toxcast_all_assay_info_perf_significant_claudeai_categories.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""### Significative specific category""")
    return


@app.cell
def _(dict_categores_significative_cell_based_claudeai):
    dict_categores_significative_cell_based_claudeai["categories"][6]["assays"]
    return


@app.cell
def _(
    create_styled_metrics_table,
    dict_categores_significative_cell_based_claudeai,
    pred_pandas,
):
    # Filtered dataframe
    _df_all_cellfree = pred_pandas[
        #(pred_pandas["Total_Compounds"] > 50) &
        #(pred_pandas["Metadata_Count_1"] > 20) &
        #(pred_pandas["Endpoint_type"] == "toxcast_cellfree") &
        (pred_pandas["Metadata_Label"].isin(dict_categores_significative_cell_based_claudeai["categories"][6]["assays"]))
    ].copy()
    create_styled_metrics_table(_df_all_cellfree)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Check assays compounds overlap""")
    return


@app.cell
def _(pred_pandas):
    pred_pandas
    return


@app.cell
def _(
    df_axiom_cellbased_binary_prediction,
    df_axiom_cellfree_binary_prediction,
    df_axiom_cytotox_binary_prediction,
    df_axiom_hits_prediction,
):
    df_axiom_hits_prediction
    df_axiom_cellbased_binary_prediction
    df_axiom_cellfree_binary_prediction
    df_axiom_cytotox_binary_prediction
    return


@app.cell
def _(df_axiom_hits_prediction):
    df_axiom_hits_prediction
    return


@app.cell
def _():
    import upsetplot
    from upsetplot import UpSet, from_contents
    return UpSet, from_contents


@app.cell
def _(
    UpSet,
    df_axiom_cellbased_binary_prediction,
    df_axiom_cellfree_binary_prediction,
    df_axiom_cytotox_binary_prediction,
    df_axiom_hits_prediction,
    from_contents,
    list_significative_performance,
    pd,
    plt,
):
    # Build one set per Metadata_Label across all assay types
    _df_all_filtered = pd.concat([
        df_axiom_hits_prediction,
        df_axiom_cellbased_binary_prediction,
        df_axiom_cellfree_binary_prediction,
        df_axiom_cytotox_binary_prediction
    ], ignore_index=True)

    _df_actual = _df_all_filtered[
        (_df_all_filtered["Model_type"] == "Actual") &
        (_df_all_filtered["Metadata_Label"].isin(list_significative_performance))
    ]

    _sets_by_label = {
        label: set(_df_actual[_df_actual["Metadata_Label"] == label]["Metadata_compound"].unique())
        for label in list_significative_performance
    }

    # Remove empty sets
    _sets_by_label = {k: v for k, v in _sets_by_label.items() if len(v) > 0}

    print("Labels and compound counts:")
    for k, v in _sets_by_label.items():
        print(f"  {k}: {len(v)}")

    # UpSet plot
    _upset_data = from_contents(_sets_by_label)

    _fig = plt.figure(figsize=(14, 6))
    UpSet(
        _upset_data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
        min_subset_size=1,
    ).plot(_fig)

    plt.suptitle("UpSet Plot: Shared Compounds Across Significant Labels", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Try XGBoost workaround: “stacked dataset” trick""")
    return


@app.cell
def _(mo):
    mo.md(r"""## input""")
    return


@app.cell
def _(pd):
    df_level6_axiom_all = pd.read_parquet("modules/03_postprocessing/outputs/level6_axiom_selected_features_batch.parquet")
    df_level6_axiom_all
    return (df_level6_axiom_all,)


@app.cell
def _(
    df_axiom_cellbased_binary_prediction,
    df_axiom_cellfree_binary_prediction,
    df_axiom_cytotox_binary_prediction,
    df_axiom_hits_prediction,
    list_significative_performance,
    pd,
):
    ## Compute one-hote encoding
    # Build one set per Metadata_Label across all assay types
    _df_all_filtered = pd.concat([
        df_axiom_hits_prediction,
        df_axiom_cellbased_binary_prediction,
        df_axiom_cellfree_binary_prediction,
        df_axiom_cytotox_binary_prediction
    ], ignore_index=True)

    _df_actual = _df_all_filtered[
        (_df_all_filtered["Model_type"] == "Actual") &
        (_df_all_filtered["Metadata_Label"].isin(list_significative_performance))
    ]
    # One-hot encode Metadata_Label (tested assay) + keep y_actual as-is
    _ohe_labels = pd.get_dummies(_df_actual["Metadata_Label"]).astype(int)

    df_ohe = pd.concat([
        _df_actual[["Metadata_compound"]].reset_index(drop=True),
        _ohe_labels.reset_index(drop=True)
    ], axis=1)
    y_true = _df_actual["y_actual"]
    print(df_ohe.shape)
    print(y_true.shape)
    df_ohe.head()
    return df_ohe, y_true


@app.cell
def _(df_level6_axiom_all, df_ohe, pd, warnings, y_true):
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    warnings.filterwarnings("ignore")

    # ── 1. Merge ──────────────────────────────────────────────────────────────────
    _df_merged = df_ohe.merge(
        df_level6_axiom_all,
        on="Metadata_compound",
        how="inner"
    )
    print(f"Merged shape: {_df_merged.shape}")

    # ── 2. Define X and y ─────────────────────────────────────────────────────────
    _meta_cols = ["Metadata_compound"]
    _label_cols = list(pd.get_dummies(_df_actual["Metadata_Label"]).columns)

    _feature_cols = [c for c in _df_merged.columns if c not in _meta_cols]

    X = _df_merged[_feature_cols].values
    y = y_true.values
    return (
        StratifiedKFold,
        X,
        XGBClassifier,
        classification_report,
        cross_val_score,
        y,
    )


@app.cell
def _(
    StratifiedKFold,
    X,
    XGBClassifier,
    classification_report,
    cross_val_score,
    np,
    pd,
    plt,
    y,
):

    print(f"X shape: {X.shape}, y shape: {y.shape}, classes: {np.unique(y)}")

    # ── 3. XGBoost with Stratified CV ─────────────────────────────────────────────
    _scale_pos = (y == 0).sum() / (y == 1).sum() if len(np.unique(y)) == 2 else 1.0

    _model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=_scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    _cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _scores = cross_val_score(_model, X, y, cv=_cv, scoring="roc_auc", n_jobs=-1)

    print(f"\nCV ROC-AUC: {_scores.mean():.3f} ± {_scores.std():.3f}")
    print(f"Per-fold:   {np.round(_scores, 3)}")

    # ── 4. Fit on full data + feature importance ───────────────────────────────────
    _model.fit(X, y)

    _feat_names = _feature_cols + _label_cols
    _importances = pd.Series(_model.feature_importances_, index=_feat_names)
    _top20 = _importances.nlargest(20)

    # ── 5. Plot top-20 feature importance ─────────────────────────────────────────
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _top20.sort_values().plot(kind="barh", ax=_ax, color="steelblue", edgecolor="white")
    _ax.set_title("XGBoost – Top 20 Feature Importances", fontsize=12)
    _ax.set_xlabel("Importance (gain)", fontsize=10)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

    # ── 6. Full-data classification report ────────────────────────────────────────
    _y_pred = _model.predict(X)
    print("\nClassification report (train, indicative):")
    if y.dtype != object:
        try:
            print(classification_report(y, _y_pred, target_names=_le.classes_))
        except Exception:
            print(classification_report(y, _y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Make level 6 DTXSID""")
    return


@app.cell
def _(pd):
    dt_level4_axiom = pd.read_parquet("modules/02_processing/outputs/level4a_axiom.parquet")
    dt_level6_axiom = pd.read_parquet("modules/02_processing/outputs/level6_axiom_all_features.parquet")
    dt_oasis_dtxsid = pd.read_csv("modules/07_classification/input/v5_oasis_03Sept2024_simple.csv")
    return dt_level4_axiom, dt_level6_axiom, dt_oasis_dtxsid


@app.cell
def _(dt_level4_axiom):
    dt_level4_axiom
    return


@app.cell
def _(dt_level6_axiom):
    dt_level6_axiom
    return


@app.cell
def _(dt_oasis_dtxsid):
    dt_oasis_dtxsid
    return


@app.cell
def _(dt_level4_axiom, dt_oasis_dtxsid):
    # Build intermediary table
    intermediary = (
        dt_level4_axiom[["Metadata_compound", "Metadata_OASIS_ID"]]
        .drop_duplicates()
        .merge(
            dt_oasis_dtxsid[["OASIS_ID", "DTXSID"]].drop_duplicates(subset="OASIS_ID"),
            left_on="Metadata_OASIS_ID",
            right_on="OASIS_ID",
            how="left"
        )
        .drop(columns="OASIS_ID")
    )

    print(intermediary.head())
    print(intermediary.shape)
    return (intermediary,)


@app.cell
def _(dt_level6_axiom, intermediary):
    lookup = (
        intermediary
        .drop_duplicates(subset="Metadata_compound")
        .set_index("Metadata_compound")["DTXSID"]
    )

    dt_level6_axiom["Metadata_dtxsid"] = dt_level6_axiom["Metadata_compound"].map(lookup)

    # Check
    matched = dt_level6_axiom["Metadata_dtxsid"].notna().sum()
    total = len(dt_level6_axiom)
    print(f"Matched: {matched}/{total} ({matched/total:.1%})")
    return (lookup,)


@app.cell
def _(dt_level6_axiom):
    dt_level6_axiom.to_parquet("modules/07_classification/outputs/dt_level6_axiom_dtxsid.parquet")
    return


@app.cell
def _(dt_level6_axiom):
    dt_level6_axiom
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Make level 5 DTXSID""")
    return


@app.cell
def _(pd):
    dt_level5_axiom = pd.read_parquet("modules/02_processing/outputs/level5_axiom_all_features.parquet")
    return (dt_level5_axiom,)


@app.cell
def _(dt_level5_axiom, lookup):
    dt_level5_axiom["Metadata_dtxsid"] = dt_level5_axiom["Metadata_compound"].map(lookup)

    # Check
    matched_level5 = dt_level5_axiom["Metadata_dtxsid"].notna().sum()
    total_level5 = len(dt_level5_axiom)
    print(f"Matched: {matched_level5}/{total_level5} ({matched_level5/total_level5:.1%})")
    return


@app.cell
def _(dt_level5_axiom):
    dt_level5_axiom
    return


@app.cell
def _(dt_level5_axiom):
    dt_level5_axiom.to_parquet("modules/07_classification/outputs/dt_level5_axiom_dtxsid.parquet")
    return


if __name__ == "__main__":
    app.run()
