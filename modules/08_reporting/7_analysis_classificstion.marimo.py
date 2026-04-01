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
    return (
        Line2D,
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
        stats,
        theme,
        theme_bw,
        theme_classic,
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
    df_axiom_hits_prediction = pd.read_parquet("modules/07_classification/outputs/axiom_axiom_hits_predictions.parquet")
    df_axiom_cellbased_binary_prediction = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cellbased_binary_predictions.parquet")
    df_axiom_cellfree_binary_prediction = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cellfree_binary_predictions.parquet")
    df_axiom_cytotox_binary_prediction = pd.read_parquet("modules/07_classification/outputs/axiom_toxcast_cytotox_binary_predictions.parquet")
    return (
        df_axiom_cellbased_binary_prediction,
        df_axiom_cellfree_binary_prediction,
        df_axiom_cytotox_binary_prediction,
        df_axiom_hits_prediction,
    )


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
        df_metrics_cv["Model_type"].isin(["Actual", "Cellcount_baseline"])
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
            auroc_data["Actual"], auroc_data["Cellcount_baseline"]
        )
        t_stat_auprg, p_val_auprg = stats.ttest_rel(
            auprg_data["Actual"], auprg_data["Cellcount_baseline"]
        )

        mean_diff_auroc = auroc_data["Actual"].mean() - auroc_data["Cellcount_baseline"].mean()
        mean_diff_auprg = auprg_data["Actual"].mean() - auprg_data["Cellcount_baseline"].mean()

        results.append({
            "Metadata_Label":        label,
            "Metadata_Count_0":      _counts_lookup.loc[label, "Metadata_Count_0"],
            "Metadata_Count_1":      _counts_lookup.loc[label, "Metadata_Count_1"],
            "AUROC_mean_Actual":     auroc_data["Actual"].mean(),
            "AUROC_mean_Cellcount":  auroc_data["Cellcount_baseline"].mean(),
            "AUROC_diff":            mean_diff_auroc,
            "AUROC_t_stat":          t_stat_auroc,
            "AUROC_p_value":         p_val_auroc,
            "AUPRG_mean_Actual":     auprg_data["Actual"].mean(),
            "AUPRG_mean_Cellcount":  auprg_data["Cellcount_baseline"].mean(),
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
    print(f"Comparing: Actual (Morphology) vs Cellcount_baseline\n")
    print(df_ttest_results)
    return (df_ttest_results,)


@app.cell
def _(df_ttest_results):
    df_ttest_results
    return


@app.cell
def _(df_ttest_results):
    df_ttest_results[df_ttest_results["AUPRG_significant"]][
        ["Metadata_Label", "AUROC_mean_Actual", "AUROC_mean_Cellcount", "AUROC_p_value", 
         "AUPRG_mean_Actual", "AUPRG_mean_Cellcount", "AUPRG_p_value", "Metadata_Count_0", "Metadata_Count_1"]
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
        (df_ttest_results["AUROC_mean_Actual"] > df_ttest_results["AUROC_mean_Cellcount"]) &
        (df_ttest_results["AUPRG_mean_Actual"] > df_ttest_results["AUPRG_mean_Cellcount"]) &
        (df_ttest_results["Metadata_Count_1"] > 20) &
        (~df_ttest_results["Metadata_Label"].isin(["tissue__liver", "tissue__prostate"]))
    ]["Metadata_Label"]
    list_significative_performance
    return (list_significative_performance,)


@app.cell
def _(df_ttest_results):
    df_ttest_results[
        (df_ttest_results["AUPRG_significant"]) & 
        (df_ttest_results["AUROC_significant"]) &
        (df_ttest_results["AUROC_mean_Actual"] > df_ttest_results["AUROC_mean_Cellcount"]) &
        (df_ttest_results["AUPRG_mean_Actual"] > df_ttest_results["AUPRG_mean_Cellcount"]) &
        (df_ttest_results["Metadata_Count_1"] > 20) &
        (~df_ttest_results["Metadata_Label"].isin(["tissue__liver", "tissue__prostate"]))
    ].to_csv("modules/07_classification/outputs/results_performance.csv")
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
def _(mo):
    mo.md(r"""# Plot comparison AUPRG AUC""")
    return


@app.cell
def _(
    Line2D,
    df_axiom_cellbased_binary_metrics_cv,
    df_axiom_cellfree_binary_metrics_cv,
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

    _ax.text(0.55, 0.4, f"Worse than random\n(n={_n_worse})",       fontsize=10, color="#c0392b", alpha=0.7, fontstyle="italic", ha="center")
    _ax.text(0.65, 0.55, f"Moderately predicted\n(n={_n_moderate})", fontsize=10, color="#b7950b", alpha=0.7, fontstyle="italic", ha="center")
    _ax.text(0.8, 0.75, f"Well predicted\n(n={_n_well})",           fontsize=10, color="#1e8449", alpha=0.7, fontstyle="italic", ha="center")
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
    #plt.savefig(Path.home() / "Downloads" / "auroc_vs_prauc_scatter.png", dpi=300, bbox_inches="tight")
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

        # Pivot for PRAUC
        _pivot_prauc = _df.pivot_table(
            index='Metadata_Label',
            columns='Model_type',
            values='PRAUC',
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
        _pivot_prauc.columns = [f'{col} PRAUC' for col in _pivot_prauc.columns]
        _pivot_auroc.columns = [f'{col} AUROC' for col in _pivot_auroc.columns]

        # Add metadata columns
        _metadata = _df[["Metadata_Label", "Metadata_Count_0", "Metadata_Count_1", "Total_Compounds"]].drop_duplicates()
        _metadata = _metadata.set_index("Metadata_Label").reindex(_ordered_labels)

        # Combine all tables: metadata + PRAUC + AUROC
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

        # Get all PRAUC and AUROC columns for styling
        _prauc_cols = [col for col in _final_table.columns if 'PRAUC' in col]
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
def _(
    create_styled_metrics_table,
    list_significative_performance,
    pred_pandas,
):
    # Filtered dataframe
    _df_all_cellfree = pred_pandas[
        (pred_pandas["Total_Compounds"] > 50) &
        (pred_pandas["Metadata_Count_1"] > 20) &
        (pred_pandas["Endpoint_type"] == "toxcast_cellfree") &
        (pred_pandas["Metadata_Label"].isin(list_significative_performance))
    ].copy()
    create_styled_metrics_table(_df_all_cellfree)
    return


if __name__ == "__main__":
    app.run()
