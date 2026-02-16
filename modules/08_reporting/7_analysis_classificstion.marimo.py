import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl 
    import pandas as pd
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
    return (
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
def _(auc, pd, precision_recall_curve, roc_auc_score):
    def compute_metrics(y_pred, y_actual, y_prob):
        try:
            auroc = roc_auc_score(y_actual, y_prob)
        except ValueError:
            auroc = None

        try:
            precision, recall, _ = precision_recall_curve(y_actual, y_prob)
            prauc = auc(recall, precision)
        except ValueError:
            prauc = None

        return auroc, prauc
    def process_metrics(pred: pd.DataFrame):
        # Get unique class balance rows
        class_balance = (
            pred[["Metadata_Label", "Metadata_Count_0", "Metadata_Count_1", "Model_type"]]
            .drop_duplicates()
        )
        # Group by Label, Model_type, AND k_fold to compute metrics per fold
        fold_metrics = (
            pred
            .groupby(["Metadata_Label", "Model_type", "k_fold"], as_index=False)
            .agg({
                "y_pred": list,
                "y_actual": list,
                "y_prob": list,
            })
        )
        # Compute metrics for each fold
        # Compute metrics for each fold (single pass)
        fold_metrics[["AUROC", "PRAUC"]] = fold_metrics.apply(
            lambda row: pd.Series(
                compute_metrics(row["y_pred"], row["y_actual"], row["y_prob"])
            ),
            axis=1
        )
        # Now aggregate across folds: compute mean and std
        result = (
            fold_metrics
            .groupby(["Metadata_Label", "Model_type"], as_index=False)
            .agg({
                "AUROC": ["mean", "std"],
                "PRAUC": ["mean", "std"]
            })
        )
        # Flatten column names
        result.columns = [
            "Metadata_Label",
            "Model_type",
            "AUROC",
            "AUROC_std",
            "PRAUC",
            "PRAUC_std"
        ]
        # Join back class balance
        result = result.merge(
            class_balance,
            on=["Metadata_Label", "Model_type"],
            how="left"
        )
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

    _plot_auroc.show()
    _plot_prauc.show()
    return


@app.cell
def _(pred_pandas):
    pred_pandas
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
    mo.md(r"""### Tisue (Total_Compounds > 100 & Metadata_Count_1 > 20)""")
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
def _(create_styled_metrics_table, pred_pandas):
    # Filtered dataframe
    _df_tissue_cytotox = pred_pandas[
        (pred_pandas["Total_Compounds"] > 100) &
        (pred_pandas["Metadata_Count_1"] > 20) &
        (pred_pandas["Endpoint_type"] == "toxcast_cytotox") &
        (pred_pandas["Metadata_Label"].str.contains("tissue", na=False))
    ].copy()
    create_styled_metrics_table(_df_tissue_cytotox)
    return


@app.cell
def _(mo):
    mo.md(r"""### Cell (Total_Compounds > 100 & Metadata_Count_1 > 20)""")
    return


@app.cell
def _(create_styled_metrics_table, pred_pandas):
    # Filtered dataframe
    _df_cell_cytotox = pred_pandas[
        (pred_pandas["Total_Compounds"] > 100) &
        (pred_pandas["Metadata_Count_1"] > 20) &
        (pred_pandas["Endpoint_type"] == "toxcast_cytotox") &
        (pred_pandas["Metadata_Label"].str.contains("cell", na=False))
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
    mo.md(r"""### Tox21""")
    return


@app.cell
def _(create_styled_metrics_table, pred_pandas):
    # Filtered dataframe
    _df_tox21_cellbased = pred_pandas[
        (pred_pandas["Endpoint_type"] == "toxcast_cellbased")  &
        (pred_pandas["Metadata_Label"].str.contains("TOX21", na=False))
    ].copy()
    create_styled_metrics_table(_df_tox21_cellbased)
    return


@app.cell
def _(mo):
    mo.md(r"""### All (Total_Compounds > 100 & Metadata_Count_1 > 20)""")
    return


@app.cell
def _(create_styled_metrics_table, pred_pandas):
    # Filtered dataframe
    _df_filter_cellbased = pred_pandas[
        (pred_pandas["Total_Compounds"] > 100) &
        (pred_pandas["Metadata_Count_1"] > 20) &
        (pred_pandas["Endpoint_type"] == "toxcast_cellbased")
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
    mo.md(r"""### All (Total_Compounds > 50 & Metadata_Count_1 > 20)""")
    return


@app.cell
def _(create_styled_metrics_table, pred_pandas):
    # Filtered dataframe
    _df_all_cellfree = pred_pandas[
        (pred_pandas["Total_Compounds"] > 50) &
        (pred_pandas["Metadata_Count_1"] > 20) &
        (pred_pandas["Endpoint_type"] == "toxcast_cellfree")
    ].copy()
    create_styled_metrics_table(_df_all_cellfree)
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

    df_pivot_prauc = df_comparison.pivot_table(
        index=["Metadata_Label", "k_fold"],
        columns="Model_type",
        values="PRAUC",
        aggfunc="first"
    ).reset_index()

    # Perform paired t-test for each Metadata_Label
    results = []

    for label in df_pivot_auroc["Metadata_Label"].unique():
        # Filter for this label
        auroc_data = df_pivot_auroc[df_pivot_auroc["Metadata_Label"] == label]
        prauc_data = df_pivot_prauc[df_pivot_prauc["Metadata_Label"] == label]
    
        # Paired t-test for AUROC
        t_stat_auroc, p_val_auroc = stats.ttest_rel(
            auroc_data["Actual"],
            auroc_data["Cellcount_baseline"]
        )
    
        # Paired t-test for PRAUC
        t_stat_prauc, p_val_prauc = stats.ttest_rel(
            prauc_data["Actual"],
            prauc_data["Cellcount_baseline"]
        )
    
        # Calculate mean differences
        mean_diff_auroc = auroc_data["Actual"].mean() - auroc_data["Cellcount_baseline"].mean()
        mean_diff_prauc = prauc_data["Actual"].mean() - prauc_data["Cellcount_baseline"].mean()
    
        results.append({
            "Metadata_Label": label,
            "AUROC_mean_Actual": auroc_data["Actual"].mean(),
            "AUROC_mean_Cellcount": auroc_data["Cellcount_baseline"].mean(),
            "AUROC_diff": mean_diff_auroc,
            "AUROC_t_stat": t_stat_auroc,
            "AUROC_p_value": p_val_auroc,
            "PRAUC_mean_Actual": prauc_data["Actual"].mean(),
            "PRAUC_mean_Cellcount": prauc_data["Cellcount_baseline"].mean(),
            "PRAUC_diff": mean_diff_prauc,
            "PRAUC_t_stat": t_stat_prauc,
            "PRAUC_p_value": p_val_prauc,
            "n_folds": len(auroc_data)
        })

    # Create results dataframe
    df_ttest_results = pd.DataFrame(results)

    # Add significance indicators (Bonferroni correction optional)
    df_ttest_results["AUROC_significant"] = df_ttest_results["AUROC_p_value"] < 0.05
    df_ttest_results["PRAUC_significant"] = df_ttest_results["PRAUC_p_value"] < 0.05

    # Sort by AUROC difference
    df_ttest_results = df_ttest_results.sort_values("AUROC_diff", ascending=False)

    print(f"\nPaired t-test results (n={df_ttest_results['n_folds'].iloc[0]} folds per label)")
    print(f"Comparing: Actual (Morphology) vs Cellcount_baseline\n")
    print(df_ttest_results)
    return (df_ttest_results,)


@app.cell
def _(df_ttest_results):
    df_ttest_results[df_ttest_results["PRAUC_significant"]][
        ["Metadata_Label", "AUROC_mean_Actual", "AUROC_mean_Cellcount", "AUROC_p_value", 
         "PRAUC_mean_Actual", "PRAUC_mean_Cellcount", "PRAUC_p_value"]
    ].sort_values("PRAUC_p_value", ascending=True)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
