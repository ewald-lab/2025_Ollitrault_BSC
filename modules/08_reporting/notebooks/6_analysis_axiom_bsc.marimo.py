import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import umap
    import plotly.express as px
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.gridspec as gridspec
    from matplotlib.gridspec import GridSpec
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    from plotnine import (
        ggplot, aes, theme_classic, geom_boxplot, geom_sina, geom_col, element_blank, scale_y_continuous, theme_minimal,
        geom_jitter,
        labs, theme, element_text, scale_x_discrete, scale_y_log10, geom_text, facet_grid, element_rect, geom_rect, scale_x_continuous
    )
    return (
        StandardScaler,
        aes,
        element_rect,
        element_text,
        facet_grid,
        geom_boxplot,
        geom_col,
        geom_sina,
        geom_text,
        ggplot,
        labs,
        mo,
        mticker,
        np,
        pd,
        plt,
        px,
        scale_y_log10,
        sns,
        theme,
        theme_classic,
        umap,
    )


@app.cell
def _(mo):
    mo.md(r"""# Analysis BSC""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Input""")
    return


@app.cell
def _(pd):
    # Cell Painting profiles
    df_level4_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level4b_bsc_filtered.parquet")
    df_level5_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level5_bsc_filtered.parquet")
    df_level6_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level6_bsc_filtered.parquet")
    return df_level4_bsc, df_level5_bsc, df_level6_bsc


@app.cell
def _(pd):
    # BSC Annotations
    df_bsc_annotations = pd.read_csv("modules/01_preprocessing/outputs/cellpainting_bsc_annotations_merged.tsv", sep="\t")
    list_botanicals = df_bsc_annotations["Plant scode"].unique()
    bsc_control = "s0617825"
    list_botanicals
    return bsc_control, df_bsc_annotations, list_botanicals


@app.cell
def _(df_bsc_annotations):
    df_bsc_annotations
    return


@app.cell
def _(pd):
    # POD Cell Painting
    df_bsc_cp_pod = pd.read_parquet("modules/06_concresponse/outputs/pods/pods_bsc_cp_original_conc.parquet")
    # POD Metadata MT
    df_bsc_mtt_pod = pd.read_parquet("modules/06_concresponse/outputs/pods/pods_bsc_Metadata_mtt_normalized_original_conc.parquet")
    # POD Metadata LDH
    df_bsc_ldh_pod = pd.read_parquet("modules/06_concresponse/outputs/pods/pods_bsc_Metadata_ldh_normalized_original_conc.parquet")
    return df_bsc_cp_pod, df_bsc_ldh_pod, df_bsc_mtt_pod


@app.cell
def _(df_bsc_annotations, df_bsc_cp_pod, df_bsc_ldh_pod, df_bsc_mtt_pod):
    # Merge BSC Annotations and POD cp
    df_bsc_cp_pod_annotations = df_bsc_cp_pod.merge(
        df_bsc_annotations,
        on="Metadata_compound",
        how="left"
    )
    # Merge BSC Annotations and POD mt
    df_bsc_mtt_pod_annotations = df_bsc_mtt_pod.merge(
        df_bsc_annotations,
        on="Metadata_compound",
        how="left"
    )
    # Merge BSC Annotations and POD ldh
    df_bsc_ldh_pod_annotations = df_bsc_ldh_pod.merge(
        df_bsc_annotations,
        on="Metadata_compound",
        how="left"
    )
    return (
        df_bsc_cp_pod_annotations,
        df_bsc_ldh_pod_annotations,
        df_bsc_mtt_pod_annotations,
    )


@app.cell
def _(pd):
    # POD Cell Painting
    df_bsc_cp_pod_channel = pd.read_parquet("modules/06_concresponse/outputs/pods/pods_bsc_cp_all_original_conc.parquet")
    df_bsc_cp_pod_channel
    return (df_bsc_cp_pod_channel,)


@app.cell
def _(df_bsc_cp_pod, df_bsc_ldh_pod, df_bsc_mtt_pod):
    df_bsc_cp_pod_renamed = df_bsc_cp_pod.rename(
        columns={
            "cp_POD": "cp_POD",
            "cp_POD_um": "cp_POD_um",
            "all.pass": "cp_all.pass",
            "adv.incr": "cp_adv.incr"
        }
    )
    df_bsc_mtt_pod_renamed = df_bsc_mtt_pod.rename(
        columns={
            "Metadata_mtt_normalized_POD": "mtt_POD",
            "Metadata_mtt_normalized_POD_um": "mtt_POD_um",
            "all.pass": "mtt_all.pass",
            "adv.incr": "mtt_adv.incr"
        }
    )

    df_bsc_ldh_pod_renamed = df_bsc_ldh_pod.rename(
        columns={
            "Metadata_ldh_normalized_POD": "ldh_POD",
            "Metadata_ldh_normalized_POD_um": "ldh_POD_um",
            "all.pass": "ldh_all.pass",
            "adv.incr": "ldh_adv.incr"
        }
    )
    return (
        df_bsc_cp_pod_renamed,
        df_bsc_ldh_pod_renamed,
        df_bsc_mtt_pod_renamed,
    )


@app.cell
def _(df_bsc_cp_pod_renamed, df_bsc_ldh_pod_renamed, df_bsc_mtt_pod_renamed):
    df_bsc_all_pod = (
        df_bsc_cp_pod_renamed
        .merge(
            df_bsc_mtt_pod_renamed,
            on="Metadata_compound",
            how="outer",
            suffixes=("", "_mttdup")  # safeguard
        )
        .merge(
            df_bsc_ldh_pod_renamed,
            on="Metadata_compound",
            how="outer",
            suffixes=("", "_ldhdup")
        )
    )
    df_bsc_all_pod
    return (df_bsc_all_pod,)


@app.cell
def _(df_bsc_all_pod, df_bsc_annotations):
    df_bsc_all_pod_annotations = df_bsc_all_pod.merge(
        df_bsc_annotations,
        on="Metadata_compound",
        how="left",
    )
    return (df_bsc_all_pod_annotations,)


@app.cell
def _(df_level4_bsc, df_level5_bsc, df_level6_bsc):
    all_features = [c for c in df_level4_bsc.columns if c.startswith("Feature_")]
    all_metadata_4 = [c for c in df_level4_bsc.columns if c.startswith("Metadata_")]
    all_metadata_5 = [c for c in df_level5_bsc.columns if c.startswith("Metadata_")]
    all_metadata_6 = [c for c in df_level6_bsc.columns if c.startswith("Metadata_")]
    return (all_features,)


@app.cell
def _(mo):
    mo.md(r"""## Plates experiments focus on Botanicals [TODO]""")
    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP level 4 filtered colored by botanicals""")
    return


@app.cell
def _(all_features, df_level4_bsc, pd, umap):
    # --- 1. Extract features and metadata
    X_bsc_4 = df_level4_bsc[all_features].values
    _compound = df_level4_bsc["Metadata_compound"].astype(str)
    _plates = df_level4_bsc["Metadata_plate"].astype(str)
    _concentration = df_level4_bsc["Metadata_concentration"].astype(float)
    # --- 2. Run UMAP for 2D projection
    reducer_bsc_4 = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    embedding_bsc_4 = reducer_bsc_4.fit_transform(X_bsc_4)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_bsc_4 = pd.DataFrame(embedding_bsc_4, columns=["UMAP_1", "UMAP_2"])
    df_umap_bsc_4["Metadata_compound"] = _compound
    df_umap_bsc_4["Metadata_plate"] = _plates
    df_umap_bsc_4["Metadata_concentration"] = _concentration
    df_umap_bsc_4["Metadata_log10conc"] = df_level4_bsc["Metadata_log10conc"].astype(float)
    return (df_umap_bsc_4,)


@app.cell
def _(df_umap_bsc_4, list_botanicals, px):
    # Create a plotting category:
    # botanicals keep their name, others become "Other"
    df_umap_bsc_4["plot_group"] = df_umap_bsc_4["Metadata_compound"].where(
        df_umap_bsc_4["Metadata_compound"].isin(list_botanicals),
        "Other"
    )

    _fig = px.scatter(
        df_umap_bsc_4.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="plot_group",
        title="UMAP of Level 4 BSC Cell Painting Profiles",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    # Force "Other" to gray
    _fig.update_traces(
        selector=dict(name="Other"),
        marker=dict(color="lightgray")
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="BSC",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP level 5 filtered colored by botanicals""")
    return


@app.cell
def _(all_features, df_level5_bsc, pd, umap):
    # --- 1. Extract features and metadata
    X_bsc_5 = df_level5_bsc[all_features].values
    _compound = df_level5_bsc["Metadata_compound"].astype(str)
    _concentration = df_level5_bsc["Metadata_concentration"].astype(float)
    # --- 2. Run UMAP for 2D projection
    reducer_bsc_5 = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    embedding_bsc_5 = reducer_bsc_5.fit_transform(X_bsc_5)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_bsc_5 = pd.DataFrame(embedding_bsc_5, columns=["UMAP_1", "UMAP_2"])
    df_umap_bsc_5["Metadata_compound"] = _compound
    df_umap_bsc_5["Metadata_concentration"] = _concentration
    #df_umap_bsc_5["Metadata_log10conc"] = df_level5_bsc["Metadata_log10conc"].astype(float)
    return (df_umap_bsc_5,)


@app.cell
def _(df_umap_bsc_5, list_botanicals, px):
    # Create a plotting category:
    # botanicals keep their name, others become "Other"
    df_umap_bsc_5["plot_group"] = df_umap_bsc_5["Metadata_compound"].where(
        df_umap_bsc_5["Metadata_compound"].isin(list_botanicals),
        "Other"
    )

    _fig = px.scatter(
        df_umap_bsc_5.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="plot_group",
        title="UMAP of Level 5 BSC Cell Painting Profiles",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    # Force "Other" to gray
    _fig.update_traces(
        selector=dict(name="Other"),
        marker=dict(color="lightgray")
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="BSC",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP level 6 aggregated""")
    return


@app.cell
def _(all_features, df_level6_bsc, pd, umap):
    # --- 1. Extract features and metadata
    X_bsc_6 = df_level6_bsc[all_features].values
    _compound = df_level6_bsc["Metadata_compound"].astype(str)
    # --- 2. Run UMAP for 2D projection
    reducer_bsc_6 = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    embedding_bsc_6 = reducer_bsc_6.fit_transform(X_bsc_6)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_bsc_6 = pd.DataFrame(embedding_bsc_6, columns=["UMAP_1", "UMAP_2"])
    df_umap_bsc_6["Metadata_compound"] = _compound
    #df_umap_bsc_5["Metadata_log10conc"] = df_level5_bsc["Metadata_log10conc"].astype(float)
    return (df_umap_bsc_6,)


@app.cell
def _(mo):
    mo.md(r"""### UMAP level 6 aggregated colored by botanicals""")
    return


@app.cell
def _(df_umap_bsc_6, list_botanicals, px):
    # Create a plotting category:
    # botanicals keep their name, others become "Other"
    df_umap_bsc_6["plot_group"] = df_umap_bsc_6["Metadata_compound"].where(
        df_umap_bsc_6["Metadata_compound"].isin(list_botanicals),
        "Other"
    )

    _fig = px.scatter(
        df_umap_bsc_6.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="plot_group",
        title="UMAP of Level 5 BSC Cell Painting Profiles",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    # Force "Other" to gray
    _fig.update_traces(
        selector=dict(name="Other"),
        marker=dict(color="lightgray")
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="BSC",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP level 6 aggregated colored by PODs""")
    return


@app.cell
def _(df_umap_bsc_6):
    df_umap_bsc_6
    return


@app.cell
def _(df_bsc_all_pod):
    df_bsc_all_pod
    return


@app.cell
def _(df_bsc_all_pod, df_umap_bsc_6, list_botanicals, np, px):
    # Merge the dataframes on Metadata_compound
    df_umap_bsc_6_pod = df_umap_bsc_6.merge(df_bsc_all_pod, on="Metadata_compound", how="left")

    # Create a plotting category: botanicals keep their name, others become "Other"
    df_umap_bsc_6_pod["plot_group"] = df_umap_bsc_6_pod["Metadata_compound"].where(
        df_umap_bsc_6_pod["Metadata_compound"].isin(list_botanicals),
        "Other"
    )

    # Function to select the color column based on the "_pass" flags
    def get_color_column(row):
        if row.get("cp_all.pass", False):
            return row.get("cp_POD_um")
        elif row.get("mtt_all.pass", False):
            return row.get("mtt_POD_um")
        elif row.get("ldh_all.pass", False):
            return row.get("ldh_POD_um")
        else:
            return np.nan  # Plotly will ignore NaNs

    df_umap_bsc_6_pod["color_value"] = df_umap_bsc_6_pod.apply(get_color_column, axis=1)

    # Apply log scale for coloring (avoid log(0))
    df_umap_bsc_6_pod["color_value_log"] = np.log10(df_umap_bsc_6_pod["color_value"] + 1e-9)

    # Determine nice tick values for colorbar in original scale
    cmin = df_umap_bsc_6_pod["color_value"].min()
    cmax = df_umap_bsc_6_pod["color_value"].max()
    # Example: 5 ticks evenly spaced in log space
    log_ticks = np.linspace(np.log10(cmin + 1e-9), np.log10(cmax + 1e-9), 6)
    original_ticks = np.round(10**log_ticks, decimals=3)

    # Plot UMAP
    _fig = px.scatter(
        df_umap_bsc_6_pod.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="color_value_log",
        color_continuous_scale=px.colors.sequential.Magma,
        title="UMAP of Level 5 BSC Cell Painting Profiles (log scale POD)",
        opacity=0.8,
        width=850,
        height=650
    )

    # Update colorbar to show original POD values
    _fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar=dict(
            title="POD_um",
            tickvals=log_ticks,
            ticktext=[str(t) for t in original_ticks]
        ),
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()

    return


@app.cell
def _(mo):
    mo.md(r"""## POD Cell Painting VS MT VS LDH""")
    return


@app.cell
def _(df_bsc_all_pod_annotations):
    df_bsc_all_pod_annotations["Metadata_compound"].nunique()
    return


@app.cell
def _(df_bsc_all_pod_annotations):
    # reshape to long format, keeping row identity
    all_pods_um = (
        df_bsc_all_pod_annotations
        .reset_index(drop=False)
        .melt(
            id_vars=["index", "cp_all.pass", "ldh_all.pass", "mtt_all.pass", "cp_adv.incr", "ldh_adv.incr", "mtt_adv.incr"],
            value_vars=["cp_POD_um", "ldh_POD_um", "mtt_POD_um"],
            var_name="POD_Type",
            value_name="POD_Value"
        )
    )
    # conditional filtering by POD type
    all_pods_um = all_pods_um[
        (
            (all_pods_um["POD_Type"] == "cp_POD_um") &
            (all_pods_um["cp_all.pass"] == True) 
        ) |
        (
            (all_pods_um["POD_Type"] == "ldh_POD_um") &
            (all_pods_um["ldh_all.pass"] == True) &
            (all_pods_um["ldh_adv.incr"] == True)
        ) |
        (
            (all_pods_um["POD_Type"] == "mtt_POD_um") &
            (all_pods_um["mtt_all.pass"] == True) &
            (all_pods_um["ldh_adv.incr"] == False)
        )
    ]

    # clean up
    all_pods_um = all_pods_um.drop(
        columns=["index", "cp_all.pass", "ldh_all.pass", "mtt_all.pass"]
    )
    all_pods_um["POD_Type"] = all_pods_um["POD_Type"].map({
        "cp_POD_um": "Morphology",
        "ldh_POD_um": "LDH",
        "mtt_POD_um": "MTT"
    })
    return (all_pods_um,)


@app.cell
def _(all_pods_um):
    # count number of active compounds per assay
    pod_counts = (
        all_pods_um
        .groupby("POD_Type", as_index=False)
        .size()
        .rename(columns={"size": "n_active"})
    )

    # choose a y-position slightly above the max POD per assay
    y_max = (
        all_pods_um
        .groupby("POD_Type")["POD_Value"]
        .max()
        .reset_index()
    )

    pod_counts = pod_counts.merge(y_max, on="POD_Type")

    # place bar slightly above data (log-safe multiplier)
    pod_counts["y_bar"] = pod_counts["POD_Value"] * 1.5
    return


@app.cell
def _(
    aes,
    all_pods_um,
    bsc_control,
    df_bsc_all_pod_annotations,
    element_rect,
    element_text,
    facet_grid,
    geom_boxplot,
    geom_col,
    geom_sina,
    geom_text,
    ggplot,
    labs,
    scale_y_log10,
    theme,
    theme_classic,
):
    # Calculate statistics
    if False:
        total_chemicals = df_bsc_all_pod_annotations[
            df_bsc_all_pod_annotations["Metadata_compound"] != bsc_control
        ]["Metadata_compound"].nunique()
    
        pod_stats = (
            all_pods_um.groupby("POD_Type")
            .agg(n_active=("POD_Value", "count"))
            .reset_index()
        )
        pod_stats["pct_active"] = (pod_stats["n_active"] / total_chemicals) * 100
    
        # Prepare data for bars (add a panel identifier)
        bar_data = pod_stats.copy()
        bar_data["panel"] = "Percentage"
        bar_data["value"] = bar_data["pct_active"]
    
        # Prepare data for boxplot
        box_data = all_pods_um.copy()
        box_data["panel"] = "Concentration"
        box_data["value"] = box_data["POD_Value"]
    
        # Combine for faceting
        combined_plot = (
            ggplot()
            # Boxplot in bottom panel
            + geom_sina(
                data=box_data,
                mapping=aes(x='POD_Type', y='value'),
                color='grey',
                alpha=0.5,
                size=1.5
            )
            + geom_boxplot(
                data=box_data,
                mapping=aes(x='POD_Type', y='value'),
                color='red',
                alpha=0.25,
                size=1,
                outlier_alpha=0
            )
            # Percentage bars in top panel
            + geom_col(
                data=bar_data,
                mapping=aes(x='POD_Type', y='value'),
                fill='black',
                width=0.5
            )
            + geom_text(
                data=bar_data,
                mapping=aes(x='POD_Type', y='value', label='pct_active'),
                va='bottom',
                size=11,
                format_string='{:.1f}%',
                nudge_y=2
            )
            # Facet into two panels
            + facet_grid('panel ~ .', scales='free_y')
            + labs(x='Assay Type', y='')
            + theme_classic()
            + theme(
                axis_text_x=element_text(angle=45, hjust=1, color='black', size=14, family='Arial'),
                axis_text_y=element_text(color='black', size=14, family='Arial'),
                axis_title_x=element_text(size=14, color='black', weight='bold', family='Arial'),
                strip_text=element_text(size=12, weight='bold', family='Arial'),
                strip_background=element_rect(fill='white', color='black'),
            )
            + scale_y_log10()  # This will only apply to the concentration panel due to free scales
        )
    
        combined_plot.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### PODB BSC Cell Painting [not filtered]""")
    return


@app.cell
def _(df_bsc_cp_pod_annotations, list_botanicals, np, pd, plt):
    # --------------------------------------------------
    # 1. Order plants by median cp_POD
    # --------------------------------------------------
    _order = (
        df_bsc_cp_pod_annotations
        .groupby("Plant name")["cp_POD"]
        .median()
        .sort_values()
        .index
    )

    df_bsc_cp_pod_annotations["Plant name"] = pd.Categorical(
        df_bsc_cp_pod_annotations["Plant name"],
        categories=_order,
        ordered=True
    )

    # --------------------------------------------------
    # 2. Build Plant scode → Plant name dictionary
    # --------------------------------------------------
    _plant_scode_to_name = (
        df_bsc_cp_pod_annotations[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )

    # --------------------------------------------------
    # 3. Prepare botanical dataframe
    # --------------------------------------------------
    df_bot = df_bsc_cp_pod_annotations[
        df_bsc_cp_pod_annotations["Metadata_compound"].isin(list_botanicals)
    ].copy()

    # CORRECT mapping: Plant scode → Plant name
    df_bot["Plant name mapped"] = df_bot["Metadata_compound"].map(_plant_scode_to_name)
    df_bot = df_bot.dropna(subset=["Plant name mapped"])

    # --------------------------------------------------
    # 4. Colors per plant (consistent)
    # --------------------------------------------------
    cmap = plt.cm.tab20
    plant_colors = {
        plant: cmap(i % cmap.N)
        for i, plant in enumerate(_order)
    }

    # --------------------------------------------------
    # 5. Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    # --- Boxplots ---
    df_bsc_cp_pod_annotations.boxplot(
        column="cp_POD",
        by="Plant name",
        grid=False
    )

    # --- Background points (circles) ---
    for i, plant in enumerate(_order, start=1):
        y = df_bsc_cp_pod_annotations.loc[
            df_bsc_cp_pod_annotations["Plant name"] == plant,
            "cp_POD"
        ].dropna()

        if y.empty:
            continue

        x = np.random.normal(i, 0.04, size=len(y))
        plt.plot(
            x,
            y,
            'o',
            alpha=0.35,
            markersize=4,
            color=plant_colors[plant]
        )

    # --- Botanical points (triangles, same color) ---
    for i, plant in enumerate(_order, start=1):
        y_bot = df_bot.loc[
            df_bot["Plant name mapped"] == plant,
            "cp_POD"
        ].dropna()

        if y_bot.empty:
            continue

        x_bot = np.random.normal(i, 0.02, size=len(y_bot))
        plt.plot(
            x_bot,
            y_bot,
            marker="^",          # triangle
            linestyle="None",
            markersize=8,
            alpha=0.95,
            color=plant_colors[plant],
            markeredgecolor="black"
        )

    # --------------------------------------------------
    # 6. Formatting
    # --------------------------------------------------
    plt.title("cp_POD by Plant Name (Botanicals highlighted)")
    plt.suptitle("")
    plt.xlabel("Plant name")
    plt.ylabel("cp_POD (log)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### PODB BSC MTT [not filtered]""")
    return


@app.cell
def _(
    df_bsc_cp_pod_annotations,
    df_bsc_mtt_pod_annotations,
    list_botanicals,
    mticker,
    np,
    pd,
    plt,
):
    # --------------------------------------------------
    # 1. Order plants by median cp_POD
    # --------------------------------------------------
    _order = (
        df_bsc_cp_pod_annotations
        .groupby("Plant name")["cp_POD"]
        .median()
        .sort_values()
        .index
    )

    df_bsc_mtt_pod_annotations["Plant name"] = pd.Categorical(
        df_bsc_mtt_pod_annotations["Plant name"],
        categories=_order,
        ordered=True
    )

    # --------------------------------------------------
    # 2. Build Plant scode → Plant name dictionary
    # --------------------------------------------------
    _plant_scode_to_name = (
        df_bsc_mtt_pod_annotations[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    # --------------------------------------------------
    # 3. Prepare botanical dataframe
    # --------------------------------------------------
    _df_bot = df_bsc_mtt_pod_annotations[
        df_bsc_mtt_pod_annotations["Metadata_compound"].isin(list_botanicals)
    ].copy()

    # CORRECT mapping: Plant scode → Plant name
    _df_bot["Plant name mapped"] = _df_bot["Metadata_compound"].map(_plant_scode_to_name)
    _df_bot = _df_bot.dropna(subset=["Plant name mapped"])

    # --------------------------------------------------
    # 4. Colors per plant (consistent)
    # --------------------------------------------------
    _cmap = plt.cm.tab20
    _plant_colors = {
        _plant: _cmap(_i % _cmap.N)
        for _i, _plant in enumerate(_order)
    }

    # --------------------------------------------------
    # 5. Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    # --- Boxplots ---
    df_bsc_mtt_pod_annotations.boxplot(
        column="Metadata_mtt_normalized_POD",
        by="Plant name",
        grid=False
    )

    # --- Background points (circles) ---
    for _i, _plant in enumerate(_order, start=1):
        _y = df_bsc_mtt_pod_annotations.loc[
            df_bsc_mtt_pod_annotations["Plant name"] == _plant,
            "Metadata_mtt_normalized_POD"
        ].dropna()

        if _y.empty:
            continue

        _x = np.random.normal(_i, 0.04, size=len(_y))
        plt.plot(
            _x,
            _y,
            'o',
            alpha=0.35,
            markersize=4,
            color=_plant_colors[_plant]
        )

    # --- Botanical points (triangles, same color) ---
    for _i, _plant in enumerate(_order, start=1):
        _y_bot = _df_bot.loc[
            _df_bot["Plant name mapped"] == _plant,
            "Metadata_mtt_normalized_POD"
        ].dropna()

        if _y_bot.empty:
            continue

        _x_bot = np.random.normal(_i, 0.02, size=len(_y_bot))
        plt.plot(
            _x_bot,
            _y_bot,
            marker="s",          # triangle
            linestyle="None",
            markersize=8,
            alpha=0.95,
            color=_plant_colors[_plant],
            markeredgecolor="black"
        )
    # --------------------------------------------------
    # Log scale + readable ticks
    # --------------------------------------------------
    plt.yscale("log")

    _ax = plt.gca()
    # Major ticks at 1, 10, 100, ...
    _ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    # Minor ticks at 2, 3, ..., 9 × 10^n
    _ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    # Formatter: show plain numbers (1, 10, 20, 30, 100, ...)
    _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda _y, _: f"{_y:g}"))
    _ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda _y, _: f"{_y:g}"))

    # --------------------------------------------------
    # 6. Formatting
    # --------------------------------------------------
    plt.title("Metadata_mtt_normalized_POD by Plant Name (Botanicals highlighted)")
    plt.suptitle("")
    plt.xlabel("Plant name")
    plt.ylabel("Metadata_mtt_normalized_POD (log)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    df_bsc_cp_pod_annotations,
    df_bsc_mtt_pod_annotations,
    list_botanicals,
    mticker,
    np,
    pd,
    plt,
):
    # --------------------------------------------------
    # 1. Order plants by median cp_POD
    # --------------------------------------------------
    _order = (
        df_bsc_cp_pod_annotations
        .groupby("Plant name")["cp_POD"]
        .median()
        .sort_values()
        .index
    )

    df_bsc_mtt_pod_annotations["Plant name"] = pd.Categorical(
        df_bsc_mtt_pod_annotations["Plant name"],
        categories=_order,
        ordered=True
    )

    # --------------------------------------------------
    # 2. Build Plant scode → Plant name dictionary
    # --------------------------------------------------
    _plant_scode_to_name = (
        df_bsc_mtt_pod_annotations[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    # --------------------------------------------------
    # 3. Prepare botanical dataframe
    # --------------------------------------------------
    _df_bot = df_bsc_mtt_pod_annotations[
        df_bsc_mtt_pod_annotations["Metadata_compound"].isin(list_botanicals)
    ].copy()

    # CORRECT mapping: Plant scode → Plant name
    _df_bot["Plant name mapped"] = _df_bot["Metadata_compound"].map(_plant_scode_to_name)
    _df_bot = _df_bot.dropna(subset=["Plant name mapped"])

    # --------------------------------------------------
    # 4. Colors per plant (consistent)
    # --------------------------------------------------
    _cmap = plt.cm.tab20
    _plant_colors = {
        _plant: _cmap(_i % _cmap.N)
        for _i, _plant in enumerate(_order)
    }

    # --------------------------------------------------
    # 5. Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    # --- Boxplots ---
    df_bsc_mtt_pod_annotations.boxplot(
        column="Metadata_mtt_normalized_POD_um",
        by="Plant name",
        grid=False
    )

    # --- Background points (circles) ---
    for _i, _plant in enumerate(_order, start=1):
        _y = df_bsc_mtt_pod_annotations.loc[
            df_bsc_mtt_pod_annotations["Plant name"] == _plant,
            "Metadata_mtt_normalized_POD"
        ].dropna()

        if _y.empty:
            continue

        _x = np.random.normal(_i, 0.04, size=len(_y))
        plt.plot(
            _x,
            _y,
            'o',
            alpha=0.35,
            markersize=4,
            color=_plant_colors[_plant]
        )

    # --- Botanical points (triangles, same color) ---
    for _i, _plant in enumerate(_order, start=1):
        _y_bot = _df_bot.loc[
            _df_bot["Plant name mapped"] == _plant,
            "Metadata_mtt_normalized_POD_um"
        ].dropna()

        if _y_bot.empty:
            continue

        _x_bot = np.random.normal(_i, 0.02, size=len(_y_bot))
        plt.plot(
            _x_bot,
            _y_bot,
            marker="s",          # triangle
            linestyle="None",
            markersize=8,
            alpha=0.95,
            color=_plant_colors[_plant],
            markeredgecolor="black"
        )
    # --------------------------------------------------
    # Log scale + readable ticks
    # --------------------------------------------------
    plt.yscale("log")

    _ax = plt.gca()
    # Major ticks at 1, 10, 100, ...
    _ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    # Minor ticks at 2, 3, ..., 9 × 10^n
    _ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    # Formatter: show plain numbers (1, 10, 20, 30, 100, ...)
    _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda _y, _: f"{_y:g}"))
    _ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda _y, _: f"{_y:g}"))

    # --------------------------------------------------
    # 6. Formatting
    # --------------------------------------------------
    plt.title("Metadata_mtt_normalized_POD_um by Plant Name (Botanicals highlighted)")
    plt.suptitle("")
    plt.xlabel("Plant name")
    plt.ylabel("Metadata_mtt_normalized_POD_um (uM)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### POD BSC LDH [not filtered]""")
    return


@app.cell
def _(
    df_bsc_cp_pod_annotations,
    df_bsc_ldh_pod_annotations,
    list_botanicals,
    mticker,
    np,
    pd,
    plt,
):
    # --------------------------------------------------
    # 1. Order plants by median cp_POD
    # --------------------------------------------------
    _order = (
        df_bsc_cp_pod_annotations
        .groupby("Plant name")["cp_POD"]
        .median()
        .sort_values()
        .index
    )

    df_bsc_ldh_pod_annotations["Plant name"] = pd.Categorical(
        df_bsc_ldh_pod_annotations["Plant name"],
        categories=_order,
        ordered=True
    )

    # --------------------------------------------------
    # 2. Build Plant scode → Plant name dictionary
    # --------------------------------------------------
    _plant_scode_to_name = (
        df_bsc_ldh_pod_annotations[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    # --------------------------------------------------
    # 3. Prepare botanical dataframe
    # --------------------------------------------------
    _df_bot = df_bsc_ldh_pod_annotations[
        df_bsc_ldh_pod_annotations["Metadata_compound"].isin(list_botanicals)
    ].copy()

    # CORRECT mapping: Plant scode → Plant name
    _df_bot["Plant name mapped"] = _df_bot["Metadata_compound"].map(_plant_scode_to_name)
    _df_bot = _df_bot.dropna(subset=["Plant name mapped"])

    # --------------------------------------------------
    # 4. Colors per plant (consistent)
    # --------------------------------------------------
    _cmap = plt.cm.tab20
    _plant_colors = {
        _plant: _cmap(_i % _cmap.N)
        for _i, _plant in enumerate(_order)
    }

    # --------------------------------------------------
    # 5. Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    # --- Boxplots ---
    df_bsc_ldh_pod_annotations.boxplot(
        column="Metadata_ldh_normalized_POD",
        by="Plant name",
        grid=False
    )

    # --- Background points (circles) ---
    for _i, _plant in enumerate(_order, start=1):
        _y = df_bsc_ldh_pod_annotations.loc[
            df_bsc_ldh_pod_annotations["Plant name"] == _plant,
            "Metadata_ldh_normalized_POD"
        ].dropna()

        if _y.empty:
            continue

        _x = np.random.normal(_i, 0.04, size=len(_y))
        plt.plot(
            _x,
            _y,
            'o',
            alpha=0.35,
            markersize=4,
            color=_plant_colors[_plant]
        )

    # --- Botanical points (triangles, same color) ---
    for _i, _plant in enumerate(_order, start=1):
        _y_bot = _df_bot.loc[
            _df_bot["Plant name mapped"] == _plant,
            "Metadata_ldh_normalized_POD"
        ].dropna()

        if _y_bot.empty:
            continue

        _x_bot = np.random.normal(_i, 0.02, size=len(_y_bot))
        plt.plot(
            _x_bot,
            _y_bot,
            marker="D",          # triangle
            linestyle="None",
            markersize=8,
            alpha=0.95,
            color=_plant_colors[_plant],
            markeredgecolor="black"
        )
    # --------------------------------------------------
    # Log scale + readable ticks
    # --------------------------------------------------
    plt.yscale("log")

    _ax = plt.gca()
    # Major ticks at 1, 10, 100, ...
    _ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    # Minor ticks at 2, 3, ..., 9 × 10^n
    _ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    # Formatter: show plain numbers (1, 10, 20, 30, 100, ...)
    _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda _y, _: f"{_y:g}"))
    _ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda _y, _: f"{_y:g}"))

    # --------------------------------------------------
    # 6. Formatting
    # --------------------------------------------------
    plt.title("Metadata_ldh_normalized_POD by Plant Name (Botanicals highlighted)")
    plt.suptitle("")
    plt.xlabel("Plant name")
    plt.ylabel("Metadata_ldh_normalized_POD (log)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    df_bsc_cp_pod_annotations,
    df_bsc_ldh_pod_annotations,
    list_botanicals,
    mticker,
    np,
    pd,
    plt,
):
    # --------------------------------------------------
    # 1. Order plants by median cp_POD
    # --------------------------------------------------
    _order = (
        df_bsc_cp_pod_annotations
        .groupby("Plant name")["cp_POD"]
        .median()
        .sort_values()
        .index
    )

    df_bsc_ldh_pod_annotations["Plant name"] = pd.Categorical(
        df_bsc_ldh_pod_annotations["Plant name"],
        categories=_order,
        ordered=True
    )

    # --------------------------------------------------
    # 2. Build Plant scode → Plant name dictionary
    # --------------------------------------------------
    _plant_scode_to_name = (
        df_bsc_ldh_pod_annotations[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    # --------------------------------------------------
    # 3. Prepare botanical dataframe
    # --------------------------------------------------
    _df_bot = df_bsc_ldh_pod_annotations[
        df_bsc_ldh_pod_annotations["Metadata_compound"].isin(list_botanicals)
    ].copy()

    # CORRECT mapping: Plant scode → Plant name
    _df_bot["Plant name mapped"] = _df_bot["Metadata_compound"].map(_plant_scode_to_name)
    _df_bot = _df_bot.dropna(subset=["Plant name mapped"])

    # --------------------------------------------------
    # 4. Colors per plant (consistent)
    # --------------------------------------------------
    _cmap = plt.cm.tab20
    _plant_colors = {
        _plant: _cmap(_i % _cmap.N)
        for _i, _plant in enumerate(_order)
    }

    # --------------------------------------------------
    # 5. Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    # --- Boxplots ---
    df_bsc_ldh_pod_annotations.boxplot(
        column="Metadata_ldh_normalized_POD_um",
        by="Plant name",
        grid=False
    )

    # --- Background points (circles) ---
    for _i, _plant in enumerate(_order, start=1):
        _y = df_bsc_ldh_pod_annotations.loc[
            df_bsc_ldh_pod_annotations["Plant name"] == _plant,
            "Metadata_ldh_normalized_POD_um"
        ].dropna()

        if _y.empty:
            continue

        _x = np.random.normal(_i, 0.04, size=len(_y))
        plt.plot(
            _x,
            _y,
            'o',
            alpha=0.35,
            markersize=4,
            color=_plant_colors[_plant]
        )

    # --- Botanical points (triangles, same color) ---
    for _i, _plant in enumerate(_order, start=1):
        _y_bot = _df_bot.loc[
            _df_bot["Plant name mapped"] == _plant,
            "Metadata_ldh_normalized_POD_um"
        ].dropna()

        if _y_bot.empty:
            continue

        _x_bot = np.random.normal(_i, 0.02, size=len(_y_bot))
        plt.plot(
            _x_bot,
            _y_bot,
            marker="D",          # triangle
            linestyle="None",
            markersize=8,
            alpha=0.95,
            color=_plant_colors[_plant],
            markeredgecolor="black"
        )
    # --------------------------------------------------
    # Log scale + readable ticks
    # --------------------------------------------------
    plt.yscale("log")

    _ax = plt.gca()
    _ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10))
    )
    _ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:g}")
    )
    _ax.yaxis.set_minor_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:g}")
    )

    # --------------------------------------------------
    # 6. Formatting
    # --------------------------------------------------
    plt.title("Metadata_ldh_normalized_POD_um by Plant Name (Botanicals highlighted)")
    plt.suptitle("")
    plt.xlabel("Plant name")
    plt.ylabel("Metadata_ldh_normalized_POD_um (uM)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### POD BSC Cell Painting MT LDH [filtered]""")
    return


@app.cell
def _(df_bsc_all_pod_annotations, list_botanicals, mticker, np, pd, plt):
    # --------------------------------------------------
    # 0. Keep only valid values for log scale
    # --------------------------------------------------
    _df = df_bsc_all_pod_annotations.copy()

    _df = _df[
        (_df["cp_POD"] > 0) |
        (_df["mtt_POD"] > 0) |
        (_df["ldh_POD"] > 0)
    ]

    # --------------------------------------------------
    # 1. Order plants by median cp_POD (constituents)
    # --------------------------------------------------
    _order = (
        _df
        .groupby("Plant name")["cp_POD"]
        .median()
        .sort_values()
        .index
    )

    _df["Plant name"] = pd.Categorical(
        _df["Plant name"],
        categories=_order,
        ordered=True
    )

    # --------------------------------------------------
    # 2. Plant scode → Plant name dictionary
    # --------------------------------------------------
    _plant_scode_to_name = (
        _df[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )

    # --------------------------------------------------
    # 3. Botanical dataframe (mapped safely)
    # --------------------------------------------------
    _df_bot = _df[_df["Metadata_compound"].isin(list_botanicals)].copy()

    _df_bot["Plant name mapped"] = _df_bot["Metadata_compound"].map(_plant_scode_to_name)
    _df_bot = _df_bot.dropna(subset=["Plant name mapped"])

    # --------------------------------------------------
    # 4. Colors per plant
    # --------------------------------------------------
    _cmap = plt.cm.tab20
    _plant_colors = {
        _plant: _cmap(_i % _cmap.N)
        for _i, _plant in enumerate(_order)
    }

    # --------------------------------------------------
    # 5. Plot (HORIZONTAL)
    # --------------------------------------------------
    plt.figure(figsize=(8, 11))

    # --- Boxplots: cp_POD (constituents) HORIZONTAL ---
    _df.boxplot(
        column="cp_POD",
        by="Plant name",
        grid=False,
        vert=False
    )

    # --- Background points (cp_POD constituents) ---
    for _i, _plant in enumerate(_order, start=1):

        _x = _df.loc[
            _df["Plant name"] == _plant,
            "cp_POD"
        ].dropna().to_numpy()

        if _x.size == 0:
            continue

        _y = np.random.normal(_i, 0.04, size=_x.size)
        plt.plot(
            _x, _y,
            "o",
            markersize=4,
            alpha=0.35,
            color=_plant_colors[_plant]
        )

    # --------------------------------------------------
    # 6. Botanical points (marker per assay, pass-filtered)
    # --------------------------------------------------
    _assays = {
        "cp_POD": {
            "shift": 0.00,
            "marker": "^",
            "label": "cp_POD",
            "pass_col": "cp_all.pass",
        },
        "mtt_POD": {
            "shift": -0.07,
            "marker": "s",
            "label": "mtt_POD",
            "pass_col": "mtt_all.pass",
        },
        "ldh_POD": {
            "shift": +0.07,
            "marker": "D",
            "label": "ldh_POD",
            "pass_col": "ldh_all.pass",
        },
    }

    _used_labels = set()

    for _i, _plant in enumerate(_order, start=1):

        _dfp = _df_bot[_df_bot["Plant name mapped"] == _plant]
        if _dfp.empty:
            continue

        for _assay, _cfg in _assays.items():

            # --------------------------------------------------
            # PASS FILTER (this is the key addition)
            # --------------------------------------------------
            _df_pass = _dfp[
                (_dfp[_cfg["pass_col"]] == True) &
                (_dfp[_assay].notna())
            ]

            if _df_pass.empty:
                continue

            _x = _df_pass[_assay].to_numpy()

            _y = np.random.normal(
                loc=_i + _cfg["shift"],
                scale=0.015,
                size=_x.size
            )

            _label = _cfg["label"] if _cfg["label"] not in _used_labels else None
            _used_labels.add(_cfg["label"])

            plt.plot(
                _x, _y,
                marker=_cfg["marker"],
                linestyle="None",
                markersize=8,
                alpha=0.95,
                color=_plant_colors[_plant],
                markeredgecolor="black",
                label=_label
            )

    # --------------------------------------------------
    # 7. Legend (OUTSIDE the plot)
    # --------------------------------------------------
    plt.legend(
        title="Botanical assays",
        frameon=True,
        fontsize=9,
        loc="center left",  # <-- CHANGED
        bbox_to_anchor=(1.05, 0.5)  # <-- CHANGED: positioned to the right
    )

    # --------------------------------------------------
    # 8. Log scale with readable ticks (NOW ON X-AXIS)
    # --------------------------------------------------
    plt.xscale("log")

    _ax = plt.gca()
    _ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.xaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10))
    )
    _ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:g}")
    )
    _ax.xaxis.set_minor_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:g}")
    )

    # --------------------------------------------------
    # 9. Formatting
    # --------------------------------------------------
    plt.title("cp_POD by Plant Name\nBotanicals: cp / mtt / ldh (triangles)")
    plt.suptitle("")
    plt.xlabel("POD (in house log scale)")
    plt.ylabel("Plant name")
    plt.tight_layout(rect=[0, 0, 1.25, 1.5])  # <-- ADDED: leave space for legend on the right
    plt.show()
    return


@app.cell
def _(df_bsc_all_pod_annotations, list_botanicals, mticker, np, pd, plt):
    # --------------------------------------------------
    # 0. Keep only valid values for log scale
    # --------------------------------------------------
    _df = df_bsc_all_pod_annotations.copy()

    _df = _df[
        (_df["cp_POD_um"] > 0) |
        (_df["mtt_POD_um"] > 0) |
        (_df["ldh_POD_um"] > 0)
    ]

    # --------------------------------------------------
    # 1. Order plants by median cp_POD (constituents)
    # --------------------------------------------------
    _order = (
        _df
        .groupby("Plant name")["cp_POD_um"]
        .median()
        .sort_values()
        .index
    )

    _df["Plant name"] = pd.Categorical(
        _df["Plant name"],
        categories=_order,
        ordered=True
    )

    # --------------------------------------------------
    # 2. Plant scode → Plant name dictionary
    # --------------------------------------------------
    _plant_scode_to_name = (
        _df[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )

    # --------------------------------------------------
    # 3. Botanical dataframe (mapped safely)
    # --------------------------------------------------
    _df_bot = _df[_df["Metadata_compound"].isin(list_botanicals)].copy()

    _df_bot["Plant name mapped"] = _df_bot["Metadata_compound"].map(_plant_scode_to_name)
    _df_bot = _df_bot.dropna(subset=["Plant name mapped"])
    mat_infos_pods_botanicals = _df_bot[["Plant name mapped","Metadata_compound","cp_POD_um", "cp_all.pass", "mtt_POD_um", "mtt_all.pass", "ldh_POD_um", "ldh_all.pass"]]
    # --------------------------------------------------
    # 4. Colors per plant
    # --------------------------------------------------
    _cmap = plt.cm.tab20
    _plant_colors = {
        _plant: _cmap(_i % _cmap.N)
        for _i, _plant in enumerate(_order)
    }

    # --------------------------------------------------
    # 5. Plot (HORIZONTAL)
    # --------------------------------------------------
    plt.figure(figsize=(8, 11))

    # --- Boxplots: cp_POD (constituents) HORIZONTAL ---
    _df.boxplot(
        column="cp_POD_um",
        by="Plant name",
        grid=False,
        vert=False  # <-- KEY CHANGE
    )

    # --- Background points (cp_POD constituents) ---
    for _i, _plant in enumerate(_order, start=1):

        _x = _df.loc[
            _df["Plant name"] == _plant,
            "cp_POD_um"
        ].dropna().to_numpy()

        if _x.size == 0:
            continue

        _y = np.random.normal(_i, 0.04, size=_x.size)
        plt.plot(
            _x, _y,  # <-- SWAPPED
            "o",
            markersize=4,
            alpha=0.35,
            color=_plant_colors[_plant]
        )

    # --------------------------------------------------
    # 6. Botanical points (marker per assay, pass-filtered)
    # --------------------------------------------------
    _assays = {
        "cp_POD_um": {
            "shift": 0.00,
            "marker": "^",
            "label": "cp_POD",
            "pass_col": "cp_all.pass",
        },
        "mtt_POD_um": {
            "shift": -0.07,
            "marker": "s",
            "label": "mtt_POD",
            "pass_col": "mtt_all.pass",
        },
        "ldh_POD_um": {
            "shift": +0.07,
            "marker": "D",
            "label": "ldh_POD",
            "pass_col": "ldh_all.pass",
        },
    }

    _used_labels = set()

    for _i, _plant in enumerate(_order, start=1):

        _dfp = _df_bot[_df_bot["Plant name mapped"] == _plant]
        if _dfp.empty:
            continue

        for _assay, _cfg in _assays.items():

            # --------------------------------------------------
            # PASS FILTER (this is the key addition)
            # --------------------------------------------------
            _df_pass = _dfp[
                (_dfp[_cfg["pass_col"]] == True) &
                (_dfp[_assay].notna())
            ]

            if _df_pass.empty:
                continue

            _x = _df_pass[_assay].to_numpy()

            _y = np.random.normal(
                loc=_i + _cfg["shift"],
                scale=0.015,
                size=_x.size
            )

            _label = _cfg["label"] if _cfg["label"] not in _used_labels else None
            _used_labels.add(_cfg["label"])

            plt.plot(
                _x, _y,  # <-- SWAPPED
                marker=_cfg["marker"],
                linestyle="None",
                markersize=8,
                alpha=0.95,
                color=_plant_colors[_plant],
                markeredgecolor="black",
                label=_label
            )

    # --------------------------------------------------
    # 7. Legend (OUTSIDE the plot)
    # --------------------------------------------------
    plt.legend(
        title="Botanical assays",
        frameon=True,
        fontsize=9,
        loc="center left",  # <-- CHANGED
        bbox_to_anchor=(1.05, 0.5)  # <-- CHANGED
    )

    # --------------------------------------------------
    # 8. Log scale with readable ticks (NOW ON X-AXIS)
    # --------------------------------------------------
    plt.xscale("log")  # <-- CHANGED FROM yscale
    def minor_formatter(x, pos):
        if x <= 0:
            return ""
        mantissa = x / 10**np.floor(np.log10(x))
        return f"{x:g}" if np.isclose(mantissa, [2,5]).any() else ""

    _ax = plt.gca()
    _ax.set_xscale("log")

    _ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    _ax.xaxis.set_minor_locator(
        mticker.LogLocator(base=10, subs=np.arange(2, 10))
    )

    _ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    _ax.xaxis.set_minor_formatter(mticker.FuncFormatter(minor_formatter))

    # --------------------------------------------------
    # 9. Formatting
    # --------------------------------------------------
    plt.title("cp_POD by Plant Name\nBotanicals: cp / mtt / ldh (triangles)")
    plt.suptitle("")
    plt.xlabel("POD (uM)")  # <-- SWAPPED
    plt.ylabel("Plant name")  # <-- SWAPPED
    plt.tight_layout(rect=[0, 0, 1.5, 1.75])  # <-- ADDED: leave space for legend
    plt.show()
    return (mat_infos_pods_botanicals,)


@app.cell
def _(mat_infos_pods_botanicals):
    mat_infos_pods_botanicals
    return


@app.cell
def _(df_bsc_all_pod_annotations):
    df_bsc_all_pod_annotations["cp_adv.incr"].value_counts()
    return


@app.cell
def _(df_bsc_all_pod_annotations):
    df_bsc_all_pod_annotations[df_bsc_all_pod_annotations["mtt_all.pass"] == True]["mtt_adv.incr"].value_counts()
    return


@app.cell
def _(df_bsc_all_pod_annotations):
    df_bsc_all_pod_annotations[df_bsc_all_pod_annotations["ldh_all.pass"] == True]["ldh_adv.incr"].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""### POD BSC Cell Painting [filtered] [channel]""")
    return


@app.cell
def _(
    df_bsc_all_pod_annotations,
    df_bsc_cp_pod_channel,
    list_botanicals,
    np,
    pd,
    plt,
):
    # --------------------------------------------------
    # 0. Compute Plant order from POD data (log-safe)
    # --------------------------------------------------
    _df_pod = df_bsc_all_pod_annotations.copy()

    _df_pod = _df_pod[
        (_df_pod["cp_POD_um"] > 0) |
        (_df_pod["mtt_POD_um"] > 0) |
        (_df_pod["ldh_POD_um"] > 0)
    ]

    _order_plants = (
        _df_pod
        .groupby("Plant name")["cp_POD_um"]
        .median()
        .sort_values()
        .index
    )
    _scode_to_name = (
        _df_pod[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    # --------------------------------------------------
    # 1. Botanical → Plant name mapping
    # --------------------------------------------------
    _bot_to_plant = (
        _df_pod[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    # --------------------------------------------------
    # 2. Filter BMD data
    # --------------------------------------------------
    _df = df_bsc_cp_pod_channel[
        df_bsc_cp_pod_channel["Metadata_compound"].isin(list_botanicals)
        & (df_bsc_cp_pod_channel["adv.incr"] == True)
        & (df_bsc_cp_pod_channel["all.pass"] == True)
    ].copy()
    # Map botanical → Plant name
    _df["Plant name"] = _df["Metadata_compound"].map(_bot_to_plant)
    # Drop botanicals without plant mapping
    _df = _df.dropna(subset=["Plant name"])

    # --------------------------------------------------
    # 3. Order botanicals by Plant order
    # --------------------------------------------------
    _botanical_order = (
        pd.Series(_df["Metadata_compound"].unique())
        .map(lambda b: _order_plants.get_loc(_bot_to_plant[b])
             if _bot_to_plant[b] in _order_plants else np.nan)
        .sort_values()
        .index
    )

    _botanical_order = (
        pd.Series(_df["Metadata_compound"].unique())
        .iloc[_botanical_order]
        .tolist()
    )

    _df["Metadata_compound"] = pd.Categorical(
        _df["Metadata_compound"],
        categories=_botanical_order,
        ordered=True
    )

    # --------------------------------------------------
    # 4. Colors per gene.id
    # --------------------------------------------------
    _genes = _df["gene.id"].dropna().unique()
    _cmap = plt.cm.tab20

    _gene_colors = {
        _g: _cmap(_i % _cmap.N)
        for _i, _g in enumerate(_genes)
    }

    # --------------------------------------------------
    # 5. Plot
    # --------------------------------------------------
    plt.figure(figsize=(11, 6))
    # --- Boxplot background ---
    _df.boxplot(
        column="cp_all_POD",
        by="Metadata_compound",
        grid=False,
        showfliers=False,
        boxprops=dict(alpha=0.25),
        medianprops=dict(color="black")
    )

    # --- Scatter points (colored by gene.id) ---
    for _i, _botanical in enumerate(_botanical_order, start=1):

        _dfb = _df[_df["Metadata_compound"] == _botanical]
        if _dfb.empty:
            continue

        for _gene, _dfg in _dfb.groupby("gene.id"):

            _y = _dfg["cp_all_POD"].dropna().to_numpy()
            if _y.size == 0:
                continue

            _x = np.random.normal(_i, 0.05, size=_y.size)

            plt.plot(
                _x, _y,
                marker="o",
                linestyle="None",
                markersize=5,
                alpha=0.85,
                color=_gene_colors[_gene],
                label=_gene
            )

    # --------------------------------------------------
    # 6. Legend (unique gene.id)
    # --------------------------------------------------
    _handles, _labels = plt.gca().get_legend_handles_labels()
    _by_label = dict(zip(_labels, _handles))

    plt.legend(
        _by_label.values(),
        _by_label.keys(),
        title="gene.id",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fontsize=9
    )
    # --------------------------------------------------
    # X-axis: show Plant name instead of botanical
    # --------------------------------------------------
    _xticks = range(1, len(_botanical_order) + 1)
    _xlabels = [
        _scode_to_name.get(_scode)  # fallback to scode if missing
        for _scode in _botanical_order
    ]
    # --------------------------------------------------
    # 7. Formatting
    # --------------------------------------------------
    plt.title("BMD values by Botanical\n(Plant order = median cp_POD_um)")
    plt.suptitle("")
    plt.xlabel("Botanical (ordered by Plant name)")
    plt.ylabel("bmd (log)")
    plt.xticks(_xticks, _xlabels, rotation=45, ha="right")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()
    return


@app.cell
def _(
    df_bsc_all_pod_annotations,
    df_bsc_cp_pod_channel,
    list_botanicals,
    mticker,
    np,
    pd,
    plt,
):
    # --------------------------------------------------
    # 0. Compute Plant order from POD data (log-safe)
    # --------------------------------------------------
    _df_pod = df_bsc_all_pod_annotations.copy()

    _df_pod = _df_pod[
        (_df_pod["cp_POD_um"] > 0) |
        (_df_pod["mtt_POD_um"] > 0) |
        (_df_pod["ldh_POD_um"] > 0)
    ]

    _order_plants = (
        _df_pod
        .groupby("Plant name")["cp_POD_um"]
        .median()
        .sort_values()
        .index
    )
    _scode_to_name = (
        _df_pod[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    # --------------------------------------------------
    # 1. Botanical → Plant name mapping
    # --------------------------------------------------
    _bot_to_plant = (
        _df_pod[["Plant scode", "Plant name"]]
        .dropna()
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    # --------------------------------------------------
    # 2. Filter BMD data
    # --------------------------------------------------
    _df = df_bsc_cp_pod_channel[
        df_bsc_cp_pod_channel["Metadata_compound"].isin(list_botanicals)
        & (df_bsc_cp_pod_channel["adv.incr"] == True)
        & (df_bsc_cp_pod_channel["all.pass"] == True)
    ].copy()
    # Map botanical → Plant name
    _df["Plant name"] = _df["Metadata_compound"].map(_bot_to_plant)
    # Drop botanicals without plant mapping
    _df = _df.dropna(subset=["Plant name"])

    # --------------------------------------------------
    # 3. Order botanicals by Plant order
    # --------------------------------------------------
    _botanical_order = (
        pd.Series(_df["Metadata_compound"].unique())
        .map(lambda b: _order_plants.get_loc(_bot_to_plant[b])
             if _bot_to_plant[b] in _order_plants else np.nan)
        .sort_values()
        .index
    )

    _botanical_order = (
        pd.Series(_df["Metadata_compound"].unique())
        .iloc[_botanical_order]
        .tolist()
    )

    _df["Metadata_compound"] = pd.Categorical(
        _df["Metadata_compound"],
        categories=_botanical_order,
        ordered=True
    )

    # --------------------------------------------------
    # 4. Colors per gene.id
    # --------------------------------------------------
    _genes = _df["gene.id"].dropna().unique()
    _cmap = plt.cm.tab20

    _gene_colors = {
        _g: _cmap(_i % _cmap.N)
        for _i, _g in enumerate(_genes)
    }

    # --------------------------------------------------
    # 5. Plot
    # --------------------------------------------------
    plt.figure(figsize=(11, 6))

    # --- Boxplot background ---
    _df.boxplot(
        column="cp_all_POD_um",
        by="Metadata_compound",
        grid=False,
        showfliers=False,
        boxprops=dict(alpha=0.25),
        medianprops=dict(color="black")
    )

    # --- Scatter points (colored by gene.id) ---
    for _i, _botanical in enumerate(_botanical_order, start=1):

        _dfb = _df[_df["Metadata_compound"] == _botanical]
        if _dfb.empty:
            continue

        for _gene, _dfg in _dfb.groupby("gene.id"):

            _y = _dfg["cp_all_POD_um"].dropna().to_numpy()
            if _y.size == 0:
                continue

            _x = np.random.normal(_i, 0.05, size=_y.size)

            plt.plot(
                _x, _y,
                marker="o",
                linestyle="None",
                markersize=5,
                alpha=0.85,
                color=_gene_colors[_gene],
                label=_gene
            )

    # --------------------------------------------------
    # 6. Legend (unique gene.id)
    # --------------------------------------------------
    _handles, _labels = plt.gca().get_legend_handles_labels()
    _by_label = dict(zip(_labels, _handles))

    plt.legend(
        _by_label.values(),
        _by_label.keys(),
        title="gene.id",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fontsize=9
    )
    # --------------------------------------------------
    # X-axis: show Plant name instead of botanical
    # --------------------------------------------------
    _xticks = range(1, len(_botanical_order) + 1)
    _xlabels = [
        _scode_to_name.get(_scode)  # fallback to scode if missing
        for _scode in _botanical_order
    ]
    # --------------------------------------------------
    # 8. Log scale with readable ticks
    # --------------------------------------------------
    plt.yscale("log")

    _ax = plt.gca()
    _ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10))
    )
    _ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:g}")
    )
    _ax.yaxis.set_minor_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:g}")
    )
    # --------------------------------------------------
    # 7. Formatting
    # --------------------------------------------------
    plt.title("BMD values by Botanical\n(Plant order = median cp_POD_um)")
    plt.suptitle("")
    plt.xlabel("Botanical (ordered by Plant name)")
    plt.ylabel("bmd (uM)")
    plt.xticks(_xticks, _xlabels, rotation=45, ha="right")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Phenotypic space Heatmap""")
    return


@app.cell
def _(df_level6_bsc):
    df_level6_bsc
    return


@app.cell
def _(all_features, df_level6_bsc):
    df_level6_bsc[all_features]
    return


@app.cell
def _(df_bsc_all_pod_annotations):
    df_bsc_all_pod_annotations[["Metadata_compound", "Plant scode"]]
    return


@app.cell
def _(df_bsc_annotations):
    df_bsc_annotations
    return


@app.cell
def _(df_level6_bsc_plantscode):
    df_level6_bsc_plantscode["Plant scode"][0]
    return


@app.cell
def _(df_level6_bsc_plantscode, np):
    # Replace empty strings with NaN first, then fill with Plant name, then with Metadata_compound
    df_level6_bsc_plantscode["Plant scode"] = (
        df_level6_bsc_plantscode["Plant scode"]
        .replace('', np.nan)
        .fillna(df_level6_bsc_plantscode["Plant name"])
        .fillna(df_level6_bsc_plantscode["Metadata_compound"])
    )

    # Verify - should have no NaN now
    print(f"Remaining NaN: {df_level6_bsc_plantscode['Plant scode'].isna().sum()}")
    return


@app.cell
def _(
    StandardScaler,
    all_features,
    df_bsc_all_pod_annotations,
    df_level6_bsc,
    pd,
    plt,
    sns,
):
    _df_level6_bsc_plantscode = df_level6_bsc.merge(df_bsc_all_pod_annotations[["Metadata_compound", "Plant scode", "Plant name", "Constituent"]], on="Metadata_compound", how="left")
    #print(_df_level6_bsc_plantscode[["Metadata_compound", "Plant scode", "Plant name", "Constituent"]])
    # Create mapping from Metadata_compound to Plant name
    _compound_to_plant_name = (
        df_bsc_all_pod_annotations[["Plant scode", "Plant name"]]
        .drop_duplicates()
        .dropna()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )

    # Fill Plant name only where it's NaN
    mask = _df_level6_bsc_plantscode["Constituent"].isna()
    _df_level6_bsc_plantscode.loc[mask, "Constituent"] = (
        _df_level6_bsc_plantscode.loc[mask, "Metadata_compound"]
        .map(_compound_to_plant_name)
    )
    mask = _df_level6_bsc_plantscode["Plant name"].isna()
    _df_level6_bsc_plantscode.loc[mask, "Plant name"] = (
        _df_level6_bsc_plantscode.loc[mask, "Metadata_compound"]
        .map(_compound_to_plant_name)
    )
    _df_level6_bsc_plantscode["Row_label"] = _df_level6_bsc_plantscode.apply(
        lambda row: f"{row['Constituent']} ({row['Metadata_compound']})" 
                    if pd.notna(row['Metadata_compound']) and pd.notna(row['Plant name'])
                    else f"{row['Constituent']} (-- Botanical --)"
                    if pd.notna(row['Metadata_compound']) and pd.isna(row['Plant name'])
                    else row['Metadata_compound'],
        axis=1
    )

    # Select features for clustering
    features_data = _df_level6_bsc_plantscode[all_features]

    # Standardize features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        features_data, #scaler.fit_transform(features_data),
        columns=all_features,
        index=_df_level6_bsc_plantscode["Metadata_compound"]
    )

    # Get row labels - use the new Row_label column
    row_labels = _df_level6_bsc_plantscode["Row_label"]

    # Create color palette for Plant scode
    unique_plant_scodes = _df_level6_bsc_plantscode["Plant name"].dropna().unique()
    n_colors = len(unique_plant_scodes)
    palette = sns.color_palette("tab20", n_colors) if n_colors <= 20 else sns.color_palette("husl", n_colors)
    plant_scode_colors = dict(zip(unique_plant_scodes, palette))

    # Create color annotation for rows
    row_colors = _df_level6_bsc_plantscode["Plant name"].map(plant_scode_colors)

    # Create cluster map with row colors
    sns.set(font_scale=0.7)
    cg = sns.clustermap(
        features_data,
        method='ward',          # linkage method
        metric='euclidean',     # distance metric
        cmap='RdBu_r',          # Red = high, Blue = low
        vmin=-2,                # force color scale
        vmax=2,
        figsize=(12, 10),
        row_cluster=True,
        col_cluster=True,
        xticklabels=False,      # hide column names
        row_colors=row_colors,  # Add color annotation
        yticklabels=row_labels, # Set row labels directly with new labels
        cbar_pos=(0.02, 0.8, 0.03, 0.15)  # Move colorbar to avoid overlap
    )

    # Adjust rotation if needed
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7)

    # Create legend for Plant scode colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plant_scode_colors[ps], label=ps) 
                       for ps in sorted(unique_plant_scodes)]

    # Add legend outside the plot
    cg.ax_heatmap.legend(
        handles=legend_elements, 
        title='Plant name',
        bbox_to_anchor=(1.35, 1), 
        loc='upper left',
        frameon=True,
        fontsize=6,
        title_fontsize=7
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Chemical space colored by POD (Cell Painting) POD (mtt) POD (ldh)""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## Clustering Profiles BSC""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## Clustering Chemicals BSC""")
    return


@app.cell
def _(mo):
    mo.md(r"""## UpSet Plot Clustering Profiles and chemicals BSC""")
    return


@app.cell
def _(mo):
    mo.md(r"""# Analysis Axiom""")
    return


@app.cell
def _():
    # annotations anlysis
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis Axiom VS BSC""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Input""")
    return


@app.cell
def _(pd):
    df_level4a_axiom = pd.read_parquet("02_processed_data/level4a_axiom.parquet")
    return (df_level4a_axiom,)


@app.cell
def _(df_level4a_axiom):
    df_level4a_axiom
    return


@app.cell
def _(df_level4a_bsc):
    df_level4a_bsc
    return


@app.cell
def _(df_level4_bsc, df_level4a_axiom, pd):
    # Add source column
    df_level4a_axiom["Dataset"] = "AXIOM"
    df_level4_bsc["Dataset"]   = "BSC"

    # Bind rows
    df_level4 = pd.concat(
        [df_level4a_axiom, df_level4_bsc],
        axis=0,
        ignore_index=True
    )
    return (df_level4,)


@app.cell
def _(mo):
    mo.md(r"""## Analysis : UMAP""")
    return


@app.cell
def _(all_features, df_level4, pd, umap):
    # --- 1. Extract features and metadata
    X = df_level4[all_features].values

    _plates = df_level4["Metadata_plate"].astype(str)
    _concentration = df_level4["Metadata_concentration"].astype(float)
    _log10conc = df_level4["Metadata_log10conc"].astype(float)
    _dataset = df_level4["Dataset"].astype(str)

    # --- 2. Run UMAP for 2D projection
    reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    embedding = reducer.fit_transform(X)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])
    df_umap["Metadata_plate"] = _plates
    df_umap["Metadata_concentration"] = _concentration
    df_umap["Metadata_log10conc"] = _log10conc
    df_umap["Dataset"] = _dataset
    return (df_umap,)


@app.cell
def _(df_umap, px):
    # --- 4. Create interactive Plotly scatter plot
    _fig = px.scatter(
        df_umap,
        x="UMAP_1",
        y="UMAP_2",
        color="Dataset",
        title="UMAP of Level 4 Cell Painting Profiles Colored by Dataset (axiom vs BSC)",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_sequence=px.colors.qualitative.Safe  # nice, distinct palette
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="Dataset",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


if __name__ == "__main__":
    app.run()
