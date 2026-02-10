import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as mticker
    import numpy as np
    return mo, mticker, np, pd, plt, sns


@app.cell
def _(mo):
    mo.md(r"""# Analysis of dose response parameters""")
    return


@app.cell
def _(mo):
    mo.md(r"""# Compare DMSO selection [closest, plate] and the filtering [SDres < 3 * SDctrl]""")
    return


@app.cell
def _(pd):
    pods_closest = pd.read_parquet("modules/06_concresponse/results/results_closest/pods/pods_bsc_cp_original_conc.parquet")
    pods_closest_sdres = pd.read_parquet("modules/06_concresponse/results/results_closest_sdres/pods/pods_bsc_cp_original_conc.parquet")
    pods_plate = pd.read_parquet("modules/06_concresponse/results/results_plate/pods/pods_bsc_cp_original_conc.parquet")
    pods_plate_sdres = pd.read_parquet("modules/06_concresponse/results/results_plate_sdres/pods/pods_bsc_cp_original_conc.parquet")
    return pods_closest, pods_closest_sdres, pods_plate, pods_plate_sdres


@app.cell
def _(pods_closest, pods_closest_sdres, pods_plate, pods_plate_sdres):
    pods_closest[["Metadata_compound","cp_POD_um"]]
    pods_closest_sdres[["Metadata_compound","cp_POD_um"]]
    pods_plate[["Metadata_compound","cp_POD_um"]]
    pods_plate_sdres[["Metadata_compound","cp_POD_um"]]
    return


@app.cell
def _(pd, pods_closest, pods_closest_sdres, pods_plate, pods_plate_sdres):
    # Add a column identifying the source
    df1 = pods_closest[["Metadata_compound", "cp_POD_um"]].copy()
    df1["Source"] = "closest"

    df2 = pods_closest_sdres[["Metadata_compound", "cp_POD_um"]].copy()
    df2["Source"] = "closest_sdres"

    df3 = pods_plate[["Metadata_compound", "cp_POD_um"]].copy()
    df3["Source"] = "plate"

    df4 = pods_plate_sdres[["Metadata_compound", "cp_POD_um"]].copy()
    df4["Source"] = "plate_sdres"

    # Concatenate
    df_long = pd.concat([df1, df2, df3, df4], ignore_index=True)

    # Keep only compounds present in all datasets
    common_compounds = (
        df_long.groupby("Metadata_compound")["Source"]
        .nunique()
    )
    common_compounds = common_compounds[common_compounds == 4].index

    df_long = df_long[df_long["Metadata_compound"].isin(common_compounds)]

    return (df_long,)


@app.cell
def _(df_long, mticker, np, plt, sns):
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df_long,
        x="Source",
        y="cp_POD_um",
        showfliers=False
    )

    sns.stripplot(
        data=df_long,
        x="Source",
        y="cp_POD_um",
        jitter=True,
        alpha=0.6,
        size=4
    )
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
    plt.title("POD comparison across methods")
    plt.suptitle("")
    plt.xlabel("")
    plt.ylabel("cp_POD_um")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(
    np,
    pd,
    plt,
    pods_closest,
    pods_closest_sdres,
    pods_plate,
    pods_plate_sdres,
    sns,
):
    def prep(df, label):
        out = df[["Metadata_compound", "cp_POD_um"]].copy()
        out["Source"] = label
        out["active"] = np.isfinite(out["cp_POD_um"])
        return out

    _df_long = pd.concat([
        prep(pods_closest, "closest"),
        prep(pods_closest_sdres, "closest_sdres"),
        prep(pods_plate, "plate"),
        prep(pods_plate_sdres, "plate_sdres"),
    ], ignore_index=True)
    active_counts = (
        _df_long.groupby("Source")["active"]
        .sum()
        .reset_index(name="n_actives")
    )
    plt.figure(figsize=(8, 5))

    ax = sns.barplot(
        data=active_counts,
        x="Source",
        y="n_actives"
    )

    # Add numbers on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2., height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points"
        )

    plt.ylabel("Number of actives")
    plt.xlabel("")
    plt.title("Number of active compounds per method")
    plt.tight_layout()
    plt.show()


    return


@app.cell
def _():
    return


@app.cell
def _(pd):
    # BSC Annotations
    df_bsc_annotations = pd.read_csv("modules/01_preprocessing/outputs/cellpainting_bsc_annotations_merged.tsv", sep="\t")
    list_botanicals = df_bsc_annotations["Plant scode"].unique()
    bsc_control = "s0617825"
    list_botanicals
    return df_bsc_annotations, list_botanicals


@app.cell
def _(df_bsc_annotations):
    df_bsc_annotations[["Metadata_compound", "Plant scode"]]
    return


@app.cell
def _(df_bsc_annotations):
    df_bsc_annotations
    return


@app.cell
def _(df_bsc_annotations, df_long):
    df_plot = df_long.merge(
        df_bsc_annotations[["Metadata_compound", "Plant scode"]],
        on="Metadata_compound",
        how="left"
    )
    df_plot
    return (df_plot,)


@app.cell
def _(df_plot, plt, sns):


    plt.figure(figsize=(14, 6))

    # Boxplot with transparency
    _ax = sns.boxplot(
        data=df_plot,
        x="Plant scode",
        y="cp_POD_um",
        hue="Source",
        showfliers=False,
        linewidth=1
    )

    # Make boxes transparent
    for patch in _ax.patches:
        patch.set_alpha(0.4)

    # Jittered points
    sns.stripplot(
        data=df_plot,
        x="Plant scode",
        y="cp_POD_um",
        hue="Source",
        dodge=True,
        jitter=True,
        alpha=1,
        size=3,
        legend=False
    )

    plt.ylabel("cp_POD_um")
    plt.xlabel("")
    plt.title("POD distribution by plant and method")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(df_bot, plant):
    df_bot.loc[
            df_bot["Plant scode"] == plant,
            "cp_POD_um"
        ].dropna()
    return


@app.cell
def _(df_plot):
    df_plot
    return


@app.cell
def _(df_plot, list_botanicals, np, plt):
    # Get unique botanicals
    botanicals = df_plot[df_plot["Metadata_compound"].isin(list_botanicals)]["Metadata_compound"].unique()

    # Color mapping for Plant scode
    plant_order = (
        df_plot
        .groupby("Plant scode")["cp_POD_um"]
        .median()
        .sort_values()
        .index
    )
    cmap = plt.cm.tab20
    plant_colors = {
        plant: cmap(i % cmap.N)
        for i, plant in enumerate(plant_order)
    }

    # Calculate grid dimensions
    n_botanicals = len(botanicals)
    n_cols = 3  # Adjust as needed
    n_rows = int(np.ceil(n_botanicals / n_cols))

    # Create one big figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten() if n_botanicals > 1 else [axes]

    for idx, botanical in enumerate(botanicals):
        # Filter data for this botanical
        df_botanical = df_plot[df_plot["Plant scode"] == botanical].copy()
    
        if df_botanical.empty:
            axes[idx].axis('off')
            continue
    
        # Get sources present in this botanical's data
        botanical_sources = df_botanical["Source"].unique()
    
        # Create boxplots by Source
        df_botanical.boxplot(
            column="cp_POD_um",
            by="Source",
            grid=False,
            showfliers=False,
            ax=axes[idx]
        )
    
        # Overlay scatter points colored by Plant scode
        for i, source in enumerate(botanical_sources, start=1):
            df_source = df_botanical[df_botanical["Source"] == source]
        
            # Regular points (all data for this source)
            for plant in df_source["Plant scode"].unique():
                y = df_source.loc[
                    df_source["Plant scode"] == plant,
                    "cp_POD_um"
                ].dropna()
            
                if y.empty:
                    continue
            
                x = np.random.normal(i, 0.04, size=len(y))
                axes[idx].plot(
                    x,
                    y,
                    'o',
                    alpha=0.35,
                    markersize=4,
                    color=plant_colors.get(plant, 'gray')
                )
        
            # Triangle markers for botanical compounds
            df_botanical_source = df_plot[
                (df_plot["Source"] == source) & 
                (df_plot["Metadata_compound"] == botanical)
            ]
        
            for plant in df_botanical_source["Metadata_compound"].unique():
                y_bot = df_botanical_source.loc[
                    df_botanical_source["Metadata_compound"] == plant,
                    "cp_POD_um"
                ].dropna()
            
                if y_bot.empty:
                    continue
            
                x_bot = np.random.normal(i, 0.02, size=len(y_bot))
                axes[idx].plot(
                    x_bot,
                    y_bot,
                    marker="^",
                    linestyle="None",
                    markersize=8,
                    alpha=0.95,
                    color=plant_colors.get(plant, 'gray'),
                    markeredgecolor="black"
                )
    
        axes[idx].set_title(f"{botanical}")
        axes[idx].set_xlabel("Source")
        axes[idx].set_ylabel("cp_POD_um (log)")
        axes[idx].set_yscale("log")
        axes[idx].tick_params(axis='x', rotation=45)

    # Hide extra subplots if any
    for idx in range(n_botanicals, len(axes)):
        axes[idx].axis('off')

    fig.suptitle("cp_POD by Source for All Botanicals", fontsize=16, y=1.00)
    plt.tight_layout()
    plt.show()
    return botanical, plant, source


@app.cell
def _(botanical, df_plot, source):
    df_plot[
                (df_plot["Source"] == source) & 
                (df_plot["Metadata_compound"] == botanical)
            ]
    return


if __name__ == "__main__":
    app.run()
