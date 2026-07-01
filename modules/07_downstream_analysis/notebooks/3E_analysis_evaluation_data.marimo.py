import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
    import umap
    return Normalize, np, pd, plt, sns, umap


@app.cell
def _(pd):
    df_level4a = pd.read_parquet("02_processed_data/level4a_normalize.parquet")
    df_level5 = pd.read_parquet("02_processed_data/level5_consensus_level4b_feature_selection.parquet")
    return df_level4a, df_level5


@app.cell
def _(df_level5):
    df_level5["Metadata_compound_scode_concentration_um"] = (
        df_level5["Metadata_compound_scode"].astype(str) + "_" +
        df_level5["Metadata_compound_concentration_um"].astype(str)
    )
    return


@app.cell
def _(df_level4a):
    col_metadata = []
    for _col in df_level4a.columns:
        if "Metadata" in _col:
            col_metadata.append(_col)
            print(_col)
    return


@app.cell
def _(df_level5):
    # Define important arguments
    meta_features = df_level5.columns[df_level5.columns.str.startswith("Metadata_")].to_list()
    features = df_level5.drop(meta_features, axis="columns").columns.tolist()
    return (features,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# [TODO] Analysis replicates""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# Analysis grit""")
    return


@app.cell
def _(pd):
    # Load data
    df_3c_grit_grp_chemical_concentration_feature_all = pd.read_parquet("03_tables/3c_grit_grp_chemical_concentration_feature_all.parquet")

    df_3c_grit_grp_chemical_concentration_feature_Feature_cp_nuclei_acid_512 = pd.read_parquet("03_tables/3c_grit_grp_chemical_concentration_feature_Feature_cp_nuclei_acid_512.parquet")
    df_3c_grit_grp_chemical_concentration_feature_Feature_hoechst_33342 = pd.read_parquet("03_tables/3c_grit_grp_chemical_concentration_feature_Feature_hoechst_33342.parquet")
    df_3c_grit_grp_chemical_concentration_feature_Feature_mito_641_narrow = pd.read_parquet("03_tables/3c_grit_grp_chemical_concentration_feature_Feature_mito_641_narrow.parquet")
    df_3c_grit_grp_chemical_concentration_feature_Feature_phenovue_fluor_488 = pd.read_parquet("03_tables/3c_grit_grp_chemical_concentration_feature_Feature_phenovue_fluor_488.parquet")

    df_3c_grit_grp_chemical_concentration_feature_Feature_brightfield = pd.read_parquet("03_tables/3c_grit_grp_chemical_concentration_feature_Feature_brightfield.parquet")
    return (
        df_3c_grit_grp_chemical_concentration_feature_Feature_brightfield,
        df_3c_grit_grp_chemical_concentration_feature_Feature_cp_nuclei_acid_512,
        df_3c_grit_grp_chemical_concentration_feature_Feature_hoechst_33342,
        df_3c_grit_grp_chemical_concentration_feature_Feature_mito_641_narrow,
        df_3c_grit_grp_chemical_concentration_feature_Feature_phenovue_fluor_488,
        df_3c_grit_grp_chemical_concentration_feature_all,
    )


@app.cell
def _(df_3c_grit_grp_chemical_concentration_feature_all, df_level4a, pd):
    # Define the feature to explore
    df_grit = df_3c_grit_grp_chemical_concentration_feature_all.copy()
    # Join table metadata
    df_grit_annotate = pd.merge(
        df_grit,
        df_level4a,
        left_on=["perturbation"],
        right_on=["Metadata_well_id"],
        how="inner"   # or "left" / "right" / "outer" depending on your need
    )
    return (df_grit_annotate,)


@app.cell
def _(mo):
    mo.md(r"""## Grit per plates""")
    return


@app.cell
def _(df_grit_annotate, plt, sns):
    # Create a common color map
    _cmap = sns.color_palette("coolwarm", as_cmap=True)
    # Compute global min and max across all data
    _vmin = df_grit_annotate["grit"].min()
    _vmax = df_grit_annotate["grit"].max()
    for _plate, df_plate_grit in df_grit_annotate.groupby("plate"):
        df_plate_grit = df_plate_grit.copy()
        df_plate_grit["grit_for_plot"] = df_plate_grit["grit"]

        # Pivot for heatmap
        plate_map_grit = (
            df_plate_grit.pivot(
                index="Metadata_row",
                columns="Metadata_col",
                values="grit_for_plot",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        plt.figure(figsize=(10, 6))

        _ax = sns.heatmap(
            plate_map_grit,
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
            cbar_kws={"label": "grit"},
            linewidths=0.5,
            linecolor="gray",
        )

        # --- Aesthetic details ---
        plt.title(f"Plate {_plate} — grit")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Grit per concentration""")
    return


@app.cell
def _(df_grit_annotate, plt, sns):
    # Define x-ticks explicitly
    x_ticks = [0, 0.01, 0.1, 1, 10, 100, 1000]

    plt.figure(figsize=(8, 6))

    # Scatter or line plot (choose one)
    sns.scatterplot(
        data=df_grit_annotate[df_grit_annotate["Metadata_compound_concentration_um"] == 0],
        x="Metadata_compound_concentration_um",
        y="grit",
        marker=".",
        alpha=0.7
    )

    # Optional: connect points per compound if needed
    sns.lineplot(
         x="Metadata_compound_concentration_um",
         data=df_grit_annotate[df_grit_annotate["Metadata_compound_concentration_um"] != 0],
         y="grit",
         hue="Metadata_compound_scode",  # or similar column
         marker=".",
         alpha=0.7,
         legend=False
     )

    # Log-scale x-axis
    plt.xscale("symlog", linthresh=0.01)
    plt.xlim(-0.002, 10000)
    # Set x-ticks manually
    plt.xticks([0, 0.01, 0.1, 1, 10, 100, 1000],
               ["DMSO", "0.01", "0.1", "1", "10", "100", "1000"])

    plt.xlabel("Compound concentration (µM, log scale)")
    plt.ylabel("Grit")
    plt.title("Grit vs Compound concentration")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## Grit violinplot per features (grouped by chemical_concentration)""")
    return


@app.cell
def _(
    df_3c_grit_grp_chemical_concentration_feature_Feature_brightfield,
    df_3c_grit_grp_chemical_concentration_feature_Feature_cp_nuclei_acid_512,
    df_3c_grit_grp_chemical_concentration_feature_Feature_hoechst_33342,
    df_3c_grit_grp_chemical_concentration_feature_Feature_mito_641_narrow,
    df_3c_grit_grp_chemical_concentration_feature_Feature_phenovue_fluor_488,
    df_3c_grit_grp_chemical_concentration_feature_all,
    pd,
    plt,
    sns,
):
    # Create a list of (label, dataframe) pairs
    dfs_grit = [
        ("All features", df_3c_grit_grp_chemical_concentration_feature_all),
        ("Nuclei Acid 512", df_3c_grit_grp_chemical_concentration_feature_Feature_cp_nuclei_acid_512),
        ("Hoechst 33342", df_3c_grit_grp_chemical_concentration_feature_Feature_hoechst_33342),
        ("Mito 641 Narrow", df_3c_grit_grp_chemical_concentration_feature_Feature_mito_641_narrow),
        ("Phenovue Fluor 488", df_3c_grit_grp_chemical_concentration_feature_Feature_phenovue_fluor_488),
        ("Brightfield", df_3c_grit_grp_chemical_concentration_feature_Feature_brightfield),
    ]

    # Combine them into one DataFrame for plotting
    dfs_grit_all = pd.concat(
        [df.assign(feature_set=label) for label, df in dfs_grit],
        ignore_index=True
    )

    # Make sure the 'grit' column exists in each DataFrame
    assert "grit" in dfs_grit_all.columns, "Column 'grit' not found in one or more DataFrames"

    # Plot violin plot
    plt.figure(figsize=(10, 6))
    _ax = sns.violinplot(
        data=dfs_grit_all,
        x="feature_set",
        y="grit",
        inner="box",       # show a box inside the violin
        cut=0,
        scale="width"
    )
    # --- Compute median per feature set ---
    _grit_medians = dfs_grit_all.groupby("feature_set")["grit"].median().reset_index()
    # --- Add median annotations ---
    for _i, _row in _grit_medians.iterrows():
        _ax.text(
            _i, _row["grit"],               # position
            f"{_row['grit']:.5f}",         # formatted median value
            color="black",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    plt.title("Distribution of GRIT values across feature sets")
    plt.xlabel("Feature set")
    plt.ylabel("GRIT")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# Analysis AP :""")
    return


@app.cell
def _(df_map, np, pd):
    # Load AP
    df_ap = pd.read_parquet("03_tables/3d_map_grp_chemical_concentration_feature_all_ap.parquet")
    df_ap["-log10(p-value)"] = -df_map["corrected_p_value"].apply(np.log10)
    return (df_ap,)


@app.cell
def _(df_ap):
    df_ap
    return


@app.cell
def _(mo):
    mo.md(r"""## AP per plates""")
    return


@app.cell
def _(df_ap, plt, sns):
    # Create a common color map
    _cmap = sns.color_palette("coolwarm", as_cmap=True)
    # Compute global min and max across all data
    _vmin = df_ap["average_precision"].min()
    _vmax = df_ap["average_precision"].max()
    for _plate, df_plate_ap in df_ap.groupby("Metadata_plate"):
        df_plate_ap = df_plate_ap.copy()
        df_plate_ap["ap_for_plot"] = df_plate_ap["average_precision"]

        # Pivot for heatmap
        plate_ap = (
            df_plate_ap.pivot(
                index="Metadata_row",
                columns="Metadata_col",
                values="ap_for_plot",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        plt.figure(figsize=(10, 6))

        _ax = sns.heatmap(
            plate_ap,
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
            cbar_kws={"label": "AP"},
            linewidths=0.5,
            linecolor="gray",
        )

        # --- Aesthetic details ---
        plt.title(f"Plate {_plate} — average precision")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## AP per concentration""")
    return


@app.cell
def _(df_ap, plt, sns):
    # Define x-ticks explicitly
    plt.figure(figsize=(8, 6))

    # Scatter or line plot (choose one)
    sns.scatterplot(
        data=df_ap[df_ap["Metadata_compound_concentration_um"] == 0],
        x="Metadata_compound_concentration_um",
        y="average_precision",
        marker=".",
        alpha=0.7
    )

    # Optional: connect points per compound if needed
    sns.lineplot(
         x="Metadata_compound_concentration_um",
         data=df_ap[df_ap["Metadata_compound_concentration_um"] != 0],
         y="average_precision",
         hue="Metadata_compound_scode",  # or similar column
         marker=".",
         alpha=0.7,
         legend=False
     )

    # Log-scale x-axis
    plt.xscale("symlog", linthresh=0.01)
    plt.xlim(-0.002, 10000)
    # Set x-ticks manually
    plt.xticks([0, 0.01, 0.1, 1, 10, 100, 1000],
               ["DMSO", "0.01", "0.1", "1", "10", "100", "1000"])

    plt.xlabel("Compound concentration (µM, log scale)")
    plt.ylabel("AP")
    plt.title("Average precision vs Compound concentration")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Analysis mAP :""")
    return


@app.cell
def _(np, pd):
    # Load mAP
    df_map = pd.read_parquet("03_tables/3d_map_grp_chemical_concentration_feature_all_map.parquet")
    df_map["-log10(p-value)"] = -df_map["corrected_p_value"].apply(np.log10)
    return (df_map,)


@app.cell
def _(df_level5, df_map, pd):
    # merge dt_level5 and map
    df_level5_map = pd.merge(
        df_level5,
        df_map,
        on=["Metadata_compound_scode_concentration_um"],
        how="left"   # or "left", "right", "outer" depending on what you need
    )
    return (df_level5_map,)


@app.cell
def _(df_map):
    df_map.head()
    return


@app.cell
def _(df_level5_map, plt, sns):
    # Define x-ticks explicitly
    plt.figure(figsize=(8, 6))

    # Scatter or line plot (choose one)
    sns.scatterplot(
        data=df_level5_map,
        x="Metadata_compound_concentration_um",
        y="mean_average_precision",
        marker=".",
        alpha=0.7
    )

    # Optional: connect points per compound if needed
    sns.lineplot(
         x="Metadata_compound_concentration_um",
         data=df_level5_map,
         y="mean_average_precision",
         hue="Metadata_compound_scode",  # or similar column
         marker=".",
         alpha=0.7,
         legend=False
     )

    # Log-scale x-axis
    plt.xscale("symlog", linthresh=0.01)
    plt.xlim(-0.002, 10000)
    # Set x-ticks manually
    plt.xticks([0, 0.01, 0.1, 1, 10, 100, 1000],
               ["DMSO", "0.01", "0.1", "1", "10", "100", "1000"])

    plt.xlabel("Compound concentration (µM, log scale)")
    plt.ylabel("mAP")
    plt.title("Mea average precision vs Compound concentration")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_level5_map, pd, plt, sns):
    # --- 1. Define concentration bins (log-scaled ranges)
    bins = [-0.001, 0.001, 0.1, 1, 10, 100, 1000, 10000]
    labels = ["0_to_0_001uM", "0_001_to_0_1uM", "0_1_to_1uM", "1_to_10uM", "10_to_100uM", "100_to_1000uM", "1000_to_10000uM", "gt_10000uM"]

    df_level5_map["concentration_bin"] = pd.cut(
        df_level5_map["Metadata_compound_concentration_um"],
        bins=bins,
        labels=labels[1:],  # exclude DMSO for numeric bins
        include_lowest=True
    )

    # Optional: manually assign DMSO where concentration == 0 or missing
    #df_level5_map.loc[df_level5_map["Metadata_compound_concentration_um"] == 0, "concentration_bin"] = "0_to_0_001uM"

    # --- 2. Create violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_level5_map,
        x="concentration_bin",
        y="mean_average_precision",
        inner="box",        # adds boxplot inside violin
        scale="width",      # uniform violin widths
        cut=0,              # limits violin to data range
        palette="viridis"   # nice gradient palette
    )

    # --- 3. Customize axes and labels
    plt.xlabel("Compound concentration range (µM)")
    plt.ylabel("Mean Average Precision (mAP)")
    plt.title("Distribution of mAP across concentration ranges")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# [TODO moove to an other file] Profile analysis""")
    return


@app.cell
def _(Normalize, df_level5_map, features, pd, plt, sns):
    # --- Extract the feature matrix ---
    feature_data = df_level5_map[features]

    # --- Extract channel info from feature names ---
    channels = [f.split("_")[0] if "_" in f else "Unknown" for f in features]

    # --- Channel color palette ---
    unique_channels = sorted(set(channels))
    palette = sns.color_palette("Set2", len(unique_channels))
    channel_colors = dict(zip(unique_channels, palette))
    col_colors = pd.Series(channels, index=features).map(channel_colors)

    # --- Clip data range ---
    feature_data_clipped = feature_data.clip(lower=-5, upper=5)

    # --- Right annotation: -log10(p-value) ---
    pvals = df_level5_map["-log10(p-value)"]
    norm = Normalize(vmin=pvals.min(), vmax=pvals.max())
    cmap_pval = plt.colormaps.get_cmap("YlOrRd")  # yellow to red color gradient
    pval_colors = [cmap_pval(norm(v)) for v in pvals]

    # Combine both left and right row annotations
    # Seaborn only supports ONE row color DataFrame, so we merge them side by side
    row_colors_combined = pd.DataFrame({
        "-log10(p-value)": pval_colors
    }, index=df_level5_map.index)

    # --- Create clustermap ---
    sns.set(style="white")
    g = sns.clustermap(
        feature_data_clipped,
        cmap="vlag",
        col_colors=col_colors,
        row_colors=row_colors_combined,
        xticklabels=False,
        yticklabels=False,
        figsize=(13, 8),
        center=0,
        vmin=-5,
        vmax=5,
        col_cluster=False,
        method='ward'
    )

    # --- Add legends ---
    # Channels legend (top)
    for label in unique_channels:
        g.ax_col_dendrogram.bar(0, 0, color=channel_colors[label], label=label, linewidth=0)
    g.ax_col_dendrogram.legend(
        title="Channels",
        loc="center",
        ncol=len(unique_channels),
        bbox_to_anchor=(0.5, 1.1)
    )

    # --- Add colorbar for -log10(p-value) ---
    sm = plt.cm.ScalarMappable(cmap=cmap_pval, norm=norm)
    sm.set_array([])

    # Instead of attaching directly to ax_heatmap,
    # use the figure's add_axes() to manually position it
    # [left, bottom, width, height] — relative to figure size
    cbar_ax = g.fig.add_axes([0.93, 0.2, 0.02, 0.6])
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("-log10(p-value)", rotation=270, labelpad=15)

    # --- Adjust layout ---
    g.fig.subplots_adjust(right=0.9)  # leave room for colorbar

    plt.show()
    return


@app.cell
def _(df_level5_map, features, umap):
    # --- Run UMAP ---
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )
    embedding = reducer.fit_transform(df_level5_map[features])
    return (embedding,)


@app.cell
def _(df_level5_map, embedding, plt):
    # --- Prepare coloring ---
    _pvals = df_level5_map["-log10(p-value)"]
    _norm = plt.Normalize(vmin=_pvals.min(), vmax=_pvals.max())
    cmap = plt.cm.YlOrRd

    # --- Plot UMAP ---
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=_pvals,
        cmap=cmap,
        norm=_norm,
        s=50,
        edgecolor="none",
        alpha=0.9
    )
    plt.title("UMAP Projection Colored by -log10(p-value)", fontsize=14)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # --- Add colorbar ---
    _cbar = plt.colorbar(sc)
    _cbar.set_label("-log10(p-value)", rotation=270, labelpad=15)

    # --- Highlight and label DMSO points ---
    dmso_mask = df_level5_map["Metadata_compound_name"] == "DMSO"
    plt.scatter(
        embedding[dmso_mask, 0],
        embedding[dmso_mask, 1],
        color="none",
        s=80,
        edgecolor="black",
        linewidth=1.5,
        label="DMSO"
    )

    # --- Add legend ---
    plt.legend(frameon=False, loc="best")

    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(df_level5_map, embedding, plt):
    # --- Prepare coloring ---
    _pvals = df_level5_map["-log10(p-value)"]
    _norm = plt.Normalize(vmin=_pvals.min(), vmax=_pvals.max())
    _cmap = plt.cm.YlOrRd

    # --- UMAP embedding (assuming already computed) ---
    plt.figure(figsize=(10, 8))
    _sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=_pvals,
        cmap=_cmap,
        norm=_norm,
        s=50,
        edgecolor="none",
        alpha=0.9
    )

    # --- Titles and labels ---
    plt.title("UMAP Projection Colored by -log10(p-value)", fontsize=14)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # --- Add colorbar ---
    _cbar = plt.colorbar(_sc)
    _cbar.set_label("-log10(p-value)", rotation=270, labelpad=15)

    # --- Highlight DMSO points ---
    _dmso_mask = df_level5_map["Metadata_compound_name"] == "DMSO"
    plt.scatter(
        embedding[_dmso_mask, 0],
        embedding[_dmso_mask, 1],
        facecolors="none",
        edgecolors="black",
        s=80,
        linewidth=1.5,
        label="DMSO"
    )

    # --- Highlight compounds with missing SMILES ---
    missing_smiles_mask = df_level5_map["Metadata_compound_smiles"].isna()
    plt.scatter(
        embedding[missing_smiles_mask, 0],
        embedding[missing_smiles_mask, 1],
        facecolors="none",
        edgecolors="blue",
        linewidth=1.5,
        s=100,
        label="Botanical extracts"
    )

    # --- Optionally label missing SMILES compounds ---
    for i, row in df_level5_map[missing_smiles_mask].iterrows():
        plt.text(
            embedding[i, 0],
            embedding[i, 1],
            str(row["Metadata_compound_name"]),
            fontsize=8,
            color="blue",
            ha="center",
            va="center",
        )

    # --- Add legend ---
    plt.legend(frameon=False, loc="best")

    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(df_level5_map, embedding, plt):
    import matplotlib.colors as mcolors

    # --- Prepare coloring (log-scaled concentration) ---
    _conc = df_level5_map["Metadata_compound_concentration_um"]

    # Handle zeros or negative values before log scale (replace 0 with small number)
    _conc_safe = _conc.copy()
    _conc_safe[_conc_safe <= 0] = _conc_safe[_conc_safe > 0].min() / 10

    _norm = mcolors.LogNorm(vmin=_conc_safe.min(), vmax=_conc_safe.max())
    _cmap = plt.cm.viridis  # perceptually uniform and great for log scales

    # --- Plot UMAP ---
    plt.figure(figsize=(10, 8))
    _sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=_conc_safe,
        cmap=_cmap,
        norm=_norm,
        s=50,
        edgecolor="none",
        alpha=0.9
    )
    plt.title("UMAP Projection Colored by Compound Concentration (log scale, µM)", fontsize=14)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # --- Add colorbar (log scale) ---
    _cbar = plt.colorbar(_sc)
    _cbar.set_label("Compound Concentration (µM, log scale)", rotation=270, labelpad=15)

    # --- Highlight DMSO points ---
    _dmso_mask = df_level5_map["Metadata_compound_name"] == "DMSO"
    plt.scatter(
        embedding[_dmso_mask, 0],
        embedding[_dmso_mask, 1],
        facecolors="none",    # transparent fill
        edgecolors="black",   # black border
        s=80,
        linewidth=1.5,
        label="DMSO"
    )

    # --- Add legend ---
    plt.legend(frameon=False, loc="best")

    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    app.run()
