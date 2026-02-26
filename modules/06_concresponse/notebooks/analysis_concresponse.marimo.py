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
    import glob
    return glob, mo, mticker, np, pd, plt, sns


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Try interdependence score filtering (X: features; Y: Metadata_concentration)""")
    return


@app.cell
def _():
    import ids
    return (ids,)


@app.cell
def _(features_cols, ids, np, subset):
    _X = subset[features_cols].values
    _Y = subset[["Metadata_concentration"]].values  # must be 2D

    _C, _p_vals = ids.compute_IDS(
        _X,
        _Y,
        p_val=True,
        num_tests=1000
    )
    _p_vals
    n_significant = np.sum(_p_vals < 0.05)

    print("Number of p-values < 0.05:", n_significant)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Python code IDS""")
    return


@app.cell
def _(np, pd):
    def well_to_coord(well):
        row_letter = well[0]
        col = int(well[1:])

        row_num = ord(row_letter) - ord("A") + 1  # A=1, B=2, ...

        return pd.Series({"row": row_num, "col": col})
    def get_closest_dmso(comp_fit_coords, dat_ctrl):

        closest_rows = []

        for _, this in comp_fit_coords.iterrows():

            ctrl_same_plate = dat_ctrl[
                dat_ctrl["Metadata_plate"] == this["Metadata_plate"]
            ]

            if ctrl_same_plate.empty:
                continue

            # compute Euclidean distance
            dist = np.sqrt(
                (ctrl_same_plate["row"] - this["row"])**2 +
                (ctrl_same_plate["col"] - this["col"])**2
            )

            ctrl_same_plate = ctrl_same_plate.assign(dist=dist)
            min_dist = dist.min()

            closest = ctrl_same_plate[ctrl_same_plate["dist"] == min_dist]
            closest_rows.append(closest)

        if len(closest_rows) == 0:
            return pd.DataFrame()

        return pd.concat(closest_rows).drop_duplicates()
    return get_closest_dmso, well_to_coord


@app.cell
def _():
    import os
    import multiprocessing as mp

    threads_per_fit = 2

    available = os.cpu_count()
    n_workers = max(1, (available - 1) // threads_per_fit)

    print(
        f"Detected {available} cores → "
        f"{n_workers} workers × {threads_per_fit} threads/fit"
    )
    return (n_workers,)


@app.cell
def _(get_closest_dmso, ids, np, pd):
    def process_compound(compound, dat, dat_ctrl, dat_ctrl_by_plate,
                         dmso_mode, features_cols, output_path):
        # Subset compound
        dat_comp = dat.loc[dat["Metadata_compound"] == compound].copy()
        if dat_comp.empty:
            print(f"No data for {compound}")
            return None
        # Control selection
        if dmso_mode == "closest":
            ctrl_fit = get_closest_dmso(dat_comp, dat_ctrl)
        else:
            plates = dat_comp["Metadata_plate"].unique()
            ctrl_fit = pd.concat(
                [dat_ctrl_by_plate[p] for p in plates if p in dat_ctrl_by_plate],
                ignore_index=True
            )
        print(
            f"Fitting IDS for {compound} "
            f"with {len(dat_comp)} compound and "
            f"{len(ctrl_fit)} DMSO points."
        )
        # Combine
        dat_fit = pd.concat([ctrl_fit, dat_comp]).sort_values(
            "Metadata_log10conc"
        )
        # Prepare IDS inputs
        X = dat_fit[features_cols].values
        Y = dat_fit[["Metadata_log10conc"]].values  # must be 2D
        # Compute IDS
        C, p_vals = ids.compute_IDS(
            X,
            Y,
            p_val=True,
            num_tests=1000
        )
        p_vals_flat = p_vals.flatten()
        C_flat = C.flatten()
        n_significant = np.sum(p_vals_flat < 0.05)
        print(f"{compound}: {n_significant} features with p < 0.05")
        print(p_vals_flat)

        # Write significant features to output file
        sig_mask = p_vals_flat < 0.05
        sig_features = np.array(features_cols)[sig_mask]
        sig_scores   = C_flat[sig_mask]
        sig_pvals    = p_vals_flat[sig_mask]

        sig_df = pd.DataFrame({
            "feature":   sig_features,
            "ids_score": sig_scores,
            "p_value":   sig_pvals
        }).sort_values("p_value")
        sig_df.to_csv(output_path, index=False)
        print(f"Significant features written to {output_path}")

        return {
            "compound":      compound,
            "n_significant": int(n_significant),
            "ids_scores":    C_flat,
            "p_values":      p_vals_flat
        }
    return (process_compound,)


@app.cell
def _(pd, well_to_coord):
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Dose-response BMD pipeline"
        )
        parser.add_argument(
            "--input_path",
            type=str,
            default="modules/03_postprocessing/outputs/level4b_bsc_filtered.parquet",
            help="Path to input parquet file"
        )
        parser.add_argument(
            "--output_path",
            type=str,
            default="modules/06_concresponse/outputs/pods/test.csv",
            help="Path to output parquet file"
        )
        parser.add_argument(
            "--ctrl",
            type=str,
            default="s0617825",
            help="Control compound name (e.g., DMSO id)"
        )
        parser.add_argument(
            "--dmso_mode",
            type=str,
            default="closest",
            choices=["closest", "plate"],
            help="How to select DMSO controls"
        )
        parser.add_argument(
            "--category",
            type=str,
            default="DNA,ER,AGP,RNA,Mito,Brightfield",
            help="Comma-separated list of categories"
        )
        parser.add_argument(
            "--log10_dose",
            type=lambda x: x.lower() == "true",
            default=True,
            help="Use log10 dose (true/false)"
        )
        parser.add_argument(
            "--compounds",
            type=str,
            default="s0654867",
            help="'all' or comma-separated compound list"
        )

        return parser.parse_args()


    args = parse_args()

    # Assign variables
    input_path = args.input_path
    output_path = args.output_path
    ctrl = args.ctrl
    dmso_mode = args.dmso_mode
    category = args.category.split(",")
    log10_dose = args.log10_dose
    compounds = [args.compounds]
    dat = pd.read_parquet(input_path)
    features_cols = [c for c in dat.columns if c.startswith("Feature_")]
    dat[["row", "col"]] = dat["Metadata_well"].apply(well_to_coord)
    dat_ctrl = dat[dat["Metadata_compound"] == ctrl].copy()
    if compounds == ["all"]:
        compounds = dat["Metadata_compound"].unique().tolist()
        compounds = [c for c in compounds if c != ctrl]
    dat_ctrl_by_plate = None
    if dmso_mode == "plate":
        dat_ctrl_by_plate = {
            plate: df
            for plate, df in dat_ctrl.groupby("Metadata_plate")
        }
    elif dmso_mode != "closest":
        raise ValueError("dmso_mode must be 'closest' or 'plate'")
    return (
        compounds,
        dat,
        dat_ctrl,
        dat_ctrl_by_plate,
        dmso_mode,
        features_cols,
        output_path,
    )


@app.cell
def _(
    compounds,
    dat,
    dat_ctrl,
    dat_ctrl_by_plate,
    dmso_mode,
    features_cols,
    n_workers,
    output_path,
    process_compound,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    results = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = {
            executor.submit(
                process_compound,
                cmpd,
                dat,
                dat_ctrl,
                dat_ctrl_by_plate,
                dmso_mode,
                features_cols,
                output_path
            ): cmpd
            for cmpd in compounds
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            compound = futures[future]
            try:
                results[compound] = future.result()
            except Exception as e:
                print(f"Error processing {compound}: {e}")
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell
def _(mo):
    mo.md(r"""## Results BSC ids filtering""")
    return


@app.cell
def _(glob, pd):
    # Define path pattern
    path_pattern_ids = "modules/06_concresponse/outputs/interdependence_score/*.txt"

    # Get all matching files
    files_ids = glob.glob(path_pattern_ids)

    print(f"Found {len(files_ids)} files.")

    # Read and concatenate
    dt_ids_all_feature = pd.concat(
        [pd.read_csv(_f) for _f in files_ids],
        ignore_index=True
    )

    print(f"Final dataframe shape: {dt_ids_all_feature.shape}")
    return (dt_ids_all_feature,)


@app.cell
def _(dt_ids_all_feature):
    dt_ids_all_feature
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Results PODs feature wise BSC ids filtering""")
    return


@app.cell
def _(pd):
    dt_pod_all_mean = pd.read_parquet("modules/06_concresponse/outputs/pods/pods_bsc_cp.parquet")
    return (dt_pod_all_mean,)


@app.cell
def _(dt_pod_all_mean):
    dt_pod_all_mean
    return


@app.cell
def _(dt_pod_all_mean):
    dt_pod_all_mean["all.pass"].value_counts()
    return


@app.cell
def _(dt_pod_all_mean):
    dt_pod_all_mean[~dt_pod_all_mean["all.pass"]]
    return


@app.cell
def _(glob, pd):
    # Define path pattern
    path_pattern = "modules/06_concresponse/outputs/pods/feature_wise/*feature_wise.parquet"

    # Get all matching files
    files = glob.glob(path_pattern)

    print(f"Found {len(files)} files.")

    # Read and concatenate
    dt_pod_all_featurewise = pd.concat(
        [pd.read_parquet(f) for f in files],
        ignore_index=True
    )

    print(f"Final dataframe shape: {dt_pod_all_featurewise.shape}")
    return (dt_pod_all_featurewise,)


@app.cell
def _(dt_pod_all_featurewise):
    dt_pod_all_featurewise
    return


@app.cell
def _(dt_pod_all_featurewise):
    dt_pod_all_featurewise.groupby("Metada_compound").size().reset_index(name="n_rows")
    return


@app.cell
def _(dt_pod_all_featurewise):
    dt_pod_all_featurewise
    return


if __name__ == "__main__":
    app.run()
