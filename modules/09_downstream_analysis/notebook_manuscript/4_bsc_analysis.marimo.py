import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import glob
    import pandas as pd
    import numpy as np
    import umap
    import plotly.express as px
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    from matplotlib.colors import LogNorm
    from scipy.stats import mannwhitneyu, kruskal, wilcoxon
    from itertools import combinations
    import dataframe_image as dfi
    from pathlib import Path
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    return (
        Line2D,
        LogNorm,
        Patch,
        Path,
        combinations,
        dfi,
        glob,
        kruskal,
        mannwhitneyu,
        mo,
        mpatches,
        mticker,
        np,
        pd,
        plt,
        px,
        umap,
        wilcoxon,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Analysis BSC

        This notebook assembles the BSC manuscript analyses from processed Cell
        Painting profiles, feature-wise concentration-response outputs, and
        cytotoxicity POD estimates. Data-preparation cells produce named tables
        and dictionaries; plotting cells consume those objects and save the
        manuscript figures.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Input

        Load processed profiles, BSC annotations, POD estimates, and shared
        plotting configuration. Summary prints are included at the point where
        result tables are created so reruns expose the sample sizes used
        downstream.
        """
    )
    return


@app.cell
def _(pd):
    # Cell Painting profiles at plate, compound, and feature-filtered levels.
    df_level4_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level4b_bsc_filtered_filtered_converted.parquet")
    df_level4_bsc = df_level4_bsc.groupby("Metadata_plate").filter(
        lambda g: ~(g["Metadata_compound_name"] == "DMSO").all()
    )
    df_level5_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level5_bsc_selected_features_filtered_converted.parquet")
    df_level6_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level6_bsc_filtered.parquet")
    df_level5_bsc_all = pd.read_parquet("modules/03_postprocessing/outputs/level5_bsc_all_features_filtered.parquet")
    df_level6_bsc_all = pd.read_parquet("modules/03_postprocessing/outputs/level6_bsc_selected_features_filtered.parquet")

    print(f"Level 4 profiles after DMSO-only plate removal: {df_level4_bsc.shape}")
    print(f"Level 5 selected-feature profiles: {df_level5_bsc.shape}")
    print(f"Level 6 profiles: {df_level6_bsc.shape}")
    return df_level4_bsc, df_level5_bsc, df_level6_bsc, df_level6_bsc_all


@app.cell
def _(pd):
    # BSC annotations link each tested compound to its botanical source.
    df_bsc_annotations = pd.read_csv("modules/01_preprocessing/outputs/cellpainting_bsc_annotations_merged.tsv", sep="\t")
    list_botanicals = df_bsc_annotations["Plant scode"].unique()
    bsc_control = "s0617825"
    print(
        "BSC annotations: "
        f"{df_bsc_annotations['Metadata_compound'].nunique()} compounds, "
        f"{len(list_botanicals)} botanicals"
    )
    return df_bsc_annotations, list_botanicals


@app.cell
def _(df_bsc_annotations):
    dic_botanicals_scode_name = (
        df_bsc_annotations[["Plant scode", "Plant name"]]
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    print(f"Botanical code-name lookup entries: {len(dic_botanicals_scode_name)}")
    return (dic_botanicals_scode_name,)


@app.cell
def _(Path, np):
    # Shared output locations and figure palettes.
    figure_dir = Path("modules/09_downstream_analysis/figures")
    table_dir = Path("modules/09_downstream_analysis/tables")
    downloads_dir = Path.home() / "Downloads"
    for _output_dir in (figure_dir, table_dir):
        _output_dir.mkdir(parents=True, exist_ok=True)

    cp_channel_categories = ["DNA", "ER", "AGP", "RNA", "Mito", "Brightfield"]
    cp_channel_categories_no_brightfield = ["DNA", "ER", "AGP", "RNA", "Mito"]
    cp_channel_colors = {
        "DNA": "#e41a1c",
        "ER": "#377eb8",
        "AGP": "#4daf4a",
        "RNA": "#984ea3",
        "Mito": "#ff7f00",
        "Brightfield": "#a65628",
    }

    botanical_colors = {
        "Kratom": "#e6194B",
        "Ashwagandha": "#3cb44b",
        "Comfrey": "#ffe119",
        "Usnea": "#4363d8",
        "Asian ginseng": "#f58231",
        "Blue cohosh": "#42d4f4",
        "Yohimbe": "#f032e6",
        "Kava": "#fabed4",
        "Milk thistle": "#469990",
        "Goldenseal": "#dcbeff",
        "Aristolochia": "#9A6324",
        "Green tea extract": "#fffac8",
        "Ephedra": "#800000",
    }

    def feature_channel(feature_name, categories=cp_channel_categories):
        """Return the Cell Painting channel encoded in a feature name."""
        for category in categories:
            if category in str(feature_name):
                return category
        return None

    def significance_stars(p_value):
        """Convert a p-value into the manuscript significance label."""
        if np.isnan(p_value):
            return "n/a"
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "ns"
    return (
        botanical_colors,
        cp_channel_categories,
        cp_channel_colors,
        downloads_dir,
        feature_channel,
        figure_dir,
        significance_stars,
        table_dir,
    )


@app.cell
def _(pd):
    # POD estimates from Cell Painting morphology and viability assays.
    df_bsc_cp_pod = pd.read_parquet("modules/04_concresponse/outputs/pods/pods_bsc_cp_original_conc.parquet")
    df_bsc_mtt_pod = pd.read_parquet("modules/04_concresponse/outputs/pods/pods_bsc_Metadata_mtt_normalized_original_conc.parquet")
    df_bsc_ldh_pod = pd.read_parquet("modules/04_concresponse/outputs/pods/pods_bsc_Metadata_ldh_normalized_original_conc.parquet")
    print(f"Cell Painting POD rows: {len(df_bsc_cp_pod)}")
    print(f"MTT POD rows: {len(df_bsc_mtt_pod)}")
    print(f"LDH POD rows: {len(df_bsc_ldh_pod)}")
    return df_bsc_cp_pod, df_bsc_ldh_pod, df_bsc_mtt_pod


@app.cell
def _(Path, glob, pd):
    # Feature-selection interdependence score, one file per compound.
    path_pattern_bsc_interdependence_score = "modules/04_concresponse/outputs/interdependence_score/feature_selection_interdependence_score_bsc_*.txt"

    _files = glob.glob(path_pattern_bsc_interdependence_score)

    df_bsc_interdependence_score = pd.concat(
        [
            pd.read_csv(f).assign(
                Metadata_compound=Path(f).stem.replace(
                    "feature_selection_interdependence_score_bsc_", ""
                )
            )
            for f in _files
        ],
        ignore_index=True,
    )

    print(f"Interdependence score files: {len(_files)}")
    print(f"Interdependence score rows/columns: {df_bsc_interdependence_score.shape}")
    return (df_bsc_interdependence_score,)


@app.cell
def _(glob, pd):
    # Feature-wise Cell Painting POD estimates.
    path_pattern_feature_wise = "modules/04_concresponse/outputs/pods/feature_wise/*feature_wise_original_conc.parquet"

    _files = glob.glob(path_pattern_feature_wise)

    print(f"Feature-wise POD files: {len(_files)}")

    df_bsc_cp_pod_all_featurewise = pd.concat(
        [pd.read_parquet(f) for f in _files],
        ignore_index=True
    )

    print(f"Feature-wise POD rows/columns: {df_bsc_cp_pod_all_featurewise.shape}")
    print(
        "Feature-wise POD compounds: "
        f"{df_bsc_cp_pod_all_featurewise['Metadata_compound'].nunique()}"
    )
    return (df_bsc_cp_pod_all_featurewise,)


@app.cell
def _():
    with open("modules/04_concresponse/outputs/pods/feature_wise/feature_wise_median_permutation_threshold.txt") as _f:
        feature_wise_permutation_threshold = float(_f.read().strip())
    print(f"Feature-wise activity threshold: {feature_wise_permutation_threshold:g} active features")
    return (feature_wise_permutation_threshold,)


@app.cell
def _(df_bsc_annotations, df_bsc_cp_pod, df_bsc_ldh_pod, df_bsc_mtt_pod):
    # Annotation-enriched POD tables, used for checking compound group labels.
    df_bsc_cp_pod_annotations = df_bsc_cp_pod.merge(
        df_bsc_annotations,
        on="Metadata_compound",
        how="left"
    )
    df_bsc_mtt_pod_annotations = df_bsc_mtt_pod.merge(
        df_bsc_annotations,
        on="Metadata_compound",
        how="left"
    )
    df_bsc_ldh_pod_annotations = df_bsc_ldh_pod.merge(
        df_bsc_annotations,
        on="Metadata_compound",
        how="left"
    )
    print(f"Annotated Cell Painting POD rows: {len(df_bsc_cp_pod_annotations)}")
    print(f"Annotated MTT POD rows: {len(df_bsc_mtt_pod_annotations)}")
    print(f"Annotated LDH POD rows: {len(df_bsc_ldh_pod_annotations)}")
    return


@app.cell
def _(pd):
    # Channel-specific Cell Painting POD estimates, retained for inspection.
    df_bsc_cp_pod_channel = pd.read_parquet("modules/04_concresponse/outputs/pods/pods_bsc_cp_all_original_conc.parquet")
    print(f"Channel-level Cell Painting POD rows: {len(df_bsc_cp_pod_channel)}")
    return


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
    print(f"Combined annotated POD table rows: {len(df_bsc_all_pod_annotations)}")
    return (df_bsc_all_pod_annotations,)


@app.cell
def _(df_level4_bsc, df_level5_bsc, df_level6_bsc, df_level6_bsc_all):
    all_selected_features = [c for c in df_level4_bsc.columns if c.startswith("Feature_")]
    all_features = [c for c in df_level6_bsc_all.columns if c.startswith("Feature_")]
    all_metadata_4 = [c for c in df_level4_bsc.columns if c.startswith("Metadata_")]
    all_metadata_5 = [c for c in df_level5_bsc.columns if c.startswith("Metadata_")]
    all_metadata_6 = [c for c in df_level6_bsc.columns if c.startswith("Metadata_")]
    print(f"Selected Cell Painting features: {len(all_selected_features)}")
    print(f"All Cell Painting features: {len(all_features)}")
    print(
        "Metadata columns by level: "
        f"L4={len(all_metadata_4)}, L5={len(all_metadata_5)}, L6={len(all_metadata_6)}"
    )
    return all_features, all_selected_features


@app.cell
def _(all_features, all_selected_features):
    print(
        "Feature selection retained "
        f"{len(all_selected_features)}/{len(all_features)} features"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Plates experiments focus on Botanicals""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Concentration range""")
    return


@app.cell
def _(df_level5_bsc, dic_botanicals_scode_name):
    # Build the concentration summary used in Figure 1A.
    _conc_stats = (
        df_level5_bsc
        .groupby("Metadata_compound")["Metadata_concentration"]
        .agg(["min", "max"])
        .rename(columns={"min": "Min (µM)", "max": "Max (µM)"})
        .reset_index()
    )

    _conc_stats["Plant name"] = _conc_stats["Metadata_compound"].map(dic_botanicals_scode_name)
    _conc_stats = _conc_stats.dropna(subset=["Plant name"])

    _n_constituents = {
        "Asian ginseng": 13, "Ashwagandha": 10, "Comfrey": 9, "Green tea extract": 9,
        "Blue cohosh": 7, "Goldenseal": 7, "Milk thistle": 7, "Kava": 6,
        "Kratom": 5, "Usnea": 5, "Yohimbe": 5, "Aristolochia": 3, "Ephedra": 1
    }

    # ── Hepatotoxicity data ───────────────────────────────────────────────────────
    _hepatotox = {
        "Asian ginseng": "No evidence",
        "Aristolochia": "Unknown",
        "Goldenseal": "Evidence",
        "Kratom": "Evidence",
        "Kava": "Evidence",
        "Milk thistle": "No evidence",
        "Yohimbe": "Some evidence",
        "Ephedra": "Evidence",
        "Green tea extract": "Evidence",
        "Blue cohosh": "Unknown",
        "Usnea": "Evidence",
        "Ashwagandha": "Unknown",
        "Comfrey": "Evidence"
    }

    df_bsc_concentration_summary = _conc_stats[["Plant name", "Min (µM)", "Max (µM)"]].copy()
    df_bsc_concentration_summary["N tested constituents"] = (
        df_bsc_concentration_summary["Plant name"].map(_n_constituents)
    )
    df_bsc_concentration_summary["Hepatotoxicity (literature)"] = (
        df_bsc_concentration_summary["Plant name"].map(_hepatotox)
    )
    df_bsc_concentration_summary = (
        df_bsc_concentration_summary
        .sort_values("N tested constituents", ascending=False)
        .reset_index(drop=True)
    )

    print(f"Figure 1A concentration summary botanicals: {len(df_bsc_concentration_summary)}")
    print(
        "Figure 1A concentration range: "
        f"{df_bsc_concentration_summary['Min (µM)'].min():.3g}-"
        f"{df_bsc_concentration_summary['Max (µM)'].max():.3g} µM"
    )
    return (df_bsc_concentration_summary,)


@app.cell
def _(df_bsc_concentration_summary, dfi, figure_dir, np):
    # Render and export Figure 1A from the prepared concentration summary.
    _df_plants = df_bsc_concentration_summary.copy()
    _global_log_min = np.log10(_df_plants["Min (µM)"].min())
    _global_log_max = np.log10(_df_plants["Max (µM)"].max())

    def _range_bar(_row):
        _log_min = np.log10(_row["Min (µM)"])
        _log_max = np.log10(_row["Max (µM)"])

        _left  = (_log_min - _global_log_min) / (_global_log_max - _global_log_min) * 100
        _width = (_log_max - _log_min)        / (_global_log_max - _global_log_min) * 100

        return (
            f'<div style="position:relative; height:16px; background:#f0f0f0; '
            f'border-radius:4px; width:100%;">'
            f'<div style="position:absolute; left:{_left:.1f}%; width:{_width:.1f}%; '
            f'height:100%; background:linear-gradient(90deg,#52b788,#1a6b3c); '
            f'border-radius:4px; opacity:0.85;"></div>'
            f'<span style="position:absolute; left:50%; transform:translateX(-50%); '
            f'font-size:10px; line-height:16px; color:#333; white-space:nowrap;">'
            f'{_row["Min (µM)"]:.3g} – {_row["Max (µM)"]:.3g} µM</span>'
            f'</div>'
        )

    _df_plants["Concentration range (µM)"] = _df_plants.apply(_range_bar, axis=1)

    _df_display = _df_plants[
        ["Plant name", "Hepatotoxicity (literature)", "N tested constituents", "Concentration range (µM)"]
    ].copy()

    def _hep_color(val):
        if val == "Evidence":
            return "background-color: #f8d7da; color: #721c24;"
        elif val == "Some evidence":
            return "background-color: #fff3cd; color: #856404;"
        elif val == "No evidence":
            return "background-color: #d4edda; color: #155724;"
        else:
            return "background-color: #e2e3e5; color: #383d41;"

    _styled = (
        _df_display.style
        .background_gradient(
            subset=["N tested constituents"],
            cmap="YlGn",
            vmin=0,
            vmax=_df_plants["N tested constituents"].max()
        )
        .applymap(_hep_color, subset=["Hepatotoxicity (literature)"])
        .set_properties(**{
            "font-size": "16px",   # ↑ from 12px
            "text-align": "center",
            "border": "1px solid #e0e0e0",
            "padding": "4px 7px",  # slightly more spacing
        })
        .set_properties(subset=["Plant name"], **{
            "text-align": "left",
            "font-weight": "bold"
        })
        .set_properties(subset=["Concentration range (µM)"], **{
            "min-width": "220px",
            "padding": "4px 8px"
        })
        .set_properties(subset=["N tested constituents"], **{
            "width": "110px",
            "min-width": "90px",
            "max-width": "120px",
        })
        .set_properties(subset=["Hepatotoxicity (literature)"], **{
            "width": "110px",
            "min-width": "90px",
            "max-width": "120px",
            "white-space": "normal",   # allows wrapping
        })
        .format({"Concentration range (µM)": lambda _x: _x})
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", "#2c6e49"),
                ("color", "white"),
                ("font-size", "16px"),
                ("font-weight", "bold"),
                ("text-align", "center"),
                ("padding", "8px 12px"),
                ("border-bottom", "2px solid #1a4731"),
                ("white-space", "normal"),  # allows header wrapping
            ]},
            {"selector": "tbody tr:nth-child(even)", "props": [
                ("background-color", "#f7f7f7"),
            ]},
            {"selector": "tbody tr:hover", "props": [
                ("background-color", "#d4edda"),
            ]},
            {"selector": "table", "props": [
                ("border-collapse", "collapse"),
                ("width", "70%"),
                ("margin", "auto"),
            ]},
        ])
        .hide(axis="index")
    )

    # ── Export ────────────────────────────────────────────────────────────────────
    dfi.export(
        _styled,
        str(figure_dir / "Figure_1_A.png"),
        dpi=300,
        table_conversion="chrome"
    )

    _styled
    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP level 4 filtered colored by botanicals""")
    return


@app.cell
def _(all_selected_features, df_level4_bsc, pd, umap):
    # Compute a reproducible 2D UMAP embedding from selected Cell Painting features.
    X_bsc_4 = df_level4_bsc[all_selected_features].values
    _compound = df_level4_bsc["Metadata_compound"].astype(str).values
    _plates = df_level4_bsc["Metadata_plate"].astype(str).values
    _concentration = df_level4_bsc["Metadata_concentration"].astype(float).values
    _concentration_log = df_level4_bsc["Metadata_log10conc"].astype(float).values

    reducer_bsc_4 = umap.UMAP(
        n_neighbors=50,
        min_dist=0.3,
        metric="cosine",
        random_state=42
    )
    embedding_bsc_4 = reducer_bsc_4.fit_transform(X_bsc_4)

    df_umap_bsc_4 = pd.DataFrame(embedding_bsc_4, columns=["UMAP_1", "UMAP_2"])
    df_umap_bsc_4["Metadata_compound"] = _compound
    df_umap_bsc_4["Metadata_plate"] = _plates
    df_umap_bsc_4["Metadata_concentration"] = _concentration
    df_umap_bsc_4["Metadata_log10conc"] = _concentration_log
    print(f"UMAP embedded profiles: {len(df_umap_bsc_4)}")
    print(f"UMAP embedded compounds: {df_umap_bsc_4['Metadata_compound'].nunique()}")
    return (df_umap_bsc_4,)


@app.cell
def _(df_umap_bsc_4, dic_botanicals_scode_name):
    # Attach plotting labels without mutating the embedding table.
    df_umap_bsc_4_plot = df_umap_bsc_4.copy()
    df_umap_bsc_4_plot["Botanical_Name"] = (
        df_umap_bsc_4_plot["Metadata_compound"].map(dic_botanicals_scode_name)
    )

    df_umap_bsc_4_plot["plot_group"] = df_umap_bsc_4_plot["Botanical_Name"].where(
        df_umap_bsc_4_plot["Botanical_Name"].isin(dic_botanicals_scode_name.values()),
        "Constituents"
    )
    print(
        "UMAP botanical profiles: "
        f"{(df_umap_bsc_4_plot['plot_group'] != 'Constituents').sum()}"
    )
    return (df_umap_bsc_4_plot,)


@app.cell
def _(botanical_colors, df_umap_bsc_4_plot, figure_dir, px):
    # Render and export Figure 2B.
    _botanical_colors = botanical_colors.copy()
    _botanical_colors["Constituents"] = "lightgrey"
    _fig = px.scatter(
        df_umap_bsc_4_plot.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="plot_group",
        title="",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_map=_botanical_colors
    )

    _fig.update_traces(
        marker=dict(size=6, line=dict(width=0))
    )

    _fig.update_layout(
        template="simple_white",
        legend_title_text="",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        font=dict(size=12, color="black"),
    )

    _fig.update_xaxes(showgrid=False, zeroline=False)
    _fig.update_yaxes(showgrid=False, zeroline=False)
    _fig.write_image(
        str(figure_dir / "Figure_2_B.png"),
        scale=4,
    )
    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Features Interdependence score""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Per botanicals""")
    return


@app.cell
def _(
    cp_channel_categories,
    df_bsc_interdependence_score,
    dic_botanicals_scode_name,
    feature_channel,
    list_botanicals,
):
    # Prepare botanical-level interdependence counts by Cell Painting channel.
    _df_inter = df_bsc_interdependence_score[
        df_bsc_interdependence_score["Metadata_compound"].isin(list_botanicals)
    ].copy()
    _df_inter["Metadata_compound"] = _df_inter["Metadata_compound"].map(
        lambda x: dic_botanicals_scode_name.get(x, x)
    )

    _df_inter["category"] = _df_inter["feature"].apply(feature_channel)

    _df_inter_counts = (
        _df_inter[_df_inter["category"].notna()]
        .groupby(["Metadata_compound", "category"])
        .size()
        .reset_index(name="n")
    )

    # Pivot
    _df_inter_pivot = (
        _df_inter_counts.pivot(index="Metadata_compound", columns="category", values="n")
        .fillna(0)
    )
    for _cat in cp_channel_categories:
        if _cat not in _df_inter_pivot.columns:
            _df_inter_pivot[_cat] = 0
    _df_inter_pivot = _df_inter_pivot[cp_channel_categories]

    _df_inter_pivot["total"] = _df_inter_pivot.sum(axis=1)
    df_interdependence_botanical_counts = (
        _df_inter_pivot
        .sort_values("total", ascending=True)
        .drop(columns="total")
    )
    print(
        "Interdependence botanical compounds with channel counts: "
        f"{len(df_interdependence_botanical_counts)}"
    )
    print(
        "Interdependence botanical selected features: "
        f"{int(df_interdependence_botanical_counts.sum(axis=1).sum())}"
    )
    return (df_interdependence_botanical_counts,)


@app.cell
def _(
    cp_channel_categories,
    cp_channel_colors,
    df_interdependence_botanical_counts,
    np,
    plt,
):
    # Render botanical interdependence score plot from prepared counts.
    _df_inter_pivot = df_interdependence_botanical_counts
    _compounds_inter = _df_inter_pivot.index.values
    _y_pos = np.arange(len(_compounds_inter))

    _fig, _ax = plt.subplots(figsize=(8, max(4, len(_compounds_inter) * 0.4)))

    _left = np.zeros(len(_compounds_inter))
    for _cat in cp_channel_categories:
        _vals_bar = _df_inter_pivot[_cat].values
        _ax.barh(_y_pos, _vals_bar, left=_left,
                 color=cp_channel_colors[_cat], height=0.6, alpha=0.85, label=_cat)
        _left += _vals_bar

    for _i, _total in enumerate(_left):
        _ax.text(_total + 0.5, _i, str(int(_total)), va="center", fontsize=8)

    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels(_compounds_inter, fontsize=10)
    _ax.set_xlabel("N features", fontsize=11)
    _ax.set_title("Interdependence score — feature count per channel per botanical", fontsize=12)
    _ax.legend(fontsize=9, loc="lower right")
    _ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""### Per constituents""")
    return


@app.cell
def _(
    cp_channel_categories,
    df_bsc_annotations,
    df_bsc_interdependence_score,
    feature_channel,
    list_botanicals,
):
    # Prepare constituent-level interdependence counts by channel.
    _df_inter = df_bsc_interdependence_score[
        ~df_bsc_interdependence_score["Metadata_compound"].isin(list_botanicals)
    ].copy()

    _df_inter["category"] = _df_inter["feature"].apply(feature_channel)

    _df_inter_counts = (
        _df_inter[_df_inter["category"].notna()]
        .groupby(["Metadata_compound", "category"])
        .size()
        .reset_index(name="n")
    )
    _df_inter_pivot = (
        _df_inter_counts.pivot(index="Metadata_compound", columns="category", values="n")
        .fillna(0)
    )
    for _cat in cp_channel_categories:
        if _cat not in _df_inter_pivot.columns:
            _df_inter_pivot[_cat] = 0
    _df_inter_pivot = _df_inter_pivot[cp_channel_categories]

    _annot_inter = df_bsc_annotations[
        ~df_bsc_annotations["Metadata_compound"].isin(list_botanicals)
    ].copy()
    _plant_order_inter = [p for p in list_botanicals if p in _annot_inter["Plant scode"].values]
    _plant_to_rank_inter = {p: i for i, p in enumerate(_plant_order_inter)}
    _annot_inter["_plant_rank"] = _annot_inter["Plant scode"].map(_plant_to_rank_inter)

    _primary_plant_inter = (
        _annot_inter.sort_values("_plant_rank")
        .drop_duplicates(subset="Metadata_compound", keep="first")
        [["Metadata_compound", "Constituent", "_plant_rank"]]
        .sort_values(["_plant_rank", "Constituent"])
    )

    df_interdependence_constituent_counts = (
        _primary_plant_inter[["Metadata_compound"]]
        .merge(_df_inter_pivot.reset_index(), on="Metadata_compound", how="left")
        .fillna(0)
        .set_index("Metadata_compound")[cp_channel_categories]
    )
    interdependence_constituent_labels = (
        _primary_plant_inter.set_index("Metadata_compound")["Constituent"].to_dict()
    )

    print(
        "Interdependence constituent compounds with channel counts: "
        f"{len(df_interdependence_constituent_counts)}"
    )
    print(
        "Interdependence constituent selected features: "
        f"{int(df_interdependence_constituent_counts.sum(axis=1).sum())}"
    )
    return (
        df_interdependence_constituent_counts,
        interdependence_constituent_labels,
    )


@app.cell
def _(
    cp_channel_categories,
    cp_channel_colors,
    df_interdependence_constituent_counts,
    interdependence_constituent_labels,
    np,
    plt,
):
    # Render constituent interdependence score plot from prepared counts.
    _df_inter_pivot = df_interdependence_constituent_counts
    _compounds_inter = _df_inter_pivot.index.values
    _constituent_labels_inter = [
        interdependence_constituent_labels.get(_compound, _compound)
        for _compound in _compounds_inter
    ]
    _x_pos_inter = np.arange(len(_compounds_inter)) * 0.6
    _fig, _ax = plt.subplots(figsize=(8, max(6, len(_compounds_inter) * 0.20)))

    _left = np.zeros(len(_compounds_inter))
    for _cat in cp_channel_categories:
        _vals_bar = _df_inter_pivot[_cat].values
        _ax.barh(_x_pos_inter, _vals_bar, left=_left,
                 color=cp_channel_colors[_cat], height=0.3, alpha=0.85, label=_cat)
        _left += _vals_bar

    for _i, _total in enumerate(_left):
        _ax.text(_total * 1.05, _x_pos_inter[_i], str(int(_total)), va="center", fontsize=6)

    _ax.set_yticks(_x_pos_inter)
    _ax.set_yticklabels(_constituent_labels_inter, fontsize=9)
    _ax.set_xlabel("N features", fontsize=11)
    _ax.set_title("Interdependence score — feature count per channel per constituent", fontsize=12)
    _ax.legend(fontsize=9, loc="lower right")
    _ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    _ax.set_yticks(_x_pos_inter)
    _ax.set_yticklabels(_constituent_labels_inter, fontsize=9)
    plt.draw()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""## POD Cell Painting VS MT VS LDH""")
    return


@app.cell
def _(mo):
    mo.md(r"""### POD CP MT LDH Table""")
    return


@app.cell
def _(
    cp_channel_categories,
    df_bsc_all_pod_annotations,
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    df_level6_bsc,
    dic_botanicals_scode_name,
    feature_channel,
    feature_wise_permutation_threshold,
    pd,
):
    # Assemble the manuscript POD summary table before styling/export.
    _df_cp_passed = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()
    _n_active = _df_cp_passed.groupby("Metadata_compound").size()
    _valid    = _n_active[_n_active >= feature_wise_permutation_threshold].index

    _all_compounds_fw = _df_cp_passed["Metadata_compound"].unique()

    _df_cp_valid = _df_cp_passed[_df_cp_passed["Metadata_compound"].isin(_valid)].copy()

    _cp_pac_series = (
        _df_cp_valid
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .rename("CP POD (µM)")
    )
    # Compounds that exist but didn't pass threshold → NaN in CP POD
    _cp_pac_series = _cp_pac_series.reindex(
        pd.Index(_all_compounds_fw).union(_cp_pac_series.index)
    )

    _df_cp_valid["category"] = _df_cp_valid["gene.id"].apply(feature_channel)

    _df_pac_cat = (
        _df_cp_valid[_df_cp_valid["category"].notna()]
        .groupby(["Metadata_compound", "category"])["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .unstack(level="category")
        .rename(columns={_cat: f"PAC {_cat} (µM)" for _cat in cp_channel_categories})
    )
    _df_pac_cat = _df_pac_cat.reindex(_cp_pac_series.index)

    _mtt_series = (
        df_bsc_mtt_pod[df_bsc_mtt_pod["all.pass"] == True]
        .groupby("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .min()
        .rename("MTT POD (µM)")
    )

    _ldh_series = (
        df_bsc_ldh_pod[df_bsc_ldh_pod["all.pass"] == True]
        .groupby("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .min()
        .rename("LDH POD (µM)")
    )

    _df_level6_bsc_plantscode = df_level6_bsc.merge(
        df_bsc_all_pod_annotations[["Metadata_compound", "Plant scode", "Plant name", "Constituent"]],
        on="Metadata_compound", how="left"
    )
    _compound_to_plant_name = (
        df_bsc_all_pod_annotations[["Plant scode", "Plant name"]]
        .drop_duplicates()
        .dropna()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    _mask = _df_level6_bsc_plantscode["Constituent"].isna()
    _df_level6_bsc_plantscode.loc[_mask, "Constituent"] = (
        _df_level6_bsc_plantscode.loc[_mask, "Metadata_compound"].map(_compound_to_plant_name)
    )
    _botanical_scodes = set(dic_botanicals_scode_name.keys())
    _constituent_lookup = (
        _df_level6_bsc_plantscode[["Metadata_compound", "Constituent"]]
        .drop_duplicates()
        .dropna(subset=["Constituent"])
        .set_index("Metadata_compound")["Constituent"]
        .to_dict()
    )

    _all_cpd_ids = (
        set(_cp_pac_series.index)
        | set(_mtt_series.index)
        | set(_ldh_series.index)
    )
    _all_cpd_index = pd.Index(sorted(_all_cpd_ids))

    _df_table = (
        pd.concat([
            _cp_pac_series.reindex(_all_cpd_index),
            _df_pac_cat.reindex(_all_cpd_index),
            _mtt_series.reindex(_all_cpd_index),
            _ldh_series.reindex(_all_cpd_index),
        ], axis=1)
        .reset_index()
        .rename(columns={"index": "Compound ID"})
    )

    _df_table["Group"] = _df_table["Compound ID"].apply(
        lambda _c: "Botanical" if _c in _botanical_scodes else "Constituent"
    )
    _df_table["Name"] = _df_table["Compound ID"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c) if _c in _botanical_scodes
                   else _constituent_lookup.get(_c, _c)
    )

    _df_table["Below threshold"] = _df_table["Compound ID"].apply(
        lambda _c: "✗" if _c in _all_compounds_fw and _c not in _valid else ""
    )

    _cat_pac_cols = [
        f"PAC {_cat} (µM)"
        for _cat in cp_channel_categories
        if f"PAC {_cat} (µM)" in _df_table.columns
    ]
    _col_order    = (
        ["Group", "Compound ID", "Name", "Below threshold", "CP POD (µM)"]
        + _cat_pac_cols
        + ["MTT POD (µM)", "LDH POD (µM)"]
    )
    _df_table = (
        _df_table[_col_order]
        .sort_values(["Group", "CP POD (µM)"])
        .reset_index(drop=True)
    )
    df_table_results_pods = _df_table
    pod_table_gradient_columns = ["CP POD (µM)"] + _cat_pac_cols + ["MTT POD (µM)", "LDH POD (µM)"]

    print(f"POD summary table rows: {len(df_table_results_pods)}")
    print(f"POD summary botanical rows: {(df_table_results_pods['Group'] == 'Botanical').sum()}")
    print(f"POD summary constituent rows: {(df_table_results_pods['Group'] == 'Constituent').sum()}")
    print(f"Compounds above CP feature threshold: {len(_valid)}")
    return df_table_results_pods, pod_table_gradient_columns


@app.cell
def _(df_table_results_pods, pd, pod_table_gradient_columns, table_dir):
    # Style and export the prepared POD summary table.
    _df_table = df_table_results_pods

    def _color_group(_val):
        return "background-color: #d4edda" if _val == "Botanical" else "background-color: #d0e8f7"

    def _color_threshold(_val):
        return "background-color: #ffe0e0; color: #cc0000; font-weight: bold" if _val == "✗" else ""

    def _fmt(_val):
        if pd.isna(_val): return "—"
        return f"{_val:.3g}"

    _gradient_cols = pod_table_gradient_columns
    _fmt_dict      = {_col: _fmt for _col in _gradient_cols}

    _styled = (
        _df_table.style
        .format(_fmt_dict)
        .applymap(_color_group,      subset=["Group"])
        .applymap(_color_threshold,  subset=["Below threshold"])
        .set_properties(**{"font-size": "10px", "text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [
                ("font-size", "10px"), ("text-align", "center"),
                ("background-color", "#f0f0f0"), ("font-weight", "bold"),
                ("white-space", "nowrap")
            ]},
            {"selector": "td:nth-child(3)", "props": [("text-align", "left")]},
        ])
        .highlight_null(color="#f5f5f5")
    )

    for _col in _gradient_cols:
        _col_vals = pd.to_numeric(_df_table[_col], errors="coerce").dropna()
        if len(_col_vals) > 0:
            _styled = _styled.background_gradient(
                subset=[_col], cmap="RdYlGn",
                vmin=_col_vals.min(), vmax=_col_vals.max()
            )
    _styled.to_excel(str(table_dir / "Tables_S1.xlsx"), engine="openpyxl")
    _df_table.to_csv(str(table_dir / "Tables_S1.csv"), index=False)
    _styled
    return


@app.cell
def _(mo):
    mo.md(r"""### POD CP MT LDH Table stats""")
    return


@app.cell
def _(
    df_bsc_annotations,
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    dic_botanicals_scode_name,
    feature_wise_permutation_threshold,
    list_botanicals,
    np,
    pd,
    table_dir,
):
    # Build Table 1: active constituent and botanical POD summaries.
    _constituents_in_data = (
        df_bsc_cp_pod_all_featurewise["Metadata_compound"]
        .unique()
    )
    _constituents_in_data = [c for c in _constituents_in_data if c not in list_botanicals]

    _annot = df_bsc_annotations[
        df_bsc_annotations["Metadata_compound"].isin(_constituents_in_data)
    ].copy()

    _df_const_passed = df_bsc_cp_pod_all_featurewise[
        (df_bsc_cp_pod_all_featurewise["all.pass"] == True) &
        (df_bsc_cp_pod_all_featurewise["Metadata_compound"].isin(_constituents_in_data))
    ]

    # Keep only constituents with n active features > threshold
    _n_active = (
        _df_const_passed.groupby("Metadata_compound")["cp_POD_um"]
        .count()
        .reset_index(name="n_active")
    )
    _active_compounds = _n_active.loc[
        _n_active["n_active"] > feature_wise_permutation_threshold,
        "Metadata_compound"
    ].values

    _constituent_pac = (
        _df_const_passed[_df_const_passed["Metadata_compound"].isin(_active_compounds)]
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reset_index(name="PAC")
    )

    _pac_lookup = _constituent_pac.set_index("Metadata_compound")["PAC"].to_dict()

    # ── 2. PAC per botanical ──────────────────────────────────────────────────────
    _df_bot_passed = df_bsc_cp_pod_all_featurewise[
        (df_bsc_cp_pod_all_featurewise["all.pass"] == True) &
        (df_bsc_cp_pod_all_featurewise["Metadata_compound"].isin(list_botanicals))
    ]

    _n_active_bot = (
        _df_bot_passed.groupby("Metadata_compound")["cp_POD_um"]
        .count()
        .reset_index(name="n_active")
    )
    _active_botanicals = _n_active_bot.loc[
        _n_active_bot["n_active"] > feature_wise_permutation_threshold,
        "Metadata_compound"
    ].values

    _botanical_pac_dict = (
        _df_bot_passed[_df_bot_passed["Metadata_compound"].isin(_active_botanicals)]
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .to_dict()
    )
    # ── MTT and LDH POD per botanical ────────────────────────────────────────────
    _mtt_pac_dict = (
        df_bsc_mtt_pod[
            (df_bsc_mtt_pod["all.pass"] == True) &
            (df_bsc_mtt_pod["Metadata_compound"].isin(list_botanicals))
        ]
        .groupby("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .min()
        .to_dict()
    )

    _ldh_pac_dict = (
        df_bsc_ldh_pod[
            (df_bsc_ldh_pod["all.pass"] == True) &
            (df_bsc_ldh_pod["Metadata_compound"].isin(list_botanicals))
        ]
        .groupby("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .min()
        .to_dict()
    )
    # ── 3. Map constituents to their botanicals ───────────────────────────────────
    _constituent_to_plants = _annot.groupby("Metadata_compound")["Plant scode"].apply(list).to_dict()

    _rows = []
    for _compound, _plants in _constituent_to_plants.items():
        if _compound not in _pac_lookup:
            continue
        for _plant in _plants:
            if _plant in list_botanicals:
                _rows.append({
                    "Metadata_compound": _compound,
                    "Plant name": dic_botanicals_scode_name[_plant],
                    "Plant scode": _plant,
                    "PAC": _pac_lookup[_compound]
                })
    _df_long = pd.DataFrame(_rows)  # ← build _df_long FIRST

    # ── NOW compute counts (depend on _df_long) ───────────────────────────────────
    _total_known = (
        df_bsc_annotations[df_bsc_annotations["Plant scode"].isin(list_botanicals)]
        .groupby("Plant scode")["Metadata_compound"]
        .nunique()
        .to_dict()
    )
    _active_counts = (
        _df_long.groupby("Plant scode")["Metadata_compound"]
        .nunique()
        .to_dict()
    )

    # ── 4. Order botanicals by median constituent PAC ─────────────────────────────
    # Base order on active constituents, then append botanicals with no active ones
    _order_active = (
        _df_long.groupby("Plant name")["PAC"]
        .median()
        .sort_values()
        .index.tolist()
    )

    # All botanical names
    _all_plant_names = [dic_botanicals_scode_name[s] for s in list_botanicals]

    # Botanicals with no active constituents — append at the end
    _order_inactive = [p for p in _all_plant_names if p not in _order_active]

    _order = _order_active + _order_inactive
    _rows_table = []

    for _plant in _order:
        _scode = next((k for k, v in dic_botanicals_scode_name.items() if v == _plant), None)
        if _scode is None:
            continue

        # ── Constituent CP POD ────────────────────────────────────────────────────
        _cp_vals = _df_long.loc[_df_long["Plant name"] == _plant, "PAC"].dropna().values
        _total_known_n = _total_known.get(_scode, 0)
        _active_cp_n   = _active_counts.get(_scode, 0)
        _cp_pct        = f"{_active_cp_n}/{_total_known_n} ({100*_active_cp_n/_total_known_n:.0f}%)" if _total_known_n > 0 else "—"
        _cp_median     = f"{np.median(_cp_vals):.2f}" if len(_cp_vals) > 0 else "—"

        # ── Botanical CP POD (whole extract) ──────────────────────────────────────
        _bot_cp  = _botanical_pac_dict.get(_scode)
        _bot_cp_str = f"{_bot_cp:.2f}" if _bot_cp is not None and pd.notna(_bot_cp) else "—"

        # ── Botanical MTT POD (whole extract) ─────────────────────────────────────
        _bot_mtt = _mtt_pac_dict.get(_scode)
        _bot_mtt_str = f"{_bot_mtt:.2f}" if _bot_mtt is not None and pd.notna(_bot_mtt) else "—"

        # ── Botanical LDH POD (whole extract) ─────────────────────────────────────
        _bot_ldh = _ldh_pac_dict.get(_scode)
        _bot_ldh_str = f"{_bot_ldh:.2f}" if _bot_ldh is not None and pd.notna(_bot_ldh) else "—"

        # ── Constituent MTT POD ───────────────────────────────────────────────────
        _mtt_vals = (
            df_bsc_mtt_pod[
                (df_bsc_mtt_pod["all.pass"] == True) &
                (df_bsc_mtt_pod["Metadata_compound"].isin(
                    _annot.loc[_annot["Plant scode"] == _scode, "Metadata_compound"]
                ))
            ]["Metadata_mtt_normalized_POD_um"].dropna().values
        )
        _active_mtt_n = len(_mtt_vals)
        _mtt_pct      = f"{_active_mtt_n}/{_total_known_n} ({100*_active_mtt_n/_total_known_n:.0f}%)" if _total_known_n > 0 else "—"
        _mtt_median   = f"{np.median(_mtt_vals):.2f}" if len(_mtt_vals) > 0 else "—"

        # ── Constituent LDH POD ───────────────────────────────────────────────────
        _ldh_vals = (
            df_bsc_ldh_pod[
                (df_bsc_ldh_pod["all.pass"] == True) &
                (df_bsc_ldh_pod["Metadata_compound"].isin(
                    _annot.loc[_annot["Plant scode"] == _scode, "Metadata_compound"]
                ))
            ]["Metadata_ldh_normalized_POD_um"].dropna().values
        )
        _active_ldh_n = len(_ldh_vals)
        _ldh_pct      = f"{_active_ldh_n}/{_total_known_n} ({100*_active_ldh_n/_total_known_n:.0f}%)" if _total_known_n > 0 else "—"
        _ldh_median   = f"{np.median(_ldh_vals):.2f}" if len(_ldh_vals) > 0 else "—"

        _rows_table.append({
            "Botanical"                      : _plant,
            "Active CP (n/total)"            : _cp_pct,
            "Median constituent CP POD (μM)" : _cp_median,
            "Botanical CP POD (μM)"          : _bot_cp_str,
            "Active MTT (n/total)"           : _mtt_pct,
            "Median constituent MTT POD (μM)": _mtt_median,
            "Botanical MTT POD (μM)"         : _bot_mtt_str,
            "Active LDH (n/total)"           : _ldh_pct,
            "Median constituent LDH POD (μM)": _ldh_median,
            "Botanical LDH POD (μM)"         : _bot_ldh_str,
        })

    # ── Summary footer row ────────────────────────────────────────────────────────
    _all_cp_vals = _df_long.drop_duplicates(subset="Metadata_compound")["PAC"].dropna().values
    _total_known_all = df_bsc_annotations[
        df_bsc_annotations["Plant scode"].isin(list_botanicals)
    ]["Metadata_compound"].nunique()
    _active_cp_all = _df_long["Metadata_compound"].nunique()

    _all_mtt_vals = (
        df_bsc_mtt_pod[
            (df_bsc_mtt_pod["all.pass"] == True) &
            (df_bsc_mtt_pod["Metadata_compound"].isin(_annot["Metadata_compound"]))
        ]
        .drop_duplicates(subset="Metadata_compound")
        ["Metadata_mtt_normalized_POD_um"].dropna().values
    )
    _active_mtt_all = len(_all_mtt_vals)

    _all_ldh_vals = (
        df_bsc_ldh_pod[
            (df_bsc_ldh_pod["all.pass"] == True) &
            (df_bsc_ldh_pod["Metadata_compound"].isin(_annot["Metadata_compound"]))
        ]
        .drop_duplicates(subset="Metadata_compound")
        ["Metadata_ldh_normalized_POD_um"].dropna().values
    )
    _active_ldh_all = len(_all_ldh_vals)

    _rows_table.append({
        "Botanical"                      : "── OVERALL ──",
        "Active CP (n/total)"            : f"{_active_cp_all}/{_total_known_all} ({100*_active_cp_all/_total_known_all:.0f}%)",
        "Median constituent CP POD (μM)" : f"{np.median(_all_cp_vals):.2f}" if len(_all_cp_vals) > 0 else "—",
        "Botanical CP POD (μM)"          : "—",
        "Active MTT (n/total)"           : f"{_active_mtt_all}/{_total_known_all} ({100*_active_mtt_all/_total_known_all:.0f}%)",
        "Median constituent MTT POD (μM)": f"{np.median(_all_mtt_vals):.2f}" if len(_all_mtt_vals) > 0 else "—",
        "Botanical MTT POD (μM)"         : "—",
        "Active LDH (n/total)"           : f"{_active_ldh_all}/{_total_known_all} ({100*_active_ldh_all/_total_known_all:.0f}%)",
        "Median constituent LDH POD (μM)": f"{np.median(_all_ldh_vals):.2f}" if len(_all_ldh_vals) > 0 else "—",
        "Botanical LDH POD (μM)"         : "—",
    })

    df_summary_table_more = pd.DataFrame(_rows_table)
    df_summary_table_more.to_csv(str(table_dir / "Table_1.csv"), index=False)
    print(f"Table 1 botanicals plus overall rows: {len(df_summary_table_more)}")
    print(f"Active CP constituents in Table 1: {_active_cp_all}/{_total_known_all}")
    print(f"Active MTT constituents in Table 1: {_active_mtt_all}/{_total_known_all}")
    print(f"Active LDH constituents in Table 1: {_active_ldh_all}/{_total_known_all}")
    df_summary_table_more
    return


@app.cell
def _(mo):
    mo.md(r"""### POD Cell Painting VS MT VS LDH - All cytotoxicity """)
    return


@app.cell
def _(
    combinations,
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    feature_wise_permutation_threshold,
    np,
    wilcoxon,
):
    # Prepare paired CP/MTT/LDH POD arrays for Figure 2D.
    _df_cp_passed = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()
    _n_active = _df_cp_passed.groupby("Metadata_compound").size()
    _valid = _n_active[_n_active >= feature_wise_permutation_threshold].index
    _cp_passed_filtered = _df_cp_passed[_df_cp_passed["Metadata_compound"].isin(_valid)]

    # ── 2. MTT POD ────────────────────────────────────────────────────────────────
    _mtt_passed = df_bsc_mtt_pod[df_bsc_mtt_pod["all.pass"] == True]

    # ── 3. LDH POD ────────────────────────────────────────────────────────────────
    _ldh_passed = df_bsc_ldh_pod[df_bsc_ldh_pod["all.pass"] == True]

    # ── 4. Keep only compounds present in ALL THREE ───────────────────────────────
    _common_compounds = (
        set(_cp_passed_filtered["Metadata_compound"].unique()) &
        set(_mtt_passed["Metadata_compound"].unique()) &
        set(_ldh_passed["Metadata_compound"].unique())
    )
    print(f"Common compounds across CP POD, MTT, LDH: {len(_common_compounds)}")

    # ── Ensure identical compound order for pairing ───────────────────────────────
    _common_sorted = sorted(_common_compounds)

    _cp_pac = (
        _cp_passed_filtered[_cp_passed_filtered["Metadata_compound"].isin(_common_sorted)]
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reindex(_common_sorted)
        .values
    )
    _mtt_vals = (
        _mtt_passed[_mtt_passed["Metadata_compound"].isin(_common_sorted)]
        .groupby("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .min()
        .reindex(_common_sorted)
        .values
    )
    _ldh_vals = (
        _ldh_passed[_ldh_passed["Metadata_compound"].isin(_common_sorted)]
        .groupby("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .min()
        .reindex(_common_sorted)
        .values
    )

    # ── Sanity check ──────────────────────────────────────────────────────────────
    assert len(_cp_pac) == len(_mtt_vals) == len(_ldh_vals) == len(_common_sorted), \
        "Arrays have different lengths — pairing is broken"
    assert not any(np.isnan(_cp_pac)) and not any(np.isnan(_mtt_vals)) and not any(np.isnan(_ldh_vals)), \
        "NaNs detected after reindex — some compounds missing from a group"

    # ── Pairwise Wilcoxon signed-rank with Bonferroni correction ─────────────────
    _data    = [_cp_pac, _mtt_vals, _ldh_vals]
    _labels  = ["CP POD", "MT POD", "LDH POD"]
    _colors  = ["steelblue", "darkorange", "green"]
    _pairs   = list(combinations(range(3), 2))
    _n_tests = len(_pairs)
    _pvals   = {}
    for _i, _j in _pairs:
        _, _p = wilcoxon(_data[_i], _data[_j], alternative="two-sided")
        _pvals[(_i, _j)] = min(_p * _n_tests, 1.0)

    pod_comparison_all = {
        "common_compounds": _common_sorted,
        "data": _data,
        "labels": _labels,
        "colors": _colors,
        "pairs": _pairs,
        "pvals": _pvals,
    }
    for (_i, _j), _p in _pvals.items():
        print(f"Figure 2D {_labels[_i]} vs {_labels[_j]}: p={_p:.4f}, n={len(_data[_i])}")
    return (pod_comparison_all,)


@app.cell
def _(figure_dir, mticker, np, plt, pod_comparison_all, significance_stars):
    # Render and export Figure 2D.
    _data = pod_comparison_all["data"]
    _labels = pod_comparison_all["labels"]
    _colors = pod_comparison_all["colors"]
    _pairs = pod_comparison_all["pairs"]
    _pvals = pod_comparison_all["pvals"]

    _fig, _ax = plt.subplots(figsize=(6, 4))

    _bp = _ax.boxplot(
        _data,
        positions=range(1, 4),
        patch_artist=True,
        vert=True,
        widths=0.5,
        flierprops=dict(marker=""),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )

    # ── Box colors ────────────────────────────────────────────────────────────────
    for _patch, _color in zip(_bp["boxes"], _colors):
        _patch.set_facecolor(_color)
        _patch.set_alpha(0.35)

    # ── Scatter points ────────────────────────────────────────────────────────────
    for _i, (_v, _color) in enumerate(zip(_data, _colors), start=1):
        _x = np.random.normal(_i, 0.08, size=len(_v))
        _ax.scatter(_x, _v, color=_color, s=25, alpha=0.6, zorder=3)

    # ── Significance brackets (IMPROVED) ──────────────────────────────────────────
    _y_max = max(max(_v) for _v in _data)
    _y_top = np.log10(_y_max)
    _bracket_gap = 0.2

    for _k, (_i, _j) in enumerate(_pairs):
        _p = _pvals[(_i, _j)]
        _sig = significance_stars(_p)
        _n = len(_data[_i])  # paired, so same n

        _x1, _x2 = _i + 1, _j + 1
        _y = 10 ** (_y_top + _bracket_gap * (_k + 1))

        _ax.plot(
            [_x1, _x1, _x2, _x2],
            [_y * 0.88, _y, _y, _y * 0.88],
            lw=0.9,
            color="black"
        )

        _ax.text(
            (_x1 + _x2) / 2,
            _y * 1.02,
            f"{_sig}  p={_p:.3f} (n={_n})",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontstyle="italic",
        )

    # ── Axes ─────────────────────────────────────────────────────────────────────
    _ax.set_yscale("log")  # IMPORTANT: first

    _ax.set_xlim(0.5, 3.5)
    _ax.set_ylim(top=10 ** (_y_top + _bracket_gap * (len(_pairs) + 2)))

    _ax.set_ylabel("Concentration (μM)", fontsize=11)

    # X labels with n
    _ax.set_xticks(range(1, 4))
    _ax.set_xticklabels(
        [f"{lab}" for lab, vals in zip(_labels, _data)],
        fontsize=10
    )

    # ── Y ticks (FIXED: visible dashes at 1,10,100,1000) ─────────────────────────
    _ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.tick_params(axis="y", which="major", colors="black", length=0, direction="out", labelsize=9)
    _ax.tick_params(axis="y", which="minor", colors="grey",  length=0, direction="out", labelsize=6)

    # X ticks
    _ax.tick_params(axis="x", which="major", length=5, width=1, direction="out")

    # ── Style ─────────────────────────────────────────────────────────────────────
    _ax.set_facecolor("white")
    _fig.patch.set_facecolor("white")

    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    _ax.spines["left"].set_color("black")
    _ax.spines["left"].set_linewidth(0.5)

    _ax.spines["bottom"].set_color("black")
    _ax.spines["bottom"].set_linewidth(0.5)

    _ax.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)

    plt.tight_layout()

    plt.savefig(
        str(figure_dir / "Figure_2_D.png"),
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    return


@app.cell
def _(
    combinations,
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    feature_wise_permutation_threshold,
    list_botanicals,
    wilcoxon,
):
    # Prepare activity-count and POD-distribution data for Figure S3B.
    _df_cp_passed = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()
    _n_active = _df_cp_passed.groupby("Metadata_compound").size()
    _valid = _n_active[_n_active >= feature_wise_permutation_threshold].index
    _cp_passed_filtered = _df_cp_passed[_df_cp_passed["Metadata_compound"].isin(_valid)]
    # ── 2. MTT POD ────────────────────────────────────────────────────────────────
    _mtt_passed = df_bsc_mtt_pod[df_bsc_mtt_pod["all.pass"] == True]

    # ── 3. LDH POD ────────────────────────────────────────────────────────────────
    _ldh_passed = df_bsc_ldh_pod[df_bsc_ldh_pod["all.pass"] == True]

    # ── 1. Constituents with all three PODs ───────────────────────────────────────
    # ── 1. Constituents with all three PODs ───────────────────────────────────────
    _cp_passed_filtered_const = _cp_passed_filtered[
        ~_cp_passed_filtered["Metadata_compound"].isin(list_botanicals)
    ]
    _mtt_passed_const = _mtt_passed[~_mtt_passed["Metadata_compound"].isin(list_botanicals)]
    _ldh_passed_const = _ldh_passed[~_ldh_passed["Metadata_compound"].isin(list_botanicals)]

    _common_const = (
        set(_cp_passed_filtered_const["Metadata_compound"].unique()) &
        set(_mtt_passed_const["Metadata_compound"].unique()) &
        set(_ldh_passed_const["Metadata_compound"].unique())
    )
    print(f"Constituents with all three PODs: {len(_common_const)}")

    _common_const_sorted = sorted(_common_const)
    _cp_const = (
        _cp_passed_filtered_const[_cp_passed_filtered_const["Metadata_compound"].isin(_common_const_sorted)]
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reindex(_common_const_sorted).values
    )
    _mtt_const = (
        _mtt_passed_const[_mtt_passed_const["Metadata_compound"].isin(_common_const_sorted)]
        .groupby("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .min().reindex(_common_const_sorted).values
    )
    _ldh_const = (
        _ldh_passed_const[_ldh_passed_const["Metadata_compound"].isin(_common_const_sorted)]
        .groupby("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .min().reindex(_common_const_sorted).values
    )

    # ── 2. Botanicals with all three PODs ─────────────────────────────────────────
    _cp_passed_filtered_bot = _cp_passed_filtered[
        _cp_passed_filtered["Metadata_compound"].isin(list_botanicals)
    ]
    _mtt_passed_bot = _mtt_passed[_mtt_passed["Metadata_compound"].isin(list_botanicals)]
    _ldh_passed_bot = _ldh_passed[_ldh_passed["Metadata_compound"].isin(list_botanicals)]

    _common_bot = (
        set(_cp_passed_filtered_bot["Metadata_compound"].unique()) &
        set(_mtt_passed_bot["Metadata_compound"].unique()) &
        set(_ldh_passed_bot["Metadata_compound"].unique())
    )
    print(f"Botanicals with all three PODs: {len(_common_bot)}")

    _common_bot_sorted = sorted(_common_bot)
    _cp_bot = (
        _cp_passed_filtered_bot[_cp_passed_filtered_bot["Metadata_compound"].isin(_common_bot_sorted)]
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reindex(_common_bot_sorted).values
    )
    _mtt_bot = (
        _mtt_passed_bot[_mtt_passed_bot["Metadata_compound"].isin(_common_bot_sorted)]
        .groupby("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .min().reindex(_common_bot_sorted).values
    )
    _ldh_bot = (
        _ldh_passed_bot[_ldh_passed_bot["Metadata_compound"].isin(_common_bot_sorted)]
        .groupby("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .min().reindex(_common_bot_sorted).values
    )
    # ── 3. All active (constituents + botanicals, any POD) ────────────────────────
    _cp_all = (
        _cp_passed_filtered
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
    )
    _mtt_all = (
        _mtt_passed
        .groupby("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .min()
    )
    _ldh_all = (
        _ldh_passed
        .groupby("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .min()
    )
    # ── Ensure identical compound order for pairing ───────────────────────────────
    # ── 4. Keep only compounds present in ALL THREE ───────────────────────────────
    _common_compounds = (
        set(_cp_passed_filtered["Metadata_compound"].unique()) &
        set(_mtt_passed["Metadata_compound"].unique()) &
        set(_ldh_passed["Metadata_compound"].unique())
    )
    print(f"Common compounds across CP POD, MTT, LDH: {len(_common_compounds)}")
    _common_sorted = sorted(_common_compounds)
    _cp_pac = (
        _cp_passed_filtered[_cp_passed_filtered["Metadata_compound"].isin(_common_sorted)]
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reindex(_common_sorted)
        .values
    )
    _mtt_vals = (
        _mtt_passed[_mtt_passed["Metadata_compound"].isin(_common_sorted)]
        .groupby("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .min()
        .reindex(_common_sorted)
        .values
    )
    _ldh_vals = (
        _ldh_passed[_ldh_passed["Metadata_compound"].isin(_common_sorted)]
        .groupby("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .min()
        .reindex(_common_sorted)
        .values
    )
    # ── 4. Wilcoxon on paired (all-three) subsets ─────────────────────────────────
    _pairs = list(combinations(range(3), 2))
    _n_tests = len(_pairs)

    def _wilcoxon_bonf(a, b):
        _, _p = wilcoxon(a, b, alternative="two-sided")
        return min(_p * _n_tests, 1.0)
    # all stats
    _pvals_all = {(_i,_j): _wilcoxon_bonf(
        [_cp_pac, _mtt_vals, _ldh_vals][_i],
        [_cp_pac, _mtt_vals, _ldh_vals][_j]
    ) for _i,_j in _pairs}

    # Get total unique counts from raw dataframes
    _totals_const = [
        df_bsc_mtt_pod[~df_bsc_mtt_pod["Metadata_compound"].isin(list_botanicals)]["Metadata_compound"].nunique(),
        df_bsc_mtt_pod[~df_bsc_mtt_pod["Metadata_compound"].isin(list_botanicals)]["Metadata_compound"].nunique(),
        df_bsc_mtt_pod[~df_bsc_mtt_pod["Metadata_compound"].isin(list_botanicals)]["Metadata_compound"].nunique(),
    ]
    _totals_bot = [
        len(list_botanicals),
        len(list_botanicals),
        len(list_botanicals),
    ]

    # Get active counts from your filtered dataframes
    # (Using the filtered DFs you already created in previous steps)
    _active_const = [
        _cp_passed_filtered_const["Metadata_compound"].nunique(),
        _mtt_passed_const["Metadata_compound"].nunique(),
        _ldh_passed_const["Metadata_compound"].nunique()
    ]

    _active_bot = [
        _cp_passed_filtered_bot["Metadata_compound"].nunique(),
        _mtt_passed_bot["Metadata_compound"].nunique(),
        _ldh_passed_bot["Metadata_compound"].nunique()
    ]

    _common_const_tox = (
        set(_mtt_passed_const["Metadata_compound"]) |
        set(_ldh_passed_const["Metadata_compound"])
    )
    _common_bot_tox = (
        set(_mtt_passed_bot["Metadata_compound"]) |
        set(_ldh_passed_bot["Metadata_compound"])
    )

    _labels  = ["CP POD", "MT POD", "LDH POD"]
    _colors  = ["steelblue", "darkorange", "green"]
    pod_comparison_breakdown = {
        "labels": _labels,
        "colors": _colors,
        "totals_const": _totals_const,
        "totals_bot": _totals_bot,
        "active_const": _active_const,
        "active_bot": _active_bot,
        "cp_pac": _cp_pac,
        "mtt_vals": _mtt_vals,
        "ldh_vals": _ldh_vals,
        "cp_all": _cp_all.values,
        "mtt_all": _mtt_all.values,
        "ldh_all": _ldh_all.values,
        "pairs": _pairs,
        "pvals_all": _pvals_all,
        "common_const_tox": _common_const_tox,
        "common_bot_tox": _common_bot_tox,
    }
    print(f"Figure S3B active constituent counts (CP/MTT/LDH): {_active_const}")
    print(f"Figure S3B active botanical counts (CP/MTT/LDH): {_active_bot}")
    print(f"Figure S3B toxic constituent union count: {len(_common_const_tox)}")
    return (pod_comparison_breakdown,)


@app.cell
def _(
    Patch,
    downloads_dir,
    figure_dir,
    mticker,
    np,
    plt,
    pod_comparison_breakdown,
    significance_stars,
):
    # Render and export Figure S3B.
    _labels = pod_comparison_breakdown["labels"]
    _colors = pod_comparison_breakdown["colors"]
    _totals_const = pod_comparison_breakdown["totals_const"]
    _totals_bot = pod_comparison_breakdown["totals_bot"]
    _active_const = pod_comparison_breakdown["active_const"]
    _active_bot = pod_comparison_breakdown["active_bot"]
    _cp_pac = pod_comparison_breakdown["cp_pac"]
    _mtt_vals = pod_comparison_breakdown["mtt_vals"]
    _ldh_vals = pod_comparison_breakdown["ldh_vals"]
    _cp_all = pod_comparison_breakdown["cp_all"]
    _mtt_all = pod_comparison_breakdown["mtt_all"]
    _ldh_all = pod_comparison_breakdown["ldh_all"]
    _pairs = pod_comparison_breakdown["pairs"]
    _pvals_all = pod_comparison_breakdown["pvals_all"]
    _common_const_tox = pod_comparison_breakdown["common_const_tox"]

    _fig, (_ax_top, _ax_all) = plt.subplots(
        2, 1, figsize=(7, 9), sharex=True, 
        gridspec_kw={'height_ratios': [1, 4]}
    )
    # --- 3. PLOT SIDE-BY-SIDE BARS WITH TOTAL BACKGROUND ---
    _x = np.arange(len(_labels)) + 1
    _width = 0.35

    # --- Background TOTAL bars (grey, wider, behind) ---
    _ax_top.bar(
        _x - _width/2, _totals_const, _width,
        color="lightgrey", alpha=0.6, zorder=1
    )

    _ax_top.bar(
        _x + _width/2, _totals_bot, _width,
        color="lightgrey", alpha=0.6, zorder=1
    )

    # --- Active bars (foreground) ---
    _rects1 = _ax_top.bar(
        _x - _width/2, _active_const, _width,
        label='Active Constituents', color=_colors, alpha=0.9, zorder=2
    )

    _rects2 = _ax_top.bar(
        _x + _width/2, _active_bot, _width,
        label='Active Botanicals', color=_colors, alpha=0.5,
        hatch='///', edgecolor="white", zorder=2
    )

    # --- Labels on active bars ---
    def _autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            _ax_top.annotate(
                f'{int(height)}',
                xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8, fontweight='bold'
            )

    _autolabel(_rects1)
    _autolabel(_rects2)
    _legend_elements = [
        Patch(facecolor="lightgrey", edgecolor="none", label="Total tested"),
        Patch(facecolor="steelblue", label="Active (Constituents)"),
        Patch(facecolor="steelblue", alpha=0.5, hatch="///", label="Active (Botanicals)")
    ]

    _ax_top.legend(handles=_legend_elements, frameon=False, fontsize=9, loc="upper right")
    # --- 4. TOP SUBPLOT FORMATTING ---
    _ax_top.set_ylabel("Active (N)", fontsize=10)
    _ax_top.set_title("Activity Breakdown: Constituents vs. Botanicals", fontsize=12, pad=15)
    _ax_top.set_xticks(_x)
    _ax_top.set_xticklabels(_labels)

    # Clean up spines
    _ax_top.spines["top"].set_visible(False)
    _ax_top.spines["right"].set_visible(False)
    _ax_top.spines["bottom"].set_visible(False) # Shared axis anyway

    for _ax, _cp_v, _mtt_v, _ldh_v, _pvals, _title, _paired_n in [
        (_ax_all, _cp_pac, _mtt_vals, _ldh_vals, _pvals_all,
         f"All\n(stats on n={len(_cp_pac)} with all 3 PODs)", len(_cp_pac)),
    ]:
        # All-active data for boxplots (not just paired)
        _data_all = [_cp_all, _mtt_all, _ldh_all]

        _bp = _ax.boxplot(
            _data_all,
            positions=range(1, 4),
            patch_artist=True,
            vert=True,
            widths=0.45,
            flierprops=dict(marker=""),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
        )
        for _patch, _color in zip(_bp["boxes"], _colors):
            _patch.set_facecolor(_color)
            _patch.set_alpha(0.25)

        # All active scatter (light, behind)
        for _i, (_v, _color) in enumerate(zip(_data_all, _colors), start=1):
            _x = np.random.normal(_i, 0.08, size=len(_v))
            _ax.scatter(_x, _v, color=_color, s=15, alpha=0.25, zorder=2)

        # Paired scatter + connecting lines
        _paired_data = [_cp_v, _mtt_v, _ldh_v]
        _x_jitter = [np.random.normal(_i+1, 0.06, size=_paired_n)
                     for _i in range(3)]

        for _k in range(_paired_n):
            _ax.plot(
                [_x_jitter[0][_k], _x_jitter[1][_k], _x_jitter[2][_k]],
                [_paired_data[0][_k], _paired_data[1][_k], _paired_data[2][_k]],
                color="grey", lw=0.5, alpha=0.35, zorder=3
            )
        for _i, (_v, _color) in enumerate(zip(_paired_data, _colors)):
            _ax.scatter(_x_jitter[_i], _v,
                        color=_color, s=30, alpha=0.85,
                        edgecolors="white", linewidths=0.4, zorder=4)

        # Significance brackets
        _y_max = max(max(_v) for _v in _data_all)
        _y_top = np.log10(_y_max)
        _bracket_gap = 0.25

        for _bk, (_i, _j) in enumerate(_pairs):
            _p   = _pvals[(_i, _j)]
            _sig = significance_stars(_p)
            _x1, _x2 = _i + 1, _j + 1
            _y = 10 ** (_y_top + _bracket_gap * (_bk + 1))
            _ax.plot([_x1, _x1, _x2, _x2], [_y*0.88, _y, _y, _y*0.88],
                     lw=0.9, color="black")
            _ax.text((_x1+_x2)/2, _y*1.02,
                     f"{_sig}  p={_p:.3f} (n={_paired_n})",
                     ha="center", va="bottom", fontsize=7.5, fontstyle="italic")

        _ax.set_yscale("log")
        _ax.set_xlim(0.5, 3.5)
        _ax.set_ylim(top=10**(_y_top + _bracket_gap * (len(_pairs) + 1)))
        _ax.set_xticks(range(1, 4))
        _ax.set_xticklabels(_labels, fontsize=10)
        #_ax.set_title(_title, fontsize=10)
        _ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        _ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
        _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
        _ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
        _ax.tick_params(axis="y", which="major", colors="black", length=0, labelsize=9)
        _ax.tick_params(axis="y", which="minor", colors="grey",  length=0, labelsize=6)
        _ax.tick_params(axis="x", which="major", length=5, width=1, direction="out")
        _ax.set_facecolor("white")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.spines["left"].set_color("black")
        _ax.spines["left"].set_linewidth(0.5)
        _ax.spines["bottom"].set_color("black")
        _ax.spines["bottom"].set_linewidth(0.5)
        _ax.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)

    _ax_all.set_ylabel("Concentration (μM)", fontsize=11)
    _fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(downloads_dir / "boxplot_cp_mtt_ldh.png",
                dpi=200, bbox_inches="tight")

    for _lab, _data in zip(_labels, _data_all):
        q1 = np.percentile(_data, 25)
        median = np.percentile(_data, 50)
        q3 = np.percentile(_data, 75)
        iqr = q3 - q1

        print(f"{_lab}:")
        print(f"  Q1 (25%)  = {q1:.3g}")
        print(f"  Median    = {median:.3g}")
        print(f"  Q3 (75%)  = {q3:.3g}")
        print(f"  IQR       = {iqr:.3g}\n")
    print("Common conts:", len(_common_const_tox))

    plt.savefig(
        str(figure_dir / "Figure_S3_B.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.show() 
    return


@app.cell
def _(mo):
    mo.md(r"""### POD Cell Painting - Botanicals VS Constituents""")
    return


@app.cell
def _(
    botanical_colors,
    df_bsc_annotations,
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    dic_botanicals_scode_name,
    feature_wise_permutation_threshold,
    list_botanicals,
    pd,
):
    # Prepare botanical-vs-constituent Cell Painting POD data.
    _constituents_in_data = (
        df_bsc_cp_pod_all_featurewise["Metadata_compound"]
        .unique()
    )
    _constituents_in_data = [c for c in _constituents_in_data if c not in list_botanicals]

    _annot = df_bsc_annotations[
        df_bsc_annotations["Metadata_compound"].isin(_constituents_in_data)
    ].copy()

    _df_const_passed = df_bsc_cp_pod_all_featurewise[
        (df_bsc_cp_pod_all_featurewise["all.pass"] == True) &
        (df_bsc_cp_pod_all_featurewise["Metadata_compound"].isin(_constituents_in_data))
    ]

    # Keep only constituents with n active features > threshold
    _n_active = (
        _df_const_passed.groupby("Metadata_compound")["cp_POD_um"]
        .count()
        .reset_index(name="n_active")
    )
    _active_compounds = _n_active.loc[
        _n_active["n_active"] > feature_wise_permutation_threshold,
        "Metadata_compound"
    ].values

    _constituent_pac = (
        _df_const_passed[_df_const_passed["Metadata_compound"].isin(_active_compounds)]
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reset_index(name="PAC")
    )

    _pac_lookup = _constituent_pac.set_index("Metadata_compound")["PAC"].to_dict()

    # ── 2. PAC per botanical ──────────────────────────────────────────────────────
    _df_bot_passed = df_bsc_cp_pod_all_featurewise[
        (df_bsc_cp_pod_all_featurewise["all.pass"] == True) &
        (df_bsc_cp_pod_all_featurewise["Metadata_compound"].isin(list_botanicals))
    ]

    _n_active_bot = (
        _df_bot_passed.groupby("Metadata_compound")["cp_POD_um"]
        .count()
        .reset_index(name="n_active")
    )
    _active_botanicals = _n_active_bot.loc[
        _n_active_bot["n_active"] > feature_wise_permutation_threshold,
        "Metadata_compound"
    ].values

    _botanical_pac_dict = (
        _df_bot_passed[_df_bot_passed["Metadata_compound"].isin(_active_botanicals)]
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .to_dict()
    )
    # ── MTT and LDH POD per botanical ────────────────────────────────────────────
    _mtt_pac_dict = (
        df_bsc_mtt_pod[
            (df_bsc_mtt_pod["all.pass"] == True) &
            (df_bsc_mtt_pod["Metadata_compound"].isin(list_botanicals))
        ]
        .groupby("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .min()
        .to_dict()
    )

    _ldh_pac_dict = (
        df_bsc_ldh_pod[
            (df_bsc_ldh_pod["all.pass"] == True) &
            (df_bsc_ldh_pod["Metadata_compound"].isin(list_botanicals))
        ]
        .groupby("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .min()
        .to_dict()
    )
    # ── 3. Map constituents to their botanicals ───────────────────────────────────
    _constituent_to_plants = _annot.groupby("Metadata_compound")["Plant scode"].apply(list).to_dict()

    _rows = []
    for _compound, _plants in _constituent_to_plants.items():
        if _compound not in _pac_lookup:
            continue
        for _plant in _plants:
            if _plant in list_botanicals:
                _rows.append({
                    "Metadata_compound": _compound,
                    "Plant name": dic_botanicals_scode_name[_plant],
                    "Plant scode": _plant,
                    "PAC": _pac_lookup[_compound]
                })
    _df_long = pd.DataFrame(_rows)

    # ── 4. Order botanicals by median constituent PAC ─────────────────────────────
    _order = (
        _df_long.groupby("Plant name")["PAC"]
        .median()
        .sort_values()
        .index.tolist()
    )

    _total_known = (
        df_bsc_annotations[df_bsc_annotations["Plant scode"].isin(list_botanicals)]
        .groupby("Plant scode")["Metadata_compound"]
        .nunique()
        .to_dict()
    )
    _active_counts = (
        _df_long.groupby("Plant scode")["Metadata_compound"]
        .nunique()
        .to_dict()
    )
    botanical_cp_constituent_plot_data = {
        "df_long": _df_long,
        "order": _order,
        "plant_colors": botanical_colors,
        "botanical_pac": _botanical_pac_dict,
        "botanical_mtt": _mtt_pac_dict,
        "botanical_ldh": _ldh_pac_dict,
        "total_known": _total_known,
        "active_counts": _active_counts,
    }
    print(f"Botanical-vs-constituent CP POD constituents: {_df_long['Metadata_compound'].nunique()}")
    print(f"Botanical-vs-constituent CP POD botanicals: {len(_order)}")
    return (botanical_cp_constituent_plot_data,)


@app.cell
def _(
    Line2D,
    botanical_cp_constituent_plot_data,
    dic_botanicals_scode_name,
    mpatches,
    mticker,
    np,
    pd,
    plt,
):
    # Render botanical-vs-constituent Cell Painting POD plot.
    _df_long = botanical_cp_constituent_plot_data["df_long"]
    _order = botanical_cp_constituent_plot_data["order"]
    _plant_colors = botanical_cp_constituent_plot_data["plant_colors"]
    _botanical_pac_dict = botanical_cp_constituent_plot_data["botanical_pac"]
    _mtt_pac_dict = botanical_cp_constituent_plot_data["botanical_mtt"]
    _ldh_pac_dict = botanical_cp_constituent_plot_data["botanical_ldh"]
    _total_known = botanical_cp_constituent_plot_data["total_known"]
    _active_counts = botanical_cp_constituent_plot_data["active_counts"]

    _data_ordered = [
        _df_long.loc[_df_long["Plant name"] == _plant, "PAC"].dropna().values
        for _plant in _order
    ]

    _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                       sharey=True,
                                       gridspec_kw={"width_ratios": [3, 1]})

    _bp = _ax.boxplot(
        _data_ordered,
        positions=range(1, len(_order) + 1),
        patch_artist=True,
        vert=False,
        widths=0.5,
        flierprops=dict(marker=""),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )

    # Color boxes
    for _patch, _plant in zip(_bp["boxes"], _order):
        _patch.set_facecolor(_plant_colors[_plant])
        _patch.set_alpha(0.4)

    # Jittered constituent dots
    for _i, _plant in enumerate(_order, start=1):
        _y = _df_long.loc[_df_long["Plant name"] == _plant, "PAC"].dropna().values
        if len(_y) == 0:
            continue
        _x = np.random.normal(_i, 0.08, size=len(_y))
        _ax.plot(_y, _x, "o", alpha=0.55, markersize=5, color=_plant_colors[_plant])

    _legend_cp_added = False
    _legend_mtt_added = False
    _legend_ldh_added = False

    for _i, _plant in enumerate(_order, start=1):
        _scode = next((k for k, v in dic_botanicals_scode_name.items() if v == _plant), None)
        if _scode is None:
            continue
        _y_cp = _botanical_pac_dict.get(_scode)
        if _y_cp is not None and pd.notna(_y_cp):
            _ax.plot([_y_cp, _y_cp], [_i - 0.35, _i + 0.35],
             color="steelblue", linewidth=2.5, solid_capstyle="round",
             linestyle="--",
             label="CP POD" if not _legend_cp_added else None)
            _legend_cp_added = True
        _y_mtt = _mtt_pac_dict.get(_scode)
        #if _y_mtt is not None and pd.notna(_y_mtt):
        #    _ax.plot([_y_mtt, _y_mtt], [_i - 0.35, _i + 0.35],
        #     color="darkorange", linewidth=2.5, solid_capstyle="round",
        #     linestyle="--",
        #     label="MTT POD" if not _legend_mtt_added else None)
        #    _legend_mtt_added = True
        #_y_ldh = _ldh_pac_dict.get(_scode)
        #if _y_ldh is not None and pd.notna(_y_ldh):
        #    _ax.plot([_y_ldh, _y_ldh], [_i - 0.35, _i + 0.35],
        #     color="green", linewidth=2.5, solid_capstyle="round",
        #     linestyle="--",
        #     label="LDH POD" if not _legend_ldh_added else None)
        #    _legend_ldh_added = True

    _ax.set_yticks(range(1, len(_order) + 1))
    _ax.set_yticklabels(_order, fontsize=14)
    _ax.tick_params(axis="y", labelleft=True)
    _ax.set_xscale("log")
    _ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    _ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    _ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))

    _ax.tick_params(axis="x", which="major", colors="black", length=0, direction="out", labelsize=12, labelrotation=30)
    _ax.tick_params(axis="x", which="minor", colors="grey",  length=0, direction="out", labelsize=10, labelrotation=30)

    #_ax.set_ylabel("Botanical")
    _ax.set_xlabel("Concentration (μM)", fontsize=16)
    #_ax.set_title("Constituent PAC distribution per botanical")
    _ax.set_ylim(0.5, len(_order) + 0.5)

    _legend_handles = [
        Line2D([0], [0], color="steelblue",        linewidth=2, linestyle="--", label="Botanical CP POD"),
        #Line2D([0], [0], color="darkorange", linewidth=2, linestyle="--", label="MT POD"),
        #Line2D([0], [0], color="green",      linewidth=2, linestyle="--", label="LDH POD"),
        Line2D([0], [0], marker="o", color="grey", markersize=6,
               linestyle="none", label="Constituent CP POD"),
    ]
    _ax.legend(
        handles=_legend_handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=12,
        borderpad=0.8,
    )

    # ── ax2: active vs total constituents per botanical ───────────────────────────
    for _i, _plant in enumerate(_order, start=1):
        _scode = next((k for k, v in dic_botanicals_scode_name.items() if v == _plant), None)
        if _scode is None:
            continue
        _color = _plant_colors[_plant]
        _total = _total_known.get(_scode, 0)
        _active = _active_counts.get(_scode, 0)

        _ax2.barh(_i, _total, color="lightgrey", height=0.5, zorder=2)
        _ax2.barh(_i, _active, color=_color, height=0.5, alpha=0.85, zorder=3)
        _ax2.tick_params(axis="x", labelsize=12)
        _ax2.text(_total + 0.3, _i, f"{_active}/{_total}", va="center", fontsize=10)  # was 7

    _ax2.set_yticks(range(1, len(_order) + 1))
    _ax2.set_yticklabels([])
    _ax2.set_xlabel("Number of constituents", fontsize=16)
    #_ax2.set_title("Active / total")

    # Shared legend for ax2
    _ax2.barh([], [], color="lightgrey", label="Total")
    _ax2.barh([], [], color="grey", alpha=0.85, label="Active (CP POD)")

    _legend_handles2 = [
        mpatches.Patch(color="lightgrey", label="Total known"),
        mpatches.Patch(color="grey", alpha=0.85, label="Active (CP POD)"),
    ]
    _ax2.legend(
        handles=_legend_handles2,
        loc="lower right",
        bbox_to_anchor=(1, 0.074),
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=12,
        borderpad=0.8,
    )

    plt.tight_layout()
    _ax.set_yticks(range(1, len(_order) + 1))
    _ax.set_yticklabels(_order, fontsize=9)
    _ax.set_facecolor("white")
    _ax2.set_facecolor("white")
    _fig.patch.set_facecolor("white")
    _ax.set_yticklabels(_order, fontsize=16)
    for _a in [_ax, _ax2]:
        _a.spines["top"].set_visible(False)
        _a.spines["right"].set_visible(False)
        _a.spines["left"].set_visible(True)
        _a.spines["left"].set_color("black")
        _a.spines["bottom"].set_color("black")
        _a.spines["bottom"].set_linewidth(0.5)
        _a.spines["left"].set_linewidth(0.5)
    plt.draw()
    _fig.subplots_adjust(wspace=0.05)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""### POD feature wise cytotoxicity""")
    return


@app.cell
def _(
    botanical_colors,
    df_bsc_annotations,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    df_level5_bsc,
    dic_botanicals_scode_name,
    list_botanicals,
    np,
    pd,
):
    # Prepare MTT/LDH botanical and constituent POD plotting data.
    _mtt_pac_dict = (
        df_bsc_mtt_pod[
            (df_bsc_mtt_pod["all.pass"] == True) &
            (df_bsc_mtt_pod["Metadata_compound"].isin(list_botanicals))
        ]
        .set_index("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .to_dict()
    )

    _ldh_pac_dict = (
        df_bsc_ldh_pod[
            (df_bsc_ldh_pod["all.pass"] == True) &
            (df_bsc_ldh_pod["Metadata_compound"].isin(list_botanicals))
        ]
        .set_index("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .to_dict()
    )

    _const_mtt_lookup = (
        df_bsc_mtt_pod[df_bsc_mtt_pod["all.pass"] == True]
        .set_index("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .to_dict()
    )
    _const_ldh_lookup = (
        df_bsc_ldh_pod[df_bsc_ldh_pod["all.pass"] == True]
        .set_index("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .to_dict()
    )

    # Before building _df_plot, compute tested counts (all constituents, not just those with POD data)
    _tested_counts = {}
    for _scode in list_botanicals:
        _plant_name = dic_botanicals_scode_name.get(_scode, _scode)
        _tested_counts[_plant_name] = len(
            df_bsc_annotations[
                df_bsc_annotations["Plant scode"] == _scode
            ]["Metadata_compound"].drop_duplicates()
        )

    # ── Build long dataframe ──────────────────────────────────────────────────────
    _rows = []
    for _scode in list_botanicals:
        _plant_name = dic_botanicals_scode_name.get(_scode, _scode)
        _bot_mtt = _mtt_pac_dict.get(_scode)
        _bot_ldh = _ldh_pac_dict.get(_scode)

        #if pd.isna(_bot_mtt) and pd.isna(_bot_ldh):
        #    continue

        _constituents = df_bsc_annotations[
            df_bsc_annotations["Plant scode"] == _scode
        ][["Metadata_compound", "Constituent"]].drop_duplicates()

        for _, _crow in _constituents.iterrows():
            _cmpd  = _crow["Metadata_compound"]
            _cname = _crow["Constituent"]
            _c_mtt = _const_mtt_lookup.get(_cmpd)
            _c_ldh = _const_ldh_lookup.get(_cmpd)
            if pd.isna(_c_mtt) and pd.isna(_c_ldh):
                continue
            _rows.append({
                "botanical":       _plant_name,
                "botanical_scode": _scode,
                "constituent":     _cname,
                "bot_mtt":         _bot_mtt,
                "bot_ldh":         _bot_ldh,
                "const_mtt":       _c_mtt,
                "const_ldh":       _c_ldh,
            })

    _df_plot = pd.DataFrame(_rows)

    # ── Order botanicals by MTT POD, constituents by min POD within botanical ─────
    _bot_order = (
        _df_plot.drop_duplicates("botanical")
        .sort_values(["bot_mtt", "bot_ldh"])
        ["botanical"].tolist()
    )

    _df_plot["_min_pod"] = _df_plot[["const_mtt", "const_ldh"]].min(axis=1)

    _rows_ordered = []
    for _plant in _bot_order:
        _sub = _df_plot[_df_plot["botanical"] == _plant].sort_values("_min_pod", ascending=False)
        _rows_ordered.append(_sub)
    _df_sorted = pd.concat(_rows_ordered).reset_index(drop=True)
    _df_sorted["_y"] = np.arange(len(_df_sorted))

    # ── Colors (CUSTOM BOTANICAL PALETTE) ─────────────────────────────────────────
    _mtt_color = "darkorange"
    _ldh_color = "green"

    # ── Build _df_ranges dynamically from df_level5_bsc ──────────────────────────
    _conc_stats = (
        df_level5_bsc
        .groupby("Metadata_compound")["Metadata_concentration"]
        .agg(Min=("min"), Max=("max"))
        .reset_index()
    )
    _conc_stats.columns = ["Metadata_compound", "Min (µM)", "Max (µM)"]
    _conc_stats["Plant name"] = _conc_stats["Metadata_compound"].map(dic_botanicals_scode_name)
    _conc_stats = _conc_stats.dropna(subset=["Plant name"])

    _df_ranges = (
        _conc_stats[["Plant name", "Min (µM)", "Max (µM)"]]
        .drop_duplicates("Plant name")
        .set_index("Plant name")
    )
    tox_pod_plot_data = {
        "df_sorted": _df_sorted,
        "bot_order": _bot_order,
        "tested_counts": _tested_counts,
        "concentration_ranges": _df_ranges,
        "plant_bg_colors": botanical_colors,
        "default_color": "#cccccc",
        "mtt_color": _mtt_color,
        "ldh_color": _ldh_color,
    }
    print(f"MTT/LDH constituent POD plot rows: {len(_df_sorted)}")
    print(f"MTT/LDH botanical groups in plot: {len(_bot_order)}")
    return (tox_pod_plot_data,)


@app.cell
def _(Line2D, downloads_dir, mticker, np, pd, plt, tox_pod_plot_data):
    # Render and export MTT/LDH botanical and constituent POD plot.
    _df_sorted = tox_pod_plot_data["df_sorted"]
    _bot_order = tox_pod_plot_data["bot_order"]
    _tested_counts = tox_pod_plot_data["tested_counts"]
    _df_ranges = tox_pod_plot_data["concentration_ranges"]
    _plant_bg_colors = tox_pod_plot_data["plant_bg_colors"]
    _default_color = tox_pod_plot_data["default_color"]
    _mtt_color = tox_pod_plot_data["mtt_color"]
    _ldh_color = tox_pod_plot_data["ldh_color"]

    _n = len(_df_sorted)
    _fig, _ax = plt.subplots(figsize=(12, max(2, _n * 0.20)))

    for _pi, _plant in enumerate(_bot_order):
        _grp  = _df_sorted[_df_sorted["botanical"] == _plant]
        _y_min = _grp["_y"].min() - 0.5
        _y_max = _grp["_y"].max() + 0.5
        _y_mid = (_y_min + _y_max) / 2
        _height = _y_max - _y_min
        _color  = _plant_bg_colors.get(_plant, _default_color)

        # ── Colored band over tested concentration range ──────────────────────
        if _plant in _df_ranges.index:
            _x_min = _df_ranges.loc[_plant, "Min (µM)"]
            _x_max = _df_ranges.loc[_plant, "Max (µM)"]
            _ax.barh(
                y=_y_mid, width=_x_max - _x_min, left=_x_min,
                height=_height, color=_color, alpha=0.12, zorder=0
            )
        else:
            _ax.axhspan(_y_min, _y_max, color=_color, alpha=0.12, zorder=0)

        # ── Botanical vertical lines ──────────────────────────────────────────
        _bot_mtt = _grp["bot_mtt"].iloc[0]
        _bot_ldh = _grp["bot_ldh"].iloc[0]

        if pd.notna(_bot_mtt):
            _ax.plot([_bot_mtt, _bot_mtt], [_y_min, _y_max],
                     color=_mtt_color, linewidth=2, linestyle="--",
                     alpha=0.8, zorder=2,
                     label="Botanical MTT POD" if _pi == 0 else None)
        if pd.notna(_bot_ldh):
            _ax.plot([_bot_ldh, _bot_ldh], [_y_min, _y_max],
                     color=_ldh_color, linewidth=2, linestyle="--",
                     alpha=0.8, zorder=2,
                     label="Botanical LDH POD" if _pi == 0 else None)

        # ── Botanical label (RIGHT SIDE) ──────────────────────────────────────
        _active_n = _grp["constituent"].nunique()
        _tested_n = _tested_counts.get(_plant, "?")
        _ax.text(
            1.02, _y_mid,
            f"{_plant}\n(N active : {_active_n}/{_tested_n})",
            transform=_ax.get_yaxis_transform(),
            fontsize=12, fontweight="bold",
            va="center", ha="left",
            color="black", clip_on=False,
        )

    # ── Constituent dots ──────────────────────────────────────────────────────────
    for _, _row in _df_sorted.iterrows():
        _y = _row["_y"]
        if pd.notna(_row["const_mtt"]):
            _ax.scatter([_row["const_mtt"]], [_y],
                        color=_mtt_color, s=40, zorder=4, alpha=0.9,
                        marker="o", edgecolors="white", linewidths=0.4,
                        label="Constituent MT" if _y == 0 else None)
        if pd.notna(_row["const_ldh"]):
            _ax.scatter([_row["const_ldh"]], [_y],
                        color=_ldh_color, s=40, zorder=4, alpha=0.9,
                        marker="s", edgecolors="white", linewidths=0.4,
                        label="Constituent LDH" if _y == 0 else None)

    # ── Axes ──────────────────────────────────────────────────────────────────────
    _ax.set_yticks(_df_sorted["_y"].tolist())
    _ax.set_yticklabels(_df_sorted["constituent"].tolist(), fontsize=10)
    _ax.set_xscale("log")
    _ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    _ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.tick_params(axis="x", which="major", colors="black", length=0, direction="out", labelsize=9, labelrotation=30)
    _ax.tick_params(axis="x", which="minor", colors="grey",  length=0, direction="out", labelsize=6, labelrotation=30)
    _ax.tick_params(axis="y", which="major", length=3, width=0.8)
    _ax.set_xlabel("Concentration (μM)", fontsize=11)
    _ax.set_ylim(-0.5, _n - 0.5)
    _ax.set_facecolor("white")
    _fig.patch.set_facecolor("white")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["left"].set_linewidth(0.5)
    _ax.spines["bottom"].set_linewidth(0.5)
    _ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    # ── Legend ────────────────────────────────────────────────────────────────────
    _handles = [
        Line2D([0], [0], color=_mtt_color, linewidth=2, linestyle="--", label="Botanical MT POD"),
        Line2D([0], [0], color=_ldh_color,  linewidth=2, linestyle="--", label="Botanical LDH POD"),
        Line2D([0], [0], marker="o", color=_mtt_color, linestyle="", markersize=6, label="Constituent MT POD"),
        Line2D([0], [0], marker="s", color=_ldh_color,  linestyle="", markersize=6, label="Constituent LDH POD"),
    ]
    _ax.legend(handles=_handles, loc="upper left", frameon=True,
               facecolor="white", edgecolor="lightgrey", framealpha=0.95,
               fontsize=9, borderpad=0.8)

    plt.tight_layout()
    _ax.set_yticks(_df_sorted["_y"].tolist())
    _ax.set_yticklabels(_df_sorted["constituent"].tolist(), fontsize=10)

    plt.savefig(downloads_dir / "botanical_constituent_plot.png",
                dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Number Categories feature wise comparison""")
    return


@app.cell
def _():
    import math
    return (math,)


@app.cell
def _(
    combinations,
    df_bsc_cp_pod_all_featurewise,
    feature_wise_permutation_threshold,
    kruskal,
    wilcoxon,
):
    # ── Categories and colors ─────────────────────────────────────────────────────
    _categories = ["DNA", "ER", "AGP", "RNA", "Mito", "Brightfield"]
    _cat_colors = {
        "DNA":         "#e41a1c",
        "ER":          "#377eb8",
        "AGP":         "#4daf4a",
        "RNA":         "#984ea3",
        "Mito":        "#ff7f00",
        "Brightfield": "#a65628",
    }

    def _get_category(_feature_name):
        for _cat in _categories:
            if _cat in str(_feature_name):
                return _cat
        return None

    # ── Filter: all.pass == True + globally valid compounds ───────────────────────
    _df_feat = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()

    _n_total = _df_feat.groupby("Metadata_compound").size()
    _valid_compounds = _n_total[_n_total >= feature_wise_permutation_threshold].index
    _df_feat = _df_feat[_df_feat["Metadata_compound"].isin(_valid_compounds)].copy()

    _df_feat["category"] = _df_feat["gene.id"].apply(_get_category)
    _df_feat = _df_feat[_df_feat["category"].notna()].copy()

    # ── PAC per compound per category: 5th percentile of cp_POD_um ───────────────
    _df_pac_cat = (
        _df_feat.groupby(["Metadata_compound", "category"])["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reset_index(name="PAC_cat")
    )

    # ── Order categories by median PAC ascending ──────────────────────────────────
    _cat_order = (
        _df_pac_cat.groupby("category")["PAC_cat"]
        .median()
        .sort_values()
        .index.tolist()
    )

    _data_box = [
        _df_pac_cat.loc[_df_pac_cat["category"] == _cat, "PAC_cat"].values
        for _cat in _cat_order
    ]

    # ── Kruskal-Wallis global test ────────────────────────────────────────────────
    _kw_stat, _kw_p = kruskal(*_data_box)
    print(f"Kruskal-Wallis: H={_kw_stat:.3f}, p={_kw_p:.4e}")

    # ── Pivot for paired Wilcoxon ─────────────────────────────────────────────────
    _df_pivot = _df_pac_cat.pivot(
        index="Metadata_compound", columns="category", values="PAC_cat"
    )

    # ── Pairwise Wilcoxon signed-rank on common compounds + Bonferroni ────────────
    _pairs = list(combinations(_cat_order, 2))
    _n_tests = len(_pairs)
    _pvals = {}
    _n_pairs_used = {}

    for _cat_a, _cat_b in _pairs:
        if _cat_a not in _df_pivot.columns or _cat_b not in _df_pivot.columns:
            continue
        _common = _df_pivot[[_cat_a, _cat_b]].dropna()
        _n_pairs_used[(_cat_a, _cat_b)] = len(_common)
        if len(_common) < 2:
            _pvals[(_cat_a, _cat_b)] = 1.0
            continue
        _stat, _p = wilcoxon(_common[_cat_a].values, _common[_cat_b].values)
        _pvals[(_cat_a, _cat_b)] = min(_p * _n_tests, 1.0)

    print(f"\nPairwise Wilcoxon signed-rank (Bonferroni corrected, n tests={_n_tests}):")
    for (_cat_a, _cat_b), _p in _pvals.items():
        _n = _n_pairs_used.get((_cat_a, _cat_b), 0)
        print(f"  {_cat_a:12s} vs {_cat_b:12s} | n_common={_n:3d} | p={_p:.4f}")

    def _stars(_p):
        if _p < 0.001:  return "***"
        elif _p < 0.01: return "**"
        elif _p < 0.05: return "*"
        else:           return "ns"

    _sig_pairs = [
        (_cat_order.index(_a), _cat_order.index(_b))
        for (_a, _b), _p in _pvals.items() if _p < 0.05
    ]

    pod_category_pac_stats = {
        "df_pac_cat": _df_pac_cat,
        "cat_order": _cat_order,
        "cat_colors": _cat_colors,
        "data_box": _data_box,
        "kw_stat": _kw_stat,
        "kw_p": _kw_p,
        "pvals": _pvals,
        "n_pairs_used": _n_pairs_used,
        "sig_pairs": _sig_pairs,
    }
    print(f"Category PAC compounds passing threshold: {len(_valid_compounds)}")
    print(f"Category PAC rows: {len(_df_pac_cat)}")
    return (pod_category_pac_stats,)


@app.cell
def _(
    downloads_dir,
    mticker,
    np,
    plt,
    pod_category_pac_stats,
    significance_stars,
):
    # Render category-level PAC comparison plot.
    _df_pac_cat = pod_category_pac_stats["df_pac_cat"]
    _cat_order = pod_category_pac_stats["cat_order"]
    _cat_colors = pod_category_pac_stats["cat_colors"]
    _data_box = pod_category_pac_stats["data_box"]
    _kw_stat = pod_category_pac_stats["kw_stat"]
    _kw_p = pod_category_pac_stats["kw_p"]
    _pvals = pod_category_pac_stats["pvals"]
    _n_pairs_used = pod_category_pac_stats["n_pairs_used"]
    _sig_pairs = pod_category_pac_stats["sig_pairs"]

    _fig, _ax = plt.subplots(figsize=(8, 5 + len(_sig_pairs) * 0.25))

    _bp = _ax.boxplot(
        _data_box,
        patch_artist=True,
        vert=True,
        widths=0.5,
        flierprops=dict(marker=""),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1, linestyle="-"),
        capprops=dict(linewidth=1),
    )
    for _patch, _cat in zip(_bp["boxes"], _cat_order):
        _patch.set_facecolor(_cat_colors[_cat])
        _patch.set_alpha(0.6)

    for _i, _cat in enumerate(_cat_order, start=1):
        _y = _df_pac_cat.loc[_df_pac_cat["category"] == _cat, "PAC_cat"].values
        _x = np.random.normal(_i, 0.08, size=len(_y))
        _ax.scatter(_x, _y, color=_cat_colors[_cat], s=20, alpha=0.6,
                    edgecolors="black", linewidths=0.3, zorder=3)

    _ax.set_yscale("log")
    _ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.set_xticks(range(1, len(_cat_order) + 1))
    _ax.set_xticklabels(_cat_order, fontsize=10)
    _ax.set_ylabel("Concentration (µM)", fontsize=11)
    _ax.set_title(
        f"Category-level PAC (5th pct cp_POD_um)\n"
        f"Kruskal-Wallis H={_kw_stat:.2f}, p={_kw_p:.2e} — "
        f"pairwise Wilcoxon signed-rank (Bonferroni, paired by compound)",
        fontsize=9
    )

    # ── Significance brackets (significant only) ──────────────────────────────────
    _y_max = max(_v.max() for _v in _data_box)
    _y_top = np.log10(_y_max)
    _bracket_gap = 0.2

    for _k, (_i, _j) in enumerate(_sig_pairs):
        _cat_a, _cat_b = _cat_order[_i], _cat_order[_j]
        _p = _pvals[(_cat_a, _cat_b)]
        _n = _n_pairs_used.get((_cat_a, _cat_b), "?")
        _x1, _x2 = _i + 1, _j + 1
        _y = 10 ** (_y_top + _bracket_gap * (_k + 1))
        _ax.plot([_x1, _x1, _x2, _x2], [_y * 0.88, _y, _y, _y * 0.88],
                 lw=0.9, color="black")
        _ax.text(
            (_x1 + _x2) / 2, _y * 1.02,
            f"{significance_stars(_p)}  p={_p:.3f} (n={_n})",
            ha="center", va="bottom", fontsize=7.5
        )

    _ax.set_ylim(top=10 ** (_y_top + _bracket_gap * (len(_sig_pairs) + 2)))

    _n_per_cat = {_cat: len(_df_pac_cat.loc[_df_pac_cat["category"] == _cat]) for _cat in _cat_order}
    _ax.set_xticks(range(1, len(_cat_order) + 1))
    _ax.set_xticklabels(
        [f"{_cat}\n(n={_n_per_cat[_cat]})" for _cat in _cat_order],
        fontsize=10,
    )
    # ── Y ticks ────────────────────────────────────────────────────────────────────
    _ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.tick_params(axis="y", which="major", colors="black", length=0, direction="out", labelsize=9)
    _ax.tick_params(axis="y", which="minor", colors="grey",  length=0, direction="out", labelsize=6)
    _ax.tick_params(axis="x", which="both",  colors="black", length=4, direction="out")

    # ── Style ──────────────────────────────────────────────────────────────────────
    _ax.set_facecolor("white")
    _fig.patch.set_facecolor("white")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["left"].set_color("black")
    _ax.spines["left"].set_linewidth(0.5)
    _ax.spines["bottom"].set_color("black")
    _ax.spines["bottom"].set_linewidth(0.5)
    _ax.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(downloads_dir / "pac_per_category.png", dpi=300, bbox_inches="tight")

    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""## Categories feature wise active comparison""")
    return


@app.cell
def _(
    combinations,
    df_bsc_cp_pod_all_featurewise,
    feature_wise_permutation_threshold,
    kruskal,
    wilcoxon,
):
    # ── Categories and colors ─────────────────────────────────────────────────────
    _categories = ["DNA", "ER", "AGP", "RNA", "Mito", "Brightfield"]
    _cat_colors = {
        "DNA":         "#e41a1c",
        "ER":          "#377eb8",
        "AGP":         "#4daf4a",
        "RNA":         "#984ea3",
        "Mito":        "#ff7f00",
        "Brightfield": "#a65628",
    }

    def _get_category(_feature_name):
        for _cat in _categories:
            if _cat in str(_feature_name):
                return _cat
        return None

    # ── Filter ────────────────────────────────────────────────────────────────────
    _df_feat = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()

    _n_total = _df_feat.groupby("Metadata_compound").size()
    _valid_compounds = _n_total[_n_total >= feature_wise_permutation_threshold].index
    _df_feat = _df_feat[_df_feat["Metadata_compound"].isin(_valid_compounds)].copy()

    _df_feat["category"] = _df_feat["gene.id"].apply(_get_category)

    _df_feat_counts = (
        _df_feat[_df_feat["category"].notna()]
        .groupby(["Metadata_compound", "category"])
        .size()
        .reset_index(name="n")
    )

    _cat_order = (
        _df_feat_counts.groupby("category")["n"]
        .median()
        .sort_values()
        .index.tolist()
    )

    # ── Pivot: one row per compound, one col per category ─────────────────────────
    _df_pivot = (
        _df_feat_counts
        .pivot(index="Metadata_compound", columns="category", values="n")
    )

    # ── Kruskal-Wallis global test ────────────────────────────────────────────────
    _data_box = [
        _df_feat_counts.loc[_df_feat_counts["category"] == _cat, "n"].values
        for _cat in _cat_order
    ]
    _kw_stat, _kw_p = kruskal(*_data_box)
    print(f"Kruskal-Wallis: H={_kw_stat:.3f}, p={_kw_p:.4e}")

    # ── Pairwise Wilcoxon signed-rank on common compounds + Bonferroni ────────────
    _pairs = list(combinations(_cat_order, 2))
    _n_tests = len(_pairs)
    _pvals = {}
    _n_pairs_used = {}

    for _cat_a, _cat_b in _pairs:
        if _cat_a not in _df_pivot.columns or _cat_b not in _df_pivot.columns:
            continue
        # keep only compounds present in BOTH categories
        _common = _df_pivot[[_cat_a, _cat_b]].dropna()
        _n_pairs_used[(_cat_a, _cat_b)] = len(_common)
        if len(_common) < 2:
            _pvals[(_cat_a, _cat_b)] = 1.0
            continue
        _stat, _p = wilcoxon(_common[_cat_a].values, _common[_cat_b].values)
        _pvals[(_cat_a, _cat_b)] = min(_p * _n_tests, 1.0)  # Bonferroni

    print(f"\nPairwise Wilcoxon signed-rank (Bonferroni corrected, n tests={_n_tests}):")
    for (_cat_a, _cat_b), _p in _pvals.items():
        _n = _n_pairs_used.get((_cat_a, _cat_b), 0)
        print(f"  {_cat_a:12s} vs {_cat_b:12s} | n_common={_n:3d} | p={_p:.4f}")

    def _stars(_p):
        if _p < 0.001:  return "***"
        elif _p < 0.01: return "**"
        elif _p < 0.05: return "*"
        else:           return "ns"

    _sig_pairs = [
        (_cat_order.index(_a), _cat_order.index(_b))
        for (_a, _b), _p in _pvals.items() if _p < 0.05
    ]

    category_active_feature_stats = {
        "df_feat_counts": _df_feat_counts,
        "cat_order": _cat_order,
        "cat_colors": _cat_colors,
        "data_box": _data_box,
        "kw_stat": _kw_stat,
        "kw_p": _kw_p,
        "pvals": _pvals,
        "n_pairs_used": _n_pairs_used,
        "sig_pairs": _sig_pairs,
    }
    print(f"Active-feature category compounds passing threshold: {len(_valid_compounds)}")
    print(f"Active-feature category rows: {len(_df_feat_counts)}")
    return (category_active_feature_stats,)


@app.cell
def _(
    category_active_feature_stats,
    downloads_dir,
    mticker,
    np,
    plt,
    significance_stars,
):
    # Render active-feature-count category comparison plot.
    _df_feat_counts = category_active_feature_stats["df_feat_counts"]
    _cat_order = category_active_feature_stats["cat_order"]
    _cat_colors = category_active_feature_stats["cat_colors"]
    _data_box = category_active_feature_stats["data_box"]
    _pvals = category_active_feature_stats["pvals"]
    _n_pairs_used = category_active_feature_stats["n_pairs_used"]
    _sig_pairs = category_active_feature_stats["sig_pairs"]

    _fig, _ax = plt.subplots(figsize=(8, 5 + len(_sig_pairs) * 0.25))

    _bp = _ax.boxplot(
        _data_box,
        patch_artist=True,
        vert=True,
        widths=0.5,
        flierprops=dict(marker="o", markersize=4, alpha=0.4),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )

    for _patch, _cat in zip(_bp["boxes"], _cat_order):
        _patch.set_facecolor(_cat_colors[_cat])
        _patch.set_alpha(0.6)

    for _i, _cat in enumerate(_cat_order, start=1):
        _y = _df_feat_counts.loc[_df_feat_counts["category"] == _cat, "n"].values
        _x = np.random.normal(_i, 0.08, size=len(_y))
        _ax.scatter(_x, _y, color=_cat_colors[_cat], s=15, alpha=0.5, zorder=3)

    # ── Significance brackets ──────────────────────────────────────────────────────
    _y_max = max(_v.max() for _v in _data_box)
    _y_top = np.log10(_y_max)
    _bracket_gap = 0.2

    for _k, (_i, _j) in enumerate(_sig_pairs):
        _cat_a, _cat_b = _cat_order[_i], _cat_order[_j]
        _p = _pvals[(_cat_a, _cat_b)]
        _n = _n_pairs_used.get((_cat_a, _cat_b), "?")
        _x1, _x2 = _i + 1, _j + 1
        _y = 10 ** (_y_top + _bracket_gap * (_k + 1))
        _ax.plot([_x1, _x1, _x2, _x2], [_y * 0.88, _y, _y, _y * 0.88], lw=0.9, color="black")
        _ax.text(
            (_x1 + _x2) / 2, _y * 1.02,
            f"{significance_stars(_p)}  p={_p:.3f} (n={_n})",
            ha="center", va="bottom", fontsize=7.5, fontstyle="italic",
        )

    # ── Axes ───────────────────────────────────────────────────────────────────────
    _ax.set_yscale("log")                          # FIRST
    _ax.set_xlim(0.5, len(_cat_order) + 0.5)
    _ax.set_ylim(top=10 ** (_y_top + _bracket_gap * (len(_sig_pairs) + 2)))
    _ax.set_ylabel("Number of active features", fontsize=11)

    _n_per_cat = {_cat: len(_df_feat_counts.loc[_df_feat_counts["category"] == _cat]) for _cat in _cat_order}
    _ax.set_xticks(range(1, len(_cat_order) + 1))
    _ax.set_xticklabels([f"{_cat}\n(n={_n_per_cat[_cat]})" for _cat in _cat_order], fontsize=10)

    # ── Y ticks ────────────────────────────────────────────────────────────────────
    _ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.tick_params(axis="y", which="major", colors="black", length=0, direction="out", labelsize=9)
    _ax.tick_params(axis="y", which="minor", colors="grey",  length=0, direction="out", labelsize=6)
    _ax.tick_params(axis="x", which="both",  colors="black", length=4, direction="out")

    # ── Style ──────────────────────────────────────────────────────────────────────
    _ax.set_facecolor("white")
    _fig.patch.set_facecolor("white")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["left"].set_color("black")
    _ax.spines["left"].set_linewidth(0.5)
    _ax.spines["bottom"].set_color("black")
    _ax.spines["bottom"].set_linewidth(0.5)
    _ax.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(downloads_dir / "boxplot_features.png", dpi=200, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(
    LogNorm,
    df_bsc_annotations,
    df_bsc_cp_pod_all_featurewise,
    dic_botanicals_scode_name,
    feature_wise_permutation_threshold,
    kruskal,
    list_botanicals,
    mannwhitneyu,
    np,
    plt,
):
    # ── Categories and colors (NO Brightfield) ────────────────────────────────────
    _categories = ["DNA", "ER", "AGP", "RNA", "Mito"]
    _cat_colors = {
        "DNA":  "#e41a1c",
        "ER":   "#377eb8",
        "AGP":  "#4daf4a",
        "RNA":  "#984ea3",
        "Mito": "#ff7f00",
    }

    def _get_category(_feature_name):
        for _cat in _categories:
            if _cat in str(_feature_name):
                return _cat
        return None

    def _stars(_p):
        if _p < 0.001:  return "***"
        elif _p < 0.01: return "**"
        elif _p < 0.05: return "*"
        else:           return "ns"

    # ── Filter: all.pass == True, exclude Brightfield ────────────────────────────
    # ── 1. Start with ALL features (no category filter yet) ───────────────────────
    _df_feat = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()

    # ── 2. Compute valid compounds using ALL channels (including Brightfield) ────
    _n_total = _df_feat.groupby("Metadata_compound").size()

    _valid_compounds = _n_total[
        _n_total >= feature_wise_permutation_threshold
    ].index

    # ── 3. Assign categories ─────────────────────────────────────────────────────
    _df_feat["category"] = _df_feat["gene.id"].apply(_get_category)

    # ── 4. NOW filter out Brightfield (category = None) ──────────────────────────
    _df_feat = _df_feat[_df_feat["category"].notna()].copy()

    # ── 5. Keep only valid compounds ─────────────────────────────────────────────
    #_df_feat = _df_feat[
    #    _df_feat["Metadata_compound"].isin(_valid_compounds)
    #].copy()

    # ── PAC per compound per category (5th pct cp_POD_um) ────────────────────────
    _df_pac_cat = (
        _df_feat.groupby(["Metadata_compound", "category"])["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reset_index(name="PAC_cat")
    )

    _df_pac_pivot = (
        _df_pac_cat.pivot(index="Metadata_compound", columns="category", values="PAC_cat")
        .reindex(columns=_categories)
    )
    _invalid = _df_pac_pivot.index.difference(_valid_compounds)
    _df_pac_pivot.loc[_invalid] = np.nan

    # ── Kruskal-Wallis + pairwise Mann-Whitney (each cat vs all others) ───────────
    # For compound X:
    #   - omnibus KW across all 5 categories
    #   - pairwise: each category vs pool of all other categories
    #     (one-sided not needed — two-sided, Bonferroni over 5 comparisons)

    _n_pairwise = len(_categories)  # 5 tests per compound (each cat vs rest)

    _kw_results = {}   # scode -> (stat, kw_p, sig, {cat: (mw_p, sig)})

    for _cpd in _valid_compounds:
        _sub = _df_feat[_df_feat["Metadata_compound"] == _cpd]

        # groups per category
        _groups = {}
        for _cat in _categories:
            _vals = _sub.loc[_sub["category"] == _cat, "cp_POD_um"].values
            if len(_vals) >= 2:
                _groups[_cat] = _vals

        # omnibus KW
        if len(_groups) >= 2:
            _kw_stat, _kw_p = kruskal(*_groups.values())
        else:
            _kw_stat, _kw_p = np.nan, np.nan

        # pairwise: each cat vs pool of all others
        _pairwise = {}
        for _cat in _categories:
            if _cat not in _groups:
                _pairwise[_cat] = (np.nan, "n/a")
                continue
            _this  = _groups[_cat]
            _other = np.concatenate([v for k, v in _groups.items() if k != _cat])
            if len(_other) < 2:
                _pairwise[_cat] = (np.nan, "n/a")
                continue
            _, _mw_p = mannwhitneyu(_this, _other, alternative="two-sided")
            _mw_p_bonf = min(_mw_p * _n_pairwise, 1.0)
            _pairwise[_cat] = (_mw_p_bonf, _stars(_mw_p_bonf))

        _kw_results[_cpd] = (
            _kw_stat,
            _kw_p,
            _stars(_kw_p) if not np.isnan(_kw_p) else "n/a",
            _pairwise,
        )

    # ── Build ordered row index ───────────────────────────────────────────────────
    _rows_ordered = []
    for _scode in list_botanicals:
        _plant_name  = dic_botanicals_scode_name.get(_scode, _scode)
        _const_scodes = df_bsc_annotations[
            df_bsc_annotations["Plant scode"] == _scode
        ][["Metadata_compound", "Constituent"]].drop_duplicates()

        _const_rows = []
        for _, _crow in _const_scodes.iterrows():
            if _crow["Metadata_compound"] in _df_pac_pivot.index:
                _const_rows.append((_crow["Metadata_compound"], f"  {_crow['Constituent']}"))

        _has_botanical = _scode in _df_pac_pivot.index
        if not _has_botanical and len(_const_rows) == 0:
            continue
        if _has_botanical:
            _rows_ordered.append((_scode, _plant_name, True))
        for _s, _lbl in _const_rows:
            _rows_ordered.append((_s, _lbl, False))

    _ordered_scodes = [r[0] for r in _rows_ordered]
    _ordered_labels = [r[1] for r in _rows_ordered]
    _is_botanical   = [r[2] for r in _rows_ordered]

    _mat_ordered = np.array([
        _df_pac_pivot.loc[_s].values if _s in _df_pac_pivot.index
        else np.full(len(_categories), np.nan)
        for _s in _ordered_scodes
    ], dtype=float)

    # ── Color scale ───────────────────────────────────────────────────────────────
    _vals_flat = _mat_ordered[~np.isnan(_mat_ordered) & (_mat_ordered > 0)]
    _norm      = LogNorm(
        vmin=max(np.nanmin(_vals_flat), 0.01),
        vmax=np.nanpercentile(_vals_flat, 95)
    )
    _cmap_feat = plt.cm.RdYlGn

    # ── Split panels ──────────────────────────────────────────────────────────────
    _half_target = len(_rows_ordered) // 2
    _bot_indices = [i for i, r in enumerate(_rows_ordered) if r[2]]
    _split_idx   = min(_bot_indices, key=lambda i: abs(i - _half_target))
    _panels_rows = [_rows_ordered[:_split_idx], _rows_ordered[_split_idx:]]

    pac_category_heatmap_data = {
        "categories": _categories,
        "cat_colors": _cat_colors,
        "kw_results": _kw_results,
        "rows_ordered": _rows_ordered,
        "mat_ordered": _mat_ordered,
        "norm": _norm,
        "cmap_feat": _cmap_feat,
        "split_idx": _split_idx,
        "panels_rows": _panels_rows,
        "n_pairwise": _n_pairwise,
    }
    print(f"PAC category heatmap rows: {len(_rows_ordered)}")
    print(f"PAC category heatmap valid compounds: {len(_valid_compounds)}")
    return (pac_category_heatmap_data,)


@app.cell
def _(
    downloads_dir,
    feature_wise_permutation_threshold,
    np,
    pac_category_heatmap_data,
    plt,
):
    # Render PAC heatmap across Cell Painting channel categories.
    _categories = pac_category_heatmap_data["categories"]
    _cat_colors = pac_category_heatmap_data["cat_colors"]
    _kw_results = pac_category_heatmap_data["kw_results"]
    _rows_ordered = pac_category_heatmap_data["rows_ordered"]
    _mat_ordered = pac_category_heatmap_data["mat_ordered"]
    _norm = pac_category_heatmap_data["norm"]
    _cmap_feat = pac_category_heatmap_data["cmap_feat"]
    _split_idx = pac_category_heatmap_data["split_idx"]
    _panels_rows = pac_category_heatmap_data["panels_rows"]
    _n_pairwise = pac_category_heatmap_data["n_pairwise"]

    # Extra columns: KW sig + 5 pairwise cols
    _n_cols      = len(_categories)
    _n_extra     = 1 #+ len(_categories)   # KW + one col per category pairwise
    _half_n      = max(len(_panels_rows[0]), len(_panels_rows[1]))
    _fig, _axes  = plt.subplots(1, 2, figsize=(10, max(8, _half_n * 0.1)), sharey=False)

    for _pi, (_ax, _panel_rows) in enumerate(zip(_axes, _panels_rows)):
        _p_labels = [r[1] for r in _panel_rows]
        _p_is_bot = [r[2] for r in _panel_rows]
        _p_scodes = [r[0] for r in _panel_rows]
        _p_start  = 0 if _pi == 0 else _split_idx
        _p_mat    = _mat_ordered[_p_start: _p_start + len(_panel_rows)]

        _p_y_pos = []
        _gap = 0.8
        _y   = 0.0
        for _i, (_lbl, _is_bot) in enumerate(zip(_p_labels, _p_is_bot)):
            if _is_bot and _i > 0:
                _y += _gap
            _p_y_pos.append(_y)
            _y += 1.0
        _p_y_pos      = np.array(_p_y_pos)
        _total_height = _p_y_pos[-1] + 1.0

        for _ri, (_y, _is_bot, _scode) in enumerate(zip(_p_y_pos, _p_is_bot, _p_scodes)):
            _row_vals = _p_mat[_ri:_ri+1, :]

            # heatmap cells
            _ax.pcolormesh(
                np.arange(_n_cols + 1),
                np.array([_y, _y + 1.0]),
                _row_vals,
                cmap=_cmap_feat, norm=_norm,
                edgecolors="white", linewidth=0.4,
            )
            _ax.axhspan(_y, _y + 1.0,
                        color="#d0d0d0" if _is_bot else "#f7f7f7",
                        alpha=0.6 if _is_bot else 0.3, zorder=0)

            # cell value annotations
            for _ci in range(_n_cols):
                _val = _p_mat[_ri, _ci]
                _cat = _categories[_ci]
                if np.isnan(_val):
                    _ax.text(_ci + 0.5, _y + 0.5, "—",
                             ha="center", va="center", fontsize=6, color="#bbbbbb")
                    continue
                _txt    = f"{_val:.0f}" if _val >= 1 else f"{_val:.2f}"
                _normed = float(_norm(_val))
                _text_col = "white" if _normed > 0.75 else "black"

                # append pairwise sig if available and significant
                if _scode in _kw_results:
                    _, _, _, _pairwise = _kw_results[_scode]
                    _mw_p, _mw_sig = _pairwise.get(_cat, (np.nan, "n/a"))
                    if _mw_sig not in ("n/a", "ns"):
                        _txt = f"{_txt} ({_mw_sig})"

                _ax.text(_ci + 0.5, _y + 0.5, _txt,
                         ha="center", va="center", fontsize=6,
                         color=_text_col, linespacing=1.2)

            # ── Stats columns (right of heatmap) — KW only ───────────────────────
            if _scode in _kw_results:
                _kw_stat, _kw_p, _kw_sig, _pairwise = _kw_results[_scode]
                _kw_col = _n_cols + 0.5
                if not np.isnan(_kw_p) and _kw_p < 0.05:
                    _ax.text(_kw_col, _y + 0.5, _kw_sig,
                             ha="center", va="center",
                             fontsize=7, fontweight="bold", color="#cc0000")
                else:
                    _ax.text(_kw_col, _y + 0.5, "ns",
                             ha="center", va="center", fontsize=6, color="#aaaaaa")

        # ── Botanical separator lines ─────────────────────────────────────────────
        _cur_lbl = _cur_y_start = _cur_y_end = None
        _bot_spans_p = []
        for _y, _lbl, _is_bot in zip(_p_y_pos, _p_labels, _p_is_bot):
            if _is_bot:
                if _cur_lbl is not None:
                    _bot_spans_p.append((_cur_lbl, _cur_y_start, _cur_y_end))
                _cur_lbl, _cur_y_start, _cur_y_end = _lbl, _y, _y + 1.0
            else:
                if _cur_lbl is not None:
                    _cur_y_end = _y + 1.0
        if _cur_lbl is not None:
            _bot_spans_p.append((_cur_lbl, _cur_y_start, _cur_y_end))

        for _bi, (_lbl, _y_start, _y_end) in enumerate(_bot_spans_p):
            if _bi > 0:
                _ax.axhline(_y_start - _gap / 2, color="#333333",
                            linewidth=1.8, linestyle="-", zorder=6)

        # ── Axes ──────────────────────────────────────────────────────────────────
        _ax.set_yticks(_p_y_pos + 0.5)
        _ax.set_yticklabels(_p_labels, fontsize=7)
        for _tick, _is_bot in zip(_ax.get_yticklabels(), _p_is_bot):
            if _is_bot:
                _tick.set_fontweight("bold")
                _tick.set_fontsize(8.5)
                _tick.set_color("#111111")
            else:
                _tick.set_fontsize(6.5)
                _tick.set_color("#555555")
                _tick.set_fontstyle("italic")
        _ax.set_ylim(0, _total_height)
        _ax.invert_yaxis()

        # x ticks: heatmap cols + stat cols

        # x ticks
        _all_xtick_pos    = list(np.arange(_n_cols) + 0.5) + [_n_cols + 0.5]
        _all_xtick_labels = _categories + ["p-value"]

        _ax.set_xticks(_all_xtick_pos)
        _ax.set_xticklabels(_all_xtick_labels, fontsize=8, fontweight="bold",
                            rotation=45, ha="left")
        for _tick, _lbl in zip(_ax.get_xticklabels(), _all_xtick_labels):
            _cat_key = _lbl.replace("vs\n", "")
            if _cat_key in _cat_colors:
                _tick.set_color(_cat_colors[_cat_key])
            else:
                _tick.set_color("black")
        _ax.xaxis.set_label_position("top")
        _ax.xaxis.tick_top()
        _ax.set_xlim(0, _n_cols + _n_extra)

        # vertical separator between heatmap and stats
        _ax.axvline(_n_cols, color="#888888", linewidth=1.2, linestyle=":", zorder=5)

        if _pi == 1:
            _sm   = plt.cm.ScalarMappable(cmap=_cmap_feat, norm=_norm)
            _cbar = _fig.colorbar(_sm, ax=_ax, shrink=0.3, pad=0.02, aspect=20)
            _cbar.set_label("CP POD Concentration (µM)", fontsize=8)
            _cbar.ax.tick_params(labelsize=7)

    _fig.suptitle(
        "PAC per CP channel category — botanicals and constituents\n"
        f"KW = omnibus Kruskal-Wallis | 'vs X' = Mann-Whitney each cat vs all others (Bonferroni n={_n_pairwise})\n"
        f"(— = below threshold of {feature_wise_permutation_threshold} features)",
        fontsize=10, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    _fig.savefig(
        downloads_dir / "heatmap_pac_category_botanicals.pdf",
        bbox_inches="tight", dpi=200
    )
    _fig
    return


@app.cell
def _(
    combinations,
    df_bsc_annotations,
    df_bsc_cp_pod_all_featurewise,
    dic_botanicals_scode_name,
    feature_wise_permutation_threshold,
    kruskal,
    mannwhitneyu,
    np,
    pd,
):
    # ── Categories and colors ─────────────────────────────────────────────────────
    _categories = ["DNA", "ER", "AGP", "RNA", "Mito", "Brightfield"]
    _cat_colors = {
        "DNA":         "#e41a1c",
        "ER":          "#377eb8",
        "AGP":         "#4daf4a",
        "RNA":         "#984ea3",
        "Mito":        "#ff7f00",
        "Brightfield": "#a65628",
    }

    def _get_category(_feature_name):
        for _cat in _categories:
            if _cat in str(_feature_name):
                return _cat
        return None

    def _stars(_p):
        if _p < 0.001:  return "***"
        elif _p < 0.01: return "**"
        elif _p < 0.05: return "*"
        else:           return "ns"

    # ── Filter: all.pass == True ──────────────────────────────────────────────────
    _df_feat = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()
    _df_feat["category"] = _df_feat["gene.id"].apply(_get_category)
    _df_feat = _df_feat[_df_feat["category"].notna()].copy()

    # ── Valid compounds (threshold filter) ────────────────────────────────────────
    _n_total         = _df_feat.groupby("Metadata_compound").size()
    _valid_compounds = _n_total[_n_total >= feature_wise_permutation_threshold].index

    # ── PAC per compound per category (5th pct cp_POD_um) ────────────────────────
    _df_pac_cat = (
        _df_feat.groupby(["Metadata_compound", "category"])["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reset_index(name="PAC_cat")
    )

    # ── Pivot ─────────────────────────────────────────────────────────────────────
    _df_pac_pivot = (
        _df_pac_cat.pivot(
            index="Metadata_compound", columns="category", values="PAC_cat"
        )
        .reindex(columns=_categories)
    )

    # ── Set to NaN compounds not meeting threshold ────────────────────────────────
    _invalid = _df_pac_pivot.index.difference(_valid_compounds)
    _df_pac_pivot.loc[_invalid] = np.nan

    # ── Kruskal-Wallis per compound ───────────────────────────────────────────────
    _kw_results = {}
    for _cpd in _valid_compounds:
        _sub    = _df_feat[_df_feat["Metadata_compound"] == _cpd]
        _groups = []
        for _cat in _categories:
            _vals = _sub.loc[_sub["category"] == _cat, "cp_POD_um"].values
            if len(_vals) >= 2:
                _groups.append(_vals)
        if len(_groups) >= 2:
            _kw_stat, _kw_p = kruskal(*_groups)
        else:
            _kw_stat, _kw_p = np.nan, np.nan
        _kw_results[_cpd] = (
            _kw_stat,
            _kw_p,
            _stars(_kw_p) if not np.isnan(_kw_p) else "n/a"
        )

    # ── Significant compounds ─────────────────────────────────────────────────────
    _sig_compounds = [
        _cpd for _cpd in _valid_compounds
        if _cpd in _kw_results
        and not np.isnan(_kw_results[_cpd][1])
        and _kw_results[_cpd][1] < 0.05
    ]

    # ── Pairwise Wilcoxon per significant compound ────────────────────────────────
    _pw_pmat = {}
    for _cpd in _sig_compounds:
        _sub  = _df_feat[_df_feat["Metadata_compound"] == _cpd]
        _pmat = np.full((len(_categories), len(_categories)), np.nan)
        _cats_available = [
            _cat for _cat in _categories
            if len(_sub.loc[_sub["category"] == _cat, "cp_POD_um"].values) >= 2
        ]
        _cpairs  = list(combinations(_cats_available, 2))
        _n_tests = len(_cpairs)
        for _cat_a, _cat_b in _cpairs:
            _i = _categories.index(_cat_a)
            _j = _categories.index(_cat_b)
            _a = _sub.loc[_sub["category"] == _cat_a, "cp_POD_um"].values
            _b = _sub.loc[_sub["category"] == _cat_b, "cp_POD_um"].values
            try:
                _, _p = mannwhitneyu(_a, _b, alternative="two-sided")
                _p_corr = min(_p * _n_tests, 1.0)
            except Exception:
                _p_corr = np.nan
            _pmat[_i, _j] = _p_corr
            _pmat[_j, _i] = _p_corr
        _pw_pmat[_cpd] = _pmat

    # ── Link compound to label: constituent + botanical ───────────────────────────
    _annot_lookup = (
        df_bsc_annotations[["Metadata_compound", "Constituent", "Plant scode"]]
        .drop_duplicates()
        .copy()
    )
    _annot_lookup["Botanical"] = _annot_lookup["Plant scode"].map(dic_botanicals_scode_name)

    _cpd_to_label = {}
    for _cpd in _sig_compounds:
        _row = _annot_lookup[_annot_lookup["Metadata_compound"] == _cpd]
        if len(_row) > 0:
            _constituent = _row["Constituent"].iloc[0]
            _botanical   = _row["Botanical"].iloc[0]
            if pd.notna(_botanical) and _botanical:
                _cpd_to_label[_cpd] = f"{_constituent}\n({_botanical})"
            else:
                _cpd_to_label[_cpd] = _constituent
        else:
            # botanical extract itself
            _cpd_to_label[_cpd] = dic_botanicals_scode_name.get(_cpd, _cpd)

    category_pairwise_matrix_data = {
        "categories": _categories,
        "cat_colors": _cat_colors,
        "sig_compounds": _sig_compounds,
        "pw_pmat": _pw_pmat,
        "kw_results": _kw_results,
        "compound_labels": _cpd_to_label,
    }
    print(f"KW-significant compounds for Figure S5C: {len(_sig_compounds)}")
    return (category_pairwise_matrix_data,)


@app.cell
def _(
    category_pairwise_matrix_data,
    figure_dir,
    math,
    np,
    plt,
    significance_stars,
):
    # Render pairwise Mann-Whitney U matrices for KW-significant compounds.
    _categories = category_pairwise_matrix_data["categories"]
    _cat_colors = category_pairwise_matrix_data["cat_colors"]
    _sig_compounds = category_pairwise_matrix_data["sig_compounds"]
    _pw_pmat = category_pairwise_matrix_data["pw_pmat"]
    _kw_results = category_pairwise_matrix_data["kw_results"]
    _cpd_to_label = category_pairwise_matrix_data["compound_labels"]

    _n_sig       = len(_sig_compounds)
    _n_cols_grid = min(6, _n_sig)
    _n_rows_grid = math.ceil(_n_sig / _n_cols_grid)

    _fig, _axes = plt.subplots(
        _n_rows_grid, _n_cols_grid,
        figsize=(_n_cols_grid * 2.4, _n_rows_grid * 2.6),
        squeeze=False
    )

    _norm_p = plt.Normalize(vmin=0, vmax=0.05)
    _cmap_p = plt.cm.Reds_r

    for _idx, _cpd in enumerate(_sig_compounds):
        _ri  = _idx // _n_cols_grid
        _ci  = _idx % _n_cols_grid
        _ax  = _axes[_ri][_ci]
        _pmat = _pw_pmat[_cpd]

        # ── Diagonal: solid category color ───────────────────────────────────────
        for _ii in range(len(_categories)):
            _ax.add_patch(plt.Rectangle(
                (_ii - 0.5, _ii - 0.5), 1, 1,
                facecolor=_cat_colors[_categories[_ii]],
                zorder=2, linewidth=0
            ))
            _ax.text(_ii, _ii, _categories[_ii][0],
                     ha="center", va="center",
                     fontsize=5, color="white", fontweight="bold", zorder=3)

        # ── Upper triangle: solid grey ────────────────────────────────────────────
        for _ii in range(len(_categories)):
            for _jj in range(_ii + 1, len(_categories)):
                _ax.add_patch(plt.Rectangle(
                    (_jj - 0.5, _ii - 0.5), 1, 1,
                    facecolor="#cccccc",
                    zorder=1, linewidth=0
                ))

        # ── Lower triangle: solid cmap background + text ──────────────────────────
        for _ii in range(len(_categories)):
            for _jj in range(len(_categories)):
                if _ii <= _jj:
                    continue
                _val = _pmat[_ii, _jj]
                if np.isnan(_val):
                    _ax.add_patch(plt.Rectangle(
                        (_jj - 0.5, _ii - 0.5), 1, 1,
                        facecolor="#f0f0f0", zorder=1, linewidth=0
                    ))
                    _ax.text(_jj, _ii, "—",
                             ha="center", va="center",
                             fontsize=4.5, color="#aaaaaa", zorder=4)
                    continue
                _bg   = _cmap_p(_norm_p(_val))
                _ax.add_patch(plt.Rectangle(
                    (_jj - 0.5, _ii - 0.5), 1, 1,
                    facecolor=_bg, zorder=1, linewidth=0
                ))
                _sig  = significance_stars(_val)
                _tcol = "white" if _val < 0.01 else "black"
                _ax.text(_jj, _ii, f"{_sig}\n{_val:.3f}",
                         ha="center", va="center",
                         fontsize=4.5, color=_tcol, fontweight="bold",
                         zorder=4, linespacing=1.3)

        # ── Axes ──────────────────────────────────────────────────────────────────
        _ax.set_xticks(range(len(_categories)))
        _ax.set_xticklabels(_categories, fontsize=4.5, fontweight="bold",
                            rotation=45, ha="right")
        for _tick, _cat in zip(_ax.get_xticklabels(), _categories):
            _tick.set_color(_cat_colors[_cat])

        _ax.set_yticks(range(len(_categories)))
        _ax.set_yticklabels(_categories, fontsize=4.5, fontweight="bold")
        for _tick, _cat in zip(_ax.get_yticklabels(), _categories):
            _tick.set_color(_cat_colors[_cat])

        _ax.set_xlim(-0.5, len(_categories) - 0.5)
        _ax.set_ylim(-0.5, len(_categories) - 0.5)
        _ax.invert_yaxis()

        # ── Title: constituent + botanical + KW result ────────────────────────────
        _, _kw_p, _kw_sig = _kw_results[_cpd]
        _cpd_lbl = _cpd_to_label.get(_cpd, _cpd)
        _title_color = (
            "#cc0000" if _kw_p < 0.001 else
            "#e07000" if _kw_p < 0.01  else
            "#333333"
        )
        _ax.set_title(
            f"{_cpd_lbl}\nKW {_kw_sig} (p={_kw_p:.3f})",
            fontsize=5, fontweight="bold", pad=3,
            color=_title_color
        )

    # ── Hide unused axes ──────────────────────────────────────────────────────────
    for _idx in range(_n_sig, _n_rows_grid * _n_cols_grid):
        _axes[_idx // _n_cols_grid][_idx % _n_cols_grid].axis("off")

    # ── Shared colorbar ───────────────────────────────────────────────────────────
    _sm   = plt.cm.ScalarMappable(cmap=_cmap_p, norm=_norm_p)
    _cbar = _fig.colorbar(_sm, ax=_axes, shrink=0.3, pad=0.02,
                          aspect=40, location="right")
    _cbar.set_label("p-value (Mann-Whitney U, Bonferroni)", fontsize=8)
    _cbar.set_ticks([0, 0.01, 0.05])
    _cbar.ax.tick_params(labelsize=7)

    _fig.suptitle(
        f"Pairwise Mann-Whitney U p-values across CP categories — "
        f"KW significant compounds (n={_n_sig})\n"
        "Lower triangle  |  diagonal = category  |  "
        "* p<0.05  ** p<0.01  *** p<0.001  (Bonferroni corrected)",
        fontsize=10, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    _fig.savefig(
        str(figure_dir / "Figure_S5_C.png"),
        bbox_inches="tight", dpi=300
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
