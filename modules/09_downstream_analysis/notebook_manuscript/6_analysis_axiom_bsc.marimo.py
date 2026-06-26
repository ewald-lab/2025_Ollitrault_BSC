import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import glob
    import pickle
    import pandas as pd
    import numpy as np
    import umap
    import plotly.express as px
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.ticker as ticker
    import matplotlib.gridspec as gridspec
    from matplotlib.gridspec import GridSpec
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from sklearn.preprocessing import StandardScaler
    from matplotlib.colors import ListedColormap, LogNorm
    import seaborn as sns
    from tqdm import tqdm
    from plotnine import (
        ggplot, aes, theme_classic, geom_boxplot, geom_sina, geom_col, element_blank, scale_y_continuous, theme_minimal,
        geom_jitter,
        labs, theme, element_text, scale_x_discrete, scale_y_log10, geom_text, facet_grid, element_rect, geom_rect, scale_x_continuous
    )

    from adjustText import adjust_text
    from scipy.stats import mannwhitneyu, kruskal, wilcoxon
    from itertools import combinations
    import plotly.graph_objects as go
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from sklearn.metrics.pairwise import cosine_similarity
    import dataframe_image as dfi
    import plotly.io as pio
    from pathlib import Path
    from matplotlib.patches import Patch
    return (
        AllChem,
        Chem,
        DataStructs,
        ListedColormap,
        LogNorm,
        Patch,
        Path,
        StandardScaler,
        adjust_text,
        aes,
        combinations,
        cosine_similarity,
        dfi,
        element_rect,
        element_text,
        facet_grid,
        geom_boxplot,
        geom_col,
        geom_sina,
        geom_text,
        ggplot,
        glob,
        go,
        kruskal,
        labs,
        mannwhitneyu,
        mlines,
        mo,
        mpatches,
        mticker,
        np,
        os,
        pd,
        pickle,
        pio,
        plt,
        px,
        scale_y_log10,
        sns,
        theme,
        theme_classic,
        ticker,
        tqdm,
        umap,
        wilcoxon,
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
    df_level4_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level4b_bsc_filtered_filtered_converted.parquet")
    if True:
        df_level4_bsc = df_level4_bsc.groupby("Metadata_plate").filter(
            lambda g: ~(g["Metadata_compound_name"] == "DMSO").all()
        )
    df_level5_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level5_bsc_selected_features_filtered_converted.parquet")
    df_level6_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level6_bsc_filtered.parquet")
    df_level5_bsc_all = pd.read_parquet("modules/03_postprocessing/outputs/level5_bsc_all_features_filtered.parquet")
    df_level6_bsc_all = pd.read_parquet("modules/03_postprocessing/outputs/level6_bsc_selected_features_filtered.parquet")
    return (
        df_level4_bsc,
        df_level5_bsc,
        df_level5_bsc_all,
        df_level6_bsc,
        df_level6_bsc_all,
    )


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
    dic_botanicals_scode_name = (
        df_bsc_annotations[["Plant scode", "Plant name"]]
        .drop_duplicates()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    dic_botanicals_scode_name
    return (dic_botanicals_scode_name,)


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
def _(glob, pd):
    # POD Cell Painting feature wise
    path_pattern_bsc_interdependence_score = "modules/06_concresponse/outputs/interdependence_score/feature_selection_interdependence_score_bsc_*.txt"

    # Get all matching files
    _files = glob.glob(path_pattern_bsc_interdependence_score)

    print(f"Found {len(_files)} files.")

    # Read and concatenate
    df_bsc_interdependence_score = pd.concat(
        [pd.read_csv(f) for f in _files],
        ignore_index=True
    )

    print(f"Final dataframe shape: {df_bsc_interdependence_score.shape}")
    return (df_bsc_interdependence_score,)


@app.cell
def _(glob, pd):
    # POD Cell Painting feature wise
    path_pattern_feature_wise = "modules/06_concresponse/outputs/pods/feature_wise/*feature_wise_original_conc.parquet"

    # Get all matching files
    _files = glob.glob(path_pattern_feature_wise)

    print(f"Found {len(_files)} files.")

    # Read and concatenate
    df_bsc_cp_pod_all_featurewise = pd.concat(
        [pd.read_parquet(f) for f in _files],
        ignore_index=True
    )

    print(f"Final dataframe shape: {df_bsc_cp_pod_all_featurewise.shape}")
    return (df_bsc_cp_pod_all_featurewise,)


@app.cell
def _():
    with open("modules/06_concresponse/outputs/pods/feature_wise/feature_wise_median_permutation_threshold.txt") as _f:
        feature_wise_permutation_threshold = float(_f.read().strip())
    return (feature_wise_permutation_threshold,)


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
def _(df_level4_bsc, df_level5_bsc, df_level6_bsc, df_level6_bsc_all):
    all_selected_features = [c for c in df_level4_bsc.columns if c.startswith("Feature_")]
    all_features = [c for c in df_level6_bsc_all.columns if c.startswith("Feature_")]
    all_metadata_4 = [c for c in df_level4_bsc.columns if c.startswith("Metadata_")]
    all_metadata_5 = [c for c in df_level5_bsc.columns if c.startswith("Metadata_")]
    all_metadata_6 = [c for c in df_level6_bsc.columns if c.startswith("Metadata_")]
    return (all_selected_features,)


@app.cell
def _(all_selected_features):
    len(all_selected_features)
    return


@app.cell
def _(mo):
    mo.md(r"""## Plates experiments focus on Botanicals """)
    return


@app.cell
def _(mo):
    mo.md(r"""## Concentration range""")
    return


@app.cell
def _(dfi, np, os, pd):
    _data = {
        "Plant name": [
            "Asian ginseng", "Ashwagandha", "Comfrey", "Green tea extract",
            "Blue cohosh", "Goldenseal", "Milk thistle", "Kava",
            "Kratom", "Usnea", "Yohimbe", "Aristolochia", "Ephedra"
        ],
        "N constituents tested": [13, 10, 9, 9, 7, 7, 7, 6, 5, 5, 5, 3, 1],
        "Min (µM)":  [0.05, 0.023, 0.05, 0.57, 0.57, 0.57, 0.76, 0.027, 0.096, 0.38, 0.29, 0.38, 0.29],
        "Max (µM)":  [110,  50,    110,  1200, 1200, 1200, 1700, 60,    210,   830,  620,  830,  620],
    }

    _df_plants = pd.DataFrame(_data)

    # ── Log-scale normalisation for bar width ─────────────────────────────────────
    _global_log_min = np.log10(_df_plants["Min (µM)"].min())
    _global_log_max = np.log10(_df_plants["Max (µM)"].max())

    def _range_bar(_row):
        _log_min = np.log10(_row["Min (µM)"])
        _log_max = np.log10(_row["Max (µM)"])
        # normalise to 0-100% of the column width
        _left  = (_log_min - _global_log_min) / (_global_log_max - _global_log_min) * 100
        _width = (_log_max - _log_min)        / (_global_log_max - _global_log_min) * 100
        _span  = _row["Max (µM)"] / _row["Min (µM)"]
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
    _df_display = _df_plants[["Plant name", "N constituents tested", "Concentration range (µM)"]].copy()

    _styled = (
        _df_display.style
        .background_gradient(
            subset=["N constituents tested"],
            cmap="YlGn",
            vmin=0,
            vmax=_df_plants["N constituents tested"].max()
        )
        .set_properties(**{
            "font-size": "12px",
            "text-align": "center",
            "border": "1px solid #e0e0e0",
            "padding": "6px 12px",
        })
        .set_properties(subset=["Plant name"], **{"text-align": "left", "font-weight": "bold"})
        .set_properties(subset=["Concentration range (µM)"], **{"min-width": "220px", "padding": "4px 8px"})
        .format({"Concentration range (µM)": lambda _x: _x})  # keep raw HTML
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", "#2c6e49"),
                ("color", "white"),
                ("font-size", "12px"),
                ("font-weight", "bold"),
                ("text-align", "center"),
                ("padding", "8px 12px"),
                ("border-bottom", "2px solid #1a4731"),
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
        .set_caption("Botanical constituents tested in Cell Painting assay")
        .set_table_styles([{
            "selector": "caption",
            "props": [
                ("font-size", "14px"),
                ("font-weight", "bold"),
                ("color", "#2c6e49"),
                ("padding-bottom", "10px"),
            ]
        }], overwrite=False)
    )
    dfi.export(
        _styled,
        os.path.expanduser("~/Downloads/botanical_constituents_table.png"),
        dpi=200,
        table_conversion="chrome"  # or "matplotlib" if chrome not available
    )
    _styled
    return


@app.cell
def _(df_level5_bsc, dfi, dic_botanicals_scode_name, np, os):
    # ── Build Min/Max from df_level5_bsc ──────────────────────────────────────────
    _conc_stats = (
        df_level5_bsc
        .groupby("Metadata_compound")["Metadata_concentration"]
        .agg(["min", "max"])
        .rename(columns={"min": "Min (µM)", "max": "Max (µM)"})
        .reset_index()
    )

    # Map Plant scode → Plant name
    _conc_stats["Plant name"] = _conc_stats["Metadata_compound"].map(dic_botanicals_scode_name)

    # Drop rows where scode had no matching plant name
    _conc_stats = _conc_stats.dropna(subset=["Plant name"])

    # ── N tested constituents (keep hardcoded or derive separately) ───────────────
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

    # ── Assemble _df_plants ───────────────────────────────────────────────────────
    _df_plants = _conc_stats[["Plant name", "Min (µM)", "Max (µM)"]].copy()
    _df_plants["N tested constituents"]        = _df_plants["Plant name"].map(_n_constituents)
    _df_plants["Hepatotoxicity (literature)"]  = _df_plants["Plant name"].map(_hepatotox)

    # Sort to match original order (by N tested constituents descending)
    _df_plants = _df_plants.sort_values("N tested constituents", ascending=False).reset_index(drop=True)


    # ── Log-scale normalisation for bar width ─────────────────────────────────────
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

    # ── Prepare display dataframe ─────────────────────────────────────────────────
    _df_display = _df_plants[
        ["Plant name", "Hepatotoxicity (literature)", "N tested constituents", "Concentration range (µM)"]
    ].copy()
    # ── Hepatotoxicity color coding ───────────────────────────────────────────────
    def _hep_color(val):
        if val == "Evidence":
            return "background-color: #f8d7da; color: #721c24;"
        elif val == "Some evidence":
            return "background-color: #fff3cd; color: #856404;"
        elif val == "No evidence":
            return "background-color: #d4edda; color: #155724;"
        else:
            return "background-color: #e2e3e5; color: #383d41;"

    # ── Styling ───────────────────────────────────────────────────────────────────
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
        os.path.expanduser("~/Downloads/botanical_constituents_table.png"),
        dpi=200,
        table_conversion="chrome"
    )

    _styled
    return


@app.cell
def _(
    df_bsc_all_pod_annotations,
    df_level5_bsc,
    df_level6_bsc,
    dic_botanicals_scode_name,
    plt,
):
    # ── 1. Rebuild _df_level6_bsc_plantscode ─────────────────────────────────────
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

    # ── 2. Classify compounds ─────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())
    _df_conc = df_level5_bsc[["Metadata_compound", "Metadata_concentration"]].drop_duplicates().copy()
    _df_conc = _df_conc[_df_conc["Metadata_compound"] != "s0617825"]
    _df_conc["Group"] = _df_conc["Metadata_compound"].apply(
        lambda _c: "Botanical" if _c in _botanical_scodes else "Constituent"
    )

    # ── 3. Aggregate min/max/doses ────────────────────────────────────────────────
    _constituent_lookup = (
        _df_level6_bsc_plantscode[["Metadata_compound", "Constituent"]]
        .drop_duplicates()
        .dropna(subset=["Constituent"])
        .set_index("Metadata_compound")["Constituent"]
        .to_dict()
    )

    _df_agg = (
        _df_conc.groupby(["Metadata_compound", "Group"])["Metadata_concentration"]
        .agg(Min="min", Max="max", N="count")
        .reset_index()
    )
    _df_agg["Label"] = _df_agg.apply(
        lambda _r: dic_botanicals_scode_name.get(_r["Metadata_compound"])
                   if _r["Group"] == "Botanical"
                   else _constituent_lookup.get(_r["Metadata_compound"], _r["Metadata_compound"]),
        axis=1
    )

    _df_bot = _df_agg[_df_agg["Group"] == "Botanical"].sort_values("Min").reset_index(drop=True)
    _df_con = _df_agg[_df_agg["Group"] == "Constituent"].sort_values("Min").reset_index(drop=True)

    # ── 4. Draw function ──────────────────────────────────────────────────────────
    def _draw_barplot(_ax, _df, _color, _title):
        _bar_height = 0.5
        for _i, _row in _df.iterrows():
            _left = _row["Min"]
            _width = _row["Max"] - _row["Min"]

            # bar spanning min to max
            _ax.barh(
                _i, _width, left=_left,
                height=_bar_height, color=_color,
                alpha=0.7, edgecolor="black", linewidth=0.4
            )
            # dot at min
            _ax.scatter(_row["Min"], _i, color="black", s=20, zorder=5)
            # dot at max
            _ax.scatter(_row["Max"], _i, color="black", s=20, zorder=5)

            # min value text (left of bar)
            _ax.text(
                _row["Min"], _i + _bar_height * 0.6,
                f'{_row["Min"]:.2g} µM',
                ha="left", va="bottom", fontsize=6.5, color="black"
            )
            # max value text (right of bar)
            _ax.text(
                _row["Max"], _i + _bar_height * 0.6,
                f'{_row["Max"]:.2g} µM',
                ha="right", va="bottom", fontsize=6.5, color="black"
            )
            # N doses text (inside bar)
            if _width > 0:
                _ax.text(
                    _left + _width / 2, _i,
                    f'n={_row["N"]}',
                    ha="center", va="center", fontsize=6, color="white", fontweight="bold"
                )

        _ax.set_yticks(range(len(_df)))
        _ax.set_yticklabels(_df["Label"], fontsize=8)
        _ax.set_xscale("log")
        _ax.set_xlabel("Concentration (µM, log scale)", fontsize=10)
        _ax.set_title(_title, fontsize=11, fontweight="bold")
        _ax.grid(axis="x", linestyle="--", alpha=0.4)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.set_ylim(-0.8, len(_df) - 0.2)

    # ── 5. Plot ───────────────────────────────────────────────────────────────────
    _n_bot = len(_df_bot)
    _n_con = len(_df_con)
    _fig, (_ax_bot, _ax_con) = plt.subplots(
        1, 2,
        figsize=(20, max(_n_bot, _n_con) * 0.4 + 3)  # auto height based on n compounds
    )

    _draw_barplot(_ax_bot, _df_bot, "#2ca02c", "Botanicals — concentration range")
    _draw_barplot(_ax_con, _df_con, "#1f77b4", "Constituents — concentration range")

    plt.suptitle("Tested concentration ranges (Cell Painting)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP level 4 filtered colored by botanicals""")
    return


@app.cell
def _(all_selected_features, df_level4_bsc, pd, umap):
    # --- 1. Extract features and metadata
    X_bsc_4 = df_level4_bsc[all_selected_features].values
    _compound = df_level4_bsc["Metadata_compound"].astype(str).values
    _plates = df_level4_bsc["Metadata_plate"].astype(str).values
    _concentration = df_level4_bsc["Metadata_concentration"].astype(float).values
    _concentration_log = df_level4_bsc["Metadata_log10conc"].astype(float).values
    # --- 2. Run UMAP for 2D projection
    reducer_bsc_4 = umap.UMAP(
        n_neighbors=50,          # local neighborhood size
        min_dist=0.3,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    embedding_bsc_4 = reducer_bsc_4.fit_transform(X_bsc_4)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_bsc_4 = pd.DataFrame(embedding_bsc_4, columns=["UMAP_1", "UMAP_2"])
    df_umap_bsc_4["Metadata_compound"] = _compound
    df_umap_bsc_4["Metadata_plate"] = _plates
    df_umap_bsc_4["Metadata_concentration"] = _concentration
    df_umap_bsc_4["Metadata_log10conc"] = _concentration_log
    return (df_umap_bsc_4,)


@app.cell
def _():
    return


@app.cell
def _(df_umap_bsc_4, dic_botanicals_scode_name, plant_scode_colors, px):
    # ── 1. Replace codes with actual botanical names ──────────────────────────────
    # (Assumes you have a mapping dict like: code -> full name)
    df_umap_bsc_4["Botanical_Name"] = df_umap_bsc_4["Metadata_compound"].map(dic_botanicals_scode_name)

    # ── 2. Create plotting group: botanicals vs others ────────────────────────────
    df_umap_bsc_4["plot_group"] = df_umap_bsc_4["Botanical_Name"].where(
        df_umap_bsc_4["Botanical_Name"].isin(dic_botanicals_scode_name.values()),
        "Constituents"
    )

    # ── 3. Define color palette (consistent with your previous figures) ───────────
    BOTANICAL_COLORS = {
        name: color for name, color in zip(
            dic_botanicals_scode_name.values(),
            px.colors.qualitative.Set2
        )
    }
    BOTANICAL_COLORS = plant_scode_colors
    BOTANICAL_COLORS["Constituents"] = "lightgrey"
    # ── 4. Plot ──────────────────────────────────────────────────────────────────
    _fig = px.scatter(
        df_umap_bsc_4.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="plot_group",
        title="UMAP of Level 4 Cell Painting Profiles",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_map=BOTANICAL_COLORS
    )

    # ── 5. Improve marker style ───────────────────────────────────────────────────
    _fig.update_traces(
        marker=dict(size=6, line=dict(width=0))
    )

    # ── 6. Layout styling (match your previous matplotlib style) ──────────────────
    _fig.update_layout(
        template="simple_white",
        legend_title_text="",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        font=dict(size=12, color="black"),
    )

    # Remove gridlines for cleaner look
    _fig.update_xaxes(showgrid=False, zeroline=False)
    _fig.update_yaxes(showgrid=False, zeroline=False)

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP level 5 filtered colored by botanicals""")
    return


@app.cell
def _(all_selected_features, df_level5_bsc, pd, umap):
    # --- 1. Extract features and metadata
    X_bsc_5 = df_level5_bsc[all_selected_features].values
    _compound = df_level5_bsc["Metadata_compound"].astype(str).values
    _concentration = df_level5_bsc["Metadata_concentration"].astype(float).values
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
def _(all_selected_features, df_level6_bsc, pd, umap):
    # --- 1. Extract features and metadata
    X_bsc_6 = df_level6_bsc[all_selected_features].values
    _compound = df_level6_bsc["Metadata_compound"].astype(str).values
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
    mo.md(r"""## POD Cell Painting VS MT VS LDH [Mahalanobis distance + FastBMD]""")
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
    mo.md(r"""## Features Interdependence score""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Per botanicals""")
    return


@app.cell
def _(
    df_bsc_interdependence_score,
    dic_botanicals_scode_name,
    list_botanicals,
    np,
    plt,
):
    # ── Categories and colors ─────────────────────────────────────────────────────
    _categories = ["DNA", "ER", "AGP", "RNA", "Mito", "Brightfield"]
    _cat_colors = {
        "DNA":        "#e41a1c",
        "ER":         "#377eb8",
        "AGP":        "#4daf4a",
        "RNA":        "#984ea3",
        "Mito":       "#ff7f00",
        "Brightfield":"#a65628",
    }

    # Feature columns: only cp_POD_um rows have a feature column
    # Extract category from feature name for each row
    def _get_category(feature_name):
        for _cat in _categories:
            if _cat in str(feature_name):
                return _cat
        return None
    # ── Filter to botanicals ──────────────────────────────────────────────────────
    _df_inter = df_bsc_interdependence_score[
        df_bsc_interdependence_score["Metadata_compound"].isin(list_botanicals)
    ].copy()
    _df_inter["Metadata_compound"] = _df_inter["Metadata_compound"].map(
        lambda x: dic_botanicals_scode_name.get(x, x)
    )

    # ── Assign category from feature column ───────────────────────────────────────
    _df_inter["category"] = _df_inter["feature"].apply(_get_category)

    # ── Count features per compound per category ──────────────────────────────────
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
    for _cat in _categories:
        if _cat not in _df_inter_pivot.columns:
            _df_inter_pivot[_cat] = 0
    _df_inter_pivot = _df_inter_pivot[_categories]

    # Order by total count
    _df_inter_pivot["total"] = _df_inter_pivot.sum(axis=1)
    _df_inter_pivot = _df_inter_pivot.sort_values("total", ascending=True)
    _df_inter_pivot = _df_inter_pivot.drop(columns="total")

    _compounds_inter = _df_inter_pivot.index.values
    _y_pos = np.arange(len(_compounds_inter))

    # ── Plot ──────────────────────────────────────────────────────────────────────
    _fig, _ax = plt.subplots(figsize=(8, max(4, len(_compounds_inter) * 0.4)))

    _left = np.zeros(len(_compounds_inter))
    for _cat in _categories:
        _vals_bar = _df_inter_pivot[_cat].values
        _ax.barh(_y_pos, _vals_bar, left=_left,
                 color=_cat_colors[_cat], height=0.6, alpha=0.85, label=_cat)
        _left += _vals_bar

    # Total label at end of each bar
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
    df_bsc_annotations,
    df_bsc_interdependence_score,
    list_botanicals,
    np,
    plt,
):
    # ── Categories and colors ─────────────────────────────────────────────────────
    _categories = ["DNA", "ER", "AGP", "RNA", "Mito", "Brightfield"]
    _cat_colors = {
        "DNA":        "#e41a1c",
        "ER":         "#377eb8",
        "AGP":        "#4daf4a",
        "RNA":        "#984ea3",
        "Mito":       "#ff7f00",
        "Brightfield":"#a65628",
    }

    def _get_category(feature_name):
        for _cat in _categories:
            if _cat in str(feature_name):
                return _cat

    # ── Filter to constituents ────────────────────────────────────────────────────
    _df_inter = df_bsc_interdependence_score[
        ~df_bsc_interdependence_score["Metadata_compound"].isin(list_botanicals)
    ].copy()

    # ── Assign category ───────────────────────────────────────────────────────────
    _df_inter["category"] = _df_inter["feature"].apply(_get_category)

    # ── Count features per constituent per category ───────────────────────────────
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
    for _cat in _categories:
        if _cat not in _df_inter_pivot.columns:
            _df_inter_pivot[_cat] = 0
    _df_inter_pivot = _df_inter_pivot[_categories]

    # ── Order constituents by botanical then name ─────────────────────────────────
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

    _compounds_inter = _primary_plant_inter["Metadata_compound"].values
    _constituent_labels_inter = _primary_plant_inter["Constituent"].values
    _x_pos_inter = np.arange(len(_compounds_inter)) * 0.6

    # ── Reorder pivot to match constituent order ──────────────────────────────────
    _df_inter_pivot = (
        _primary_plant_inter[["Metadata_compound"]]
        .merge(_df_inter_pivot.reset_index(), on="Metadata_compound", how="left")
        .fillna(0)
        .set_index("Metadata_compound")[_categories]
    )

    # ── Plot ──────────────────────────────────────────────────────────────────────
    _fig, _ax = plt.subplots(figsize=(8, max(6, len(_compounds_inter) * 0.20)))

    _left = np.zeros(len(_compounds_inter))
    for _cat in _categories:
        _vals_bar = _df_inter_pivot[_cat].values
        _ax.barh(_x_pos_inter, _vals_bar, left=_left,
                 color=_cat_colors[_cat], height=0.3, alpha=0.85, label=_cat)
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
    mo.md(r"""## POD Cell Painting VS MT VS LDH [Feature wise + FastBMD]""")
    return


@app.cell
def _(mo):
    mo.md(r"""### POD CP MT LDH Table""")
    return


@app.cell
def _(
    df_bsc_all_pod_annotations,
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    df_level6_bsc,
    dic_botanicals_scode_name,
    feature_wise_permutation_threshold,
    os,
    pd,
):
    # ── 1. CP POD global per compound ─────────────────────────────────────────────
    _df_cp_passed = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()
    _n_active = _df_cp_passed.groupby("Metadata_compound").size()
    _valid    = _n_active[_n_active >= feature_wise_permutation_threshold].index

    # All compounds ever seen in the featurewise data (pass or not)
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

    # ── 2. CP POD per compound per category ───────────────────────────────────────
    _categories = ["DNA", "ER", "AGP", "RNA", "Mito", "Brightfield"]

    def _get_category(_feature_name):
        for _cat in _categories:
            if _cat in str(_feature_name):
                return _cat
        return None

    _df_cp_valid["category"] = _df_cp_valid["gene.id"].apply(_get_category)

    _df_pac_cat = (
        _df_cp_valid[_df_cp_valid["category"].notna()]
        .groupby(["Metadata_compound", "category"])["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .unstack(level="category")
        .rename(columns={_cat: f"PAC {_cat} (µM)" for _cat in _categories})
    )
    # Reindex to include all compounds (NaN for those below threshold)
    _df_pac_cat = _df_pac_cat.reindex(_cp_pac_series.index)

    # ── 3. MTT POD per compound ───────────────────────────────────────────────────
    _mtt_series = (
        df_bsc_mtt_pod[df_bsc_mtt_pod["all.pass"] == True]
        .groupby("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .min()
        .rename("MTT POD (µM)")
    )

    # ── 4. LDH POD per compound ───────────────────────────────────────────────────
    _ldh_series = (
        df_bsc_ldh_pod[df_bsc_ldh_pod["all.pass"] == True]
        .groupby("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .min()
        .rename("LDH POD (µM)")
    )

    # ── 5. Rebuild compound display names ─────────────────────────────────────────
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

    # ── 6. Merge all into one table ───────────────────────────────────────────────
    # Union of all compound IDs across all sources
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

    # ── Flag compounds below threshold ───────────────────────────────────────────
    _df_table["Below threshold"] = _df_table["Compound ID"].apply(
        lambda _c: "✗" if _c in _all_compounds_fw and _c not in _valid else ""
    )

    # ── 7. Reorder + sort ─────────────────────────────────────────────────────────
    _cat_pac_cols = [f"PAC {_cat} (µM)" for _cat in _categories if f"PAC {_cat} (µM)" in _df_table.columns]
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

    # ── 8. Style ──────────────────────────────────────────────────────────────────
    def _color_group(_val):
        return "background-color: #d4edda" if _val == "Botanical" else "background-color: #d0e8f7"

    def _color_threshold(_val):
        return "background-color: #ffe0e0; color: #cc0000; font-weight: bold" if _val == "✗" else ""

    def _fmt(_val):
        if pd.isna(_val): return "—"
        return f"{_val:.3g}"

    _gradient_cols = ["CP POD (µM)"] + _cat_pac_cols + ["MTT POD (µM)", "LDH POD (µM)"]
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
    _styled.to_excel(os.path.expanduser("~/Downloads/styled_table.xlsx"), engine="openpyxl")
    _df_table.to_csv("modules/06_concresponse/outputs/pods/df_pods_bsc_cp_mt_ldh.csv")
    df_table_results_pods = _df_table
    _styled
    return (df_table_results_pods,)


@app.cell
def _(df_table_results_pods):
    df_table_results_pods["Below threshold"] = df_table_results_pods["Below threshold"] != "✗"
    df_table_results_pods
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
):
    # ── 1. PAC per constituent ────────────────────────────────────────────────────
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
    len(_pac_lookup)
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
    _order = (
        _df_long.groupby("Plant name")["PAC"]
        .median()
        .sort_values()
        .index.tolist()
    )
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
        _bot_cp = _botanical_pac_dict.get(_scode)
        _bot_cp_str = f"{_bot_cp:.2f}" if _bot_cp is not None and pd.notna(_bot_cp) else "—"

        # ── MTT POD ───────────────────────────────────────────────────────────────
        # Active MTT = compounds in df_bsc_mtt_pod that pass for this botanical
        _mtt_vals = (
            df_bsc_mtt_pod[
                (df_bsc_mtt_pod["all.pass"] == True) &
                (df_bsc_mtt_pod["Metadata_compound"].isin(
                    _annot.loc[_annot["Plant scode"] == _scode, "Metadata_compound"]
                ))
            ]["Metadata_mtt_normalized_POD_um"].dropna().values
        )
        _active_mtt_n  = len(_mtt_vals)
        _mtt_pct       = f"{_active_mtt_n}/{_total_known_n} ({100*_active_mtt_n/_total_known_n:.0f}%)" if _total_known_n > 0 else "—"
        _mtt_median    = f"{np.median(_mtt_vals):.2f}" if len(_mtt_vals) > 0 else "—"

        # ── LDH POD ───────────────────────────────────────────────────────────────
        _ldh_vals = (
            df_bsc_ldh_pod[
                (df_bsc_ldh_pod["all.pass"] == True) &
                (df_bsc_ldh_pod["Metadata_compound"].isin(
                    _annot.loc[_annot["Plant scode"] == _scode, "Metadata_compound"]
                ))
            ]["Metadata_ldh_normalized_POD_um"].dropna().values
        )
        _active_ldh_n  = len(_ldh_vals)
        _ldh_pct       = f"{_active_ldh_n}/{_total_known_n} ({100*_active_ldh_n/_total_known_n:.0f}%)" if _total_known_n > 0 else "—"
        _ldh_median    = f"{np.median(_ldh_vals):.2f}" if len(_ldh_vals) > 0 else "—"

        _rows_table.append({
            "Botanical"                      : _plant,
            "Active CP (n/total)"            : _cp_pct,
            "Median constituent CP POD (μM)" : _cp_median,
            "Botanical CP POD (μM)"          : _bot_cp_str,
            "Active MTT (n/total)"           : _mtt_pct,
            "Median MTT POD (μM)"            : _mtt_median,
            "Active LDH (n/total)"           : _ldh_pct,
            "Median LDH POD (μM)"            : _ldh_median,
        })

    df_summary_table = pd.DataFrame(_rows_table)
    df_summary_table
    # ── Summary footer row ────────────────────────────────────────────────────────

    # CP: deduplicate constituents (a constituent shared by multiple botanicals counts once)
    _all_cp_vals = (
        _df_long.drop_duplicates(subset="Metadata_compound")["PAC"].dropna().values
    )
    _total_known_all = df_bsc_annotations[
        df_bsc_annotations["Plant scode"].isin(list_botanicals)
    ]["Metadata_compound"].nunique()
    _active_cp_all = _df_long["Metadata_compound"].nunique()

    # MTT: deduplicate constituents across all botanicals
    _all_mtt_vals = (
        df_bsc_mtt_pod[
            (df_bsc_mtt_pod["all.pass"] == True) &
            (df_bsc_mtt_pod["Metadata_compound"].isin(_annot["Metadata_compound"]))
        ]
        .drop_duplicates(subset="Metadata_compound")
        ["Metadata_mtt_normalized_POD_um"].dropna().values
    )
    _active_mtt_all = len(_all_mtt_vals)

    # LDH: deduplicate constituents across all botanicals
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
        "Median MTT POD (μM)"            : f"{np.median(_all_mtt_vals):.2f}" if len(_all_mtt_vals) > 0 else "—",
        "Active LDH (n/total)"           : f"{_active_ldh_all}/{_total_known_all} ({100*_active_ldh_all/_total_known_all:.0f}%)",
        "Median LDH POD (μM)"            : f"{np.median(_all_ldh_vals):.2f}" if len(_all_ldh_vals) > 0 else "—",
    })

    df_summary_table = pd.DataFrame(_rows_table)
    df_summary_table
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
):
    # ── 1. PAC per constituent ────────────────────────────────────────────────────
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
    len(_pac_lookup)
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
    df_summary_table_more
    return


@app.cell
def _(mo):
    mo.md(r"""### POD Cell Painting VS MT VS LDH - All cytotoxicity [Feature wise + FastBMD]""")
    return


@app.cell
def _(
    combinations,
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    feature_wise_permutation_threshold,
    mannwhitneyu,
    mticker,
    np,
    plt,
):
    # ── 1. CP POD ─────────────────────────────────────────────────────────────────
    _df_cp_passed = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()
    _n_active = _df_cp_passed.groupby("Metadata_compound").size()
    _valid = _n_active[_n_active >= feature_wise_permutation_threshold].index
    _cp_pac = (
        _df_cp_passed[_df_cp_passed["Metadata_compound"].isin(_valid)]
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .values
    )

    # ── 2. MTT POD ────────────────────────────────────────────────────────────────
    _mtt_vals = (
        df_bsc_mtt_pod[df_bsc_mtt_pod["all.pass"] == True]
        .groupby("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
        .min()
        .values
    )

    # ── 3. LDH POD ────────────────────────────────────────────────────────────────
    _ldh_vals = (
        df_bsc_ldh_pod[df_bsc_ldh_pod["all.pass"] == True]
        .groupby("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
        .min()
        .values
    )

    # ── 4. Pairwise Mann-Whitney U with Bonferroni correction ────────────────────
    _data   = [_cp_pac, _mtt_vals, _ldh_vals]
    _labels = ["CP POD", "MTT POD", "LDH POD"]
    _colors = ["steelblue", "darkorange", "green"]

    _pairs = list(combinations(range(3), 2))  # (0,1), (0,2), (1,2)
    _n_tests = len(_pairs)

    _pvals = {}
    for _i, _j in _pairs:
        _, _p = mannwhitneyu(_data[_i], _data[_j], alternative="two-sided")
        _p_corrected = min(_p * _n_tests, 1.0)  # Bonferroni
        _pvals[(_i, _j)] = _p_corrected

    def _stars(_p):
        if _p < 0.001: return "***"
        elif _p < 0.01: return "**"
        elif _p < 0.05: return "*"
        else: return "ns"

    # ── 5. Plot ───────────────────────────────────────────────────────────────────
    _fig, _ax = plt.subplots(figsize=(6, 6))

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
    for _patch, _color in zip(_bp["boxes"], _colors):
        _patch.set_facecolor(_color)
        _patch.set_alpha(0.35)

    for _i, (_v, _color) in enumerate(zip(_data, _colors), start=1):
        _x = np.random.normal(_i, 0.08, size=len(_v))
        _ax.scatter(_x, _v, color=_color, s=30, alpha=0.7, zorder=3)

    _ax.set_yscale("log")
    _ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.set_xticks(range(1, 4))
    _ax.set_xticklabels(_labels, fontsize=11)
    _ax.set_ylabel("POD (μM, log scale)", fontsize=11)
    _ax.set_title("CP POD vs MTT vs LDH POD\n(Mann-Whitney U, Bonferroni corrected)", fontsize=11)

    # ── 6. Significance brackets ──────────────────────────────────────────────────
    _y_max = max(max(_v) for _v in _data)
    _y_top = np.log10(_y_max)
    _bracket_gap = 0.18  # gap between brackets in log space

    for _k, (_i, _j) in enumerate(_pairs):
        _p = _pvals[(_i, _j)]
        _sig = _stars(_p)
        _x1, _x2 = _i + 1, _j + 1
        _y = 10 ** (_y_top + _bracket_gap * (_k + 1))

        # horizontal bracket
        _ax.plot([_x1, _x1, _x2, _x2], [_y * 0.85, _y, _y, _y * 0.85],
                 lw=1.0, color="black")
        # stars + p-value
        _ax.text(
            (_x1 + _x2) / 2, _y * 1.02,
            f"{_sig}  (p={_p:.3f})",
            ha="center", va="bottom", fontsize=8
        )

    plt.draw()
    _ax.set_ylim(top=10 ** (_y_top + _bracket_gap * (_n_tests + 1.5)))

    for _i, _v in enumerate(_data, start=1):
        _ax.text(_i, _ax.get_ylim()[0] * 0.7, f"n={len(_v)}",
                 ha="center", va="top", fontsize=8, color="dimgrey")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(
    Path,
    combinations,
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    feature_wise_permutation_threshold,
    mticker,
    np,
    plt,
    wilcoxon,
):
    # ── 1. CP POD ─────────────────────────────────────────────────────────────────
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

    def _stars(_p):
        if _p < 0.001:  return "***"
        elif _p < 0.01: return "**"
        elif _p < 0.05: return "*"
        else:           return "ns"

    # ── 6. Plot ───────────────────────────────────────────────────────────────────
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
        _sig = _stars(_p)
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
        Path.home() / "Downloads" / "boxplot_cp_mtt_ldh.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    return


@app.cell
def _(
    Patch,
    Path,
    combinations,
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    feature_wise_permutation_threshold,
    list_botanicals,
    mticker,
    np,
    plt,
    wilcoxon,
):
    # ── 1. CP POD ─────────────────────────────────────────────────────────────────
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
    def _stars(_p):
        if _p < 0.001:  return "***"
        elif _p < 0.01: return "**"
        elif _p < 0.05: return "*"
        else:           return "ns"

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
    # ── 5. Plot ───────────────────────────────────────────────────────────────────
    _labels  = ["CP POD", "MT POD", "LDH POD"]
    _colors  = ["steelblue", "darkorange", "green"]

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
        _data_all = [_cp_all.values, _mtt_all.values, _ldh_all.values]

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
            _sig = _stars(_p)
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
    plt.savefig(Path.home() / "Downloads" / "boxplot_cp_mtt_ldh.png",
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
    _common_const_tox = (
        set(_mtt_passed_const["Metadata_compound"]) |
        set(_ldh_passed_const["Metadata_compound"])
    )
    _common_bot_tox = (
        set(_mtt_passed_bot["Metadata_compound"]) |
        set(_ldh_passed_bot["Metadata_compound"])
    )

    print("Common conts:", len(_common_const_tox))
    print("Common botanicals:", len(_common_bot_tox))
    plt.show() 
    return


@app.cell
def _(
    Patch,
    combinations,
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    feature_wise_permutation_threshold,
    list_botanicals,
    np,
    plt,
    wilcoxon,
):
    # ── 1. CP POD ─────────────────────────────────────────────────────────────────
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
    def _stars(_p):
        if _p < 0.001:  return "***"
        elif _p < 0.01: return "**"
        elif _p < 0.05: return "*"
        else:           return "ns"

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
    # ── 5. Plot ─────────────────────────────────
    _labels  = ["CP Active", "MT Active", "LDH Active"]
    _colors  = ["steelblue", "darkorange", "green"]
    _fig, _ax_top = plt.subplots(figsize=(7, 2))

    _x = np.arange(len(_labels)) + 1
    _width = 0.35

    _ax_top.bar(_x - _width/2, _totals_const, _width, color="lightgrey", alpha=0.6, zorder=1)
    _ax_top.bar(_x + _width/2, _totals_bot, _width, color="lightgrey", alpha=0.6, zorder=1)

    _rects1 = _ax_top.bar(_x - _width/2, _active_const, _width, label='Active Constituents', color=_colors, alpha=0.9, zorder=2)
    _rects2 = _ax_top.bar(_x + _width/2, _active_bot, _width, label='Active Botanicals', color=_colors, alpha=0.5, hatch='///', edgecolor="white", zorder=2)

    def _autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            _ax_top.annotate(f'{int(height)}',
                xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    _autolabel(_rects1)
    _autolabel(_rects2)

    _legend_elements = [
        Patch(facecolor="lightgrey", edgecolor="none", label="Total tested"),
        Patch(facecolor="steelblue", label="Active (Constituents)"),
        Patch(facecolor="steelblue", alpha=0.5, hatch="///", label="Active (Botanicals)")
    ]

    _ax_top.legend(handles=_legend_elements, frameon=False, fontsize=10, loc="upper right")
    _ax_top.set_ylabel("Active (N)", fontsize=14)
    #_ax_top.set_title("Activity Breakdown: Constituents vs. Botanicals", fontsize=12, pad=15)
    _ax_top.set_xticks(_x)
    _ax_top.set_xticklabels(_labels, fontsize=14)
    _ax_top.spines["top"].set_visible(False)
    _ax_top.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### POD Cell Painting  - Botanicals [Feature wise + FastBMD]""")
    return


@app.cell
def _(
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    dic_botanicals_scode_name,
    feature_wise_permutation_threshold,
    list_botanicals,
    np,
    pd,
    plt,
    ticker,
):
    # Filter to botanicals and rename
    _df_passed = df_bsc_cp_pod_all_featurewise[df_bsc_cp_pod_all_featurewise["all.pass"] == True]
    _df_passed = _df_passed[_df_passed["Metadata_compound"].isin(list_botanicals)].copy()
    _df_passed["Metadata_compound"] = _df_passed["Metadata_compound"].map(lambda x: dic_botanicals_scode_name.get(x, x))

    _df_all = df_bsc_cp_pod_all_featurewise[df_bsc_cp_pod_all_featurewise["Metadata_compound"].isin(list_botanicals)].copy()
    _df_all["Metadata_compound"] = _df_all["Metadata_compound"].map(lambda x: dic_botanicals_scode_name.get(x, x))

    _mtt_lookup = df_bsc_mtt_pod[
        (df_bsc_mtt_pod["all.pass"] == True) & (df_bsc_mtt_pod["Metadata_compound"].isin(list_botanicals))
    ].set_index("Metadata_compound")["Metadata_mtt_normalized_POD_um"]
    _mtt_lookup.index = _mtt_lookup.index.map(lambda x: dic_botanicals_scode_name.get(x, x))

    _ldh_lookup = df_bsc_ldh_pod[
        (df_bsc_ldh_pod["all.pass"] == True) & (df_bsc_ldh_pod["Metadata_compound"].isin(list_botanicals))
    ].set_index("Metadata_compound")["Metadata_ldh_normalized_POD_um"]
    _ldh_lookup.index = _ldh_lookup.index.map(lambda x: dic_botanicals_scode_name.get(x, x))

    _df_counts = _df_all.groupby("Metadata_compound").size().reset_index(name="n_rows")
    _compounds = _df_counts["Metadata_compound"].values
    _x_pos = np.arange(len(_compounds))

    # Compute PAC only on passed features
    _df_pac = _df_passed.groupby("Metadata_compound")["cp_POD_um"].quantile(0.05, interpolation="linear").reset_index(name="PAC")

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(_compounds) * 0.4)),
                                        sharey=True,
                                        gridspec_kw={"width_ratios": [4, 1]})
    _legend_mtt_added = False
    _legend_ldh_added = False

    for _i, _compound in enumerate(_compounds):
        _vals = _df_passed[_df_passed["Metadata_compound"] == _compound]["cp_POD_um"].values
        _pac_row = _df_pac[_df_pac["Metadata_compound"] == _compound]["PAC"].values

        if len(_vals) == 0:
            _ax1.scatter([0], [_i], alpha=0.4, color="grey", s=15, zorder=2, marker="x")
        else:
            _pac = _pac_row[0]
            _jitter = np.random.normal(_i, 0.07, size=len(_vals))
            _ax1.scatter(_vals, _jitter, alpha=0.4, color="steelblue", s=15, zorder=2)
            _ax1.boxplot(_vals, positions=[_i], widths=0.4, patch_artist=True, vert=False,
                        boxprops=dict(facecolor="lightblue", alpha=0.5),
                        flierprops=dict(marker=""), medianprops=dict(color="navy"),
                        zorder=3)
            _ax1.plot([_pac, _pac], [_i - 0.3, _i + 0.3], color="red", linewidth=1.5, zorder=4)

        # MTT line
        if _compound in _mtt_lookup.index:
            _mtt_val = _mtt_lookup[_compound]
            if pd.notna(_mtt_val):
                _label = "MTT POD" if not _legend_mtt_added else None
                _ax1.plot([_mtt_val, _mtt_val], [_i - 0.3, _i + 0.3], color="darkorange", linewidth=1.5, zorder=5, label=_label)
                _legend_mtt_added = True

        # LDH line
        if _compound in _ldh_lookup.index:
            _ldh_val = _ldh_lookup[_compound]
            if pd.notna(_ldh_val):
                _label = "LDH POD" if not _legend_ldh_added else None
                _ax1.plot([_ldh_val, _ldh_val], [_i - 0.3, _i + 0.3], color="green", linewidth=1.5, zorder=5, label=_label)
                _legend_ldh_added = True

    _ax1.set_yticks(_x_pos)
    _ax1.set_yticklabels(_compounds, fontsize=8)
    _ax1.set_xlabel("POD (μM)")
    _ax1.set_title("POD distribution per compound BMD (log)")
    _ax1.plot([], [], color="red", linewidth=1.5, label="PAC (5th percentile)")
    _ax1.legend(fontsize=8)
    _ax1.set_xscale("log")
    _ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    _ax1.tick_params(axis="x", rotation=45, labelsize=8)

    # Bar plot — count of passing features per compound (0 if none)
    _df_counts_passed = _df_passed.groupby("Metadata_compound").size().reset_index(name="n_rows")
    _df_counts = _df_counts.merge(_df_counts_passed.rename(columns={"n_rows": "n_passed"}), on="Metadata_compound", how="left").fillna({"n_passed": 0})
    _df_counts["n_passed"] = _df_counts["n_passed"].astype(int)

    _ax2.barh(_x_pos, _df_counts["n_passed"], color="steelblue", alpha=0.7, height=0.4)
    for _i, _v in enumerate(_df_counts["n_passed"]):
        _ax2.text(_v * 1.05, _i, str(_v), va="center", fontsize=7)
    _ax2.axvline(x=feature_wise_permutation_threshold, color="black", linewidth=1.5, label="Permutation threshold")
    _ax2.legend(fontsize=8)
    _ax2.set_xlabel("N features")
    _ax2.set_title("Count")
    _ax2.set_xscale("log")
    _ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    _ax2.tick_params(axis="x", rotation=45, labelsize=8)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""### POD Cell Painting  - Botanicals per Categories [Feature wise + FastBMD]""")
    return


@app.cell
def _(df_bsc_cp_pod_all_featurewise):
    df_bsc_cp_pod_all_featurewise
    return


@app.cell
def _(
    df_bsc_cp_pod_all_featurewise,
    dic_botanicals_scode_name,
    feature_wise_permutation_threshold,
    list_botanicals,
    mlines,
    np,
    pd,
    plt,
    ticker,
):
    # ── Categories and colors ─────────────────────────────────────────────────────
    _categories = ["DNA", "ER", "AGP", "RNA", "Mito", "Brightfield"]
    _cat_colors = {
        "DNA":        "#e41a1c",
        "ER":         "#377eb8",
        "AGP":        "#4daf4a",
        "RNA":        "#984ea3",
        "Mito":       "#ff7f00",
        "Brightfield":"#a65628",
    }

    # Feature columns: only cp_POD_um rows have a feature column
    # Extract category from feature name for each row
    def _get_category(feature_name):
        for _cat in _categories:
            if _cat in str(feature_name):
                return _cat
        return None

    # ── Filter to botanicals ──────────────────────────────────────────────────────
    _df_passed = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()
    _df_passed = _df_passed[_df_passed["Metadata_compound"].isin(list_botanicals)].copy()
    _df_passed["Metadata_compound"] = _df_passed["Metadata_compound"].map(
        lambda x: dic_botanicals_scode_name.get(x, x)
    )

    _df_all = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["Metadata_compound"].isin(list_botanicals)
    ].copy()
    _df_all["Metadata_compound"] = _df_all["Metadata_compound"].map(
        lambda x: dic_botanicals_scode_name.get(x, x)
    )

    # ── Assign category from feature column name ──────────────────────────────────
    # feature name is stored in a column e.g. "feature" — adjust if column name differs
    _df_passed["category"] = _df_passed["gene.id"].apply(_get_category)
    _df_all["category"] = _df_all["gene.id"].apply(_get_category)

    # ── Order compounds ───────────────────────────────────────────────────────────
    _df_counts = _df_all.groupby("Metadata_compound").size().reset_index(name="n_rows")
    _compounds = _df_counts["Metadata_compound"].values
    _x_pos = np.arange(len(_compounds))

    # ── PAC per compound (all features, for boxplot) ──────────────────────────────
    _df_pac_all = (
        _df_passed.groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reset_index(name="PAC")
    )

    # ── PAC per compound per category ────────────────────────────────────────────
    _df_pac_cat = (
        _df_passed[_df_passed["category"].notna()]
        .groupby(["Metadata_compound", "category"])["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reset_index(name="PAC_cat")
    )

    # ── Feature counts per compound per category (for stacked bar) ───────────────
    _df_cat_counts = (
        _df_passed[_df_passed["category"].notna()]
        .groupby(["Metadata_compound", "category"])
        .size()
        .reset_index(name="n")
    )
    # Pivot for stacking
    _df_cat_pivot = (
        _df_cat_counts.pivot(index="Metadata_compound", columns="category", values="n")
        .reindex(_compounds)
        .fillna(0)
    )
    # Ensure all categories present
    for _cat in _categories:
        if _cat not in _df_cat_pivot.columns:
            _df_cat_pivot[_cat] = 0
    _df_cat_pivot = _df_cat_pivot[_categories]

    # ── Plot ──────────────────────────────────────────────────────────────────────
    _fig, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(16, max(6, len(_compounds) * 0.4)),
        sharey=True,
        gridspec_kw={"width_ratios": [4, 1]}
    )

    # Boxplot + jittered dots per compound
    for _i, _compound in enumerate(_compounds):
        _vals = _df_passed[_df_passed["Metadata_compound"] == _compound]["cp_POD_um"].values
        _cats = _df_passed[_df_passed["Metadata_compound"] == _compound]["category"].values

        if len(_vals) == 0:
            _ax1.scatter([0], [_i], alpha=0.4, color="grey", s=15, zorder=2, marker="x")
        else:
            # Boxplot (all features, grey)
            _ax1.boxplot(
                _vals, positions=[_i], widths=0.35, patch_artist=True, vert=False,
                boxprops=dict(facecolor="lightgrey", alpha=0.4),
                flierprops=dict(marker=""),
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                zorder=2
            )
            # Jittered dots colored by category — vectorized
            _jitter = np.random.normal(0, 0.07, size=len(_vals))

            # Grey transparent layer for all points
            _ax1.scatter(
                _vals, _i + _jitter,
                alpha=0.15, color="grey", s=12, zorder=2
            )

        # PAC lines per category
        for _cat in _categories:
            _pac_row = _df_pac_cat[
                (_df_pac_cat["Metadata_compound"] == _compound) &
                (_df_pac_cat["category"] == _cat)
            ]["PAC_cat"].values
            if len(_pac_row) == 0 or pd.isna(_pac_row[0]):
                continue
            _ax1.plot(
                [_pac_row[0], _pac_row[0]], [_i - 0.3, _i + 0.3],
                color=_cat_colors[_cat], linewidth=1.5, zorder=4,
                label=f"{_cat} PAC" if _i == 0 else None
            )

    _ax1.set_yticks(_x_pos)
    _ax1.set_yticklabels(_compounds, fontsize=8)
    _ax1.set_xlabel("POD (μM)")
    _ax1.set_title("POD distribution per botanical — colored by channel")
    _ax1.set_xscale("log")
    _ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    _ax1.tick_params(axis="x", rotation=45, labelsize=8)

    # Legend for categories
    _handles = [
        mlines.Line2D([0], [0], color=_cat_colors[_cat], linewidth=2, label=f"{_cat} PAC")
        for _cat in _categories
    ] 
    _ax1.legend(handles=_handles, fontsize=7, loc="lower right", ncol=2)

    # ── ax2: stacked bar — n features per category ───────────────────────────────
    _left = np.zeros(len(_compounds))
    for _cat in _categories:
        _vals_bar = _df_cat_pivot[_cat].values
        _ax2.barh(_x_pos, _vals_bar, left=_left, color=_cat_colors[_cat],
                  height=0.5, alpha=0.85, label=_cat)
        _left += _vals_bar

    _ax2.axvline(x=feature_wise_permutation_threshold, color="black",
                 linewidth=1.5, linestyle="--", label="Threshold")
    for _i, _total in enumerate(_left):
        _ax2.text(_total + 0.5, _i, str(int(_total)), va="center", fontsize=7)

    _ax2.set_xlabel("N features")
    _ax2.set_title("Features\nper channel")
    _ax2.set_xscale("log")
    _ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    _ax2.tick_params(axis="x", rotation=45, labelsize=8)
    _ax2.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    _ax1.set_yticks(_x_pos)
    _ax1.set_yticklabels(_compounds, fontsize=8)
    plt.draw()
    _fig.subplots_adjust(wspace=0.05)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""### POD Cell Painting - Constituents [Feature wise + FastBMD]""")
    return


@app.cell
def _(df_bsc_annotations):
    df_bsc_annotations
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
    os,
    pd,
    plt,
    ticker,
):
    # Build constituent list filtered to those in df_bsc_cp_pod_all_featurewise
    _constituents_in_data = df_bsc_cp_pod_all_featurewise["Metadata_compound"].unique()
    _annot = df_bsc_annotations[df_bsc_annotations["Metadata_compound"].isin(_constituents_in_data)].copy()

    # Assign each constituent to its first plant following list_botanicals order
    _plant_order = [p for p in list_botanicals if p in _annot["Plant scode"].values]
    _plant_to_rank = {p: i for i, p in enumerate(_plant_order)}
    _annot["_plant_rank"] = _annot["Plant scode"].map(_plant_to_rank)

    # Primary plant = lowest rank in list_botanicals
    _primary_plant = (
        _annot.sort_values("_plant_rank")
        .drop_duplicates(subset="Metadata_compound", keep="first")
        [["Metadata_compound", "Plant scode", "Plant name", "Constituent", "_plant_rank"]]
        .rename(columns={"Plant scode": "primary_plant_scode", "Plant name": "primary_plant_name"})
    )

    # Sort constituents: by primary plant rank, then constituent name
    _primary_plant = _primary_plant.sort_values(["_plant_rank", "Constituent"])
    _compounds = _primary_plant["Metadata_compound"].values
    _constituent_labels = _primary_plant["Constituent"].values
    #_x_pos = np.arange(len(_compounds))
    spacing = 0.6   # try 0.5–0.7
    _x_pos = np.arange(len(_compounds)) * spacing

    # All plants per constituent (for colored markers)
    _constituent_to_plants = (
        _annot.groupby("Metadata_compound")["Plant name"].apply(list).to_dict()
    )

    # Color palette per plant
    _unique_plants = [dic_botanicals_scode_name[p] for p in _plant_order]
    _plant_colors = dict(zip(_unique_plants, plt.cm.tab20.colors[:len(_unique_plants)]))

    # Filter passed features to constituents only
    _df_passed = df_bsc_cp_pod_all_featurewise[
        (df_bsc_cp_pod_all_featurewise["all.pass"] == True) &
        (df_bsc_cp_pod_all_featurewise["Metadata_compound"].isin(_compounds))
    ].copy()

    _df_all = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["Metadata_compound"].isin(_compounds)
    ].copy()

    _mtt_lookup = df_bsc_mtt_pod[
        (df_bsc_mtt_pod["all.pass"] == True) & (df_bsc_mtt_pod["Metadata_compound"].isin(_compounds))
    ].set_index("Metadata_compound")["Metadata_mtt_normalized_POD_um"]

    _ldh_lookup = df_bsc_ldh_pod[
        (df_bsc_ldh_pod["all.pass"] == True) & (df_bsc_ldh_pod["Metadata_compound"].isin(_compounds))
    ].set_index("Metadata_compound")["Metadata_ldh_normalized_POD_um"]

    _df_pac = _df_passed.groupby("Metadata_compound")["cp_POD_um"].quantile(0.05, interpolation="linear").reset_index(name="PAC")
    _df_counts = _df_all.groupby("Metadata_compound").size().reset_index(name="n_rows")
    _df_counts_passed = _df_passed.groupby("Metadata_compound").size().reset_index(name="n_rows")
    _df_counts = _df_counts.merge(_df_counts_passed.rename(columns={"n_rows": "n_passed"}), on="Metadata_compound", how="left").fillna({"n_passed": 0})
    _df_counts["n_passed"] = _df_counts["n_passed"].astype(int)
    # Reorder to match _compounds
    _df_counts = _primary_plant[["Metadata_compound"]].merge(_df_counts, on="Metadata_compound", how="left").fillna({"n_passed": 0})
    _df_counts["n_passed"] = _df_counts["n_passed"].astype(int)

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(16, max(6, len(_compounds) * 0.20)),
                                              sharey=True,
                                              gridspec_kw={"width_ratios": [3, 1, 0.2]})

    _legend_mtt_added = False
    _legend_ldh_added = False
    for _i, _compound in enumerate(_compounds):
        _y = _x_pos[_i]
        _vals = _df_passed[_df_passed["Metadata_compound"] == _compound]["cp_POD_um"].values
        _pac_row = _df_pac[_df_pac["Metadata_compound"] == _compound]["PAC"].values
        if len(_vals) == 0:
            _ax1.scatter([0], [_y], alpha=0.4, color="grey", s=8, zorder=2, marker="x")
        else:
            _pac = _pac_row[0]
            _jitter = np.random.normal(_y, 0.04, size=len(_vals))
            _ax1.scatter(_vals, _jitter, alpha=0.3, color="steelblue", s=4, zorder=2)
            _ax1.boxplot(_vals, positions=[_y], widths=0.15, patch_artist=True, vert=False,
                        boxprops=dict(facecolor="lightblue", alpha=0.5),
                        flierprops=dict(marker=""), medianprops=dict(color="navy", linewidth=0.8),
                        whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6),
                        zorder=3)
            _ax1.plot([_pac, _pac], [_y - 0.15, _y + 0.15], color="red", linewidth=1.0, zorder=4)
        if _compound in _mtt_lookup.index:
            _mtt_val = _mtt_lookup[_compound]
            if pd.notna(_mtt_val):
                _label = "MTT POD" if not _legend_mtt_added else None
                _ax1.plot([_mtt_val, _mtt_val], [_y - 0.15, _y + 0.15], color="darkorange", linewidth=1.0, zorder=5, label=_label)
                _legend_mtt_added = True
        if _compound in _ldh_lookup.index:
            _ldh_val = _ldh_lookup[_compound]
            if pd.notna(_ldh_val):
                _label = "LDH POD" if not _legend_ldh_added else None
                _ax1.plot([_ldh_val, _ldh_val], [_y - 0.15, _y + 0.15], color="green", linewidth=1.0, zorder=5, label=_label)
                _legend_ldh_added = True
        _plants_for_compound = _constituent_to_plants.get(_compound, [])
        for _j, _plant_name in enumerate(_plants_for_compound):
            _color = _plant_colors.get(_plant_name, "grey")
            _ax3.scatter([_j], [_y], color=_color, s=20, zorder=3, marker="s")
    # Ensure labels align with _compounds order
    _label_map = dict(zip(_primary_plant["Metadata_compound"], _primary_plant["Constituent"]))
    _constituent_labels = [_label_map[c] for c in _compounds]

    _ax1.set_yticks(_x_pos)
    _ax1.set_yticklabels(_constituent_labels, fontsize=9)
    _ax1.set_xlabel("POD (μM)")
    _ax1.set_title("POD distribution per constituent (log)")
    _ax1.plot([], [], color="red", linewidth=1.5, label="PAC (5th percentile)")
    _ax1.legend(fontsize=8)
    _ax1.set_xscale("log")
    _ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    _ax1.tick_params(axis="x", rotation=45, labelsize=8)

    # Bar plot
    _ax2.barh(_x_pos, _df_counts["n_passed"], color="steelblue", alpha=0.7, height=0.4)
    for _i, _v in enumerate(_df_counts["n_passed"]):
        _y = _x_pos[_i]
        _ax2.text(_v * 1.05, _y, str(_v), va="center", fontsize=7)
    _ax2.axvline(x=feature_wise_permutation_threshold, color="black", linewidth=1.5, label="Permutation threshold")
    _ax2.legend(fontsize=8)
    _ax2.set_xlabel("N features")
    _ax2.set_title("Count")
    _ax2.set_xscale("log")
    _ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    _ax2.tick_params(axis="x", rotation=45, labelsize=8)

    # Plant marker legend panel
    _ax3.set_xlim(-0.5, 4)
    _ax3.set_yticks(_x_pos)
    _ax3.set_yticklabels([])
    _ax3.set_xticks([])
    _ax3.set_title("Plant")
    for _plant_name, _color in _plant_colors.items():
        _ax3.scatter([], [], color=_color, s=40, marker="s", label=_plant_name)
    _ax3.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.2, 1), frameon=True)
    _ax3.axis("off")

    _ax1.set_yticks(_x_pos)
    _ax1.set_yticklabels(_constituent_labels, fontsize=9)
    plt.draw()  # force render before save
    _fig.subplots_adjust(wspace=0.02)
    _fig.savefig(os.path.expanduser("~/Downloads/constituent_POD_distribution.pdf"))
    return


@app.cell
def _(mo):
    mo.md(r"""### POD Cell Painting  - Constituents per categories [Feature wise + FastBMD]""")
    return


@app.cell
def _(
    df_bsc_annotations,
    df_bsc_cp_pod_all_featurewise,
    dic_botanicals_scode_name,
    feature_wise_permutation_threshold,
    list_botanicals,
    np,
    os,
    pd,
    plt,
    ticker,
):
    # ── Categories and colors ─────────────────────────────────────────────────────
    _categories = ["DNA", "ER", "AGP", "RNA", "Mito", "Brightfield"]
    _cat_colors = {
        "DNA":        "#e41a1c",
        "ER":         "#377eb8",
        "AGP":        "#4daf4a",
        "RNA":        "#984ea3",
        "Mito":       "#ff7f00",
        "Brightfield":"#a65628",
    }

    # Feature columns: only cp_POD_um rows have a feature column
    # Extract category from feature name for each row
    def _get_category(feature_name):
        for _cat in _categories:
            if _cat in str(feature_name):
                return _cat
        return None
    # Build constituent list filtered to those in df_bsc_cp_pod_all_featurewise
    _constituents_in_data = df_bsc_cp_pod_all_featurewise["Metadata_compound"].unique()
    _annot = df_bsc_annotations[df_bsc_annotations["Metadata_compound"].isin(_constituents_in_data)].copy()

    # Assign each constituent to its first plant following list_botanicals order
    _plant_order = [p for p in list_botanicals if p in _annot["Plant scode"].values]
    _plant_to_rank = {p: i for i, p in enumerate(_plant_order)}
    _annot["_plant_rank"] = _annot["Plant scode"].map(_plant_to_rank)

    # Primary plant = lowest rank in list_botanicals
    _primary_plant = (
        _annot.sort_values("_plant_rank")
        .drop_duplicates(subset="Metadata_compound", keep="first")
        [["Metadata_compound", "Plant scode", "Plant name", "Constituent", "_plant_rank"]]
        .rename(columns={"Plant scode": "primary_plant_scode", "Plant name": "primary_plant_name"})
    )

    # Sort constituents: by primary plant rank, then constituent name
    _primary_plant = _primary_plant.sort_values(["_plant_rank", "Constituent"])
    _compounds = _primary_plant["Metadata_compound"].values
    _constituent_labels = _primary_plant["Constituent"].values
    #_x_pos = np.arange(len(_compounds))
    _spacing = 0.6   # try 0.5–0.7
    _x_pos = np.arange(len(_compounds)) * _spacing

    # All plants per constituent (for colored markers)
    _constituent_to_plants = (
        _annot.groupby("Metadata_compound")["Plant name"].apply(list).to_dict()
    )

    # Color palette per plant
    _unique_plants = [dic_botanicals_scode_name[p] for p in _plant_order]
    _plant_colors = dict(zip(_unique_plants, plt.cm.tab20.colors[:len(_unique_plants)]))

    # Filter passed features to constituents only
    _df_passed = df_bsc_cp_pod_all_featurewise[
        (df_bsc_cp_pod_all_featurewise["all.pass"] == True) &
        (df_bsc_cp_pod_all_featurewise["Metadata_compound"].isin(_compounds))
    ].copy()

    _df_all = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["Metadata_compound"].isin(_compounds)
    ].copy()
    # ── Assign category ───────────────────────────────────────────────────────────
    _df_passed["category"] = _df_passed["gene.id"].apply(_get_category)
    _df_all["category"] = _df_all["gene.id"].apply(_get_category)

    # ── PAC per constituent per category ─────────────────────────────────────────
    _df_pac_cat = (
        _df_passed[_df_passed["category"].notna()]
        .groupby(["Metadata_compound", "category"])["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reset_index(name="PAC_cat")
    )

    # ── Feature counts per constituent per category ───────────────────────────────
    _df_cat_counts = (
        _df_passed[_df_passed["category"].notna()]
        .groupby(["Metadata_compound", "category"])
        .size()
        .reset_index(name="n")
    )
    _df_cat_pivot = (
        _df_cat_counts.pivot(index="Metadata_compound", columns="category", values="n")
        .fillna(0)
    )
    for _cat in _categories:
        if _cat not in _df_cat_pivot.columns:
            _df_cat_pivot[_cat] = 0
    _df_cat_pivot = _df_cat_pivot[_categories]
    # Reorder to match _compounds
    _df_cat_pivot = (
        _primary_plant[["Metadata_compound"]]
        .merge(_df_cat_pivot.reset_index(), on="Metadata_compound", how="left")
        .fillna(0)
        .set_index("Metadata_compound")[_categories]
    )

    # ── Figure ────────────────────────────────────────────────────────────────────
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(
        1, 3, figsize=(16, max(6, len(_compounds) * 0.20)),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1, 0.2]}
    )

    _legend_cat_added = set()

    for _i, _compound in enumerate(_compounds):
        _y = _x_pos[_i]
        _vals = _df_passed[_df_passed["Metadata_compound"] == _compound]["cp_POD_um"].values
        _cats = _df_passed[_df_passed["Metadata_compound"] == _compound]["category"].values

        if len(_vals) == 0:
            _ax1.scatter([0], [_y], alpha=0.4, color="grey", s=8, zorder=2, marker="x")
        else:
            # Boxplot (grey outline)
            _ax1.boxplot(
                _vals, positions=[_y], widths=0.15, patch_artist=True, vert=False,
                boxprops=dict(facecolor="lightgrey", alpha=0.3),
                flierprops=dict(marker=""),
                medianprops=dict(color="black", linewidth=0.8),
                whiskerprops=dict(linewidth=0.6),
                capprops=dict(linewidth=0.6),
                zorder=2
            )
            # Grey transparent dots underneath
            _jitter = np.random.normal(0, 0.04, size=len(_vals))
            _ax1.scatter(_vals, _y + _jitter, alpha=0.15, color="grey", s=4, zorder=2)

            # Colored dots by category
            _cats_arr = np.array(_cats)
            for _cat in _categories:
                _mask = _cats_arr == _cat
                if _mask.sum() == 0:
                    continue
                _ax1.scatter(
                    _vals[_mask], _y + _jitter[_mask],
                    alpha=0.5, color=_cat_colors[_cat], s=4, zorder=3
                )

            # PAC lines per category
            for _cat in _categories:
                _pac_row = _df_pac_cat[
                    (_df_pac_cat["Metadata_compound"] == _compound) &
                    (_df_pac_cat["category"] == _cat)
                ]["PAC_cat"].values
                if len(_pac_row) == 0 or pd.isna(_pac_row[0]):
                    continue
                _label = _cat if _cat not in _legend_cat_added else None
                _ax1.plot(
                    [_pac_row[0], _pac_row[0]], [_y - 0.12, _y + 0.12],
                    color=_cat_colors[_cat], linewidth=1.0, zorder=4, label=_label
                )
                _legend_cat_added.add(_cat)

        # Plant markers ax3
        for _j, _plant_name in enumerate(_constituent_to_plants.get(_compound, [])):
            _color = _plant_colors.get(_plant_name, "grey")
            _ax3.scatter([_j], [_y], color=_color, s=20, zorder=3, marker="s")

    # ── ax1 formatting ────────────────────────────────────────────────────────────
    _label_map = dict(zip(_primary_plant["Metadata_compound"], _primary_plant["Constituent"]))
    _constituent_labels = [_label_map[c] for c in _compounds]
    _ax1.set_yticks(_x_pos)
    _ax1.set_yticklabels(_constituent_labels, fontsize=9)
    _ax1.set_xlabel("POD (μM)")
    _ax1.set_title("POD distribution per constituent — colored by channel")
    _ax1.legend(fontsize=7, loc="lower right", ncol=2)
    _ax1.set_xscale("log")
    _ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    _ax1.tick_params(axis="x", rotation=45, labelsize=8)

    # ── ax2: stacked bar per category ────────────────────────────────────────────
    _left = np.zeros(len(_compounds))
    for _cat in _categories:
        _vals_bar = _df_cat_pivot[_cat].values
        _ax2.barh(_x_pos, _vals_bar, left=_left,
                  color=_cat_colors[_cat], height=0.3, alpha=0.85, label=_cat)
        _left += _vals_bar
    _ax2.axvline(x=feature_wise_permutation_threshold, color="black",
                 linewidth=1.2, linestyle="--", label="Threshold")
    for _i, _total in enumerate(_left):
        _y = _x_pos[_i]
        _ax2.text(_total * 1.05, _y, str(int(_total)), va="center", fontsize=6)
    _ax2.legend(fontsize=6, loc="lower right")
    _ax2.set_xlabel("N features")
    _ax2.set_title("Count")
    _ax2.set_xscale("log")
    _ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    _ax2.tick_params(axis="x", rotation=45, labelsize=8)

    # ── ax3: plant marker panel ───────────────────────────────────────────────────
    _ax3.set_xlim(-0.5, 4)
    _ax3.set_yticks(_x_pos)
    _ax3.set_yticklabels([])
    _ax3.set_xticks([])
    _ax3.set_title("Plant")
    for _plant_name, _color in _plant_colors.items():
        _ax3.scatter([], [], color=_color, s=40, marker="s", label=_plant_name)
    _ax3.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.2, 1), frameon=True)
    _ax3.axis("off")

    _ax1.set_yticks(_x_pos)
    _ax1.set_yticklabels(_constituent_labels, fontsize=9)
    plt.draw()
    _fig.subplots_adjust(wspace=0.02)
    _fig.savefig(os.path.expanduser("~/Downloads/constituent_POD_distribution.pdf"))
    return


@app.cell
def _(mo):
    mo.md(r"""### POD Cell Painting - Botanicals VS Constituents [Feature wise + FastBMD]""")
    return


@app.cell
def _(df_bsc_annotations):
    df_bsc_annotations[df_bsc_annotations["Plant name"] == "Asian ginseng"]
    return


@app.cell
def _(
    Line2D,
    df_bsc_annotations,
    df_bsc_cp_pod_all_featurewise,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    dic_botanicals_scode_name,
    feature_wise_permutation_threshold,
    list_botanicals,
    mpatches,
    mticker,
    np,
    pd,
    plt,
):
    # ── 1. PAC per constituent ────────────────────────────────────────────────────
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
    len(_pac_lookup)
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

    # ── 5. Colors ─────────────────────────────────────────────────────────────────
    _cmap = plt.cm.tab20
    _plant_colors = {plant: _cmap(_i % _cmap.N) for _i, plant in enumerate(_order)}
    # ── Custom botanical colors ───────────────────────────────────────────────────
    _plant_colors = {
        "Kratom":"#e6194B",
        "Ashwagandha":"#3cb44b",
        "Comfrey":"#ffe119",
        "Usnea":"#4363d8",
        "Asian ginseng":"#f58231",
        "Blue cohosh":"#42d4f4",
        "Yohimbe":"#f032e6",
        "Kava":"#fabed4",
        "Milk thistle":"#469990",
        "Goldenseal":"#dcbeff",
        "Aristolochia":"#9A6324",
        "Green tea extract":"#fffac8",
        "Ephedra":"#800000",
    }
    # ── 6. Plot ───────────────────────────────────────────────────────────────────
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
    # Total known constituents per botanical (from annotations, regardless of data)
    _total_known = (
        df_bsc_annotations[df_bsc_annotations["Plant scode"].isin(list_botanicals)]
        .groupby("Plant scode")["Metadata_compound"]
        .nunique()
        .to_dict()
    )

    # Active = has a PAC (all.pass == True AND present in _pac_lookup)
    _active_counts = (
        _df_long.groupby("Plant scode")["Metadata_compound"]
        .nunique()
        .to_dict()
    )

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
    mo.md(r"""## POD feature wise cytotoxicity""")
    return


@app.cell
def _(
    df_bsc_annotations,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    dic_botanicals_scode_name,
    list_botanicals,
    np,
    pd,
    plt,
    ticker,
):
    # ── Build constituent MTT/LDH toxicity lookup ─────────────────────────────────
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
    # ── Build rows: one per (botanical, constituent, endpoint) ────────────────────
    _rows = []
    for _scode in list_botanicals:
        _plant_name = dic_botanicals_scode_name.get(_scode, _scode)

        # Botanical POD
        _bot_mtt = _mtt_pac_dict.get(_scode)
        _bot_ldh = _ldh_pac_dict.get(_scode)

        # Constituents of this botanical
        _constituents = df_bsc_annotations[
            df_bsc_annotations["Plant scode"] == _scode
        ]["Metadata_compound"].unique()

        for _cmpd in _constituents:
            _cname = df_bsc_annotations[
                df_bsc_annotations["Metadata_compound"] == _cmpd
            ]["Constituent"].iloc[0]
            _rows.append({
                "botanical":       _plant_name,
                "botanical_scode": _scode,
                "constituent":     _cname,
                "constituent_scode": _cmpd,
                "bot_mtt":         _bot_mtt,
                "bot_ldh":         _bot_ldh,
                "const_mtt":       _const_mtt_lookup.get(_cmpd),
                "const_ldh":       _const_ldh_lookup.get(_cmpd),
            })

    _df_plot = pd.DataFrame(_rows)

    # ── Keep only botanicals that are toxic in at least one endpoint ──────────────
    _toxic_botanicals = _df_plot[
        _df_plot["bot_mtt"].notna() | _df_plot["bot_ldh"].notna()
    ]["botanical"].unique()
    _df_plot = _df_plot[_df_plot["botanical"].isin(_toxic_botanicals)].copy()

    # Order botanicals by MTT POD (then LDH)
    _bot_order = (
        _df_plot.drop_duplicates("botanical")
        .sort_values(["bot_mtt", "bot_ldh"])
        ["botanical"].tolist()
    )

    # ── Color: constituent toxic (POD ≤ botanical POD) vs not ─────────────────────
    def _dot_color(const_val, bot_val):
        if pd.isna(const_val):
            return "lightgrey"   # no toxicity data
        if pd.isna(bot_val):
            return "steelblue"   # constituent toxic but botanical not tested
        return "crimson" if const_val <= bot_val else "steelblue"

    _df_plot["color_mtt"] = _df_plot.apply(lambda r: _dot_color(r["const_mtt"], r["bot_mtt"]), axis=1)
    _df_plot["color_ldh"] = _df_plot.apply(lambda r: _dot_color(r["const_ldh"], r["bot_ldh"]), axis=1)

    # ── Plot ──────────────────────────────────────────────────────────────────────
    _n_bot = len(_bot_order)
    _fig, (_ax_mtt, _ax_ldh) = plt.subplots(
        1, 2,
        figsize=(16, max(5, _n_bot * 0.6)),
        sharey=True
    )

    for _ax, _endpoint, _bot_col, _const_col, _color_col in [
        (_ax_mtt, "MTT", "bot_mtt", "const_mtt", "color_mtt"),
        (_ax_ldh, "LDH", "bot_ldh", "const_ldh", "color_ldh"),
    ]:
        for _i, _plant in enumerate(_bot_order):
            _sub = _df_plot[_df_plot["botanical"] == _plant]
            _bot_val = _sub[_bot_col].iloc[0]

            # Botanical reference line
            if pd.notna(_bot_val):
                _ax.axvline(_bot_val, ymin=(_i) / _n_bot, ymax=(_i + 1) / _n_bot,
                            color="black", linewidth=1.2, linestyle="--", zorder=2)
                _ax.plot([_bot_val], [_i], marker="D", color="black",
                         markersize=6, zorder=4, label="Botanical POD" if _i == 0 else None)

            # Constituent dots
            for _, _row in _sub.iterrows():
                _val = _row[_const_col]
                if pd.isna(_val):
                    continue
                _jitter = np.random.uniform(-0.25, 0.25)
                _ax.scatter([_val], [_i + _jitter],
                            color=_row[_color_col], s=30, alpha=0.8, zorder=3,
                            edgecolors="white", linewidths=0.3)

        _ax.set_yticks(range(_n_bot))
        _ax.set_yticklabels(_bot_order, fontsize=9)
        _ax.set_xscale("log")
        _ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        _ax.tick_params(axis="x", rotation=45, labelsize=8)
        _ax.set_xlabel("POD (μM, log scale)", fontsize=9)
        _ax.set_title(f"{_endpoint} — Botanical POD vs Constituent POD", fontsize=10)
        _ax.set_ylim(-0.5, _n_bot - 0.5)

        # Legend
        from matplotlib.lines import Line2D
        _handles = [
            Line2D([0], [0], marker="D", color="black", linestyle="--", markersize=6, label="Botanical POD"),
            Line2D([0], [0], marker="o", color="crimson",   linestyle="", markersize=6, label="Constituent ≤ botanical (more/equally toxic)"),
            Line2D([0], [0], marker="o", color="steelblue", linestyle="", markersize=6, label="Constituent > botanical (less toxic)"),
            Line2D([0], [0], marker="o", color="lightgrey", linestyle="", markersize=6, label="No constituent data"),
        ]
        _ax.legend(handles=_handles, fontsize=7, loc="lower right")

    plt.tight_layout()
    _fig.subplots_adjust(wspace=0.05)
    _fig
    return (Line2D,)


@app.cell
def _(
    Line2D,
    Path,
    df_bsc_annotations,
    df_bsc_ldh_pod,
    df_bsc_mtt_pod,
    df_level5_bsc,
    dic_botanicals_scode_name,
    list_botanicals,
    mticker,
    np,
    pd,
    plt,
):
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

    _plant_bg_colors = {
        "Kratom":"#e6194B",
        "Ashwagandha":"#3cb44b",
        "Comfrey":"#ffe119",
        "Usnea":"#4363d8",
        "Asian ginseng":"#f58231",
        "Blue cohosh":"#42d4f4",
        "Yohimbe":"#f032e6",
        "Kava":"#fabed4",
        "Milk thistle":"#469990",
        "Goldenseal":"#dcbeff",
        "Aristolochia":"#9A6324",
        "Green tea extract":"#fffac8",
        "Ephedra":"#800000",
    }

    # fallback if new plant appears
    _default_color = "#cccccc"
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
    # ── Plot ──────────────────────────────────────────────────────────────────────
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

    plt.savefig(Path.home() / "Downloads" / "botanical_constituent_plot.png",
                dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Number Categories feature wise comparison""")
    return


@app.cell
def _(
    combinations,
    df_bsc_annotations,
    df_bsc_cp_pod_all_featurewise,
    dic_botanicals_scode_name,
    feature_wise_permutation_threshold,
    kruskal,
    np,
    pd,
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

    # ── Link compounds to botanical / constituent via df_bsc_annotations ──────────
    _annot = (
        df_bsc_annotations[["Metadata_compound", "Constituent", "Plant scode"]]
        .drop_duplicates()
        .copy()
    )
    _annot["Botanical"] = _annot["Plant scode"].map(dic_botanicals_scode_name)

    # ── PAC per compound per category: 5th percentile of cp_POD_um ───────────────
    _df_pac = (
        _df_feat.groupby(["Metadata_compound", "category"])["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
        .reset_index(name="PAC")
    )

    # ── Pivot: rows = compounds, columns = categories ─────────────────────────────
    _df_pivot = _df_pac.pivot(
        index="Metadata_compound", columns="category", values="PAC"
    ).reindex(columns=_categories)

    # ── ANOVA per compound: Kruskal-Wallis across categories ─────────────────────
    # (each compound has one PAC per category → use the feature-level values)
    # ── Define helper first ───────────────────────────────────────────────────────
    def _stars(_p):
        if _p < 0.001:  return "***"
        elif _p < 0.01: return "**"
        elif _p < 0.05: return "*"
        else:           return "ns"
    _anova_rows = []

    for _cpd in _valid_compounds:
        _sub = _df_feat[_df_feat["Metadata_compound"] == _cpd]

        # Per-category raw POD values for KW test
        _groups = []
        _cat_pacs = {}
        for _cat in _categories:
            _vals = _sub.loc[_sub["category"] == _cat, "cp_POD_um"].values
            if len(_vals) >= 2:
                _groups.append(_vals)
            _cat_pacs[_cat] = np.quantile(_vals, 0.05) if len(_vals) >= 1 else np.nan

        # Kruskal-Wallis (need at least 2 groups with >=2 values)
        if len(_groups) >= 2:
            _kw_stat, _kw_p = kruskal(*_groups)
        else:
            _kw_stat, _kw_p = np.nan, np.nan

        # Pairwise Wilcoxon on raw feature values within compound
        _pair_results = {}
        _cats_available = [
            _cat for _cat in _categories
            if len(_sub.loc[_sub["category"] == _cat, "cp_POD_um"].values) >= 2
        ]
        _cpairs = list(combinations(_cats_available, 2))
        _n_tests = len(_cpairs)

        for _cat_a, _cat_b in _cpairs:
            _a = _sub.loc[_sub["category"] == _cat_a, "cp_POD_um"].values
            _b = _sub.loc[_sub["category"] == _cat_b, "cp_POD_um"].values
            # Trim to same length for Wilcoxon (use min length, sorted)
            _min_n = min(len(_a), len(_b))
            try:
                _, _p = wilcoxon(
                    np.sort(_a)[:_min_n],
                    np.sort(_b)[:_min_n],
                    alternative="two-sided"
                )
                _p_corr = min(_p * _n_tests, 1.0)
            except Exception:
                _p_corr = np.nan
            _pair_results[f"{_cat_a} vs {_cat_b}"] = _p_corr

        # Lookup botanical / constituent
        _annot_row = _annot[_annot["Metadata_compound"] == _cpd]
        _constituent = _annot_row["Constituent"].iloc[0]   if len(_annot_row) else _cpd
        _botanical   = _annot_row["Botanical"].iloc[0]     if len(_annot_row) else ""
        _plant_scode = _annot_row["Plant scode"].iloc[0]   if len(_annot_row) else ""

        _row = {
            "Botanical":        _botanical,
            "Plant scode":      _plant_scode,
            "Constituent":      _constituent,
            "Metadata_compound": _cpd,
            "KW_stat":          round(_kw_stat, 3) if not np.isnan(_kw_stat) else np.nan,
            "KW_p":             _kw_p,
            "KW_sig":           _stars(_kw_p) if not np.isnan(_kw_p) else "n/a",
        }
        # Add PAC per category
        for _cat in _categories:
            _row[f"PAC_{_cat}"] = round(_cat_pacs[_cat], 3) if not np.isnan(_cat_pacs.get(_cat, np.nan)) else np.nan
        # Add pairwise p-values
        for _pair_key, _pv in _pair_results.items():
            _row[f"p_{_pair_key}"] = round(_pv, 4) if not np.isnan(_pv) else np.nan
            _row[f"sig_{_pair_key}"] = _stars(_pv) if not np.isnan(_pv) else "n/a"

        _anova_rows.append(_row)

    _df_anova = (
        pd.DataFrame(_anova_rows)
        .sort_values(["Botanical", "Constituent"])
        .reset_index(drop=True)
    )

    # ── Display styled table ──────────────────────────────────────────────────────
    _pac_cols  = [f"PAC_{_cat}" for _cat in _categories]
    _sig_cols  = [c for c in _df_anova.columns if c.startswith("sig_")]
    _p_cols    = [c for c in _df_anova.columns if c.startswith("p_")]
    _id_cols   = ["Botanical", "Constituent", "KW_stat", "KW_p", "KW_sig"]
    _show_cols = _id_cols + _pac_cols + _sig_cols

    _df_display = _df_anova[_show_cols].copy()

    def _color_sig(_val):
        _cmap = {"***": "background-color: #d73027; color: white; font-weight: bold",
                 "**":  "background-color: #fc8d59; font-weight: bold",
                 "*":   "background-color: #fee090",
                 "ns":  "background-color: #f0f0f0; color: #999",
                 "n/a": "color: #cccccc"}
        return _cmap.get(str(_val), "")

    def _color_pac(_val):
        if pd.isna(_val): return "color: #cccccc"
        return ""

    def _color_kw(_val):
        try:
            _p = float(_val)
            if _p < 0.001:  return "background-color: #d73027; color: white; font-weight: bold"
            elif _p < 0.01: return "background-color: #fc8d59; font-weight: bold"
            elif _p < 0.05: return "background-color: #fee090"
            else:           return "background-color: #f0f0f0; color: #999"
        except:
            return ""

    _styled = (
        _df_display.style
        .format({
            **{_c: lambda x: f"{x:.2f}" if not pd.isna(x) else "—" for _c in _pac_cols},
            "KW_stat": lambda x: f"{x:.2f}" if not pd.isna(x) else "—",
            "KW_p":    lambda x: f"{x:.2e}" if not pd.isna(x) else "—",
        })
        .applymap(_color_sig, subset=_sig_cols + ["KW_sig"])
        .applymap(_color_pac, subset=_pac_cols)
        .applymap(_color_kw,  subset=["KW_p"])
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#2c3e50"), ("color", "white"),
                ("font-weight", "bold"), ("font-size", "11px"),
                ("text-align", "center"), ("padding", "6px")
            ]},
            {"selector": "td", "props": [
                ("font-size", "10px"), ("padding", "4px 8px"), ("text-align", "center")
            ]},
            {"selector": "tr:nth-child(even)", "props": [("background-color", "#f7f7f7")]},
            {"selector": "tr:hover", "props": [("background-color", "#eaf4fb")]},
        ])
        .set_caption(
            f"ANOVA-style results: Kruskal-Wallis + pairwise Wilcoxon per compound "
            f"(Bonferroni corrected) — {len(_df_anova)} compounds"
        )
    )

    _styled
    return


@app.cell
def _():
    import math
    return (math,)


@app.cell
def _(
    df_bsc_annotations,
    df_bsc_cp_pod_all_featurewise,
    dic_botanicals_scode_name,
    list_botanicals,
    math,
    np,
    os,
    plt,
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

    # ── Filter: all.pass == True ──────────────────────────────────────────────────
    _df_feat = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()
    _df_feat["category"] = _df_feat["gene.id"].apply(_get_category)
    _df_feat = _df_feat[_df_feat["category"].notna()].copy()

    # ── Feature count per compound per category ───────────────────────────────────
    _df_feat_count = (
        _df_feat.groupby(["Metadata_compound", "category"])
        .size()
        .reset_index(name="n_features")
    )
    _df_feat_pivot = (
        _df_feat_count.pivot(
            index="Metadata_compound", columns="category", values="n_features"
        )
        .reindex(columns=_categories)
        .fillna(0)
        .astype(int)
    )

    # ── Build ordered row index: botanicals + constituents ────────────────────────
    _rows_ordered = []
    for _scode in list_botanicals:
        _plant_name   = dic_botanicals_scode_name.get(_scode, _scode)
        _const_scodes = df_bsc_annotations[
            df_bsc_annotations["Plant scode"] == _scode
        ][["Metadata_compound", "Constituent"]].drop_duplicates()

        _const_rows = []
        for _, _crow in _const_scodes.iterrows():
            if _crow["Metadata_compound"] in _df_feat_pivot.index:
                _const_rows.append((_crow["Metadata_compound"], f"  {_crow['Constituent']}"))

        _has_botanical = _scode in _df_feat_pivot.index
        if not _has_botanical and len(_const_rows) == 0:
            continue

        if _has_botanical:
            _rows_ordered.append((_scode, _plant_name, True))
        for _s, _lbl in _const_rows:
            _rows_ordered.append((_s, _lbl, False))

    _ordered_scodes = [r[0] for r in _rows_ordered]
    _ordered_labels = [r[1] for r in _rows_ordered]
    _is_botanical   = [r[2] for r in _rows_ordered]

    # ── Build matrix row by row (avoids duplicate index issue) ────────────────────
    _mat_ordered = np.array([
        _df_feat_pivot.loc[_s].values if _s in _df_feat_pivot.index
        else np.zeros(len(_categories), dtype=int)
        for _s in _ordered_scodes
    ], dtype=float)

    # ── Shared color scale ────────────────────────────────────────────────────────
    _vmax      = np.nanpercentile(np.where(_mat_ordered == 0, np.nan, _mat_ordered), 95)
    _cmap_feat = plt.cm.YlOrRd

    # ── Split into two panels ─────────────────────────────────────────────────────
    _half        = math.ceil(len(_rows_ordered) / 2)
    _panels_rows = [_rows_ordered[:_half], _rows_ordered[_half:]]

    # ── Figure ────────────────────────────────────────────────────────────────────
    _n_cols     = len(_categories)
    _fig, _axes = plt.subplots(1, 2, figsize=(14, max(8, _half * 0.22)), sharey=False)

    for _pi, (_ax, _panel_rows) in enumerate(zip(_axes, _panels_rows)):

        _p_scodes = [r[0] for r in _panel_rows]
        _p_labels = [r[1] for r in _panel_rows]
        _p_is_bot = [r[2] for r in _panel_rows]

        # Global indices into _mat_ordered for this panel
        _global_offset = _half * _pi
        _p_mat = _mat_ordered[_global_offset: _global_offset + len(_panel_rows)]

        # ── y positions ───────────────────────────────────────────────────────────
        _p_y_pos = []
        _gap     = 0.8
        _y       = 0.0
        for _i, (_lbl, _is_bot) in enumerate(zip(_p_labels, _p_is_bot)):
            if _is_bot and _i > 0:
                _y += _gap
            _p_y_pos.append(_y)
            _y += 1.0
        _p_y_pos      = np.array(_p_y_pos)
        _total_height = _p_y_pos[-1] + 1.0

        # ── Draw rows ─────────────────────────────────────────────────────────────
        for _ri, (_y, _is_bot) in enumerate(zip(_p_y_pos, _p_is_bot)):
            _row_vals        = _p_mat[_ri:_ri+1, :]
            _row_vals_masked = np.where(_row_vals == 0, np.nan, _row_vals)
            _ax.pcolormesh(
                np.arange(_n_cols + 1),
                np.array([_y, _y + 1.0]),
                _row_vals_masked,
                cmap=_cmap_feat,
                edgecolors="white",
                linewidth=0.4,
                vmin=0, vmax=_vmax
            )
            if _is_bot:
                _ax.axhspan(_y, _y + 1.0, color="#d0d0d0", alpha=0.6, zorder=0)
            else:
                _ax.axhspan(_y, _y + 1.0, color="#f7f7f7", alpha=0.3, zorder=0)

            for _ci in range(_n_cols):
                _val = int(_p_mat[_ri, _ci])
                if _val == 0:
                    continue
                _text_color = "white" if _val > _vmax * 0.7 else "black"
                _ax.text(
                    _ci + 0.5, _y + 0.5, str(_val),
                    ha="center", va="center",
                    fontsize=6.5, color=_text_color
                )

        # ── Botanical spans ───────────────────────────────────────────────────────
        _bot_spans_p = []
        _cur_lbl     = None
        _cur_y_start = None
        _cur_y_end   = None
        for _y, _lbl, _is_bot in zip(_p_y_pos, _p_labels, _p_is_bot):
            if _is_bot:
                if _cur_lbl is not None:
                    _bot_spans_p.append((_cur_lbl, _cur_y_start, _cur_y_end))
                _cur_lbl     = _lbl
                _cur_y_start = _y
                _cur_y_end   = _y + 1.0
            else:
                if _cur_lbl is not None:
                    _cur_y_end = _y + 1.0
        if _cur_lbl is not None:
            _bot_spans_p.append((_cur_lbl, _cur_y_start, _cur_y_end))

        # ── Separator lines + brackets ────────────────────────────────────────────
        for _bi, (_lbl, _y_start, _y_end) in enumerate(_bot_spans_p):
            if _bi > 0:
                _ax.axhline(
                    _y_start - _gap / 2,
                    color="#333333", linewidth=1.8, linestyle="-", zorder=6
                )
            _const_ys = [
                _y for _y, _lbl2, _is_bot in zip(_p_y_pos, _p_labels, _p_is_bot)
                if not _is_bot and _y >= _y_start and _y < _y_end
            ]

        # ── y axis ────────────────────────────────────────────────────────────────
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

        # ── x axis ────────────────────────────────────────────────────────────────
        _ax.set_xticks(np.arange(_n_cols) + 0.5)
        _ax.set_xticklabels(_categories, fontsize=9, fontweight="bold",
                            rotation=45, ha="left")
        for _tick, _cat in zip(_ax.get_xticklabels(), _categories):
            _tick.set_color(_cat_colors[_cat])
        _ax.xaxis.set_label_position("top")
        _ax.xaxis.tick_top()
        _ax.set_xlim(0, _n_cols)
        _ax.set_title(
            f"N features per CP category — panel {_pi + 1}",
            fontsize=10, fontweight="bold", pad=30
        )

        # Colorbar on last panel only
        if _pi == 1:
            _sm   = plt.cm.ScalarMappable(cmap=_cmap_feat, norm=plt.Normalize(vmin=0, vmax=_vmax))
            _cbar = _fig.colorbar(_sm, ax=_ax, shrink=0.3, pad=0.02, aspect=20)
            _cbar.set_label("N features", fontsize=8)
            _cbar.ax.tick_params(labelsize=7)

    _fig.suptitle(
        "N features per CP channel category — botanicals and constituents",
        fontsize=11, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    _fig.savefig(
        os.path.expanduser("~/Downloads/heatmap_n_features_botanicals_2panels.pdf"),
        bbox_inches="tight", dpi=200
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""## POD Categories feature wise active comparison""")
    return


@app.cell
def _(
    Path,
    combinations,
    df_bsc_cp_pod_all_featurewise,
    feature_wise_permutation_threshold,
    kruskal,
    mticker,
    np,
    plt,
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

    # ── Plot ──────────────────────────────────────────────────────────────────────
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
            f"{_stars(_p)}  p={_p:.3f} (n={_n})",
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
    plt.savefig(Path.home() / "Downloads" / "pac_per_category.png", dpi=300, bbox_inches="tight")

    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""## Categories feature wise active comparison""")
    return


@app.cell
def _(df_bsc_cp_pod_all_featurewise):
    df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["Metadata_compound"] == "s0654868"
    ]
    return


@app.cell
def _(
    Path,
    combinations,
    df_bsc_cp_pod_all_featurewise,
    feature_wise_permutation_threshold,
    kruskal,
    mticker,
    np,
    plt,
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

    # ── Plot ──────────────────────────────────────────────────────────────────────
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
            f"{_stars(_p)}  p={_p:.3f} (n={_n})",
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
    plt.savefig(Path.home() / "Downloads" / "boxplot_features.png", dpi=200, bbox_inches="tight")
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
    np,
    os,
    plt,
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

    # ── Valid compounds (threshold filter) ───────────────────────────────────────
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
    _kw_results = {}   # scode -> (stat, p, sig)

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

    # ── Build ordered row index: botanicals + constituents ────────────────────────
    _rows_ordered = []
    for _scode in list_botanicals:
        _plant_name   = dic_botanicals_scode_name.get(_scode, _scode)
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

    # ── Build matrix row by row ───────────────────────────────────────────────────
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

    # ── Split at nearest botanical boundary to midpoint ───────────────────────────
    _half_target = len(_rows_ordered) // 2
    _bot_indices = [i for i, r in enumerate(_rows_ordered) if r[2]]
    _split_idx   = min(_bot_indices, key=lambda i: abs(i - _half_target))
    _panels_rows = [_rows_ordered[:_split_idx], _rows_ordered[_split_idx:]]

    # ── Figure ────────────────────────────────────────────────────────────────────
    _n_cols     = len(_categories)
    _half_n     = max(len(_panels_rows[0]), len(_panels_rows[1]))
    _fig, _axes = plt.subplots(1, 2, figsize=(16, max(8, _half_n * 0.25)), sharey=False)

    for _pi, (_ax, _panel_rows) in enumerate(zip(_axes, _panels_rows)):
        _p_labels = [r[1] for r in _panel_rows]
        _p_is_bot = [r[2] for r in _panel_rows]
        _p_scodes = [r[0] for r in _panel_rows]
        _p_start  = 0 if _pi == 0 else _split_idx
        _p_mat    = _mat_ordered[_p_start: _p_start + len(_panel_rows)]

        # ── y positions ───────────────────────────────────────────────────────────
        _p_y_pos = []
        _gap     = 0.8
        _y       = 0.0
        for _i, (_lbl, _is_bot) in enumerate(zip(_p_labels, _p_is_bot)):
            if _is_bot and _i > 0:
                _y += _gap
            _p_y_pos.append(_y)
            _y += 1.0
        _p_y_pos      = np.array(_p_y_pos)
        _total_height = _p_y_pos[-1] + 1.0

        # ── Draw heatmap rows ─────────────────────────────────────────────────────
        for _ri, (_y, _is_bot, _scode) in enumerate(zip(_p_y_pos, _p_is_bot, _p_scodes)):
            _row_vals = _p_mat[_ri:_ri+1, :]
            _ax.pcolormesh(
                np.arange(_n_cols + 1),
                np.array([_y, _y + 1.0]),
                _row_vals,
                cmap=_cmap_feat, norm=_norm,
                edgecolors="white", linewidth=0.4,
            )
            if _is_bot:
                _ax.axhspan(_y, _y + 1.0, color="#d0d0d0", alpha=0.6, zorder=0)
            else:
                _ax.axhspan(_y, _y + 1.0, color="#f7f7f7", alpha=0.3, zorder=0)

            # Cell value annotations
            for _ci in range(_n_cols):
                _val = _p_mat[_ri, _ci]
                if np.isnan(_val):
                    _ax.text(_ci + 0.5, _y + 0.5, "—",
                             ha="center", va="center", fontsize=6, color="#bbbbbb")
                    continue
                _txt      = f"{_val:.0f}" if _val >= 1 else f"{_val:.2f}"
                _normed   = float(_norm(_val))
                _text_col = "white" if _normed > 0.75 else "black"
                _ax.text(_ci + 0.5, _y + 0.5, _txt,
                         ha="center", va="center", fontsize=6, color=_text_col)

            # ── KW significance on the right margin ───────────────────────────────
            if _scode in _kw_results:
                _, _kw_p, _kw_sig = _kw_results[_scode]
                if not np.isnan(_kw_p) and _kw_p < 0.05:
                    _ax.text(
                        _n_cols + 0.3, _y + 0.5, _kw_sig,
                        ha="center", va="center",
                        fontsize=7, fontweight="bold", color="#cc0000"
                    )

        # ── Botanical spans ───────────────────────────────────────────────────────
        _bot_spans_p = []
        _cur_lbl = _cur_y_start = _cur_y_end = None
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
            _const_ys = [
                _y for _y, _l2, _ib in zip(_p_y_pos, _p_labels, _p_is_bot)
                if not _ib and _y >= _y_start and _y < _y_end
            ]
            _has_hdr = any(
                _ib and _y >= _y_start and _y < _y_end
                for _y, _ib in zip(_p_y_pos, _p_is_bot)
            )

        # ── y axis ────────────────────────────────────────────────────────────────
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

        # ── x axis ────────────────────────────────────────────────────────────────
        _ax.set_xticks(np.arange(_n_cols) + 0.5)
        _ax.set_xticklabels(_categories, fontsize=9, fontweight="bold",
                            rotation=45, ha="left")
        for _tick, _cat in zip(_ax.get_xticklabels(), _categories):
            _tick.set_color(_cat_colors[_cat])
        _ax.xaxis.set_label_position("top")
        _ax.xaxis.tick_top()
        _ax.set_xlim(0, _n_cols + 0.8)
        _ax.set_title(
            f"PAC (5th pct cp_POD_um) per CP category — panel {_pi + 1}\n"
            "red * = Kruskal-Wallis significant across categories",
            fontsize=9, fontweight="bold", pad=30
        )

        if _pi == 1:
            _sm   = plt.cm.ScalarMappable(cmap=_cmap_feat, norm=_norm)
            _cbar = _fig.colorbar(_sm, ax=_ax, shrink=0.3, pad=0.02, aspect=20)
            _cbar.set_label("PAC — 5th pct cp_POD_um (µM, log)", fontsize=8)
            _cbar.ax.tick_params(labelsize=7)

    _fig.suptitle(
        "PAC per CP channel category — botanicals and constituents\n"
        f"(— = compound below threshold of {feature_wise_permutation_threshold} features)",
        fontsize=11, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    _fig.savefig(
        os.path.expanduser("~/Downloads/heatmap_pac_category_botanicals.pdf"),
        bbox_inches="tight", dpi=200
    )
    _fig
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
    os,
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

    # ── Figure ────────────────────────────────────────────────────────────────────
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
        os.path.expanduser("~/Downloads/heatmap_pac_category_botanicals.pdf"),
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
    math,
    np,
    os,
    pd,
    plt,
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

    # ── Layout ────────────────────────────────────────────────────────────────────
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
                _sig  = _stars(_val)
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
        os.path.expanduser("~/Downloads/pairwise_pval_grid.png"),
        bbox_inches="tight", dpi=300
    )
    _fig
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
def _(all_selected_features, df_level6_bsc):
    df_level6_bsc[all_selected_features]
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
def _(
    Patch,
    Path,
    StandardScaler,
    all_selected_features,
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
        lambda row: f"{row['Constituent']}" 
                    if pd.notna(row['Metadata_compound']) and pd.notna(row['Plant name'])
                    else f"{row['Constituent']}"
                    if pd.notna(row['Metadata_compound']) and pd.isna(row['Plant name'])
                    else row['Metadata_compound'],
        axis=1
    )
    _df_level6_bsc_plantscode["Row_label"] = _df_level6_bsc_plantscode.apply(
        lambda row: "DMSO" if pd.isna(row["Constituent"])
        else row["Constituent"],
        axis=1
    )

    # Select features for clustering
    features_data = _df_level6_bsc_plantscode[all_selected_features]

    # Standardize features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        features_data, #scaler.fit_transform(features_data),
        columns=all_selected_features,
        index=_df_level6_bsc_plantscode["Metadata_compound"]
    )

    # Get row labels - use the new Row_label column
    row_labels = _df_level6_bsc_plantscode["Row_label"]

    # Create color palette for Plant scode
    unique_plant_scodes = _df_level6_bsc_plantscode["Plant name"].dropna().unique()
    #n_colors = len(unique_plant_scodes)
    #palette = sns.color_palette("tab20", n_colors) if n_colors <= 20 else sns.color_palette("husl", n_colors)
    #plant_scode_colors = dict(zip(unique_plant_scodes, palette))
    plant_scode_colors = {
        "Kratom":              "#e6194B",
        "Ashwagandha":         "#3cb44b",
        "Comfrey":             "#ffe119",
        "Usnea":               "#4363d8",
        "Asian ginseng":       "#f58231",
        "Blue cohosh":         "#42d4f4",
        "Yohimbe":             "#f032e6",
        "Kava":                "#fabed4",
        "Milk thistle":        "#469990",
        "Goldenseal":          "#dcbeff",
        "Aristolochia":        "#9A6324",
        "Green tea extract":   "#fffac8",
        "Ephedra":             "#800000",
    }
    # Create color annotation for rows
    row_colors = _df_level6_bsc_plantscode["Plant name"].map(plant_scode_colors)

    # Create cluster map with row colors
    sns.set(font_scale=1)
    cg = sns.clustermap(
        features_data,
        method='average',          # linkage method
        metric='cosine',     # distance metric
        cmap='RdBu_r',          # Red = high, Blue = low
        vmin=-2,                # force color scale
        vmax=2,
        figsize=(14, 12),
        row_cluster=True,
        col_cluster=True,
        xticklabels=False,      # hide column names
        row_colors=row_colors,  # Add color annotation
        yticklabels=row_labels, # Set row labels directly with new labels
        #cbar_pos=(0.02, 0.8, 0.01, 0.05)  # Move colorbar to avoid overlap
    )

    _botanical_names = set(plant_scode_colors.keys())
    _compound_to_plant = (
        _df_level6_bsc_plantscode
        .set_index("Metadata_compound")["Plant name"]
        .to_dict()
    )

    for _tick in cg.ax_heatmap.get_yticklabels():
        _label = _tick.get_text()
        if _label in plant_scode_colors.keys():
            _tick.set_fontweight("bold")
            _tick.set_fontsize(8)
        else:
            _tick.set_fontsize(7)

    # Move row dendrogram closer and shrink it
    # Positions
    _hm_pos  = cg.ax_heatmap.get_position()
    _rd_pos  = cg.ax_row_dendrogram.get_position()
    _rc_pos  = cg.ax_row_colors.get_position()

    _new_rd_width = _rd_pos.width * 0.3
    _new_rc_width = _rc_pos.width * 0.4

    cg.ax_row_colors.set_xticklabels([])
    cg.ax_row_colors.set_yticklabels([])

    # Dendrogram on the far left
    cg.ax_row_dendrogram.set_position([
        _hm_pos.x0 - _new_rd_width - _new_rc_width - 0.002,
        _rd_pos.y0,
        _new_rd_width,
        _rd_pos.height
    ])

    # Row colors between dendrogram and heatmap
    cg.ax_row_colors.set_position([
        _hm_pos.x0 - _new_rc_width - 0.001,
        _rc_pos.y0,
        _new_rc_width,
        _rc_pos.height
    ])

    # Label the row colors axis
    #cg.ax_row_colors.set_title("Botanical extracts", fontsize=9, fontweight="bold", pad=4)

    # Hide the column dendrogram (the "tree")
    cg.ax_col_dendrogram.set_visible(False)

    cg.ax_cbar.set_position((1, 0.3, 0.02, 0.05))

    # Adjust rotation if needed
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7)

    # Create legend for Plant scode colors

    legend_elements = [Patch(facecolor=plant_scode_colors[ps], label=ps) 
                       for ps in sorted(unique_plant_scodes)]

    # Add legend outside the plot

    cg.ax_heatmap.legend(
        handles=legend_elements,
        loc="upper left",        # or "best" to let matplotlib decide
        bbox_to_anchor=(1.2, 1),
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=14,
        borderpad=0.8,
        title="Botanical extracts",
        title_fontsize=16,
    ).get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(Path.home() / "Downloads" / "clustermap_level4_bsc.png", format="png", bbox_inches="tight", dpi = 300)
    plt.show()
    return legend_elements, plant_scode_colors


@app.cell
def _(mo):
    mo.md(r"""## Phenotypic space colored by POD (Cell Painting) POD (mtt) POD (ldh)""")
    return


@app.cell
def _(
    Line2D,
    LogNorm,
    adjust_text,
    all_selected_features,
    df_bsc_all_pod_annotations,
    df_bsc_cp_pod_all_featurewise,
    df_level6_bsc,
    dic_botanicals_scode_name,
    feature_wise_permutation_threshold,
    pd,
    plt,
    umap,
):
    # ── 1. Rebuild _df_level6_bsc_plantscode ─────────────────────────────────────
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

    _mask = _df_level6_bsc_plantscode["Plant name"].isna()
    _df_level6_bsc_plantscode.loc[_mask, "Plant name"] = (
        _df_level6_bsc_plantscode.loc[_mask, "Metadata_compound"].map(_compound_to_plant_name)
    )

    _df_level6_bsc_plantscode["Row_label"] = _df_level6_bsc_plantscode.apply(
        lambda row: f"{row['Constituent']} ({row['Metadata_compound']})"
                    if pd.notna(row['Metadata_compound']) and pd.notna(row['Plant name'])
                    else f"{row['Constituent']} (-- Botanical --)"
                    if pd.notna(row['Metadata_compound']) and pd.isna(row['Plant name'])
                    else row['Metadata_compound'],
        axis=1
    )

    # ── 2. Build PAC lookup ───────────────────────────────────────────────────────
    _df_cp_passed = df_bsc_cp_pod_all_featurewise[
        df_bsc_cp_pod_all_featurewise["all.pass"] == True
    ].copy()

    _n_active = _df_cp_passed.groupby("Metadata_compound").size()
    _valid = _n_active[_n_active >= feature_wise_permutation_threshold].index

    _cp_pac_series = (
        _df_cp_passed[_df_cp_passed["Metadata_compound"].isin(_valid)]
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05, interpolation="linear")
    )

    # ── 3. Prepare feature matrix + group classification ─────────────────────────
    _df_plot = _df_level6_bsc_plantscode.copy()
    _df_plot["PAC_cp"] = _df_plot["Metadata_compound"].map(_cp_pac_series)

    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _classify(_row):
        if _row["Metadata_compound"] == "s0617825":
            return "DMSO"
        elif _row["Metadata_compound"] in _botanical_scodes:
            return "Botanical"
        elif pd.notna(_row["PAC_cp"]):
            return "Active constituent"
        else:
            return "Inactive constituent"

    _df_plot["Group"] = _df_plot.apply(_classify, axis=1)

    # ── 4. UMAP embedding ─────────────────────────────────────────────────────────
    _features_matrix = _df_plot[all_selected_features].values
    _reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    _embedding = _reducer.fit_transform(_features_matrix)
    _df_plot["UMAP1"] = _embedding[:, 0]
    _df_plot["UMAP2"] = _embedding[:, 1]

    # ── 5. Shared color scale across all PAC values ───────────────────────────────
    _all_pac = _df_plot["PAC_cp"].dropna().values
    _norm = LogNorm(vmin=_all_pac.min(), vmax=_all_pac.max())
    _cmap = plt.cm.plasma_r

    # ── 6. Plot ───────────────────────────────────────────────────────────────────
    _fig, _ax = plt.subplots(figsize=(12, 9))

    _group_config = {
        "Inactive constituent": ("o",  35, 2),
        "Active constituent":   ("o",  60, 4),
        "Botanical":            ("D",  55, 3),
        "DMSO":                 ("*", 180, 5),
    }

    # Inactive constituents (grey)
    _sub_inactive = _df_plot[_df_plot["Group"] == "Inactive constituent"]
    _mk, _sz, _zo = _group_config["Inactive constituent"]
    _ax.scatter(
        _sub_inactive["UMAP1"], _sub_inactive["UMAP2"],
        c="lightgrey", s=_sz, marker=_mk, alpha=0.4,
        linewidths=0.3, edgecolors="grey", zorder=_zo,
        label="Inactive constituent"
    )

    # Botanicals without PAC (grey diamonds)
    _sub_bot = _df_plot[_df_plot["Group"] == "Botanical"]
    _mk, _sz, _zo = _group_config["Botanical"]
    _has_pac_bot = _sub_bot["PAC_cp"].notna()
    _ax.scatter(
        _sub_bot.loc[~_has_pac_bot, "UMAP1"], _sub_bot.loc[~_has_pac_bot, "UMAP2"],
        c="lightgrey", s=_sz, marker=_mk, alpha=0.6,
        linewidths=0.4, edgecolors="grey", zorder=_zo
    )

    # Botanicals with PAC (colored diamonds)
    if _has_pac_bot.any():
        _ax.scatter(
            _sub_bot.loc[_has_pac_bot, "UMAP1"], _sub_bot.loc[_has_pac_bot, "UMAP2"],
            c=_sub_bot.loc[_has_pac_bot, "PAC_cp"].values, cmap=_cmap, norm=_norm,
            s=_sz, marker=_mk, alpha=0.85,
            linewidths=0.4, edgecolors="black", zorder=_zo
        )

    # Active constituents (colored circles)
    _sub_active = _df_plot[_df_plot["Group"] == "Active constituent"]
    _mk, _sz, _zo = _group_config["Active constituent"]
    _sc = _ax.scatter(
        _sub_active["UMAP1"], _sub_active["UMAP2"],
        c=_sub_active["PAC_cp"].values, cmap=_cmap, norm=_norm,
        s=_sz, marker=_mk, alpha=0.85,
        linewidths=0.4, edgecolors="black", zorder=_zo,
        label="Active constituent"
    )

    # DMSO (black star)
    _sub_dmso = _df_plot[_df_plot["Group"] == "DMSO"]
    _mk, _sz, _zo = _group_config["DMSO"]
    _ax.scatter(
        _sub_dmso["UMAP1"], _sub_dmso["UMAP2"],
        c="black", s=_sz, marker=_mk, alpha=1.0,
        linewidths=0.5, edgecolors="black", zorder=_zo,
        label="DMSO"
    )

    # ── 7. Labels for botanicals + active constituents ────────────────────────────
    _to_label = _df_plot[_df_plot["Group"].isin(["Botanical", "Active constituent"])].copy()
    _to_label["Label"] = _to_label.apply(
        lambda _row: dic_botanicals_scode_name.get(_row["Metadata_compound"], _row["Constituent"]),
        axis=1
    )

    _texts = []
    for _, _row in _to_label.iterrows():
        _texts.append(_ax.text(
            _row["UMAP1"] + 0.05, _row["UMAP2"] + 0.05, _row["Label"],
            fontsize=5.5, ha="left", va="bottom"
        ))

    adjust_text(
        _texts,
        ax=_ax,
        expand_points=(2.0, 2.0),   # push labels away from points
        expand_text=(1.5, 1.5),     # push labels away from each other
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.4)
    )

    # ── 8. Colorbar + legend ──────────────────────────────────────────────────────
    _cbar = _fig.colorbar(_sc, ax=_ax, pad=0.02, shrink=0.65)
    _cbar.set_label("CP POD – 5th pct (µM, log scale)", fontsize=10)

    _legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey",
               markeredgecolor="grey", markersize=7, label="Inactive constituent"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="purple",
               markeredgecolor="black", markersize=7, label="Active constituent"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="purple",
               markeredgecolor="black", markersize=7, label="Botanical (active)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="lightgrey",
               markeredgecolor="grey", markersize=7, label="Botanical (no PAC)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
               markeredgecolor="black", markersize=10, label="DMSO"),
    ]
    _ax.legend(handles=_legend_elements, loc="lower right", fontsize=8, frameon=True)

    _ax.set_xlabel("UMAP 1", fontsize=11)
    _ax.set_ylabel("UMAP 2", fontsize=11)
    _ax.set_title("Phenotypic space (Cell Painting)\nBotanicals · Active constituents · DMSO", fontsize=13)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Analysis : Heatmap cosine similarity""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Analysis : Heatmap cosine similarity level 4 [OOM Too BIG]""")
    return


@app.cell
def _(
    Patch,
    all_selected_features,
    cosine_similarity,
    df_bsc_all_pod_annotations,
    df_level4_bsc,
    dic_botanicals_scode_name,
    np,
    os,
    pd,
    plt,
    sns,
):
    if False:
        # ── 1. Setup ──────────────────────────────────────────────────────────────────
        _botanical_scodes = set(dic_botanicals_scode_name.keys())

        def _classify_point(_compound):
            if _compound in ("s0617825", "DMSO"):
                return "DMSO"
            elif _compound in _botanical_scodes:
                return "Botanical"
            else:
                return "Compound"

        _df = df_level4_bsc.copy()
        _df["PointType"] = _df["Metadata_compound"].apply(_classify_point)

        _compound_to_plant = (
            df_bsc_all_pod_annotations[["Metadata_compound", "Plant name"]]
            .drop_duplicates()
            .dropna()
            .set_index("Metadata_compound")["Plant name"]
            .to_dict()
        )
        _df["Plant name"] = _df["Metadata_compound"].map(lambda _c: _compound_to_plant.get(_c, "—"))

        _constituent_lookup = (
            df_bsc_all_pod_annotations[["Metadata_compound", "Constituent"]]
            .drop_duplicates()
            .dropna(subset=["Constituent"])
            .set_index("Metadata_compound")["Constituent"]
            .to_dict()
        )

        # DisplayName includes concentration
        _df["DisplayName"] = _df.apply(
            lambda _r: f"{dic_botanicals_scode_name.get(_r['Metadata_compound'])} @ {_r['Metadata_concentration']:.2g} µM"
                       if _r["Metadata_compound"] in _botanical_scodes
                       else f"{_constituent_lookup.get(_r['Metadata_compound'], _r['Metadata_compound'])} @ {_r['Metadata_concentration']:.2g} µM"
                       if _r["Metadata_compound"] not in ("s0617825", "DMSO")
                       else f"{_r['Metadata_compound']} @ {_r['Metadata_concentration']:.2g} µM",
            axis=1
        )

        # ── 2. Cosine similarity matrix ───────────────────────────────────────────────
        _X = _df[all_selected_features].values
        _cos_sim = cosine_similarity(_X)
        _df_cos = pd.DataFrame(
            _cos_sim,
            index=_df["DisplayName"].values,
            columns=_df["DisplayName"].values,
        )

        # ── 3. Row color annotations ──────────────────────────────────────────────────
        _ptype_palette = {
            "Compound":  "#aec7e8",
            "Botanical": "#2ca02c",
            "DMSO":      "#000000",
        }
        _unique_plants = [_p for _p in _df["Plant name"].unique() if _p != "—"]
        _plant_palette = dict(zip(
            _unique_plants,
            sns.color_palette("tab20", len(_unique_plants))
        ))
        _plant_palette["—"] = "#f0f0f0"

        # Concentration color (log scale)
        _conc_vals = _df["Metadata_concentration"].values
        _conc_norm = plt.Normalize(vmin=np.log10(_conc_vals[_conc_vals > 0].min()),
                                    vmax=np.log10(_conc_vals[_conc_vals > 0].max()))
        _conc_cmap = plt.cm.YlOrRd
        _conc_colors = [
            _conc_cmap(_conc_norm(np.log10(_v))) if _v > 0 else (0.9, 0.9, 0.9, 1.0)
            for _v in _conc_vals
        ]

        _row_colors = pd.DataFrame({
            "Type":          _df["PointType"].map(_ptype_palette).values,
            "Plant name":    _df["Plant name"].map(_plant_palette).values,
            "Concentration": _conc_colors,
        }, index=_df_cos.index)

        _ytick_labels = _df["DisplayName"].values.tolist()
        _n_rows = len(_df)
        # ── 4. Clustermap ─────────────────────────────────────────────────────────────
        sns.set(font_scale=0.65)
        _cg = sns.clustermap(
            _df_cos,
            method="ward",
            metric="euclidean",
            cmap="RdBu_r",
            vmin=-1, vmax=1,
            figsize=(max(14, _n_rows * 0.1)+2, max(14, _n_rows * 0.1)),
            row_cluster=True,
            col_cluster=True,
            row_colors=_row_colors,
            xticklabels=False,
            yticklabels=_ytick_labels,
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
        )
        _cg.ax_heatmap.set_yticklabels(
            _cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=5
        )
        _cg.ax_heatmap.set_xlabel("Well", fontsize=9)
        _cg.ax_heatmap.set_ylabel("Compound @ Concentration", fontsize=9)
        _cg.fig.suptitle(
            "Cosine similarity clustermap — Level 5 BSC profiles (per well)",
            y=1.01, fontsize=12
        )

        # ── 5. Legends ────────────────────────────────────────────────────────────────
        _legend_ptype = [Patch(facecolor=_col, label=_pt) for _pt, _col in _ptype_palette.items()]
        _legend_plant = [Patch(facecolor=_col, label=_pl) for _pl, _col in _plant_palette.items() if _pl != "—"]

        # Concentration colorbar legend
        _sm = plt.cm.ScalarMappable(cmap=_conc_cmap, norm=_conc_norm)
        _sm.set_array([])
        _cax = _cg.fig.add_axes([0.82, 0.72, 0.03, 0.12])
        _cg.fig.colorbar(_sm, cax=_cax, orientation="vertical")
        _cax.set_ylabel("log10 Conc (µM)", fontsize=6)
        _cax.tick_params(labelsize=5)

        _leg2 = _cg.fig.add_axes([0.82, 0.52, 0.15, 0.15])
        _leg2.axis("off")
        _leg2.legend(handles=_legend_ptype, title="Type", loc="upper left",
                     frameon=True, fontsize=6, title_fontsize=7)

        _leg3 = _cg.fig.add_axes([0.82, 0.20, 0.15, 0.28])
        _leg3.axis("off")
        _leg3.legend(handles=_legend_plant, title="Plant name", loc="upper left",
                     frameon=True, fontsize=6, title_fontsize=7)

        plt.tight_layout()
        _cg.fig.savefig(os.path.expanduser("~/Downloads/clustermap_level5_bsc.pdf"), bbox_inches="tight", dpi=150)
        plt.close(_cg.fig)
    return


@app.cell
def _(mo):
    mo.md(r"""### Analysis : Heatmap cosine similarity level 5""")
    return


@app.cell
def _(
    Patch,
    all_selected_features,
    cosine_similarity,
    df_bsc_all_pod_annotations,
    df_level5_bsc,
    dic_botanicals_scode_name,
    np,
    pd,
    plt,
    sns,
):
    # ── 1. Setup ──────────────────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _classify_point(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    _df = df_level5_bsc.copy()
    _df["PointType"] = _df["Metadata_compound"].apply(_classify_point)

    _compound_to_plant = (
        df_bsc_all_pod_annotations[["Metadata_compound", "Plant name"]]
        .drop_duplicates()
        .dropna()
        .set_index("Metadata_compound")["Plant name"]
        .to_dict()
    )
    _df["Plant name"] = _df["Metadata_compound"].map(lambda _c: _compound_to_plant.get(_c, "—"))

    _constituent_lookup = (
        df_bsc_all_pod_annotations[["Metadata_compound", "Constituent"]]
        .drop_duplicates()
        .dropna(subset=["Constituent"])
        .set_index("Metadata_compound")["Constituent"]
        .to_dict()
    )

    # DisplayName includes concentration
    _df["DisplayName"] = _df.apply(
        lambda _r: f"{dic_botanicals_scode_name.get(_r['Metadata_compound'])} @ {_r['Metadata_concentration']:.2g} µM"
                   if _r["Metadata_compound"] in _botanical_scodes
                   else f"{_constituent_lookup.get(_r['Metadata_compound'], _r['Metadata_compound'])} @ {_r['Metadata_concentration']:.2g} µM"
                   if _r["Metadata_compound"] not in ("s0617825", "DMSO")
                   else f"{_r['Metadata_compound']} @ {_r['Metadata_concentration']:.2g} µM",
        axis=1
    )

    # ── 2. Cosine similarity matrix ───────────────────────────────────────────────
    _X = _df[all_selected_features].values
    _cos_sim = cosine_similarity(_X)
    _df_cos = pd.DataFrame(
        _cos_sim,
        index=_df["DisplayName"].values,
        columns=_df["DisplayName"].values,
    )

    # ── 3. Row color annotations ──────────────────────────────────────────────────
    _ptype_palette = {
        "Compound":  "#aec7e8",
        "Botanical": "#2ca02c",
        "DMSO":      "#000000",
    }
    _unique_plants = [_p for _p in _df["Plant name"].unique() if _p != "—"]
    _plant_palette = dict(zip(
        _unique_plants,
        sns.color_palette("tab20", len(_unique_plants))
    ))
    _plant_palette["—"] = "#f0f0f0"

    # Concentration color (log scale)
    _conc_vals = _df["Metadata_concentration"].values
    _conc_norm = plt.Normalize(vmin=np.log10(_conc_vals[_conc_vals > 0].min()),
                                vmax=np.log10(_conc_vals[_conc_vals > 0].max()))
    _conc_cmap = plt.cm.YlOrRd
    _conc_colors = [
        _conc_cmap(_conc_norm(np.log10(_v))) if _v > 0 else (0.9, 0.9, 0.9, 1.0)
        for _v in _conc_vals
    ]

    _row_colors = pd.DataFrame({
        "Type":          _df["PointType"].map(_ptype_palette).values,
        "Plant name":    _df["Plant name"].map(_plant_palette).values,
        "Concentration": _conc_colors,
    }, index=_df_cos.index)

    _ytick_labels = _df["DisplayName"].values.tolist()
    _n_rows = len(_df)
    # ── 4. Clustermap ─────────────────────────────────────────────────────────────
    sns.set(font_scale=0.65)
    _cg = sns.clustermap(
        _df_cos,
        method="ward",
        metric="euclidean",
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        figsize=(max(14, _n_rows * 0.1)+2, max(14, _n_rows * 0.1)),
        row_cluster=True,
        col_cluster=True,
        row_colors=_row_colors,
        xticklabels=False,
        yticklabels=_ytick_labels,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )
    _cg.ax_heatmap.set_yticklabels(
        _cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=5
    )
    _cg.ax_heatmap.set_xlabel("Well", fontsize=9)
    _cg.ax_heatmap.set_ylabel("Compound @ Concentration", fontsize=9)
    _cg.fig.suptitle(
        "Cosine similarity clustermap — Level 5 BSC profiles (per well)",
        y=1.01, fontsize=12
    )

    # ── 5. Legends ────────────────────────────────────────────────────────────────
    _legend_ptype = [Patch(facecolor=_col, label=_pt) for _pt, _col in _ptype_palette.items()]
    _legend_plant = [Patch(facecolor=_col, label=_pl) for _pl, _col in _plant_palette.items() if _pl != "—"]

    # Concentration colorbar legend
    _sm = plt.cm.ScalarMappable(cmap=_conc_cmap, norm=_conc_norm)
    _sm.set_array([])
    _cax = _cg.fig.add_axes([0.82, 0.72, 0.03, 0.12])
    _cg.fig.colorbar(_sm, cax=_cax, orientation="vertical")
    _cax.set_ylabel("log10 Conc (µM)", fontsize=6)
    _cax.tick_params(labelsize=5)

    _leg2 = _cg.fig.add_axes([0.82, 0.52, 0.15, 0.15])
    _leg2.axis("off")
    _leg2.legend(handles=_legend_ptype, title="Type", loc="upper left",
                 frameon=True, fontsize=6, title_fontsize=7)

    _leg3 = _cg.fig.add_axes([0.82, 0.20, 0.15, 0.28])
    _leg3.axis("off")
    _leg3.legend(handles=_legend_plant, title="Plant name", loc="upper left",
                 frameon=True, fontsize=6, title_fontsize=7)

    plt.tight_layout()
    _cg.fig
    return


@app.cell
def _(mo):
    mo.md(r"""### Analysis : Heatmap cosine similarity level 6""")
    return


@app.cell
def _(df_table_results_pods):
    df_table_results_pods
    return


@app.cell
def _(
    Path,
    all_selected_features,
    cosine_similarity,
    df_bsc_all_pod_annotations,
    df_level6,
    df_level6_bsc,
    df_table_results_pods,
    dic_botanicals_scode_name,
    legend_elements,
    pd,
    plant_scode_colors,
    plt,
    sns,
):
    # ── 1. Cosine similarity matrix ───────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _classify_point(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    _df = df_level6_bsc.merge(
        df_table_results_pods[df_table_results_pods["Below threshold"] == True][["Compound ID"]],
        left_on="Metadata_compound",
        right_on="Compound ID",
        how="inner"
    ).drop("Compound ID", axis=1).copy()
    print(_df.shape)
    _df["PointType"] = _df["Metadata_compound"].apply(_classify_point)

    _compound_to_plant = (
        df_bsc_all_pod_annotations[["Metadata_compound", "Plant name"]]
        .drop_duplicates()
        .dropna()
        .set_index("Metadata_compound")["Plant name"]
        .to_dict()
    )
    _df["Plant name"] = _df["Metadata_compound"].apply(
        lambda _c: _compound_to_plant.get(_c) 
                   if _c in _compound_to_plant
                   else dic_botanicals_scode_name.get(_c) if _c in _botanical_scodes
                   else "DMSO" if _c in ("s0617825", "DMSO")
                   else "—"
    )
    _constituent_lookup = (
        df_bsc_all_pod_annotations[["Metadata_compound", "Constituent"]]
        .drop_duplicates()
        .dropna(subset=["Constituent"])
        .set_index("Metadata_compound")["Constituent"]
        .to_dict()
    )

    _df["DisplayName"] = _df["Metadata_compound"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c)    if _c in _botanical_scodes
                   else _constituent_lookup.get(_c, _c)  if _c not in ("s0617825", "DMSO")
                   else _c
    )
    # ── 2. Cosine similarity matrix ───────────────────────────────────────────────
    _X = _df[all_selected_features].values
    _cos_sim = cosine_similarity(_X)
    _df_cos = pd.DataFrame(
        _cos_sim,
        index=_df["DisplayName"].values,
        columns=_df["DisplayName"].values,
    )

    # ── 3. Row color annotations ──────────────────────────────────────────────────
    _ptype_palette = {
        "Compound":  "#aec7e8",
        "Botanical": "#2ca02c",
        "DMSO":      "#000000",
    }
    _unique_plants = [_p for _p in _df["Plant name"].unique() if _p != "—"]

    _row_colors = pd.DataFrame({
        "Plant name": _df["Plant name"].map(plant_scode_colors).values,
    }, index=_df_cos.index)
    _bsc_compounds = set(df_level6[df_level6["Dataset"] == "BSC"]["Metadata_compound"].values)

    _ytick_labels = _df["DisplayName"].values.tolist()
    # ── 4. Clustermap ─────────────────────────────────────────────────────────────
    sns.set(font_scale=0.65)
    _cg = sns.clustermap(
        _df_cos,
        method="ward",
        metric="euclidean",
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        figsize=(14, 12),
        row_cluster=True,
        col_cluster=True,
        row_colors=_row_colors,
        xticklabels=_ytick_labels,
        yticklabels=_ytick_labels,
        #cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )
    _cg.ax_heatmap.set_yticklabels(
        _cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6
    )
    _cg.ax_heatmap.set_xticklabels(
        _cg.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6
    )

    for _tick in _cg.ax_heatmap.get_yticklabels():
        _label = _tick.get_text()
        if _label in plant_scode_colors.keys():
            _tick.set_fontweight("bold")
            _tick.set_fontsize(10)
        else:
            _tick.set_fontsize(9)

    # Mirror the same bold/size logic for x labels
    for _tick in _cg.ax_heatmap.get_xticklabels():
        _label = _tick.get_text()
        if _label in plant_scode_colors.keys():
            _tick.set_fontweight("bold")
            _tick.set_fontsize(10)
        else:
            _tick.set_fontsize(9)

    # Move row dendrogram closer and shrink it
    _hm_pos  = _cg.ax_heatmap.get_position()
    _rd_pos  = _cg.ax_row_dendrogram.get_position()
    _rc_pos  = _cg.ax_row_colors.get_position()
    _cd_pos  = _cg.ax_col_dendrogram.get_position()

    _new_rd_width = _rd_pos.width * 0.3
    _new_cd_height = _cd_pos.height * 0.3
    _new_rc_width = _rc_pos.width * 0.4

    _cg.ax_row_colors.set_xticklabels([])
    _cg.ax_row_colors.set_yticklabels([])

    # Dendrogram on the far left
    _cg.ax_row_dendrogram.set_position([
        _hm_pos.x0 - _new_rd_width - _new_rc_width - 0.002,
        _rd_pos.y0,
        _new_rd_width,
        _rd_pos.height
    ])

    # Row colors between dendrogram and heatmap
    _cg.ax_row_colors.set_position([
        _hm_pos.x0 - _new_rc_width - 0.001,
        _rc_pos.y0,
        _new_rc_width,
        _rc_pos.height
    ])

    # Dendrogram on the top
    _cg.ax_col_dendrogram.set_position([
        _cd_pos.x0,
        _hm_pos.y0 + _hm_pos.height + 0.001,
        _cd_pos.width,
        _new_cd_height
    ])

    # Row colors between dendrogram and heatmap
    _cg.ax_row_colors.set_position([
        _hm_pos.x0 - _new_rc_width - 0.001,
        _rc_pos.y0,
        _new_rc_width,
        _rc_pos.height
    ])

    _cg.ax_cbar.set_position((1.06, 0.2, 0.1, 0.2))


    #_cg.ax_heatmap.set_xlabel("Compound", fontsize=9)
    #_cg.ax_heatmap.set_ylabel("Compound", fontsize=9)
    #_cg.fig.suptitle("Cosine similarity clustermap — Level 6 compound profiles",y=1.01, fontsize=12)

    # ── 5. Legends ────────────────────────────────────────────────────────────────
    # Add legend outside the plot

    _cg.ax_heatmap.legend(
        handles=legend_elements,
        loc="upper left",        # or "best" to let matplotlib decide
        bbox_to_anchor=(1.15, 1),
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=14,
        borderpad=0.8,
        title="Botanical extracts",
        title_fontsize=16,
    ).get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(Path.home() / "Downloads" / "clustermap_cosine_level6_bsc.png", format="png", bbox_inches="tight", dpi = 300)
    _cg.fig
    return


@app.cell
def _(mo):
    mo.md(r"""## Chemical space colored by POD (Cell Painting) POD (mtt) POD (ldh)""")
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
    df_metadata_compound_oasis = pd.read_csv("modules/01_preprocessing/inputs/v5_oasis_03Sept2024_simple.csv")
    return (df_metadata_compound_oasis,)


@app.cell
def _(pd):
    df_level4a_axiom = pd.read_parquet("modules/03_postprocessing/outputs/level4b_axiom_batch.parquet")
    df_level5_axiom_all = pd.read_parquet("modules/03_postprocessing/outputs/level5_axiom_selected_features_batch.parquet")
    df_level6_axiom_all = pd.read_parquet("modules/03_postprocessing/outputs/level6_axiom_selected_features_batch.parquet")
    return df_level4a_axiom, df_level5_axiom_all, df_level6_axiom_all


@app.cell
def _(pd):
    # Cell Painting profiles
    df_level4_bsc_batch = pd.read_parquet("modules/03_postprocessing/outputs/level4b_bsc_batch.parquet")
    if True:
        df_level4_bsc_batch = df_level4_bsc_batch.groupby("Metadata_plate").filter(
            lambda g: ~(g["Metadata_compound_name"] == "DMSO").all()
        )
    df_level5_bsc_batch = pd.read_parquet("modules/03_postprocessing/outputs/level5_bsc_selected_features_batch.parquet")
    df_level6_bsc_batch = pd.read_parquet("modules/03_postprocessing/outputs/level6_bsc_selected_features_batch.parquet")
    return df_level4_bsc_batch, df_level5_bsc_batch, df_level6_bsc_batch


@app.cell
def _(df_level4_bsc_batch, df_level4a_axiom, pd):
    # Add source column
    df_level4a_axiom["Dataset"] = "AXIOM"
    df_level4_bsc_batch["Dataset"]   = "BSC"

    # Bind rows
    df_level4 = pd.concat(
        [df_level4a_axiom, df_level4_bsc_batch],
        axis=0,
        ignore_index=True
    )
    return (df_level4,)


@app.cell
def _(df_level5_axiom_all, df_level5_bsc_batch, pd):
    # Add source column
    df_level5_axiom_all["Dataset"] = "AXIOM"
    df_level5_bsc_batch["Dataset"]   = "BSC"

    # Bind rows
    df_level5 = pd.concat(
        [df_level5_axiom_all, df_level5_bsc_batch],
        axis=0,
        ignore_index=True
    )
    return (df_level5,)


@app.cell
def _(df_level6_axiom_all, df_level6_bsc_batch, pd):
    # Add source column
    df_level6_axiom_all["Dataset"] = "AXIOM"
    df_level6_bsc_batch["Dataset"]   = "BSC"

    # Bind rows
    df_level6 = pd.concat(
        [df_level6_axiom_all, df_level6_bsc_batch],
        axis=0,
        ignore_index=True
    )
    print(df_level6_axiom_all.shape, df_level6_bsc_batch.shape, df_level6.shape)
    return (df_level6,)


@app.cell
def _(df_level6):
    all_features_bsc_axiom = [c for c in df_level6.columns if c.startswith("Feature_")]
    all_features_bsc_axiom
    return (all_features_bsc_axiom,)


@app.cell
def _(mo):
    mo.md(r"""## Analysis : UMAP Chemical space""")
    return


@app.cell
def _(
    AllChem,
    Chem,
    DataStructs,
    df_level4_bsc_batch,
    df_level4a_axiom,
    df_metadata_compound_oasis,
    dic_botanicals_scode_name,
    np,
    pd,
    umap,
):


    # ── 1. Build SMILES lookup for Axiom ─────────────────────────────────────────
    _axiom_oasis_ids = df_level4a_axiom["Metadata_OASIS_ID"].unique()
    _df_axiom_smiles = (
        df_metadata_compound_oasis[
            df_metadata_compound_oasis["OASIS_ID"].isin(_axiom_oasis_ids)
        ][["OASIS_ID", "SMILES"]]
        .drop_duplicates()
        .rename(columns={"OASIS_ID": "Metadata_compound", "SMILES": "SMILES"})
    )
    _df_axiom_smiles["Dataset"] = "Axiom"

    # ── 2. Build SMILES lookup for BSC ────────────────────────────────────────────
    _df_bsc_smiles = (
        df_level4_bsc_batch[["Metadata_compound", "Metadata_compound_smiles"]]
        .drop_duplicates()
        .dropna(subset=["Metadata_compound_smiles"])
        .rename(columns={"Metadata_compound_smiles": "SMILES"})
    )
    _df_bsc_smiles["Dataset"] = "BSC"

    # ── 3. Combine ────────────────────────────────────────────────────────────────
    df_smiles = (
        pd.concat([_df_axiom_smiles, _df_bsc_smiles], axis=0, ignore_index=True)
        .drop_duplicates(subset=["Metadata_compound"])
        .dropna(subset=["SMILES"])
        .reset_index(drop=True)
    )

    # ── 4. Classify point type ────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _classify_point(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    df_smiles["PointType"] = df_smiles["Metadata_compound"].apply(_classify_point)
    df_smiles["DisplayName"] = df_smiles["Metadata_compound"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c, _c) if _c in _botanical_scodes else _c
    )

    # ── 5. Compute Morgan fingerprints (radius=2, nBits=2048, binary) ─────────────
    def _smiles_to_fp(_smi):
        try:
            _mol = Chem.MolFromSmiles(_smi)
            if _mol is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(_mol, radius=2, nBits=2048)
        except:
            return None

    df_smiles["FP"] = df_smiles["SMILES"].apply(_smiles_to_fp)
    df_smiles = df_smiles[df_smiles["FP"].notna()].reset_index(drop=True)

    print(f"Valid structures: {len(df_smiles)}")

    # ── 6. Tanimoto distance matrix ───────────────────────────────────────────────
    _fps = df_smiles["FP"].tolist()
    _n = len(_fps)
    _dist_matrix = np.zeros((_n, _n))

    for _i in range(_n):
        for _j in range(_i + 1, _n):
            _sim = DataStructs.TanimotoSimilarity(_fps[_i], _fps[_j])
            _dist_matrix[_i, _j] = 1 - _sim
            _dist_matrix[_j, _i] = 1 - _sim

    # ── 7. UMAP on precomputed Tanimoto distance ──────────────────────────────────
    _reducer = umap.UMAP(
        n_components=2,
        metric="precomputed",
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    _embedding = _reducer.fit_transform(_dist_matrix)
    df_smiles["UMAP_1"] = _embedding[:, 0]
    df_smiles["UMAP_2"] = _embedding[:, 1]
    return (df_smiles,)


@app.cell
def _(df_smiles, go):
    # ── 8. Plot ───────────────────────────────────────────────────────────────────
    _dataset_colors = {
        _ds: _col for _ds, _col in zip(
            df_smiles["Dataset"].unique(),
            ["#1f77b4", "#ff7f0e"]
        )
    }

    _marker_config = {
        "Compound":  dict(symbol="circle",  size=6,  opacity=0.7),
        "Botanical": dict(symbol="diamond", size=10, opacity=0.9),
        "DMSO":      dict(symbol="star",    size=14, opacity=1.0),
    }

    _fig = go.Figure()

    for _dataset in df_smiles["Dataset"].unique():
        _df_ds = df_smiles[df_smiles["Dataset"] == _dataset]
        _color = _dataset_colors[_dataset]

        for _ptype, _mconf in _marker_config.items():
            _df_sub = _df_ds[_df_ds["PointType"] == _ptype]
            if _df_sub.empty:
                continue

            _fig.add_trace(go.Scatter(
                x=_df_sub["UMAP_1"],
                y=_df_sub["UMAP_2"],
                mode="markers",
                name=f"{_dataset} — {_ptype}",
                marker=dict(
                    symbol=_mconf["symbol"],
                    size=_mconf["size"],
                    opacity=_mconf["opacity"],
                    color=_color,
                    line=dict(width=0.8, color="black") if _ptype != "Compound" else dict(width=0),
                ),
                customdata=_df_sub[["Metadata_compound", "DisplayName", "SMILES"]].values,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "ID: %{customdata[0]}<br>"
                    f"Dataset: {_dataset}<br>"
                    "SMILES: %{customdata[2]}<br>"
                    "UMAP1: %{x:.2f} | UMAP2: %{y:.2f}"
                    "<extra></extra>"
                ),
            ))

    # Botanical text labels
    _df_bot = df_smiles[df_smiles["PointType"] == "Botanical"]
    _fig.add_trace(go.Scatter(
        x=_df_bot["UMAP_1"],
        y=_df_bot["UMAP_2"],
        mode="text",
        text=_df_bot["DisplayName"],
        textposition="top center",
        textfont=dict(size=8, color="black"),
        showlegend=False,
        hoverinfo="skip",
    ))

    _fig.update_layout(
        template="plotly_white",
        title="Chemical space UMAP — Morgan fingerprints (Tanimoto)<br><sup>◆ Botanicals · ★ DMSO · ● Compounds</sup>",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title_text="Dataset — Type",
        width=950,
        height=700,
        legend=dict(itemsizing="constant", font=dict(size=10)),
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Analysis : UMAP Cell Painting""")
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP Cell Painting level 4 Controls""")
    return


@app.cell
def _(df_level4_bsc_batch, df_level4a_axiom, pd):
    # Add source column
    df_level4a_axiom["Dataset"] = "AXIOM"
    df_level4_bsc_batch["Dataset"]   = "BSC"

    # Bind rows
    df_level4_control = pd.concat(
        [df_level4a_axiom[df_level4a_axiom["Metadata_compound"] == "DMSO"], df_level4_bsc_batch[df_level4_bsc_batch["Metadata_compound"] == "s0617825"]],
        axis=0,
        ignore_index=True
    )
    return (df_level4_control,)


@app.cell
def _(all_selected_features, df_level4_control, pd, umap):
    # --- 1. Extract features and metadata
    _X = df_level4_control[all_selected_features].values

    _plates = df_level4_control["Metadata_plate"].astype(str).values
    _concentration = df_level4_control["Metadata_concentration"].astype(float).values
    _log10conc = df_level4_control["Metadata_log10conc"].astype(float).values
    _dataset = df_level4_control["Dataset"].astype(str).values

    # --- 2. Run UMAP for 2D projection
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    _embedding = _reducer.fit_transform(_X)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_level4_control = pd.DataFrame(_embedding, columns=["UMAP_1", "UMAP_2"])
    df_umap_level4_control["Metadata_plate"] = _plates
    df_umap_level4_control["Metadata_concentration"] = _concentration
    df_umap_level4_control["Metadata_log10conc"] = _log10conc
    df_umap_level4_control["Dataset"] = _dataset
    return (df_umap_level4_control,)


@app.cell
def _(df_umap_level4_control, px):
    # --- 4. Create interactive Plotly scatter plot
    _fig = px.scatter(
        df_umap_level4_control,
        x="UMAP_1",
        y="UMAP_2",
        color="Dataset",
        title="UMAP of Level 4 Cell Painting Profiles of CONTROLS Colored by Dataset (axiom vs BSC)",
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


@app.cell
def _(mo):
    mo.md(r"""### UMAP Cell Painting level 4""")
    return


@app.cell
def _(all_selected_features, df_level4, pd, umap):
    # --- 1. Extract features and metadata
    _X = df_level4[all_selected_features].values

    _plates = df_level4["Metadata_plate"].astype(str).values
    _concentration = df_level4["Metadata_concentration"].astype(float).values
    _log10conc = df_level4["Metadata_log10conc"].astype(float).values
    _dataset = df_level4["Dataset"].astype(str).values

    # --- 2. Run UMAP for 2D projection
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    _embedding = _reducer.fit_transform(_X)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_level4 = pd.DataFrame(_embedding, columns=["UMAP_1", "UMAP_2"])
    df_umap_level4["Metadata_plate"] = _plates
    df_umap_level4["Metadata_concentration"] = _concentration
    df_umap_level4["Metadata_log10conc"] = _log10conc
    df_umap_level4["Dataset"] = _dataset
    return (df_umap_level4,)


@app.cell
def _(df_umap_level4, px):
    # --- 4. Create interactive Plotly scatter plot
    _fig = px.scatter(
        df_umap_level4,
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


@app.cell
def _(mo):
    mo.md(r"""### UMAP Cell Painting level 5""")
    return


@app.cell
def _(all_features_bsc_axiom, df_level5, pd, umap):
    _X = df_level5[all_features_bsc_axiom].values

    _dataset = df_level5["Dataset"].astype(str).values
    _compounds = df_level5["Metadata_compound"].astype(str).values
    # --- 2. Run UMAP for 2D projection
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    _embedding = _reducer.fit_transform(_X)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_level5 = pd.DataFrame(_embedding, columns=["UMAP_1", "UMAP_2"])
    df_umap_level5["Dataset"] = _dataset
    df_umap_level5["Metadata_compound"] = _compounds
    return (df_umap_level5,)


@app.cell
def _(Path, df_umap_level5, dic_botanicals_scode_name, go, pio, px):
    # ── Classify points ───────────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _classify_point(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    df_umap_level5["PointType"] = df_umap_level5["Metadata_compound"].apply(_classify_point)

    df_umap_level5["DisplayName"] = df_umap_level5["Metadata_compound"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c, _c) if _c in _botanical_scodes else _c
    )

    # ── Color per dataset ─────────────────────────────────────────────────────────
    _dataset_colors = {
        _ds: _col for _ds, _col in zip(
            df_umap_level5["Dataset"].unique(),
            px.colors.qualitative.Safe
        )
    }

    # ── Marker config per point type ──────────────────────────────────────────────
    _marker_config = {
        "Compound":  dict(symbol="circle",       size=6,  opacity=0.7),
        "Botanical": dict(symbol="diamond",      size=10, opacity=0.9),
        "DMSO":      dict(symbol="star",         size=14, opacity=1.0),
    }

    # ── Build figure manually with go.Scatter ─────────────────────────────────────
    _fig = go.Figure()

    for _dataset in df_umap_level5["Dataset"].unique():
        _df_ds = df_umap_level5[df_umap_level5["Dataset"] == _dataset]
        _color = _dataset_colors[_dataset]

        for _ptype, _mconf in _marker_config.items():
            _df_sub = _df_ds[_df_ds["PointType"] == _ptype]
            if _df_sub.empty:
                continue

            _fig.add_trace(go.Scatter(
                x=_df_sub["UMAP_1"],
                y=_df_sub["UMAP_2"],
                mode="markers",
                name=f"{_dataset} — {_ptype}",
                marker=dict(
                    symbol=_mconf["symbol"],
                    size=_mconf["size"],
                    opacity=_mconf["opacity"],
                    color=_color,
                    line=dict(width=0.8, color="black") if _ptype != "Compound" else dict(width=0),
                ),
                customdata=_df_sub[["Metadata_compound", "DisplayName"]].values,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "ID: %{customdata[0]}<br>"
                    f"Dataset: {_dataset}<br>"
                    "UMAP1: %{x:.2f} | UMAP2: %{y:.2f}"
                    "<extra></extra>"
                ),
            ))

    # ── Botanical text labels ─────────────────────────────────────────────────────
    _df_bot_labels = df_umap_level5[df_umap_level5["PointType"] == "Botanical"]
    _fig.add_trace(go.Scatter(
        x=_df_bot_labels["UMAP_1"],
        y=_df_bot_labels["UMAP_2"],
        mode="text",
        text=_df_bot_labels["DisplayName"],
        textposition="top center",
        textfont=dict(size=8, color="black"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────────
    _fig.update_layout(
        template="plotly_white",
        title="UMAP of Level 5 Cell Painting Profiles<br><sup>◆ Botanicals · ★ DMSO · ● Compounds</sup>",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title_text="Dataset — Type",
        width=950,
        height=700,
        legend=dict(
            itemsizing="constant",
            font=dict(size=10),
        )
    )
    # ── Save high-resolution image ────────────────────────────────────────────────
    pio.write_image(_fig, Path.home() / "Downloads" / "umap_level5_axiom_bsc.png", width=950, height=700, scale=6)
    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP Cell Painting level 6""")
    return


@app.cell
def _(all_features_bsc_axiom, df_level6, pd, umap):
    _X = df_level6[all_features_bsc_axiom].values

    _dataset = df_level6["Dataset"].astype(str)
    _compounds = df_level6["Metadata_compound"].astype(str)
    # --- 2. Run UMAP for 2D projection
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    _embedding = _reducer.fit_transform(_X)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_level6 = pd.DataFrame(_embedding, columns=["UMAP_1", "UMAP_2"])
    df_umap_level6["Dataset"] = _dataset
    df_umap_level6["Metadata_compound"] = _compounds
    return (df_umap_level6,)


@app.cell
def _(df_umap_level6, dic_botanicals_scode_name, go, px):
    # ── Classify points ───────────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _classify_point(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    df_umap_level6["PointType"] = df_umap_level6["Metadata_compound"].apply(_classify_point)

    df_umap_level6["DisplayName"] = df_umap_level6["Metadata_compound"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c, _c) if _c in _botanical_scodes else _c
    )

    # ── Color per dataset ─────────────────────────────────────────────────────────
    _dataset_colors = {
        _ds: _col for _ds, _col in zip(
            df_umap_level6["Dataset"].unique(),
            px.colors.qualitative.Safe
        )
    }

    # ── Marker config per point type ──────────────────────────────────────────────
    _marker_config = {
        "Compound":  dict(symbol="circle",       size=6,  opacity=0.7),
        "Botanical": dict(symbol="diamond",      size=10, opacity=0.9),
        "DMSO":      dict(symbol="star",         size=14, opacity=1.0),
    }

    # ── Build figure manually with go.Scatter ─────────────────────────────────────
    _fig = go.Figure()

    for _dataset in df_umap_level6["Dataset"].unique():
        _df_ds = df_umap_level6[df_umap_level6["Dataset"] == _dataset]
        _color = _dataset_colors[_dataset]

        for _ptype, _mconf in _marker_config.items():
            _df_sub = _df_ds[_df_ds["PointType"] == _ptype]
            if _df_sub.empty:
                continue

            _fig.add_trace(go.Scatter(
                x=_df_sub["UMAP_1"],
                y=_df_sub["UMAP_2"],
                mode="markers",
                name=f"{_dataset} — {_ptype}",
                marker=dict(
                    symbol=_mconf["symbol"],
                    size=_mconf["size"],
                    opacity=_mconf["opacity"],
                    color=_color,
                    line=dict(width=0.8, color="black") if _ptype != "Compound" else dict(width=0),
                ),
                customdata=_df_sub[["Metadata_compound", "DisplayName"]].values,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "ID: %{customdata[0]}<br>"
                    f"Dataset: {_dataset}<br>"
                    "UMAP1: %{x:.2f} | UMAP2: %{y:.2f}"
                    "<extra></extra>"
                ),
            ))

    # ── Botanical text labels ─────────────────────────────────────────────────────
    _df_bot_labels = df_umap_level6[df_umap_level6["PointType"] == "Botanical"]
    _fig.add_trace(go.Scatter(
        x=_df_bot_labels["UMAP_1"],
        y=_df_bot_labels["UMAP_2"],
        mode="text",
        text=_df_bot_labels["DisplayName"],
        textposition="top center",
        textfont=dict(size=8, color="black"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────────
    _fig.update_layout(
        template="plotly_white",
        title="UMAP of Level 6 Cell Painting Profiles<br><sup>◆ Botanicals · ★ DMSO · ● Compounds</sup>",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title_text="Dataset — Type",
        width=950,
        height=700,
        legend=dict(
            itemsizing="constant",
            font=dict(size=10),
        )
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP Cell Painting level 6 (colored by cytotoxicity)""")
    return


@app.cell
def _(pd):
    df_axiom_hits_prediction = pd.read_parquet("modules/07_classification/outputs/axiom_axiom_hits_predictions_batch.parquet")
    return (df_axiom_hits_prediction,)


@app.cell
def _(df_axiom_hits_prediction):
    df_axiom_hits_prediction
    return


@app.cell
def _(df_axiom_hits_prediction, df_umap_level6, dic_botanicals_scode_name):
    # ── Classify points ───────────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())
    def _classify_point(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    df_umap_level6["PointType"] = df_umap_level6["Metadata_compound"].apply(_classify_point)
    df_umap_level6["DisplayName"] = df_umap_level6["Metadata_compound"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c, _c) if _c in _botanical_scodes else _c
    )

    # ── Toxicity lookup (Axiom only) ──────────────────────────────────────────────
    _tox_lookup = (
        df_axiom_hits_prediction[df_axiom_hits_prediction["Metadata_Label"] == "LDH"]
        [["Metadata_compound", "y_actual"]]
        .drop_duplicates()
        .set_index("Metadata_compound")["y_actual"]
        .to_dict()
    )
    df_umap_level6["Tox"] = df_umap_level6["Metadata_compound"].map(_tox_lookup)  # NaN for non-Axiom
    df_umap_level6
    return


@app.cell
def _(df_umap_level6):
    df_umap_level6["Tox"].value_counts()
    return


@app.cell
def _(df_axiom_hits_prediction):
    df_axiom_hits_prediction[df_axiom_hits_prediction["Metadata_Label"] == "LDH"][["Metadata_compound", "y_actual"]].drop_duplicates()
    return


@app.cell
def _(
    df_axiom_hits_prediction,
    df_umap_level6,
    dic_botanicals_scode_name,
    go,
    px,
):
    # ── Classify points ───────────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())
    def _classify_point(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    df_umap_level6["PointType"] = df_umap_level6["Metadata_compound"].apply(_classify_point)
    df_umap_level6["DisplayName"] = df_umap_level6["Metadata_compound"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c, _c) if _c in _botanical_scodes else _c
    )

    # ── Toxicity lookup (Axiom only) ──────────────────────────────────────────────
    _tox_lookup = (
        df_axiom_hits_prediction[df_axiom_hits_prediction["Metadata_Label"] == "LDH"]
        [["Metadata_compound", "y_actual"]]
        .drop_duplicates()
        .set_index("Metadata_compound")["y_actual"]
        .to_dict()
    )
    df_umap_level6["Tox"] = df_umap_level6["Metadata_compound"].map(_tox_lookup)  # NaN for non-Axiom

    # ── Color per dataset ─────────────────────────────────────────────────────────
    _dataset_colors = {
        _ds: _col for _ds, _col in zip(
            df_umap_level6["Dataset"].unique(),
            px.colors.qualitative.Safe
        )
    }
    _tox_colors = {1: "black", 0: "#1f77b4"}  # red = toxic, blue = non-toxic

    # ── Marker config per point type ──────────────────────────────────────────────
    _marker_config = {
        "Compound":  dict(symbol="circle",  size=6,  opacity=0.7),
        "Botanical": dict(symbol="diamond", size=10, opacity=0.9),
        "DMSO":      dict(symbol="star",    size=14, opacity=1.0),
    }

    # ── Build figure ──────────────────────────────────────────────────────────────
    _fig = go.Figure()

    for _dataset in df_umap_level6["Dataset"].unique():
        _df_ds = df_umap_level6[df_umap_level6["Dataset"] == _dataset]
        _is_axiom = _dataset == "AXIOM"

        for _ptype, _mconf in _marker_config.items():
            _df_sub = _df_ds[_df_ds["PointType"] == _ptype]
            if _df_sub.empty:
                continue

            if _is_axiom:
                # Split Axiom points into toxic / non-toxic / unknown
                for _tox_val, _tox_label, _tox_color in [
                    (1,   "Toxic",     _tox_colors[1]),
                    (0,   "Non-toxic", _tox_colors[0]),
                    (None,"Unknown",   "#aaaaaa"),
                ]:
                    if _tox_val is None:
                        _df_tox = _df_sub[_df_sub["Tox"].isna()]
                    else:
                        _df_tox = _df_sub[_df_sub["Tox"] == _tox_val]
                    if _df_tox.empty:
                        continue
                    _fig.add_trace(go.Scatter(
                        x=_df_tox["UMAP_1"],
                        y=_df_tox["UMAP_2"],
                        mode="markers",
                        name=f"Axiom — {_ptype} — {_tox_label}",
                        marker=dict(
                            symbol=_mconf["symbol"],
                            size=_mconf["size"],
                            opacity=_mconf["opacity"],
                            color=_tox_color,
                            line=dict(width=0.8, color="black") if _ptype != "Compound" else dict(width=0),
                        ),
                        customdata=_df_tox[["Metadata_compound", "DisplayName", "Tox"]].values,
                        hovertemplate=(
                            "<b>%{customdata[1]}</b><br>"
                            "ID: %{customdata[0]}<br>"
                            f"Dataset: Axiom<br>"
                            "Toxicity (MTT): %{customdata[2]}<br>"
                            "UMAP1: %{x:.2f} | UMAP2: %{y:.2f}"
                            "<extra></extra>"
                        ),
                    ))
            else:
                # Non-Axiom datasets: original dataset color
                _fig.add_trace(go.Scatter(
                    x=_df_sub["UMAP_1"],
                    y=_df_sub["UMAP_2"],
                    mode="markers",
                    name=f"{_dataset} — {_ptype}",
                    marker=dict(
                        symbol=_mconf["symbol"],
                        size=_mconf["size"],
                        opacity=_mconf["opacity"],
                        color=_dataset_colors[_dataset],
                        line=dict(width=0.8, color="black") if _ptype != "Compound" else dict(width=0),
                    ),
                    customdata=_df_sub[["Metadata_compound", "DisplayName"]].values,
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "ID: %{customdata[0]}<br>"
                        f"Dataset: {_dataset}<br>"
                        "UMAP1: %{x:.2f} | UMAP2: %{y:.2f}"
                        "<extra></extra>"
                    ),
                ))

    # ── Botanical text labels ─────────────────────────────────────────────────────
    _df_bot_labels = df_umap_level6[df_umap_level6["PointType"] == "Botanical"]
    _fig.add_trace(go.Scatter(
        x=_df_bot_labels["UMAP_1"],
        y=_df_bot_labels["UMAP_2"],
        mode="text",
        text=_df_bot_labels["DisplayName"],
        textposition="top center",
        textfont=dict(size=8, color="black"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────────
    _fig.update_layout(
        template="plotly_white",
        title="UMAP of Level 6 Cell Painting Profiles<br><sup>◆ Botanicals · ★ DMSO · ● Compounds</sup>",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title_text="Dataset — Type",
        width=950,
        height=700,
        legend=dict(
            itemsizing="constant",
            font=dict(size=10),
        )
    )
    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP Cell Painting level 6 & level 5 (two panel plot)""")
    return


@app.cell
def _(
    Path,
    df_umap_level5,
    df_umap_level6,
    dic_botanicals_scode_name,
    go,
    pio,
):
    from plotly.subplots import make_subplots

    # ── Shared config ─────────────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _classify_point(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    # Scientific color palette — colorblind-safe
    _dataset_palette = [
        "#2166AC", "#D6604D", "#4DAC26", "#8073AC",
        "#E08214", "#1A9641", "#D01C8B", "#4393C3",
    ]

    _marker_config = {
        "Compound":  dict(symbol="circle",  size=5,  opacity=0.55, line_width=0),
        "Botanical": dict(symbol="diamond", size=11, opacity=0.95, line_width=0.8),
        "DMSO":      dict(symbol="star",    size=13, opacity=1.0,  line_width=0.8),
    }

    _scientific_axis = dict(
        showgrid=True,
        gridcolor="#e8e8e8",
        gridwidth=0.5,
        zeroline=False,
        showline=True,
        linecolor="#333333",
        linewidth=1.2,
        tickfont=dict(size=11, family="Arial"),
        title_font=dict(size=13, family="Arial"),
        mirror=True,
    )

    # ── Process both dataframes ───────────────────────────────────────────────────
    _panels = [
        (df_umap_level5, "Level 5 profiles"),
        (df_umap_level6, "Level 6 profiles"),
    ]

    for _df, _ in _panels:
        _df["PointType"]   = _df["Metadata_compound"].apply(_classify_point)
        _df["DisplayName"] = _df["Metadata_compound"].apply(
            lambda _c: dic_botanicals_scode_name.get(_c, _c) if _c in _botanical_scodes else _c
        )

    # Unified dataset color mapping across both panels
    _all_datasets = sorted(set(
        df_umap_level5["Dataset"].unique().tolist() +
        df_umap_level6["Dataset"].unique().tolist()
    ))
    _dataset_colors = {
        _ds: _dataset_palette[_i % len(_dataset_palette)]
        for _i, _ds in enumerate(_all_datasets)
    }

    # ── Build two-panel figure ────────────────────────────────────────────────────
    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["<b>A</b>  Level 5 profiles", "<b>B</b>  Level 6 profiles"],
        horizontal_spacing=0.08,
    )

    _legend_shown = set()  # track which (dataset, ptype) combos already have a legend entry

    for _col_idx, (_df, _panel_title) in enumerate(_panels, start=1):

        # ── Compound / Botanical / DMSO traces ───────────────────────────────────
        for _ptype in ["Compound", "Botanical", "DMSO"]:   # draw order: compounds first
            _mconf = _marker_config[_ptype]
            for _dataset in _all_datasets:
                _df_sub = _df[
                    (_df["Dataset"]    == _dataset) &
                    (_df["PointType"]  == _ptype)
                ]
                if _df_sub.empty:
                    continue

                _legend_key  = (_dataset, _ptype)
                _show_legend = _legend_key not in _legend_shown
                if _show_legend:
                    _legend_shown.add(_legend_key)

                _color = _dataset_colors[_dataset]

                _fig.add_trace(go.Scatter(
                    x=_df_sub["UMAP_1"],
                    y=_df_sub["UMAP_2"],
                    mode="markers",
                    name=f"{_dataset} — {_ptype}",
                    legendgroup=_legend_key[0],
                    showlegend=_show_legend,
                    marker=dict(
                        symbol=_mconf["symbol"],
                        size=_mconf["size"],
                        opacity=_mconf["opacity"],
                        color=_color,
                        line=dict(width=_mconf["line_width"], color="#222222"),
                    ),
                    customdata=_df_sub[["Metadata_compound", "DisplayName", "Dataset"]].values,
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "ID: %{customdata[0]}<br>"
                        "Dataset: %{customdata[2]}<br>"
                        "UMAP1: %{x:.2f} | UMAP2: %{y:.2f}"
                        "<extra></extra>"
                    ),
                ), row=1, col=_col_idx)

        # ── Botanical text labels ─────────────────────────────────────────────────
        _df_bot = _df[_df["PointType"] == "Botanical"].copy()
        # deduplicate labels at identical positions
        _df_bot = _df_bot.drop_duplicates(subset=["Metadata_compound"])

        _fig.add_trace(go.Scatter(
            x=_df_bot["UMAP_1"],
            y=_df_bot["UMAP_2"],
            mode="text",
            text=_df_bot["DisplayName"],
            textposition="top center",
            textfont=dict(size=8, family="Arial", color="#111111"),
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=_col_idx)

    # ── Apply scientific axis style to both panels ────────────────────────────────
    for _axis_prefix in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
        _is_x = _axis_prefix.startswith("x")
        _fig.update_layout(**{
            _axis_prefix: dict(
                **_scientific_axis,
                title=dict(text="UMAP 1" if _is_x else "UMAP 2"),
            )
        })

    # ── Legend: show point type via symbol in legend title ───────────────────────
    # Add invisible dummy traces to show symbol legend
    for _ptype, _symbol, _label in [
        ("Compound",  "circle",  "Compound  ●"),
        ("Botanical", "diamond", "Botanical  ◆"),
        ("DMSO",      "star",    "DMSO  ★"),
    ]:
        _fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol=_symbol, size=9, color="#555555",
                        line=dict(width=0.8, color="#222222")),
            name=f"<i>{_label}</i>",
            legendgroup="__type__",
            showlegend=True,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────────
    _fig.update_layout(
        template="plotly_white",
        title=dict(
            text=(
                "UMAP of Cell Painting Profiles<br>"
                "<sup>Morphological similarity of botanical extracts and pure constituents</sup>"
            ),
            font=dict(size=15, family="Arial", color="#111111"),
            x=0.5, xanchor="center",
        ),
        width=1400,
        height=650,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            title=dict(text="<b>Dataset / Type</b>", font=dict(size=11)),
            font=dict(size=10, family="Arial"),
            itemsizing="constant",
            bordercolor="#cccccc",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            tracegroupgap=4,
        ),
        margin=dict(l=60, r=180, t=90, b=60),
    )

    # ── Subplot title styling ─────────────────────────────────────────────────────
    for _ann in _fig.layout.annotations:
        _ann.update(font=dict(size=13, family="Arial", color="#111111"))

    # ── Export ────────────────────────────────────────────────────────────────────
    pio.write_image(
        _fig,
        Path.home() / "Downloads" / "umap_level5_6_bsc.png",
        width=1400, height=650, scale=4
    )
    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Analysis : Heatmap cosine similarity""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Analysis : Heatmap cosine similarity level 6""")
    return


@app.cell
def _(
    Patch,
    all_features_bsc_axiom,
    cosine_similarity,
    df_bsc_all_pod_annotations,
    df_level6,
    dic_botanicals_scode_name,
    pd,
    plt,
    sns,
):
    # ── 1. Cosine similarity matrix ───────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _classify_point(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    _df = df_level6.copy()
    _df["PointType"] = _df["Metadata_compound"].apply(_classify_point)

    _compound_to_plant = (
        df_bsc_all_pod_annotations[["Metadata_compound", "Plant name"]]
        .drop_duplicates()
        .dropna()
        .set_index("Metadata_compound")["Plant name"]
        .to_dict()
    )
    _df["Plant name"] = _df["Metadata_compound"].map(lambda _c: _compound_to_plant.get(_c, "—"))
    _constituent_lookup = (
        df_bsc_all_pod_annotations[["Metadata_compound", "Constituent"]]
        .drop_duplicates()
        .dropna(subset=["Constituent"])
        .set_index("Metadata_compound")["Constituent"]
        .to_dict()
    )

    _df["DisplayName"] = _df["Metadata_compound"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c)    if _c in _botanical_scodes
                   else _constituent_lookup.get(_c, _c)  if _c not in ("s0617825", "DMSO")
                   else _c
    )
    # ── 2. Cosine similarity matrix ───────────────────────────────────────────────
    _X = _df[all_features_bsc_axiom].values
    _cos_sim = cosine_similarity(_X)
    _df_cos = pd.DataFrame(
        _cos_sim,
        index=_df["DisplayName"].values,
        columns=_df["DisplayName"].values,
    )

    # ── 3. Row color annotations ──────────────────────────────────────────────────
    _dataset_palette = dict(zip(
        _df["Dataset"].unique(),
        sns.color_palette("Set1", len(_df["Dataset"].unique()))
    ))
    _ptype_palette = {
        "Compound":  "#aec7e8",
        "Botanical": "#2ca02c",
        "DMSO":      "#000000",
    }
    _unique_plants = [_p for _p in _df["Plant name"].unique() if _p != "—"]
    _plant_palette = dict(zip(
        _unique_plants,
        sns.color_palette("tab20", len(_unique_plants))
    ))
    _plant_palette["—"] = "#f0f0f0"

    _row_colors = pd.DataFrame({
        "Dataset":    _df["Dataset"].map(_dataset_palette).values,
        "Type":       _df["PointType"].map(_ptype_palette).values,
        "Plant name": _df["Plant name"].map(_plant_palette).values,
    }, index=_df_cos.index)
    _bsc_compounds = set(df_level6[df_level6["Dataset"] == "BSC"]["Metadata_compound"].values)

    _ytick_labels = [
        _label if _compound in _bsc_compounds else ""
        for _label, _compound in zip(_df["DisplayName"].values, _df["Metadata_compound"].values)
    ]
    _n_rows = len(_df_cos)
    # ── 4. Clustermap ─────────────────────────────────────────────────────────────
    sns.set(font_scale=0.65)
    _cg = sns.clustermap(
        _df_cos,
        method="ward",
        metric="euclidean",
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        figsize=(max(14, _n_rows * 0.1)+2, max(14, _n_rows * 0.1)),
        row_cluster=True,
        col_cluster=True,
        row_colors=_row_colors,
        xticklabels=False,
        yticklabels=_ytick_labels,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )
    _cg.ax_heatmap.set_yticklabels(
        _cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6
    )
    _cg.ax_heatmap.set_xlabel("Compound", fontsize=9)
    _cg.ax_heatmap.set_ylabel("Compound", fontsize=9)
    _cg.fig.suptitle(
        "Cosine similarity clustermap — Level 6 compound profiles",
        y=1.01, fontsize=12
    )

    # ── 5. Legends ────────────────────────────────────────────────────────────────
    _legend_dataset = [Patch(facecolor=_col, label=_ds) for _ds, _col in _dataset_palette.items()]
    _legend_ptype   = [Patch(facecolor=_col, label=_pt) for _pt, _col in _ptype_palette.items()]
    _legend_plant   = [Patch(facecolor=_col, label=_pl) for _pl, _col in _plant_palette.items() if _pl != "—"]

    _cg.ax_heatmap.legend(
        handles=_legend_dataset, title="Dataset",
        bbox_to_anchor=(1.35, 1.0), loc="upper left",
        frameon=True, fontsize=6, title_fontsize=7
    )
    _leg2 = _cg.fig.add_axes([0.82, 0.55, 0.15, 0.15])
    _leg2.axis("off")
    _leg2.legend(handles=_legend_ptype, title="Type", loc="upper left",
                 frameon=True, fontsize=6, title_fontsize=7)

    _leg3 = _cg.fig.add_axes([0.82, 0.25, 0.15, 0.25])
    _leg3.axis("off")
    _leg3.legend(handles=_legend_plant, title="Plant name", loc="upper left",
                 frameon=True, fontsize=6, title_fontsize=7)

    plt.tight_layout()
    _cg.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Classification BSC with Axiom (Toxcast models)""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Load models""")
    return


@app.cell
def _(glob, os, pickle):
    _models_path = "modules/07_classification/outputs/models_batch/"

    # Find all matching model files
    _model_files = glob.glob(os.path.join(_models_path, "*_actual_model.pkl"))

    loaded_models = {}

    for _file in _model_files:
        # Extract clean model name
        _filename = os.path.basename(_file)
        _model_name = _filename.replace("_actual_model.pkl", "")

        # Load model
        with open(_file, "rb") as f:
            loaded_models[_model_name] = pickle.load(f)
    return (loaded_models,)


@app.cell
def _(loaded_models):
    loaded_models.keys()
    return


@app.cell
def _(mo):
    mo.md(r"""### Predict BSC""")
    return


@app.cell
def _(df_level6_bsc_all, list_botanicals, loaded_models, pd, tqdm):
    # Subset once
    X_base = df_level6_bsc_all.loc[
        df_level6_bsc_all["Metadata_compound"].isin(list_botanicals)
    ].copy()

    # Prepare prediction matrix with compound names
    pred_matrix = pd.DataFrame({
        "Metadata_compound": X_base["Metadata_compound"].values
    }, index=X_base.index)

    # Loop with progress bar
    for model_name, model in tqdm(loaded_models.items(), 
                                  total=len(loaded_models), 
                                  desc="Predicting"):

        # Get training feature names
        trained_features = model.get_booster().feature_names

        # Align features exactly
        X_pred = X_base.reindex(columns=trained_features)

        # Predict
        preds = model.predict(X_pred)

        # Store predictions
        pred_matrix[model_name] = preds

    # Optional: reset index
    pred_matrix = pred_matrix.reset_index(drop=True)

    pred_matrix.head()
    return (pred_matrix,)


@app.cell
def _(mo):
    mo.md(r"""### Visualise BSC predictions""")
    return


@app.cell
def _(ListedColormap, df_bsc_all_pod_annotations):
    compound_to_plant_name = (
        df_bsc_all_pod_annotations[["Plant scode", "Plant name"]]
        .drop_duplicates()
        .dropna()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    binary_cmap = ListedColormap(["lightgray", "red"])
    return binary_cmap, compound_to_plant_name


@app.cell
def _(mo):
    mo.md(r"""#### All""")
    return


@app.cell
def _(binary_cmap, compound_to_plant_name, plt, pred_matrix, sns):
    # Set compound as index
    _heatmap_data = pred_matrix.set_index("Metadata_compound")
    # Map Metadata_compound (Plant scode) to Plant name
    _heatmap_data.index = _heatmap_data.index.map(compound_to_plant_name)

    plt.figure(figsize=(20,4))
    sns.heatmap(
        _heatmap_data,
        cmap=binary_cmap,
        cbar=False,           # optional: hide colorbar for binary
        linewidths=0.5,       # grid lines
        linecolor='white'
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Active vs Inactive Predictions")
    plt.show()
    return


@app.cell
def _(
    binary_cmap,
    compound_to_plant_name,
    linkage,
    plt,
    pred_matrix,
    sns,
    ssd,
):
    # Set compound as index
    _heatmap_data = pred_matrix.set_index("Metadata_compound")
    # Map Metadata_compound (Plant scode) to Plant name
    _heatmap_data.index = _heatmap_data.index.map(compound_to_plant_name)
    # Row clustering: compounds
    _row_dist = ssd.pdist(_heatmap_data.values, metric='jaccard')
    _row_linkage = linkage(_row_dist, method='average')  # 'ward' can't be used with Jaccard

    # Column clustering: models
    _col_dist = ssd.pdist(_heatmap_data.T.values, metric='jaccard')
    _col_linkage = linkage(_col_dist, method='average')

    # Plot clustered heatmap
    sns.clustermap(
        _heatmap_data,
        row_linkage=_row_linkage,
        col_linkage=_col_linkage,
        cmap=binary_cmap,
        cbar=False,
        linewidths=0.5,
        linecolor='white',
        figsize=(18, 6)
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""#### Cytotox""")
    return


@app.cell
def _(binary_cmap, compound_to_plant_name, plt, pred_matrix, sns):
    # Set compound as index
    _heatmap_data = (
        pred_matrix
        .set_index("Metadata_compound")          # step 1: index
        .loc[:, lambda df: df.columns.str.contains("tissue|MTT|LDH|cell_count")]  # step 2: filter
    )
    # Map Metadata_compound (Plant scode) to Plant name
    _heatmap_data.index = _heatmap_data.index.map(compound_to_plant_name)

    plt.figure(figsize=(6,4))
    sns.heatmap(
        _heatmap_data,
        cmap=binary_cmap,
        cbar=False,           # optional: hide colorbar for binary
        linewidths=0.5,       # grid lines
        linecolor='white'
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Active vs Inactive Predictions (Tissue level)")
    plt.show()
    return


@app.cell
def _(binary_cmap, compound_to_plant_name, plt, pred_matrix, sns):
    # Set compound as index
    _heatmap_data = (
        pred_matrix
        .set_index("Metadata_compound")          # step 1: index
        .loc[:, lambda df: df.columns.str.contains("cell|MTT|LDH|cell_count")]  # step 2: filter
    )
    # Map Metadata_compound (Plant scode) to Plant name
    _heatmap_data.index = _heatmap_data.index.map(compound_to_plant_name)

    plt.figure(figsize=(6,4))
    sns.heatmap(
        _heatmap_data,
        cmap=binary_cmap,
        cbar=False,           # optional: hide colorbar for binary
        linewidths=0.5,       # grid lines
        linecolor='white'
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Active vs Inactive Predictions (cell level)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Cell based""")
    return


@app.cell
def _(binary_cmap, compound_to_plant_name, plt, pred_matrix, sns):
    # Set compound as index
    _heatmap_data = (
        pred_matrix
        .set_index("Metadata_compound")          # step 1: index
        .loc[:, lambda df: ~df.columns.str.contains("cell|MTT|LDH|cell_count|NVS|tissue")]  # step 2: filter
    )
    # Map Metadata_compound (Plant scode) to Plant name
    _heatmap_data.index = _heatmap_data.index.map(compound_to_plant_name)

    plt.figure(figsize=(20,4))
    sns.heatmap(
        _heatmap_data,
        cmap=binary_cmap,
        cbar=False,           # optional: hide colorbar for binary
        linewidths=0.5,       # grid lines
        linecolor='white'
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Active vs Inactive Predictions (cell based)")
    plt.show()
    return


@app.cell
def _(binary_cmap, compound_to_plant_name, plt, pred_matrix, sns):
    # Set compound as index
    _heatmap_data = (
        pred_matrix
        .set_index("Metadata_compound")          # step 1: index
        .loc[:, lambda df: df.columns.str.contains("TOX21")]  # step 2: filter
    )
    # Map Metadata_compound (Plant scode) to Plant name
    _heatmap_data.index = _heatmap_data.index.map(compound_to_plant_name)

    plt.figure(figsize=(15,4))
    sns.heatmap(
        _heatmap_data,
        cmap=binary_cmap,
        cbar=False,           # optional: hide colorbar for binary
        linewidths=0.5,       # grid lines
        linecolor='white'
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Active vs Inactive Predictions (cell based)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Cell free""")
    return


@app.cell
def _(binary_cmap, compound_to_plant_name, plt, pred_matrix, sns):
    # Set compound as index
    _heatmap_data = (
        pred_matrix
        .set_index("Metadata_compound")          # step 1: index
        .loc[:, lambda df: df.columns.str.contains("NVS|CCTE_GLTED_hIYD")]  # step 2: filter
    )
    # Map Metadata_compound (Plant scode) to Plant name
    _heatmap_data.index = _heatmap_data.index.map(compound_to_plant_name)

    plt.figure(figsize=(6,4))
    sns.heatmap(
        _heatmap_data,
        cmap=binary_cmap,
        cbar=False,           # optional: hide colorbar for binary
        linewidths=0.5,       # grid lines
        linecolor='white'
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Active vs Inactive Predictions (cell free)")
    plt.show()
    return


@app.cell
def _(binary_cmap, compound_to_plant_name, plt, pred_matrix, sns):
    # Set compound as index
    _heatmap_data = (
        pred_matrix
        .set_index("Metadata_compound")          # step 1: index
        .loc[:, lambda df: df.columns.str.contains("NVS_TR_hNET|NVS_TR_hDAT|NVS_ADME_hCYP2C19|NVS_NR_hGR|NVS_MP_hPBR")]  # step 2: filter
    )
    # Map Metadata_compound (Plant scode) to Plant name
    _heatmap_data.index = _heatmap_data.index.map(compound_to_plant_name)

    plt.figure(figsize=(4,4))
    sns.heatmap(
        _heatmap_data,
        cmap=binary_cmap,
        cbar=False,           # optional: hide colorbar for binary
        linewidths=0.5,       # grid lines
        linecolor='white'
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Active vs Inactive Predictions (cell free)")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""#### CYP""")
    return


@app.cell
def _(binary_cmap, compound_to_plant_name, plt, pred_matrix, sns):
    # Set compound as index
    _heatmap_data = (
        pred_matrix
        .set_index("Metadata_compound")          # step 1: index
        .loc[:, lambda df: df.columns.str.contains("kidney|HEK|HK|498|TERT1", case=False)]  # step 2: filter
    )
    # Map Metadata_compound (Plant scode) to Plant name
    _heatmap_data.index = _heatmap_data.index.map(compound_to_plant_name)

    plt.figure(figsize=(6,4))
    sns.heatmap(
        _heatmap_data,
        cmap=binary_cmap,
        cbar=False,           # optional: hide colorbar for binary
        linewidths=0.5,       # grid lines
        linecolor='white'
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Active vs Inactive Predictions (Tissue level)")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Predict BSC & constituents""")
    return


@app.cell
def _(df_level6_bsc_all, loaded_models, pd, tqdm):
    # Subset once
    _X_all = df_level6_bsc_all.copy()

    # Prepare prediction matrix with compound names
    pred_matrix_all = pd.DataFrame({
        "Metadata_compound": _X_all["Metadata_compound"].values
    }, index=_X_all.index)

    # Loop with progress bar
    for _model_name, _model in tqdm(loaded_models.items(), 
                                  total=len(loaded_models), 
                                  desc="Predicting"):

        # Get training feature names
        _trained_features = _model.get_booster().feature_names

        # Align features exactly
        _X_pred = _X_all.reindex(columns=_trained_features)

        # Predict
        _preds = _model.predict(_X_pred)

        # Store predictions
        pred_matrix_all[_model_name] = _preds

    # Optional: reset index
    pred_matrix_all = pred_matrix_all.reset_index(drop=True)

    pred_matrix_all.head()
    return (pred_matrix_all,)


@app.cell
def _(pred_matrix_all):
    pred_matrix_all
    return


@app.cell
def _(mo):
    mo.md(r"""#### All""")
    return


@app.cell
def _():

    from scipy.cluster.hierarchy import linkage
    import scipy.spatial.distance as ssd
    return linkage, ssd


@app.cell
def _(
    binary_cmap,
    compound_to_plant_name,
    linkage,
    plt,
    pred_matrix_all,
    sns,
    ssd,
):
    # Set compound as index
    _heatmap_data = pred_matrix_all.set_index("Metadata_compound")
    # Map Metadata_compound (Plant scode) to Plant name
    _heatmap_data.index = _heatmap_data.index.map(lambda x: compound_to_plant_name.get(x, x))
    # Row clustering: compounds
    _row_dist = ssd.pdist(_heatmap_data.values, metric='jaccard')
    _row_linkage = linkage(_row_dist, method='average')  # 'ward' can't be used with Jaccard

    # Column clustering: models
    _col_dist = ssd.pdist(_heatmap_data.T.values, metric='jaccard')
    _col_linkage = linkage(_col_dist, method='average')

    # Plot clustered heatmap
    sns.clustermap(
        _heatmap_data,
        row_linkage=_row_linkage,
        col_linkage=_col_linkage,
        cmap=binary_cmap,
        cbar=False,
        linewidths=0.5,
        linecolor='white',
        figsize=(18, 16)
    )
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Predict BSC & constituents significative""")
    return


@app.cell
def _():
    from scipy.cluster.hierarchy import leaves_list
    return (leaves_list,)


@app.cell
def _(pd):
    df_ttest_results = pd.read_csv("modules/07_classification/outputs/results_performance.csv")
    df_ttest_results
    return (df_ttest_results,)


@app.cell
def _(df_ttest_results):
    # Filter rows labels : cell_count, cell free assays, cytotox
    # Filter df_ttest_results based on Metadata_Label criteria:
    # - Exclude rows where Metadata_Label == "cell_count"
    # - Exclude rows where Metadata_Label starts with "NVS_"
    # - Exclude rows containing "tissue" or "cell_type"
    # - EXCEPT include "cell_type__HepG2"

    df_ttest_results_filtered = df_ttest_results[
        # Exclude cell_count
        (df_ttest_results["Metadata_Label"] != "cell_count") &
        # Exclude NVS_* labels
        (~df_ttest_results["Metadata_Label"].str.startswith("NVS_", na=False)) &
        # Exclude rows with "tissue" or "cell_type" UNLESS it's "cell_type__HepG2"
        (
            ~(
                (df_ttest_results["Metadata_Label"].str.contains("tissue", case=False, na=False)) |
                (df_ttest_results["Metadata_Label"].str.contains("cell_type", case=False, na=False))
            ) |
            (df_ttest_results["Metadata_Label"] == "cell_type__HepG2")
        )
    ].copy()

    print(f"Original df_ttest_results: {len(df_ttest_results)} rows")
    print(f"Filtered df_ttest_results: {len(df_ttest_results_filtered)} rows\n")

    print("Filtered results:")
    print(df_ttest_results_filtered)

    # Optional: Show what was filtered out
    print("\n" + "="*80)
    print("FILTERED OUT LABELS:")
    print("="*80)
    filtered_out = df_ttest_results[~df_ttest_results.index.isin(df_ttest_results_filtered.index)]
    print(f"Total removed: {len(filtered_out)} rows")
    print("\nRemoved labels:")
    for label in sorted(filtered_out["Metadata_Label"].unique()):
        print(f"  - {label}")
    return (df_ttest_results_filtered,)


@app.cell
def _(
    binary_cmap,
    df_bsc_annotations,
    df_ttest_results_filtered,
    dic_botanicals_scode_name,
    leaves_list,
    linkage,
    list_botanicals,
    mpatches,
    np,
    os,
    pd,
    plt,
    pred_matrix_all,
):
    n_active_filter = 20
    # ── Filter columns to those present in df_ttest_results_filtered["Metadata_Label"] ────
    _valid_labels = set(df_ttest_results_filtered["Metadata_Label"].values)

    # Keep only heatmap columns that appear in df_ttest_results_filtered
    _pred_matrix_all = pred_matrix_all.rename(columns={"MTT": "MT"})
    _all_cols_before_filter = list(_pred_matrix_all.columns)
    _all_cols_before_filter = [c for c in _all_cols_before_filter if c != "Metadata_compound"]
    _filtered_cols = [c for c in _all_cols_before_filter if c in _valid_labels]

    # ── Filter to labels with n_active > n_active_filter ──────────────────────────────────────
    _active_lookup = df_ttest_results_filtered.set_index("Metadata_Label")["Metadata_Count_1"]
    _filtered_cols = [c for c in _filtered_cols if _active_lookup.get(c, 0) > n_active_filter]
    print(_filtered_cols)
    # Rebuild pred_matrix_all with only filtered columns
    pred_matrix_filtered = _pred_matrix_all[["Metadata_compound"] + _filtered_cols]

    # ── Build ordered row index ───────────────────────────────────────────────────
    def _cluster_row_indices(data_matrix, indices):
        """Return row indices reordered by hierarchical clustering, or original if too few."""
        if len(indices) < 2:
            return indices
        sub = data_matrix[indices, :]
        valid_rows = [i for i, idx in enumerate(indices) if not np.all(np.isnan(sub[i, :]))]
        if len(valid_rows) < 2:
            return indices
        sub_valid = np.nan_to_num(sub[valid_rows, :], nan=0.0)
        Z = linkage(sub_valid, method="average", metric="jaccard")
        order = leaves_list(Z)
        return [indices[valid_rows[o]] for o in order]

    # ── Build ordered row index with clustering within botanical groups ────────────
    _rows_ordered = []

    # Build a temporary matrix indexed by scode for clustering
    _all_scodes_in_matrix = list(
        pred_matrix_filtered.set_index("Metadata_compound").index
    )
    _scode_to_rowidx = {s: i for i, s in enumerate(_all_scodes_in_matrix)}
    _matrix_values = pred_matrix_filtered.set_index("Metadata_compound").values.astype(float)

    for _scode in list_botanicals:
        _plant_name = dic_botanicals_scode_name.get(_scode, _scode)

        # Collect constituent scodes for this botanical
        _const_scodes = df_bsc_annotations[
            df_bsc_annotations["Plant scode"] == _scode
        ][["Metadata_compound", "Constituent"]].drop_duplicates()

        _const_rows = []
        for _, _crow in _const_scodes.iterrows():
            if _crow["Metadata_compound"] in pred_matrix_filtered["Metadata_compound"].values:
                _const_rows.append((_crow["Metadata_compound"], f"  {_crow['Constituent']}"))

        # Only add the botanical header if it or any constituent is present
        has_botanical = _scode in pred_matrix_filtered["Metadata_compound"].values
        if not has_botanical and len(_const_rows) == 0:
            continue

        if has_botanical:
            _rows_ordered.append((_scode, _plant_name, True))

        if len(_const_rows) > 0:
            # Cluster constituents by their activity row vectors
            _const_matrix_indices = [
                _scode_to_rowidx[s] for s, _ in _const_rows
                if s in _scode_to_rowidx
            ]
            _clustered_idx = _cluster_row_indices(_matrix_values, _const_matrix_indices)

            # Map back to (scode, label) pairs
            _idx_to_row = {_scode_to_rowidx[s]: (s, lbl) for s, lbl in _const_rows if s in _scode_to_rowidx}
            for _ridx in _clustered_idx:
                _s, _lbl = _idx_to_row[_ridx]
                _rows_ordered.append((_s, _lbl, False))

    _ordered_scodes = [r[0] for r in _rows_ordered]
    _ordered_labels = [r[1] for r in _rows_ordered]
    _is_botanical   = [r[2] for r in _rows_ordered]

    # ── Build heatmap data ────────────────────────────────────────────────────────
    _heatmap_data = pred_matrix_filtered.set_index("Metadata_compound").reindex(_ordered_scodes)
    _heatmap_data.index = _ordered_labels

    # ── Assign columns to groups ──────────────────────────────────────────────────
    _cols = pd.Series(_heatmap_data.columns)

    _mask_cell_free  = _cols.str.contains("NVS|CCTE_GLTED_hIYD", regex=True)
    _mask_hit_calls  = _cols.str.contains("MT|LDH|cell_count", regex=True)
    _mask_cytotoxic  = _cols.str.contains("cell|tissue", regex=True) & ~_mask_hit_calls
    _mask_cell_based = ~_mask_cell_free & ~_mask_hit_calls & ~_mask_cytotoxic

    _group_order = ["Cell-free", "Cell-based", "Cytotoxic", "Hit calls"]
    _group_masks = {
        "Cell-free":  _mask_cell_free.values,
        "Cell-based": _mask_cell_based.values,
        "Cytotoxic":  _mask_cytotoxic.values,
        "Hit calls":  _mask_hit_calls.values,
    }

    _SOURCE_COLORS = {
        "Hits":       "#4C72B0",
        "Cell-based": "#DD8452",
        "Cell-free":  "#55A868",
        "Cytotox":    "#C44E52",
    }
    _LABEL_MAP = {
        "Hits":       "Cytotox",
        "Cell-based": "ToxCast cell-based",
        "Cell-free":  "ToxCast cell-free",
        "Cytotox":    "ToxCast cytotox",
    }

    # Replace _group_colors with _SOURCE_COLORS mapped via _LABEL_MAP
    _group_colors = {
        "Cell-free":  _SOURCE_COLORS["Cell-free"],
        "Cell-based": _SOURCE_COLORS["Cell-based"],
        "Cytotoxic":  _SOURCE_COLORS["Cytotox"],
        "Hit calls":  _SOURCE_COLORS["Hits"],
    }
    _group_display = {
        "Cell-free":  _LABEL_MAP["Cell-free"],
        "Cell-based": _LABEL_MAP["Cell-based"],
        "Cytotoxic":  _LABEL_MAP["Cytotox"],
        "Hit calls":  _LABEL_MAP["Hits"],
    }

    # ── Reorder columns by group + cluster within each group ─────────────────────
    def _cluster_indices(data_matrix, indices):
        """Return indices reordered by hierarchical clustering, or original if too few."""
        if len(indices) < 2:
            return indices
        sub = data_matrix[:, indices]
        # Drop all-NaN columns before clustering
        valid = [i for i, idx in enumerate(indices) if not np.all(np.isnan(sub[:, i]))]
        if len(valid) < 2:
            return indices
        sub_valid = sub[:, valid]
        sub_valid = np.nan_to_num(sub_valid, nan=0.0)
        Z = linkage(sub_valid.T, method="average", metric="jaccard")
        order = leaves_list(Z)
        return [indices[valid[o]] for o in order]

    _col_indices_ordered = []
    _col_group_labels    = []
    for _grp in _group_order:
        _idx = list(np.where(_group_masks[_grp])[0])
        _idx_clustered = _cluster_indices(_heatmap_data.values, _idx)
        _col_indices_ordered.extend(_idx_clustered)
        _col_group_labels.extend([_grp] * len(_idx_clustered))

    _heatmap_data     = _heatmap_data.iloc[:, _col_indices_ordered]
    _col_group_labels = [_col_group_labels[i] for i in range(len(_col_indices_ordered))]

    # ── Build AUROC values aligned to (reordered) heatmap columns ────────────────
    _auroc_lookup = df_ttest_results_filtered.set_index("Metadata_Label")["AUROC_mean_Actual"]
    _auroc_values = np.array([
        _auroc_lookup.get(col, np.nan) for col in _heatmap_data.columns
    ])
    _bar_colors = [_group_colors[g] for g in _col_group_labels]

    # ── Row separators: spacing between botanical groups ──────────────────────────
    _y_positions = []
    _y_labels    = []
    _y_is_bot    = []
    _gap         = 0.4
    _y           = 0.0

    for _i, (_label, _is_bot) in enumerate(zip(_ordered_labels, _is_botanical)):
        if _is_bot and _i > 0:
            _y += _gap
        _y_positions.append(_y)
        _y_labels.append(_label)
        _y_is_bot.append(_is_bot)
        _y += 1.0

    _y_positions  = np.array(_y_positions)
    _total_height = _y_positions[-1] + 1.0

    # ── Plot layout: band | barplot | heatmap ─────────────────────────────────────
    _n_rows = len(_heatmap_data)
    _n_cols = len(_heatmap_data.columns)

    _fig = plt.figure(figsize=(max(14, _n_cols * 0.1), max(10, _total_height * 0.16) + 2))
    _gs = _fig.add_gridspec(
        2, 1,
        height_ratios=[0.01, 1],   # band | heatmap
        hspace=0.005
    )
    _ax_band = _fig.add_subplot(_gs[0])
    _ax      = _fig.add_subplot(_gs[1])


    # ── Color band (no text) ──────────────────────────────────────────────────────
    _band_colors_list = [_group_colors[_g] for _g in _col_group_labels]
    _ax_band.imshow(
        [list(range(_n_cols))],
        aspect="auto",
        cmap=plt.matplotlib.colors.ListedColormap(_band_colors_list),
        interpolation="none"
    )
    _ax_band.set_xlim(-0.5, _n_cols - 0.5)

    # Put x labels on band axis
    _ax_band.xaxis.set_label_position("top")
    _ax_band.xaxis.tick_top()
    _ax_band.set_xlim(-0.5, _n_cols - 0.5)

    _ax_band.spines[["left", "right", "bottom"]].set_visible(False)
    _ax_band.spines["top"].set_visible(False)
    _ax_band.set_yticks([])
    _ax_band.set_xticks([])
    # ── Separator lines on band ───────────────────────────────────────────────────
    _prev_grp = _col_group_labels[0]
    for _ci, _grp in enumerate(_col_group_labels):
        if _grp != _prev_grp:
            _ax_band.axvline(_ci - 0.5, color="black", linewidth=1.5, zorder=5)
            _prev_grp = _grp

    # ── Legend ────────────────────────────────────────────────────────────────────
    _legend_patches = [
        mpatches.Patch(facecolor=_group_colors[_grp], label=_group_display.get(_grp, _grp))
        for _grp in _group_order
    ]
    _ax_band.legend(
        handles=_legend_patches,
        loc="upper left",
        bbox_to_anchor=(0, -0.8),
        ncol=len(_group_order),
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=9,
        borderpad=0.6,
        title="Assays",
        title_fontsize=10,
    ).get_title().set_fontweight("bold")

    # ── Heatmap rows ──────────────────────────────────────────────────────────────
    for _ri, (_y, _is_bot) in enumerate(zip(_y_positions, _y_is_bot)):
        _row_vals = _heatmap_data.values[_ri:_ri+1, :]
        _ax.pcolormesh(
            np.arange(_n_cols + 1),
            np.array([_y, _y + 1.0]),
            _row_vals,
            cmap=binary_cmap,
            edgecolors="white",
            linewidth=0.3,
            vmin=0, vmax=1
        )
        if _is_bot:
            _ax.axhspan(_y, _y + 1.0, color="lightgrey", alpha=0.35, zorder=0)

    # ── y axis ────────────────────────────────────────────────────────────────────
    _ax.set_yticks(_y_positions + 0.25)
    _ax.set_yticklabels(_y_labels, fontsize=9)
    for _tick, _is_bot in zip(_ax.get_yticklabels(), _y_is_bot):
        if _is_bot:
            _tick.set_fontweight("bold")
            _tick.set_fontsize(11)
            _tick.set_color("black")
        else:
            _tick.set_fontsize(9)
            _tick.set_color("#444444")

    _ax.set_ylim(0, _total_height)
    _ax.invert_yaxis()

    # ── x axis ────────────────────────────────────────────────────────────────────
    _ax.set_xticks(np.arange(_n_cols) + 0.25)
    _ax.set_xticklabels(_heatmap_data.columns, rotation=90, fontsize=8)
    _ax.xaxis.set_label_position("top")
    _ax.xaxis.tick_top()
    _ax.tick_params(axis="x", length=0, pad=17)  # ← pad pushes labels up above the band
    # ── Column group separator lines on heatmap ───────────────────────────────────
    _prev_grp = _col_group_labels[0]
    for _ci, _grp in enumerate(_col_group_labels):
        if _grp != _prev_grp:
            _ax.axvline(_ci, color="black", linewidth=1.5, zorder=5)
            _prev_grp = _grp
    # ── Legend ────────────────────────────────────────────────────────────────────
    _legend_patches = [
        mpatches.Patch(facecolor=_group_colors[_grp], label=_group_display.get(_grp, _grp))
        for _grp in _group_order
    ]
    _ax.legend(
        handles=_legend_patches,
        loc="upper left",
        bbox_to_anchor=(-0.005, 0),
        ncol=len(_group_order),
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=9,
        borderpad=0.6,
        title="Assays",
        title_fontsize=10,
    ).get_title().set_fontweight("bold")

    _fig.savefig(os.path.expanduser("~/Downloads/prediction_heatmap_botanical_constituents_batch.png"),
                 bbox_inches="tight", dpi=300)
    _fig
    return (pred_matrix_filtered,)


@app.cell
def _(mo):
    mo.md(r"""### Predict BSC & constituents significative with claudeai categories""")
    return


@app.cell
def _(pd):
    # load data 
    df_toxcast_cellbased_assay_info_significant = pd.read_csv("modules/07_classification/outputs/df_toxcast_cellbased_assay_info_significant_claudeai_categories.csv")
    df_toxcast_cellbased_assay_info_significant
    return (df_toxcast_cellbased_assay_info_significant,)


@app.cell
def _(
    binary_cmap,
    df_bsc_annotations,
    df_toxcast_cellbased_assay_info_significant,
    df_ttest_results_filtered,
    dic_botanicals_scode_name,
    leaves_list,
    linkage,
    list_botanicals,
    mpatches,
    np,
    os,
    pd,
    plt,
    pred_matrix_filtered,
):
    # ── Build ordered row index ───────────────────────────────────────────────────
    def _cluster_row_indices(data_matrix, indices):
        """Return row indices reordered by hierarchical clustering, or original if too few."""
        if len(indices) < 2:
            return indices
        sub = data_matrix[indices, :]
        valid_rows = [i for i, idx in enumerate(indices) if not np.all(np.isnan(sub[i, :]))]
        if len(valid_rows) < 2:
            return indices
        sub_valid = np.nan_to_num(sub[valid_rows, :], nan=0.0)
        Z = linkage(sub_valid, method="average", metric="jaccard")
        order = leaves_list(Z)
        return [indices[valid_rows[o]] for o in order]

    # ── Build ordered row index with clustering within botanical groups ────────────
    _rows_ordered = []

    # Build a temporary matrix indexed by scode for clustering
    _all_scodes_in_matrix = list(
        pred_matrix_filtered.set_index("Metadata_compound").index
    )
    _scode_to_rowidx = {s: i for i, s in enumerate(_all_scodes_in_matrix)}
    _matrix_values = pred_matrix_filtered.set_index("Metadata_compound").values.astype(float)

    for _scode in list_botanicals:
        _plant_name = dic_botanicals_scode_name.get(_scode, _scode)

        # Collect constituent scodes for this botanical
        _const_scodes = df_bsc_annotations[
            df_bsc_annotations["Plant scode"] == _scode
        ][["Metadata_compound", "Constituent"]].drop_duplicates()

        _const_rows = []
        for _, _crow in _const_scodes.iterrows():
            if _crow["Metadata_compound"] in pred_matrix_filtered["Metadata_compound"].values:
                _const_rows.append((_crow["Metadata_compound"], f"  {_crow['Constituent']}"))

        # Only add the botanical header if it or any constituent is present
        _has_botanical = _scode in pred_matrix_filtered["Metadata_compound"].values
        if not _has_botanical and len(_const_rows) == 0:
            continue

        if _has_botanical:
            _rows_ordered.append((_scode, _plant_name, True))

        if len(_const_rows) > 0:
            # Cluster constituents by their activity row vectors
            _const_matrix_indices = [
                _scode_to_rowidx[s] for s, _ in _const_rows
                if s in _scode_to_rowidx
            ]
            _clustered_idx = _cluster_row_indices(_matrix_values, _const_matrix_indices)

            # Map back to (scode, label) pairs
            _idx_to_row = {_scode_to_rowidx[s]: (s, lbl) for s, lbl in _const_rows if s in _scode_to_rowidx}
            for _ridx in _clustered_idx:
                _s, _lbl = _idx_to_row[_ridx]
                _rows_ordered.append((_s, _lbl, False))

    _ordered_scodes = [r[0] for r in _rows_ordered]
    _ordered_labels = [r[1] for r in _rows_ordered]
    _is_botanical   = [r[2] for r in _rows_ordered]

    # ── Build heatmap data ────────────────────────────────────────────────────────
    _heatmap_data = pred_matrix_filtered.set_index("Metadata_compound").reindex(_ordered_scodes)
    _heatmap_data.index = _ordered_labels

    # ── Assign columns to groups ──────────────────────────────────────────────────
    _cols = pd.Series(_heatmap_data.columns)

    _mask_cell_free  = _cols.str.contains("NVS|CCTE_GLTED_hIYD", regex=True)
    _mask_hit_calls  = _cols.str.contains("MT|LDH|cell_count", regex=True)
    _mask_cytotoxic  = _cols.str.contains("cell|tissue", regex=True) & ~_mask_hit_calls
    _mask_cell_based = ~_mask_cell_free & ~_mask_hit_calls & ~_mask_cytotoxic

    _group_order = ["Cell-free", "Cell-based", "Cytotoxic", "Hit calls"]
    _group_masks = {
        "Cell-free":  _mask_cell_free.values,
        "Cell-based": _mask_cell_based.values,
        "Cytotoxic":  _mask_cytotoxic.values,
        "Hit calls":  _mask_hit_calls.values,
    }

    _SOURCE_COLORS = {
        "Hits":       "#4C72B0",
        "Cell-based": "#DD8452",
        "Cell-free":  "#55A868",
        "Cytotox":    "#C44E52",
    }
    _LABEL_MAP = {
        "Hits":       "Cytotox",
        "Cell-based": "ToxCast cell-based",
        "Cell-free":  "ToxCast cell-free",
        "Cytotox":    "ToxCast cytotox",
    }

    # Replace _group_colors with _SOURCE_COLORS mapped via _LABEL_MAP
    _group_colors = {
        "Cell-free":  _SOURCE_COLORS["Cell-free"],
        "Cell-based": _SOURCE_COLORS["Cell-based"],
        "Cytotoxic":  _SOURCE_COLORS["Cytotox"],
        "Hit calls":  _SOURCE_COLORS["Hits"],
    }
    _group_display = {
        "Cell-free":  _LABEL_MAP["Cell-free"],
        "Cell-based": _LABEL_MAP["Cell-based"],
        "Cytotoxic":  _LABEL_MAP["Cytotox"],
        "Hit calls":  _LABEL_MAP["Hits"],
    }

    # ── Reorder columns by group + claudeai_label within Cell-based (no clustering) ──
    _col_indices_ordered = []
    _col_group_labels    = []

    for _grp in _group_order:
        _idx = list(np.where(_group_masks[_grp])[0])

        if _grp == "Cell-based":
            # Sort by claudeai_label instead of clustering
            _cols_in_grp = [_heatmap_data.columns[i] for i in _idx]
            _label_order = (
                df_toxcast_cellbased_assay_info_significant
                .drop_duplicates("assay_component_endpoint_name")
                .set_index("assay_component_endpoint_name")["claudeai_label"]
            )
            _idx_sorted = sorted(
                _idx,
                key=lambda i: _label_order.get(_heatmap_data.columns[i], "zzz")
            )
            _col_indices_ordered.extend(_idx_sorted)
        else:
            # Keep original order for other groups (no clustering)
            _col_indices_ordered.extend(_idx)

        _col_group_labels.extend([_grp] * len(_idx))

    _heatmap_data     = _heatmap_data.iloc[:, _col_indices_ordered]
    _col_group_labels = [_col_group_labels[i] for i in range(len(_col_indices_ordered))]

    # ── Build AUROC values aligned to (reordered) heatmap columns ────────────────
    _auroc_lookup = df_ttest_results_filtered.set_index("Metadata_Label")["AUROC_mean_Actual"]
    _auroc_values = np.array([
        _auroc_lookup.get(col, np.nan) for col in _heatmap_data.columns
    ])
    _bar_colors = [_group_colors[g] for g in _col_group_labels]

    # ── Row separators: spacing between botanical groups ──────────────────────────
    _y_positions = []
    _y_labels    = []
    _y_is_bot    = []
    _gap         = 0.4
    _y           = 0.0

    for _i, (_label, _is_bot) in enumerate(zip(_ordered_labels, _is_botanical)):
        if _is_bot and _i > 0:
            _y += _gap
        _y_positions.append(_y)
        _y_labels.append(_label)
        _y_is_bot.append(_is_bot)
        _y += 1.0

    _y_positions  = np.array(_y_positions)
    _total_height = _y_positions[-1] + 1.0
    # ── Category annotation lookup for cell-based columns ────────────────────────
    _cat_color_lookup = (
        df_toxcast_cellbased_assay_info_significant
        .drop_duplicates("assay_component_endpoint_name")
        .set_index("assay_component_endpoint_name")[["claudeai_color", "claudeai_label"]]
        .to_dict(orient="index")
    )

    # Per-column category color and label (NaN for non-cell-based columns)
    _col_cat_colors = []
    _col_cat_labels = []
    for _col in _heatmap_data.columns:
        _info = _cat_color_lookup.get(_col)
        _col_cat_colors.append(_info["claudeai_color"] if _info else "#dddddd")
        _col_cat_labels.append(_info["claudeai_label"] if _info else None)
    # ── Plot layout: band | barplot | heatmap ─────────────────────────────────────
    _n_rows = len(_heatmap_data)
    _n_cols = len(_heatmap_data.columns)

    _fig = plt.figure(figsize=(max(14, _n_cols * 0.1), max(10, _total_height * 0.16) + 2))
    _gs = _fig.add_gridspec(
        3, 1,
        height_ratios=[0.01, 0.01, 1],   # group band | category band | heatmap
        hspace=0.005
    )
    _ax_band     = _fig.add_subplot(_gs[1])
    _ax_cat_band = _fig.add_subplot(_gs[0])
    _ax          = _fig.add_subplot(_gs[2])


    # ── Color band (no text) ──────────────────────────────────────────────────────
    _band_colors_list = [_group_colors[_g] for _g in _col_group_labels]
    _ax_band.imshow(
        [list(range(_n_cols))],
        aspect="auto",
        cmap=plt.matplotlib.colors.ListedColormap(_band_colors_list),
        interpolation="none"
    )
    _ax_band.set_xlim(-0.5, _n_cols - 0.5)

    # Put x labels on band axis
    _ax_band.xaxis.set_label_position("top")
    _ax_band.xaxis.tick_top()
    _ax_band.set_xlim(-0.5, _n_cols - 0.5)

    _ax_band.spines[["left", "right", "bottom"]].set_visible(False)
    _ax_band.spines["top"].set_visible(False)
    _ax_band.set_yticks([])
    _ax_band.set_xticks([])
    # ── Separator lines on band ───────────────────────────────────────────────────
    _prev_grp = _col_group_labels[0]
    for _ci, _grp in enumerate(_col_group_labels):
        if _grp != _prev_grp:
            _ax_band.axvline(_ci - 0.5, color="black", linewidth=1.5, zorder=5)
            _prev_grp = _grp
    # ── Category color band ───────────────────────────────────────────────────────
    _ax_cat_band.imshow(
        [list(range(_n_cols))],
        aspect="auto",
        cmap=plt.matplotlib.colors.ListedColormap(_col_cat_colors),
        interpolation="none"
    )
    _ax_cat_band.set_xlim(-0.5, _n_cols - 0.5)
    _ax_cat_band.set_yticks([])
    _ax_cat_band.set_xticks([])
    _ax_cat_band.spines[["left", "right", "bottom", "top"]].set_visible(False)

    # Separator lines between claudeai_label groups within cell-based
    _prev_cat = _col_cat_labels[0]
    for _ci, _cat in enumerate(_col_cat_labels):
        if _cat != _prev_cat:
            _ax_cat_band.axvline(_ci - 0.5, color="black", linewidth=1.0, zorder=5)
            _prev_cat = _cat

    # ── Heatmap rows ──────────────────────────────────────────────────────────────
    for _ri, (_y, _is_bot) in enumerate(zip(_y_positions, _y_is_bot)):
        _row_vals = _heatmap_data.values[_ri:_ri+1, :]
        _ax.pcolormesh(
            np.arange(_n_cols + 1),
            np.array([_y, _y + 1.0]),
            _row_vals,
            cmap=binary_cmap,
            edgecolors="white",
            linewidth=0.3,
            vmin=0, vmax=1
        )
        if _is_bot:
            _ax.axhspan(_y, _y + 1.0, color="lightgrey", alpha=0.35, zorder=0)

    # ── y axis ────────────────────────────────────────────────────────────────────
    _ax.set_yticks(_y_positions + 0.25)
    _ax.set_yticklabels(_y_labels, fontsize=9)
    for _tick, _is_bot in zip(_ax.get_yticklabels(), _y_is_bot):
        if _is_bot:
            _tick.set_fontweight("bold")
            _tick.set_fontsize(11)
            _tick.set_color("black")
        else:
            _tick.set_fontsize(9)
            _tick.set_color("#444444")

    _ax.set_ylim(0, _total_height)
    _ax.invert_yaxis()

    # ── x axis ────────────────────────────────────────────────────────────────────
    _ax.set_xticks(np.arange(_n_cols) + 0.5)
    _ax.set_xticklabels(_heatmap_data.columns, rotation=90, fontsize=8)
    _ax.xaxis.set_label_position("top")
    _ax.xaxis.tick_top()
    _ax.tick_params(axis="x", length=0, pad=30)  # ← pad pushes labels up above the band
    # ── Column group separator lines on heatmap ───────────────────────────────────
    _prev_grp = _col_group_labels[0]
    for _ci, _grp in enumerate(_col_group_labels):
        if _grp != _prev_grp:
            _ax.axvline(_ci, color="black", linewidth=1.5, zorder=5)
            _prev_grp = _grp
    # ── Legend ────────────────────────────────────────────────────────────────────
    _legend_patches = [
        mpatches.Patch(facecolor=_group_colors[_grp], label=_group_display.get(_grp, _grp))
        for _grp in _group_order
    ]
    _ax.legend(
        handles=_legend_patches,
        loc="upper left",
        bbox_to_anchor=(-0.005, 0),
        ncol=2, #len(_group_order),
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=9,
        borderpad=0.6,
        title="Assays",
        title_fontsize=10,
    ).get_title().set_fontweight("bold")
    # Category legend
    _seen = {}
    for _col, _color, _label in zip(_heatmap_data.columns, _col_cat_colors, _col_cat_labels):
        if _label and _label not in _seen:
            _seen[_label] = _color

    _cat_legend_patches = [
        mpatches.Patch(facecolor=_color, label=_label)
        for _label, _color in _seen.items()
    ]
    _ax_cat_band.legend(
        handles=_cat_legend_patches,
        loc="upper left",
        bbox_to_anchor=(0.3, -101.5),  # offset right to sit next to the assay legend
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

    _fig.savefig(os.path.expanduser("~/Downloads/prediction_heatmap_botanical_constituents_batch_groups_claudeai.png"),
                 bbox_inches="tight", dpi=300)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""### Stats prediction BSC & constituents significative""")
    return


@app.cell
def _(
    df_bsc_annotations,
    df_toxcast_cellbased_assay_info_significant,
    df_ttest_results_filtered,
    dic_botanicals_scode_name,
    leaves_list,
    linkage,
    list_botanicals,
    np,
    pd,
    pred_matrix_filtered,
):
    # ── Build ordered row index ───────────────────────────────────────────────────
    def _cluster_row_indices(data_matrix, indices):
        """Return row indices reordered by hierarchical clustering, or original if too few."""
        if len(indices) < 2:
            return indices
        sub = data_matrix[indices, :]
        valid_rows = [i for i, idx in enumerate(indices) if not np.all(np.isnan(sub[i, :]))]
        if len(valid_rows) < 2:
            return indices
        sub_valid = np.nan_to_num(sub[valid_rows, :], nan=0.0)
        Z = linkage(sub_valid, method="average", metric="jaccard")
        order = leaves_list(Z)
        return [indices[valid_rows[o]] for o in order]

    # ── Build ordered row index with clustering within botanical groups ────────────
    _rows_ordered = []

    # Build a temporary matrix indexed by scode for clustering
    _all_scodes_in_matrix = list(
        pred_matrix_filtered.set_index("Metadata_compound").index
    )
    _scode_to_rowidx = {s: i for i, s in enumerate(_all_scodes_in_matrix)}
    _matrix_values = pred_matrix_filtered.set_index("Metadata_compound").values.astype(float)

    for _scode in list_botanicals:
        _plant_name = dic_botanicals_scode_name.get(_scode, _scode)

        # Collect constituent scodes for this botanical
        _const_scodes = df_bsc_annotations[
            df_bsc_annotations["Plant scode"] == _scode
        ][["Metadata_compound", "Constituent"]].drop_duplicates()

        _const_rows = []
        for _, _crow in _const_scodes.iterrows():
            if _crow["Metadata_compound"] in pred_matrix_filtered["Metadata_compound"].values:
                _const_rows.append((_crow["Metadata_compound"], f"  {_crow['Constituent']}"))

        # Only add the botanical header if it or any constituent is present
        _has_botanical = _scode in pred_matrix_filtered["Metadata_compound"].values
        if not _has_botanical and len(_const_rows) == 0:
            continue

        if _has_botanical:
            _rows_ordered.append((_scode, _plant_name, True))

        if len(_const_rows) > 0:
            # Cluster constituents by their activity row vectors
            _const_matrix_indices = [
                _scode_to_rowidx[s] for s, _ in _const_rows
                if s in _scode_to_rowidx
            ]
            _clustered_idx = _cluster_row_indices(_matrix_values, _const_matrix_indices)

            # Map back to (scode, label) pairs
            _idx_to_row = {_scode_to_rowidx[s]: (s, lbl) for s, lbl in _const_rows if s in _scode_to_rowidx}
            for _ridx in _clustered_idx:
                _s, _lbl = _idx_to_row[_ridx]
                _rows_ordered.append((_s, _lbl, False))

    _ordered_scodes = [r[0] for r in _rows_ordered]
    _ordered_labels = [r[1] for r in _rows_ordered]
    _is_botanical   = [r[2] for r in _rows_ordered]

    # ── Build heatmap data ────────────────────────────────────────────────────────
    _heatmap_data = pred_matrix_filtered.set_index("Metadata_compound").reindex(_ordered_scodes)
    _heatmap_data.index = _ordered_labels

    # ── Assign columns to groups ──────────────────────────────────────────────────
    _cols = pd.Series(_heatmap_data.columns)

    _mask_cell_free  = _cols.str.contains("NVS|CCTE_GLTED_hIYD", regex=True)
    _mask_hit_calls  = _cols.str.contains("MT|LDH|cell_count", regex=True)
    _mask_cytotoxic  = _cols.str.contains("cell|tissue", regex=True) & ~_mask_hit_calls
    _mask_cell_based = ~_mask_cell_free & ~_mask_hit_calls & ~_mask_cytotoxic

    _group_order = ["Cell-free", "Cell-based", "Cytotoxic", "Hit calls"]
    _group_masks = {
        "Cell-free":  _mask_cell_free.values,
        "Cell-based": _mask_cell_based.values,
        "Cytotoxic":  _mask_cytotoxic.values,
        "Hit calls":  _mask_hit_calls.values,
    }

    _SOURCE_COLORS = {
        "Hits":       "#4C72B0",
        "Cell-based": "#DD8452",
        "Cell-free":  "#55A868",
        "Cytotox":    "#C44E52",
    }
    _LABEL_MAP = {
        "Hits":       "Cytotox",
        "Cell-based": "ToxCast cell-based",
        "Cell-free":  "ToxCast cell-free",
        "Cytotox":    "ToxCast cytotox",
    }

    # Replace _group_colors with _SOURCE_COLORS mapped via _LABEL_MAP
    _group_colors = {
        "Cell-free":  _SOURCE_COLORS["Cell-free"],
        "Cell-based": _SOURCE_COLORS["Cell-based"],
        "Cytotoxic":  _SOURCE_COLORS["Cytotox"],
        "Hit calls":  _SOURCE_COLORS["Hits"],
    }
    _group_display = {
        "Cell-free":  _LABEL_MAP["Cell-free"],
        "Cell-based": _LABEL_MAP["Cell-based"],
        "Cytotoxic":  _LABEL_MAP["Cytotox"],
        "Hit calls":  _LABEL_MAP["Hits"],
    }

    # ── Reorder columns by group + claudeai_label within Cell-based (no clustering) ──
    _col_indices_ordered = []
    _col_group_labels    = []

    for _grp in _group_order:
        _idx = list(np.where(_group_masks[_grp])[0])

        if _grp == "Cell-based":
            # Sort by claudeai_label instead of clustering
            _cols_in_grp = [_heatmap_data.columns[i] for i in _idx]
            _label_order = (
                df_toxcast_cellbased_assay_info_significant
                .drop_duplicates("assay_component_endpoint_name")
                .set_index("assay_component_endpoint_name")["claudeai_label"]
            )
            _idx_sorted = sorted(
                _idx,
                key=lambda i: _label_order.get(_heatmap_data.columns[i], "zzz")
            )
            _col_indices_ordered.extend(_idx_sorted)
        else:
            # Keep original order for other groups (no clustering)
            _col_indices_ordered.extend(_idx)

        _col_group_labels.extend([_grp] * len(_idx))

    _heatmap_data     = _heatmap_data.iloc[:, _col_indices_ordered]
    _col_group_labels = [_col_group_labels[i] for i in range(len(_col_indices_ordered))]

    # ── Build AUROC values aligned to (reordered) heatmap columns ────────────────
    _auroc_lookup = df_ttest_results_filtered.set_index("Metadata_Label")["AUROC_mean_Actual"]
    _auroc_values = np.array([
        _auroc_lookup.get(col, np.nan) for col in _heatmap_data.columns
    ])
    _bar_colors = [_group_colors[g] for g in _col_group_labels]

    # ── Row separators: spacing between botanical groups ──────────────────────────
    _y_positions = []
    _y_labels    = []
    _y_is_bot    = []
    _gap         = 0.4
    _y           = 0.0

    for _i, (_label, _is_bot) in enumerate(zip(_ordered_labels, _is_botanical)):
        if _is_bot and _i > 0:
            _y += _gap
        _y_positions.append(_y)
        _y_labels.append(_label)
        _y_is_bot.append(_is_bot)
        _y += 1.0

    _y_positions  = np.array(_y_positions)
    _total_height = _y_positions[-1] + 1.0
    # ── Category annotation lookup for cell-based columns ────────────────────────
    _cat_color_lookup = (
        df_toxcast_cellbased_assay_info_significant
        .drop_duplicates("assay_component_endpoint_name")
        .set_index("assay_component_endpoint_name")[["claudeai_color", "claudeai_label"]]
        .to_dict(orient="index")
    )

    # Per-column category color and label (NaN for non-cell-based columns)
    _col_cat_colors = []
    _col_cat_labels = []
    for _col in _heatmap_data.columns:
        _info = _cat_color_lookup.get(_col)
        _col_cat_colors.append(_info["claudeai_color"] if _info else "#dddddd")
        _col_cat_labels.append(_info["claudeai_label"] if _info else None)
    # Melt pred_matrix_filtered to long format
    _pred_long = pred_matrix_filtered.set_index("Metadata_compound").reset_index().melt(
        id_vars="Metadata_compound",
        var_name="assay",
        value_name="active"
    )

    # Map compound to botanical
    _const_to_botanical = (
        df_bsc_annotations[["Metadata_compound", "Plant scode"]]
        .drop_duplicates()
        .set_index("Metadata_compound")["Plant scode"]
        .to_dict()
    )
    _pred_long["botanical_scode"] = _pred_long["Metadata_compound"].map(_const_to_botanical)
    _pred_long["botanical"] = _pred_long["botanical_scode"].map(dic_botanicals_scode_name)

    df_pred_long_all = _pred_long
    # Drop rows with no botanical mapping (botanicals themselves or DMSO)
    _pred_long = _pred_long.dropna(subset=["botanical"])
    # ── Table 1: % active per botanical (across all assays) ──────────────────────
    df_pct_active_by_botanical = (
        _pred_long
        .groupby("botanical")["active"]
        .agg(
            n_active=lambda x: x.eq(1).sum(),
            n_tested=lambda x: x.notna().sum(),
        )
        .assign(pct_active=lambda x: (x["n_active"] / x["n_tested"] * 100).round(1))
        .sort_values("pct_active", ascending=False)
        .reset_index()
    )
    print("=== % Active predictions by botanical (all assays) ===")
    print(df_pct_active_by_botanical.to_string(index=False))

    # ── Table 2: % active per botanical × assay group ────────────────────────────
    _col_group_series = pd.Series(_col_group_labels, index=_heatmap_data.columns, name="group")
    _pred_long["group"] = _pred_long["assay"].map(_col_group_series)
    df_pred_long_botanical = _pred_long
    df_pct_active_by_botanical_group = (
        _pred_long.dropna(subset=["group"])
        .groupby(["botanical", "group"])["active"]
        .agg(
            n_active=lambda x: x.eq(1).sum(),
            n_tested=lambda x: x.notna().sum(),
        )
        .assign(pct_active=lambda x: (x["n_active"] / x["n_tested"] * 100).round(1))
        .reset_index()
        .pivot(index="botanical", columns="group", values="pct_active")
        .sort_values("Cell-based", ascending=False)
        .reset_index()
    )
    print("\n=== % Active predictions by botanical × assay group ===")
    print(df_pct_active_by_botanical_group.to_string(index=False))

    # ── Table 3: % active per botanical × claudeai_label (cell-based only) ───────
    _cat_series = pd.Series(
        _col_cat_labels,
        index=_heatmap_data.columns,
        name="claudeai_label"
    )
    _pred_long["claudeai_label"] = _pred_long["assay"].map(_cat_series)

    df_pct_active_by_botanical_category = (
        _pred_long.dropna(subset=["claudeai_label"])
        .groupby(["botanical", "claudeai_label"])["active"]
        .agg(
            n_active=lambda x: x.eq(1).sum(),
            n_tested=lambda x: x.notna().sum(),
        )
        .assign(pct_active=lambda x: (x["n_active"] / x["n_tested"] * 100).round(1))
        .reset_index()
        .pivot(index="botanical", columns="claudeai_label", values="pct_active")
        .sort_values(
            by=df_pct_active_by_botanical_category.columns.tolist()[0]
            if "df_pct_active_by_botanical_category" in dir() else "botanical",
            ascending=False
        )
        .reset_index()
    )
    print("\n=== % Active predictions by botanical × cell-based category ===")
    print(df_pct_active_by_botanical_category.to_string(index=False))
    return (
        df_pct_active_by_botanical,
        df_pct_active_by_botanical_category,
        df_pct_active_by_botanical_group,
        df_pred_long_all,
        df_pred_long_botanical,
    )


@app.cell
def _(df_pred_long):
    df_pred_long
    return


@app.cell
def _(df_pred_long_botanical):
    df_pred_long_botanical
    return


@app.cell
def _(df_pred_long_all):
    _df_summary = (
        df_pred_long_all
        .groupby("assay")["active"]
        .agg(
            active_count="sum",     # assumes active = 1/0 or True/False
            total_count="count"
        )
    )

    _df_summary["percent_active"] = (
        _df_summary["active_count"] / _df_summary["total_count"] * 100
    )

    _df_summary
    return


@app.cell
def _():
    return


@app.cell
def _(df_pct_active_by_botanical):
    df_pct_active_by_botanical
    return


@app.cell
def _(df_pct_active_by_botanical_group):
    df_pct_active_by_botanical_group
    return


@app.cell
def _(df_pct_active_by_botanical_category):
    df_pct_active_by_botanical_category
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Classification BSC with Axiom (models) - Level 5""")
    return


@app.cell
def _(df_level5_bsc_all, loaded_models, pd, tqdm):
    # Subset once - using correct feature matrix
    X_base_lvl5 = df_level5_bsc_all.copy() #.loc[df_level5_bsc_all["Metadata_compound"].isin(list_botanicals)].copy()

    # Prepare prediction matrix with compound names
    pred_matrix_lvl5 = pd.DataFrame({
        "Metadata_compound": X_base_lvl5["Metadata_compound"].values,
        "Metadata_concentration": X_base_lvl5["Metadata_concentration"].values,
    }, index=X_base_lvl5.index)

    # Get feature matrix (same rows as pred_matrix_lvl5)
    X_pred_lvl5 = X_base_lvl5[[col for col in X_base_lvl5.columns 
                                if col not in ["Metadata_compound", "Metadata_concentration"]]]

    # Filter loaded_models to only MTT and LDH
    loaded_models_filtered = {
        k: v for k, v in loaded_models.items() 
        if any(endpoint in k.lower() for endpoint in ["mtt", "ldh", "cell_type__hepg2"])
    }

    print(f"Predicting for {len(loaded_models_filtered)} models: {list(loaded_models_filtered.keys())}\n")

    # Loop with progress bar
    for _model_name, _model in tqdm(loaded_models_filtered.items(), 
                                   total=len(loaded_models_filtered), 
                                   desc="Predicting"):
        # Get training feature names
        _trained_features = _model.get_booster().feature_names

        # Align features exactly
        _X_pred = X_pred_lvl5.reindex(columns=_trained_features)

        # Predict
        _preds = _model.predict(_X_pred)

        # Store predictions
        pred_matrix_lvl5[_model_name] = _preds

    # Reset index
    pred_matrix_lvl5 = pred_matrix_lvl5.reset_index(drop=True)

    print(f"\nPrediction matrix shape: {pred_matrix_lvl5.shape}")
    print(f"Models included: {[col for col in pred_matrix_lvl5.columns if col not in ['Metadata_compound', 'Metadata_concentration']]}")
    print(f"\nFirst rows:")
    print(pred_matrix_lvl5.head())
    return (pred_matrix_lvl5,)


@app.cell
def _(mo):
    mo.md(r"""### Prediction - MT and LDH""")
    return


@app.cell
def _():
    from matplotlib.path import Path as matplotlib_path
    return (matplotlib_path,)


@app.cell
def _(
    df_bsc_annotations,
    dic_botanicals_scode_name,
    leaves_list,
    linkage,
    list_botanicals,
    matplotlib_path,
    mticker,
    np,
    os,
    pd,
    plt,
    pred_matrix_lvl5,
):
    # Get MTT and LDH model columns
    _mtt_cols = [col for col in pred_matrix_lvl5.columns if "mtt" in col.lower()]
    _ldh_cols = [col for col in pred_matrix_lvl5.columns if "ldh" in col.lower()]
    _mtt_model = _mtt_cols[0] if _mtt_cols else None
    _ldh_model = _ldh_cols[0] if _ldh_cols else None
    def half_circle_marker(side="left"):
        theta = np.linspace(0, np.pi, 50)
        if side == "right":
            theta = np.linspace(np.pi, 2*np.pi, 50)

        verts = np.column_stack([np.cos(theta), np.sin(theta)])
        verts = np.vstack(([0, 0], verts, [0, 0]))
        codes = [matplotlib_path.MOVETO] + [matplotlib_path.LINETO]*len(theta) + [matplotlib_path.CLOSEPOLY]

        return matplotlib_path(verts, codes)

    # Prepare data
    _plot_data = pred_matrix_lvl5.copy()
    _plot_data["_concentration_log"] = _plot_data["Metadata_concentration"]# np.log10(_plot_data["Metadata_concentration"])
    _plot_data["_mtt_active"] = (_plot_data[_mtt_model] >= 0.5).astype(int)
    _plot_data["_ldh_active"] = (_plot_data[_ldh_model] >= 0.5).astype(int)

    # ── Build ordered row index with botanicals and constituents ────────────────────
    _rows_ordered = []

    # Build temporary matrix for clustering
    _all_scodes_in_matrix = list(_plot_data["Metadata_compound"].unique())
    _scode_to_rowidx = {s: i for i, s in enumerate(_all_scodes_in_matrix)}
    _matrix_values = _plot_data.set_index("Metadata_compound").values.astype(float)

    def _cluster_row_indices(data_matrix, indices):
        """Return row indices reordered by hierarchical clustering."""
        if len(indices) < 2:
            return indices
        sub = data_matrix[indices, :]
        valid_rows = [i for i, idx in enumerate(indices) if not np.all(np.isnan(sub[i, :]))]
        if len(valid_rows) < 2:
            return indices
        sub_valid = np.nan_to_num(sub[valid_rows, :], nan=0.0)
        Z = linkage(sub_valid, method="average", metric="jaccard")
        order = leaves_list(Z)
        return [indices[valid_rows[o]] for o in order]

    for _scode in list_botanicals:
        _plant_name = dic_botanicals_scode_name.get(_scode, _scode)

        # Collect constituent scodes for this botanical
        _const_scodes = df_bsc_annotations[
            df_bsc_annotations["Plant scode"] == _scode
        ][["Metadata_compound", "Constituent"]].drop_duplicates()

        _const_rows = []
        for _, _crow in _const_scodes.iterrows():
            if _crow["Metadata_compound"] in _plot_data["Metadata_compound"].values:
                _const_rows.append((_crow["Metadata_compound"], f"  {_crow['Constituent']}"))

        # Only add the botanical header if it or any constituent is present
        _has_botanical = _scode in _plot_data["Metadata_compound"].values
        if not _has_botanical and len(_const_rows) == 0:
            continue

        if _has_botanical:
            _rows_ordered.append((_scode, _plant_name, True))

        if len(_const_rows) > 0:
            # Cluster constituents by their activity row vectors
            _const_matrix_indices = [
                _scode_to_rowidx[s] for s, _ in _const_rows
                if s in _scode_to_rowidx
            ]
            _clustered_idx = _cluster_row_indices(_matrix_values, _const_matrix_indices)

            # Map back to (scode, label) pairs
            _idx_to_row = {_scode_to_rowidx[s]: (s, lbl) for s, lbl in _const_rows if s in _scode_to_rowidx}
            for _ridx in _clustered_idx:
                _s, _lbl = _idx_to_row[_ridx]
                _rows_ordered.append((_s, _lbl, False))

    _ordered_scodes = [r[0] for r in _rows_ordered]
    _ordered_labels = [r[1] for r in _rows_ordered]
    _is_botanical   = [r[2] for r in _rows_ordered]

    # ── Build scatter plot with concentration on x (log), botanicals on y ──────────
    _fig, _ax = plt.subplots(figsize=(12, max(8, len(_ordered_labels) * 0.2)))

    # Map botanicals to y-positions (with gaps between groups)
    _y_positions = []
    _y_labels    = []
    _y_is_bot    = []
    _gap         = 0.4
    _y           = 0.0

    for _i, (_label, _is_bot) in enumerate(zip(_ordered_labels, _is_botanical)):
        if _is_bot and _i > 0:
            _y += _gap
        _y_positions.append(_y)
        _y_labels.append(_label)
        _y_is_bot.append(_is_bot)
        _y += 1.0

    _y_positions = np.array(_y_positions)
    _total_height = _y_positions[-1] + 1.0

    # 1. Create a reference DataFrame from your ordered metadata
    # This captures the EXACT y-position for every instance of an scode
    _metadata_map = pd.DataFrame({
        "Metadata_compound": _ordered_scodes,
        "y_coord": _y_positions,
        "is_botanical": _is_botanical
    })

    # 2. Merge this map with your plotting data
    # 'inner' ensures we only plot things that exist in our ordered list
    # 'left' ensures we keep all plot data (use left if plot_data is the master)
    _merged_plot_data = _plot_data.merge(_metadata_map, on="Metadata_compound")

    # 3. Plot using the newly merged y_coord
    for _idx, _row in _merged_plot_data.iterrows():
        _x = _row["_concentration_log"]
        _y = _row["y_coord"]
        _mtt_active = _row["_mtt_active"]
        _ldh_active = _row["_ldh_active"]

        # Determine color based on MTT + LDH combination
        if _mtt_active == 0 and _ldh_active == 0:
            # Both inactive: green
            _ax.scatter(_x, _y + 0.5, color="green", s=120, alpha=0.7, 
                       edgecolors="black", linewidth=0.5, zorder=3)

        elif _mtt_active == 0 and _ldh_active == 1:
            # MTT inactive, LDH active: black
            _ax.scatter(_x, _y + 0.5, color="black", s=120, alpha=0.7, 
                       edgecolors="black", linewidth=0.5, zorder=3)

        elif _mtt_active == 1 and _ldh_active == 0:
            # MTT active, LDH inactive: red
            _ax.scatter(_x, _y + 0.5, color="red", s=120, alpha=0.7, 
                       edgecolors="black", linewidth=0.5, zorder=3)

        else:  # _mtt_active == 1 and _ldh_active == 1
            # Both active: split circle (left=red for MTT, right=black for LDH)
            # Plot left half (red)
            _ax.scatter(_x, _y + 0.5, marker=half_circle_marker("left"),
                        color="red", s=120, edgecolors="black", zorder=3)
            # Plot right half (black)
            _ax.scatter(_x, _y + 0.5, marker=half_circle_marker("right"),
                        color="black", s=120, edgecolors="black", zorder=3)

    # ── Botanical group background shading ─────────────────────────────────────────
    for _i, (_y, _is_bot) in enumerate(zip(_y_positions, _y_is_bot)):
        if _is_bot:
            _ax.axhspan(_y, _y + 1.0, color="lightgrey", alpha=0.25, zorder=0)

    # ── y-axis: botanical names (bold) and constituents (normal) ──────────────────
    _ax.set_yticks(_y_positions + 0.5)
    _ax.set_yticklabels(_y_labels, fontsize=9)
    for _tick, _is_bot in zip(_ax.get_yticklabels(), _y_is_bot):
        if _is_bot:
            _tick.set_fontweight("bold")
            _tick.set_fontsize(11)
            _tick.set_color("black")
        else:
            _tick.set_fontsize(9)
            _tick.set_color("#444444")

    _ax.set_ylim(0, _total_height)
    _ax.invert_yaxis()

    # ── x-axis: log concentration ────────────────────────────────────────────────────
    _ax.set_ylabel("Botanical / Constituent", fontsize=11, fontweight="bold")
    _ax.set_title(f"MTT + LDH Predictions", 
                  fontsize=12, fontweight="bold", pad=15)

    _ax.grid(True, alpha=0.2, axis="x", linestyle="--")


    # ── Axes formatting ───────────────────────────────────────────────────────────

    _ax.set_xscale("log")

    # ✅ LOG TICKS (major + minor like your previous plot)
    _ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))

    _ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.tick_params(axis="x", which="major", colors="black", length=0, direction="out", labelsize=9, labelrotation=30)
    _ax.tick_params(axis="x", which="minor", colors="grey",  length=0, direction="out", labelsize=6, labelrotation=30)

    _ax.set_xlabel("Concentration (μM)", fontsize=11)

    # ── Add legend ────────────────────────────────────────────────────────────────
    _legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", 
                   markersize=10, label="Both inactive", markeredgecolor="black", markeredgewidth=0.5),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black", 
                   markersize=10, label="LDH active only", markeredgecolor="black", markeredgewidth=0.5),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", 
                   markersize=10, label="MT active only", markeredgecolor="black", markeredgewidth=0.5),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", 
                   markerfacecoloralt="black", markersize=10, fillstyle="left",
                   label="Both active", 
                   markeredgecolor="black", markeredgewidth=0.5),
    ]
    _ax.legend(handles=_legend_elements, loc="upper right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/Downloads/mtt_ldh_predictions_botanical_constituents.png"),
                bbox_inches="tight", dpi=300)
    plt.show()

    print(f"Plot saved. Displayed {len(_plot_data)} samples across {len(_ordered_scodes)} botanicals/constituents")
    print(f"MTT model: {_mtt_model}")
    print(f"LDH model: {_ldh_model}")
    return (half_circle_marker,)


@app.cell
def _(mo):
    mo.md(r"""### Prediction - HepG2""")
    return


@app.cell
def _(pred_matrix_lvl5):
    _mtt_cols = [col for col in pred_matrix_lvl5.columns if "_hepg2" in col.lower()]
    _mtt_cols
    return


@app.cell
def _(
    df_bsc_annotations,
    dic_botanicals_scode_name,
    leaves_list,
    linkage,
    list_botanicals,
    mticker,
    np,
    os,
    pd,
    plt,
    pred_matrix_lvl5,
):
    # Get MTT and LDH model columns
    _mtt_cols = [col for col in pred_matrix_lvl5.columns if "cell_type__HepG2" in col]
    _mtt_model = _mtt_cols[0] if _mtt_cols else None

    # Prepare data
    _plot_data = pred_matrix_lvl5.copy()
    _plot_data["_concentration_log"] = _plot_data["Metadata_concentration"]# np.log10(_plot_data["Metadata_concentration"])
    _plot_data["_mtt_active"] = (_plot_data[_mtt_model] >= 0.5).astype(int)

    # ── Build ordered row index with botanicals and constituents ────────────────────
    _rows_ordered = []

    # Build temporary matrix for clustering
    _all_scodes_in_matrix = list(_plot_data["Metadata_compound"].unique())
    _scode_to_rowidx = {s: i for i, s in enumerate(_all_scodes_in_matrix)}
    _matrix_values = _plot_data.set_index("Metadata_compound").values.astype(float)

    def _cluster_row_indices(data_matrix, indices):
        """Return row indices reordered by hierarchical clustering."""
        if len(indices) < 2:
            return indices
        sub = data_matrix[indices, :]
        valid_rows = [i for i, idx in enumerate(indices) if not np.all(np.isnan(sub[i, :]))]
        if len(valid_rows) < 2:
            return indices
        sub_valid = np.nan_to_num(sub[valid_rows, :], nan=0.0)
        Z = linkage(sub_valid, method="average", metric="jaccard")
        order = leaves_list(Z)
        return [indices[valid_rows[o]] for o in order]

    for _scode in list_botanicals:
        _plant_name = dic_botanicals_scode_name.get(_scode, _scode)

        # Collect constituent scodes for this botanical
        _const_scodes = df_bsc_annotations[
            df_bsc_annotations["Plant scode"] == _scode
        ][["Metadata_compound", "Constituent"]].drop_duplicates()

        _const_rows = []
        for _, _crow in _const_scodes.iterrows():
            if _crow["Metadata_compound"] in _plot_data["Metadata_compound"].values:
                _const_rows.append((_crow["Metadata_compound"], f"  {_crow['Constituent']}"))

        # Only add the botanical header if it or any constituent is present
        _has_botanical = _scode in _plot_data["Metadata_compound"].values
        if not _has_botanical and len(_const_rows) == 0:
            continue

        if _has_botanical:
            _rows_ordered.append((_scode, _plant_name, True))

        if len(_const_rows) > 0:
            # Cluster constituents by their activity row vectors
            _const_matrix_indices = [
                _scode_to_rowidx[s] for s, _ in _const_rows
                if s in _scode_to_rowidx
            ]
            _clustered_idx = _cluster_row_indices(_matrix_values, _const_matrix_indices)

            # Map back to (scode, label) pairs
            _idx_to_row = {_scode_to_rowidx[s]: (s, lbl) for s, lbl in _const_rows if s in _scode_to_rowidx}
            for _ridx in _clustered_idx:
                _s, _lbl = _idx_to_row[_ridx]
                _rows_ordered.append((_s, _lbl, False))

    _ordered_scodes = [r[0] for r in _rows_ordered]
    _ordered_labels = [r[1] for r in _rows_ordered]
    _is_botanical   = [r[2] for r in _rows_ordered]

    # ── Build scatter plot with concentration on x (log), botanicals on y ──────────
    _fig, _ax = plt.subplots(figsize=(12, max(8, len(_ordered_labels) * 0.2)))

    # Map botanicals to y-positions (with gaps between groups)
    _y_positions = []
    _y_labels    = []
    _y_is_bot    = []
    _gap         = 0.4
    _y           = 0.0

    for _i, (_label, _is_bot) in enumerate(zip(_ordered_labels, _is_botanical)):
        if _is_bot and _i > 0:
            _y += _gap
        _y_positions.append(_y)
        _y_labels.append(_label)
        _y_is_bot.append(_is_bot)
        _y += 1.0

    _y_positions = np.array(_y_positions)
    _total_height = _y_positions[-1] + 1.0

    # 1. Create a reference DataFrame from your ordered metadata
    # This captures the EXACT y-position for every instance of an scode
    _metadata_map = pd.DataFrame({
        "Metadata_compound": _ordered_scodes,
        "y_coord": _y_positions,
        "is_botanical": _is_botanical
    })

    # 2. Merge this map with your plotting data
    # 'inner' ensures we only plot things that exist in our ordered list
    # 'left' ensures we keep all plot data (use left if plot_data is the master)
    _merged_plot_data = _plot_data.merge(_metadata_map, on="Metadata_compound")

    # 3. Plot using the newly merged y_coord
    for _idx, _row in _merged_plot_data.iterrows():
        _x = _row["_concentration_log"]
        _y = _row["y_coord"]
        _mtt_active = _row["_mtt_active"]

        # Determine color based on MTT + LDH combination
        if _mtt_active == 0:
            # Both inactive: green
            _ax.scatter(_x, _y + 0.5, color="green", s=120, alpha=0.7, 
                       edgecolors="black", linewidth=0.5, zorder=3)

        elif _mtt_active == 1:
            # MTT active, LDH inactive: red
            _ax.scatter(_x, _y + 0.5, color="red", s=120, alpha=0.7, 
                       edgecolors="black", linewidth=0.5, zorder=3)

    # ── Botanical group background shading ─────────────────────────────────────────
    for _i, (_y, _is_bot) in enumerate(zip(_y_positions, _y_is_bot)):
        if _is_bot:
            _ax.axhspan(_y, _y + 1.0, color="lightgrey", alpha=0.25, zorder=0)

    # ── y-axis: botanical names (bold) and constituents (normal) ──────────────────
    _ax.set_yticks(_y_positions + 0.5)
    _ax.set_yticklabels(_y_labels, fontsize=9)
    for _tick, _is_bot in zip(_ax.get_yticklabels(), _y_is_bot):
        if _is_bot:
            _tick.set_fontweight("bold")
            _tick.set_fontsize(11)
            _tick.set_color("black")
        else:
            _tick.set_fontsize(9)
            _tick.set_color("#444444")

    _ax.set_ylim(0, _total_height)
    _ax.invert_yaxis()

    # ── x-axis: log concentration ────────────────────────────────────────────────────
    _ax.set_ylabel("Botanical / Constituent", fontsize=11, fontweight="bold")
    _ax.set_title(f"HepG2 Predictions", 
                  fontsize=12, fontweight="bold", pad=15)

    _ax.grid(True, alpha=0.2, axis="x", linestyle="--")


    # ── Axes formatting ───────────────────────────────────────────────────────────

    _ax.set_xscale("log")

    # ✅ LOG TICKS (major + minor like your previous plot)
    _ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    _ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))

    _ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    _ax.tick_params(axis="x", which="major", colors="black", length=0, direction="out", labelsize=9, labelrotation=30)
    _ax.tick_params(axis="x", which="minor", colors="grey",  length=0, direction="out", labelsize=6, labelrotation=30)

    _ax.set_xlabel("Concentration (μM)", fontsize=11)

    # ── Add legend ────────────────────────────────────────────────────────────────
    _legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", 
                   markersize=10, label="Inactive", markeredgecolor="black", markeredgewidth=0.5),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", 
                   markersize=10, label="HepG2 active", markeredgecolor="black", markeredgewidth=0.5),
    ]
    _ax.legend(handles=_legend_elements, loc="upper right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/Downloads/hepg2_predictions_botanical_constituents.png"),
                bbox_inches="tight", dpi=300)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Classification BSC with Axiom (models) - Level 4""")
    return


@app.cell
def _(df_level4_bsc, pd):
    # Cell Painting profiles
    df_level4a_bsc = pd.read_parquet("modules/02_processing/outputs/level4a_bsc.parquet")
    if True:
        # Keep only plates that exist in df_level4_bsc
        df_level4a_bsc = df_level4a_bsc[
            df_level4a_bsc["Metadata_plate"].isin(df_level4_bsc["Metadata_plate"])
        ]
        # Your existing filter (unchanged)
        df_level4a_bsc = df_level4a_bsc.groupby("Metadata_plate").filter(
            lambda g: ~(g["Metadata_compound_name"] == "DMSO").all()
        )
        df_level4a_bsc["Metadata_replicates"] = (
            df_level4a_bsc.groupby("Metadata_perturbation")
            .cumcount() + 1
        )
    return (df_level4a_bsc,)


@app.cell
def _(df_level4a_bsc, loaded_models, pd, tqdm):
    # Subset once - using correct feature matrix
    X_base_lvl4 = df_level4a_bsc.copy() #.loc[df_level5_bsc_all["Metadata_compound"].isin(list_botanicals)].copy()

    # Prepare prediction matrix with compound names
    pred_matrix_lvl4 = pd.DataFrame({
        "Metadata_compound": X_base_lvl4["Metadata_compound"].values,
        "Metadata_concentration": X_base_lvl4["Metadata_concentration"].values,
        "Metadata_replicates": X_base_lvl4["Metadata_replicates"].values,
    }, index=X_base_lvl4.index)

    # Get feature matrix (same rows as pred_matrix_lvl5)
    X_pred_lvl4 = X_base_lvl4[
        [col for col in X_base_lvl4.columns if col.startswith("Feature")]
    ]

    # Filter loaded_models to only MTT and LDH
    _loaded_models_filtered = {
        k: v for k, v in loaded_models.items() 
        if any(endpoint in k.lower() for endpoint in ["mtt", "ldh", "cell_type__hepg2"])
    }

    print(f"Predicting for {len(_loaded_models_filtered)} models: {list(_loaded_models_filtered.keys())}\n")

    # Loop with progress bar
    for _model_name, _model in tqdm(_loaded_models_filtered.items(), 
                                   total=len(_loaded_models_filtered), 
                                   desc="Predicting"):
        # Get training feature names
        _trained_features = _model.get_booster().feature_names

        # Align features exactly
        _X_pred = X_pred_lvl4.reindex(columns=_trained_features)

        # Predict
        _preds = _model.predict(_X_pred)

        # Store predictions
        pred_matrix_lvl4[_model_name] = _preds

    # Reset index
    pred_matrix_lvl4 = pred_matrix_lvl4.reset_index(drop=True)

    print(f"\nPrediction matrix shape: {pred_matrix_lvl4.shape}")
    print(f"Models included: {[col for col in pred_matrix_lvl4.columns if col not in ['Metadata_compound', 'Metadata_concentration']]}")
    print(f"\nFirst rows:")
    print(pred_matrix_lvl4.head())
    return (pred_matrix_lvl4,)


@app.cell
def _(
    df_bsc_annotations,
    dic_botanicals_scode_name,
    half_circle_marker,
    leaves_list,
    linkage,
    list_botanicals,
    mticker,
    np,
    os,
    pd,
    plt,
    pred_matrix_lvl4,
):
    # ── Identify MTT and LDH model columns ─────────────────────────────────────────
    _mtt_cols = [col for col in pred_matrix_lvl4.columns if "mtt" in col.lower()]
    _ldh_cols = [col for col in pred_matrix_lvl4.columns if "ldh" in col.lower()]
    _mtt_model = _mtt_cols[0] if _mtt_cols else None
    _ldh_model = _ldh_cols[0] if _ldh_cols else None

    # ── Prepare data ─────────────────────────────────────────────────────────────
    _plot_data = pred_matrix_lvl4.copy()
    _plot_data["_concentration_log"] = _plot_data["Metadata_concentration"]
    _plot_data["_mtt_active"] = (_plot_data[_mtt_model] >= 0.5).astype(int)
    _plot_data["_ldh_active"] = (_plot_data[_ldh_model] >= 0.5).astype(int)

    # Ensure replicates are integers for merging
    _plot_data["Metadata_replicates"] = _plot_data["Metadata_replicates"].astype(int)

    # ── Build ordered row index with botanicals and constituents ─────────────────
    _rows_ordered = []

    _all_scodes_in_matrix = list(_plot_data["Metadata_compound"].unique())
    _scode_to_rowidx = {s: i for i, s in enumerate(_all_scodes_in_matrix)}

    _numeric_cols = _plot_data.select_dtypes(include=[np.number]).columns
    _matrix_values = _plot_data.set_index("Metadata_compound")[_numeric_cols].values.astype(float)

    def _cluster_row_indices(data_matrix, indices):
        if len(indices) < 2:
            return indices
        sub = data_matrix[indices, :]
        valid_rows = [i for i, idx in enumerate(indices) if not np.all(np.isnan(sub[i, :]))]
        if len(valid_rows) < 2:
            return indices
        sub_valid = np.nan_to_num(sub[valid_rows, :], nan=0.0)
        Z = linkage(sub_valid, method="average", metric="jaccard")
        order = leaves_list(Z)
        return [indices[valid_rows[o]] for o in order]

    for _scode in list_botanicals:
        _plant_name = dic_botanicals_scode_name.get(_scode, _scode)

        # Find associated constituents
        _const_scodes = df_bsc_annotations[
            df_bsc_annotations["Plant scode"] == _scode
        ][["Metadata_compound", "Constituent"]].drop_duplicates()

        _const_rows = []
        for _, _crow in _const_scodes.iterrows():
            if _crow["Metadata_compound"] in _plot_data["Metadata_compound"].values:
                _const_rows.append((_crow["Metadata_compound"], _crow["Constituent"]))

        _has_botanical = _scode in _plot_data["Metadata_compound"].values
        if not _has_botanical and len(_const_rows) == 0:
            continue

        # 1. Add Botanical Replicates as separate rows
        if _has_botanical:
            _n_reps_bot = int(_plot_data[_plot_data["Metadata_compound"] == _scode]["Metadata_replicates"].max())
            for rep in range(1, _n_reps_bot + 1):
                _rows_ordered.append((_scode, f"{_plant_name} (Rep {rep})", True, rep))

        # 2. Add Constituent Replicates as separate rows
        if len(_const_rows) > 0:
            _const_matrix_indices = [_scode_to_rowidx[s] for s, _ in _const_rows if s in _scode_to_rowidx]
            _clustered_idx = _cluster_row_indices(_matrix_values, _const_matrix_indices)

            _idx_to_row = {_scode_to_rowidx[s]: (s, lbl) for s, lbl in _const_rows if s in _scode_to_rowidx}
            for _ridx in _clustered_idx:
                _s, _lbl = _idx_to_row[_ridx]
                _n_reps_const = int(_plot_data[_plot_data["Metadata_compound"] == _s]["Metadata_replicates"].max())
                for rep in range(1, _n_reps_const + 1):
                    _rows_ordered.append((_s, f"  {_lbl} (Rep {rep})", False, rep))

    # ── Build Y-coordinates ──────────────────────────────────────────────────────
    _ordered_scodes = [r[0] for r in _rows_ordered]
    _ordered_labels = [r[1] for r in _rows_ordered]
    _is_botanical   = [r[2] for r in _rows_ordered]
    _replicate_ids  = [r[3] for r in _rows_ordered]

    _y_positions = []
    _y_is_bot = []
    _gap = 0.5
    _y = 0.0

    for _i, _is_bot in enumerate(_is_botanical):
        # Add a visual gap when we encounter a NEW botanical group
        if _i > 0 and _is_bot and not _is_botanical[_i-1]:
            _y += _gap
        elif _i > 0 and _is_bot and _ordered_scodes[_i] != _ordered_scodes[_i-1]:
            # Gap between different botanical types if they follow each other
            _y += _gap

        _y_positions.append(_y)
        _y_is_bot.append(_is_bot)
        _y += 1.0

    _y_positions = np.array(_y_positions)
    _total_height = _y + 1.0

    # ── Merge Coordinates with Data ──────────────────────────────────────────────
    _metadata_map = pd.DataFrame({
        "Metadata_compound": _ordered_scodes,
        "Metadata_replicates": _replicate_ids,
        "y_coord": _y_positions,
        "is_botanical_row": _is_botanical,
        "row_label": _ordered_labels
    })

    # Crucial: Merge on both Compound and Replicate ID
    _merged_plot_data = _plot_data.merge(
        _metadata_map, 
        on=["Metadata_compound", "Metadata_replicates"], 
        how="inner"
    )

    # ── Build Plot ───────────────────────────────────────────────────────────────
    _fig, _ax = plt.subplots(figsize=(14, max(10, len(_ordered_labels) * 0.2)))

    # Plot background shading for botanical rows
    for _y, _is_bot in zip(_y_positions, _y_is_bot):
        if _is_bot:
            _ax.axhspan(_y - 0.1, _y + 0.9, color="lightgrey", alpha=0.2, zorder=0)

    # Plot markers
    for _, _row in _merged_plot_data.iterrows():
        _x = _row["_concentration_log"]
        _y = _row["y_coord"]
        mtt = _row["_mtt_active"]
        ldh = _row["_ldh_active"]

        # Center points in the row "lane"
        y_mid = _y + 0.4 

        if mtt == 0 and ldh == 0: # Both Inactive
            _ax.scatter(_x, y_mid, color="green", s=130, alpha=0.8, edgecolors="black", linewidth=0.6, zorder=3)
        elif mtt == 0 and ldh == 1: # LDH only
            _ax.scatter(_x, y_mid, color="black", s=130, alpha=0.8, edgecolors="white", linewidth=0.6, zorder=3)
        elif mtt == 1 and ldh == 0: # MTT only
            _ax.scatter(_x, y_mid, color="red", s=130, alpha=0.8, edgecolors="black", linewidth=0.6, zorder=3)
        else: # Both Active
            _ax.scatter(_x, y_mid, marker=half_circle_marker("left"), color="red", s=130, edgecolors="black", zorder=3)
            _ax.scatter(_x, y_mid, marker=half_circle_marker("right"), color="black", s=130, edgecolors="black", zorder=3)

    # ── Formatting ───────────────────────────────────────────────────────────────
    _ax.set_yticks(_y_positions + 0.4)
    _ax.set_yticklabels(_ordered_labels)

    for _tick, _is_bot in zip(_ax.get_yticklabels(), _y_is_bot):
        if _is_bot:
            _tick.set_fontweight("bold")
            _tick.set_fontsize(10)
            _tick.set_color("black")
        else:
            _tick.set_fontsize(8.5)
            _tick.set_color("#555555")

    _ax.set_ylim(-0.5, _total_height)
    _ax.invert_yaxis()
    _ax.set_xscale("log")

    _ax.set_xlabel("Concentration (μM)", fontsize=12, labelpad=10)
    _ax.set_ylabel("Botanicals & Constituents (Replicate-level)", fontsize=12)
    _ax.set_title("MTT and LDH Toxicity Predictions", fontsize=14, fontweight="bold", pad=20)

    _ax.grid(True, axis="x", linestyle="--", alpha=0.3, zorder=1)
    _ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # ── Legend ──────────────────────────────────────────────────────────────────
    _legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, 
                   label="Both Inactive", markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=10, 
                   label="LDH Active", markeredgecolor="white"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, 
                   label="MTT Active", markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markerfacecoloralt="black",
                   markersize=10, fillstyle="left", label="Both Active", markeredgecolor="black"),
    ]
    _ax.legend(handles=_legend_elements, loc="upper right", frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/Downloads/mt_ldh_predictions_botanical_constituents_level4.png"),
                bbox_inches="tight", dpi=300)
    plt.show()

    print(f"Plotting complete. Rows generated: {len(_ordered_labels)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Correlation prediction vs measured BSC (models) - Level 6 - MT - LDH""")
    return


@app.cell
def _(df_level6_axiom_all):
    df_level6_axiom_all
    return


@app.cell
def _(pred_matrix_filtered):
    pred_matrix_filtered[["Metadata_compound", "MT", "LDH"]]
    return


@app.cell
def _(df_table_results_pods):
    df_table_results_pods[["Compound ID", "MTT POD (µM)", "LDH POD (µM)"]]
    return


@app.cell
def _():
    from sklearn.metrics import roc_auc_score
    return (roc_auc_score,)


@app.cell
def _(df_table_results_pods, pd, pred_matrix_filtered):
    # Copy to avoid modifying original
    df_true = df_table_results_pods.copy()

    # Convert POD columns to binary
    df_true["MT_label"] = df_true["MTT POD (µM)"].notna().astype(int)
    df_true["LDH_label"] = df_true["LDH POD (µM)"].notna().astype(int)
    df_pred = pred_matrix_filtered.copy()

    # Keep only needed columns
    df_pred = df_pred[["Metadata_compound", "MT", "LDH"]]
    df_pred_true_mt_ldh = pd.merge(
        df_pred,
        df_true,
        left_on="Metadata_compound",
        right_on="Compound ID",
        how="inner"   # only matched compounds
    )
    df_pred_true_mt_ldh
    print("Merged shape:", df_pred_true_mt_ldh.shape)
    print("Missing MT preds:", df_pred_true_mt_ldh["MT"].isna().sum())
    print("Missing LDH preds:", df_pred_true_mt_ldh["LDH"].isna().sum())
    return (df_pred_true_mt_ldh,)


@app.cell
def _(df_pred_true_mt_ldh, roc_auc_score):
    # Drop rows with missing predictions
    df_mt = df_pred_true_mt_ldh.dropna(subset=["MT", "MT_label"])
    df_ldh = df_pred_true_mt_ldh.dropna(subset=["LDH", "LDH_label"])

    auroc_mt = roc_auc_score(df_mt["MT_label"], df_mt["MT"])
    auroc_ldh = roc_auc_score(df_ldh["LDH_label"], df_ldh["LDH"])

    print(f"AUROC MT:  {auroc_mt:.3f}")
    print(f"AUROC LDH: {auroc_ldh:.3f}")
    return df_ldh, df_mt


@app.cell
def _(df_ldh, df_mt, np):
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
    auprg_mt = prg_auc(df_mt["MT_label"], df_mt["MT"])
    auprg_ldh = prg_auc(df_ldh["LDH_label"], df_ldh["LDH"])

    print(f"AUPRG MT:  {auprg_mt:.3f}")
    print(f"AUPRG LDH: {auprg_ldh:.3f}")
    return


if __name__ == "__main__":
    app.run()
