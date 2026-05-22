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
    import matplotlib.pyplot as plt
    # mAP
    from copairs import map
    from copairs.matching import assign_reference_index
    from copairs.map.average_precision import p_values
    return assign_reference_index, map, mo, np, p_values, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis BSC mAP""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Input""")
    return


@app.cell
def _(pd):
    # Cell Painting profiles
    df_level4_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level4b_bsc_filtered.parquet")
    if True:
        df_level4_bsc = df_level4_bsc.groupby("Metadata_plate").filter(
            lambda g: ~(g["Metadata_compound_name"] == "DMSO").all()
        )
    #df_level5_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level5_bsc_filtered.parquet")
    return (df_level4_bsc,)


@app.cell
def _(pd):
    # BSC Annotations
    df_bsc_annotations = pd.read_csv("modules/01_preprocessing/outputs/cellpainting_bsc_annotations_merged.tsv", sep="\t")
    list_botanicals = df_bsc_annotations["Plant scode"].unique()
    bsc_control = "s0617825"
    list_botanicals
    return (df_bsc_annotations,)


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
def _(df_bsc_annotations):
    compound_to_plant = (
        df_bsc_annotations[["Metadata_compound", "Plant name"]]
        .drop_duplicates()
        .dropna()
        .set_index("Metadata_compound")["Plant name"]
        .to_dict()
    )
    compound_to_plant
    return (compound_to_plant,)


@app.cell
def _(df_level4_bsc):
    all_selected_features = [c for c in df_level4_bsc.columns if c.startswith("Feature_")]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Add columns infos and source_botanical 
    Metadata_type : botanical,constituent,dmso
    Metadata_source_botanical : botanical, corresponding botanical of constiteunts
    """
    )
    return


@app.cell
def _(compound_to_plant, df_level4_bsc, dic_botanicals_scode_name):
    # Add metadata columns
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _classify_compound(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Constituent"

    def _get_botanical_name(_compound):
        if _compound in _botanical_scodes:
            return dic_botanicals_scode_name.get(_compound, "—")
        elif _compound in compound_to_plant:
            return compound_to_plant[_compound]
        else:
            return "DMSO"

    # Add columns to level4
    df_level4_bsc["Metadata_type"] = df_level4_bsc["Metadata_compound"].apply(_classify_compound)
    df_level4_bsc["Metadata_source_botanical"] = df_level4_bsc["Metadata_compound"].apply(_get_botanical_name)

    # Add columns to level5
    #df_level5_bsc["Metadata_type"] = df_level5_bsc["Metadata_compound"].apply(_classify_compound)
    #df_level5_bsc["Metadata_source_botanical"] = df_level5_bsc["Metadata_compound"].apply(_get_botanical_name)
    return


@app.cell
def _(df_level4_bsc):
    df_level4_bsc["Metadata_type"]
    return


@app.cell
def _(df_level4_bsc):
    df_level4_bsc["Metadata_source_botanical"]
    return


@app.cell
def _(mo):
    mo.md(r"""## Filter profiles concentrations above POD""")
    return


@app.cell
def _(pd):
    df_pod = pd.read_csv("modules/06_concresponse/outputs/pods/df_pods_bsc_cp_mt_ldh.csv")
    return (df_pod,)


@app.cell
def _(df_pod):
    df_pod[["Compound ID", "CP POD (µM)"]]
    return


@app.cell
def _(df_level4_bsc):
    df_level4_bsc
    return


@app.cell
def _(df_level4_bsc, df_pod):
    filter_concentration_above_pod = True
    if filter_concentration_above_pod:
        df_level4_bsc_pod = (
            df_level4_bsc
            .merge(
                df_pod[["Compound ID", "CP POD (µM)"]].drop_duplicates(subset="Compound ID").rename(columns={"CP POD (µM)": "Metadata_cp_pod_um"}),
                left_on="Metadata_compound",
                right_on="Compound ID",
                how="left"
            )
            .drop(columns="Compound ID")
        )
        df_level4_bsc_pod = df_level4_bsc_pod[
            (df_level4_bsc_pod["Metadata_concentration"] >= df_level4_bsc_pod["Metadata_cp_pod_um"])
            | (df_level4_bsc_pod["Metadata_type"] == "DMSO")
        ]
    else:
        df_level4_bsc_pod = df_level4_bsc
    return (df_level4_bsc_pod,)


@app.cell
def _(mo):
    mo.md(r"""## mAP level 4 : phenotypic consistency botanicals""")
    return


@app.cell
def _(assign_reference_index, df_level4_bsc_pod):
    reference_col = "Metadata_reference_index"

    df_activity = assign_reference_index(
        df_level4_bsc_pod[df_level4_bsc_pod["Metadata_type"] != "DMSO"],
        "Metadata_type == 'Constituent'",  # condition to get reference profiles (neg controls)
        reference_col=reference_col,
        default_value=-1,
    )
    return (df_activity,)


@app.cell
def _(df_activity):
    df_activity[["Metadata_compound", "Metadata_source_botanical", "Metadata_reference_index", "Metadata_type"]]
    return


@app.cell
def _():
    # positive pairs are replicates of the same treatment
    pos_sameby = ["Metadata_source_botanical"]
    pos_diffby = ["Metadata_type"]

    neg_sameby = []
    # negative pairs are replicates of different treatments
    neg_diffby = ["Metadata_compound", "Metadata_source_botanical", "Metadata_type"]
    return neg_diffby, neg_sameby, pos_diffby, pos_sameby


@app.cell
def _(df_activity, map, neg_diffby, neg_sameby, pos_diffby, pos_sameby):
    metadata = df_activity.filter(regex="^Metadata")
    profiles = df_activity.filter(regex="^(?!Metadata)").values

    activity_ap = map.average_precision(
        metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
    )
    activity_ap = activity_ap.query("Metadata_type != 'DMSO'")  # remove DMSO
    return (activity_ap,)


@app.cell
def _(activity_ap, np, p_values, plt):
    # Calculate p-values using the same null_size and seed as mean_average_precision
    activity_ap["p_value"] = p_values(activity_ap, null_size=1000000, seed=0)
    activity_ap["-log10(p-value)"] = -activity_ap["p_value"].apply(np.log10)
    activity_ap["below_p"] = activity_ap["p_value"] < 0.05

    active_ratio_ap = activity_ap["below_p"].mean()

    # Plot raw and normalized AP scores vs -log10 p-values
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Raw AP vs -log10(p-value)
    axes[0].scatter(
        data=activity_ap,
        x="average_precision",
        y="-log10(p-value)",
        c="below_p",
        cmap="tab10",
        s=10,
    )
    axes[0].axhline(-np.log10(0.05), color="black", linestyle="--")
    axes[0].set_xlabel("AP")
    axes[0].set_ylabel("-log10(p-value)")
    axes[0].set_title("Replicate retrieval")
    axes[0].text(
        0.65,
        1.5,
        f"Retrieved = {100 * active_ratio_ap:.2f}%",
        va="center",
        ha="left",
    )
    if False :
        # Plot 2: Normalized AP vs -log10(p-value)
        axes[1].scatter(
            data=activity_ap,
            x="normalized_average_precision",
            y="-log10(p-value)",
            c="below_p",
            cmap="tab10",
            s=10,
        )
        axes[1].axhline(-np.log10(0.05), color="black", linestyle="--")
        axes[1].axvline(0, color="black", linestyle="--", linewidth=0.5)
        axes[1].set_xlabel("normalized AP")
        axes[1].set_ylabel("-log10(p-value)")
        axes[1].set_title("Replicate retrieval (normalized AP)")
        axes[1].text(
            0.65,
            1.5,
            f"Retrieved = {100 * active_ratio_ap:.2f}%",
            va="center",
            ha="left",
        )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(activity_ap, map, np, pos_sameby):
    # m AP compound level
    activity_map = map.mean_average_precision(
        activity_ap, pos_sameby, null_size=1000000, threshold=0.05, seed=0
    )
    activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(np.log10)
    activity_map[["Metadata_source_botanical", "below_p","below_corrected_p","-log10(p-value)"]]
    return


@app.cell
def _(mo):
    mo.md(r"""## mAP level 4 : phenotypic consistency""")
    return


@app.cell
def _(activity_ap):
    activity_ap
    return


if __name__ == "__main__":
    app.run()
