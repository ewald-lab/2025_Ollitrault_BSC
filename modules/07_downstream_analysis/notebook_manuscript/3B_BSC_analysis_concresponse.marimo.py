import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import argparse
    import sys
    return argparse, mo, sys


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from rdkit import Chem
    import matplotlib.pyplot as plt
    return np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""# 3. B. Analysis of Cell Painting dose response analysis: BSC""")
    return


@app.cell
def _(argparse, mo, sys):
    # Define the parser
    parser = argparse.ArgumentParser(description="My Marimo Report")
    parser.add_argument("--name", type=str, default="Default")
    parser.add_argument("--input", type=str, default="modules/03_postprocessing/outputs/level5_bsc_all_features_filtered.parquet") # TODO : adapt for level6
    parser.add_argument("--input_pods_cp_all", type=str, default=
        "modules/04_concresponse/outputs/pods/pods_bsc_cp_feature_wise_all.parquet")
    parser.add_argument("--input_pods_cp", type=str, default=
        "modules/04_concresponse/outputs/pods/pods_bsc_cp.parquet")
    parser.add_argument("--input_pods_metadata", nargs="+", default=[
        "modules/04_concresponse/outputs/pods/pods_bsc_Metadata_ldh_normalized_original_conc.parquet",
        "modules/04_concresponse/outputs/pods/pods_bsc_Metadata_mtt_normalized_original_conc.parquet"
                                                           ])
    parser.add_argument("--control", type=str, default="s0617825")
    parser.add_argument("--metadata_cell_viability", nargs="+", default=[
        "Metadata_mtt_normalized", "Metadata_ldh_normalized"])

    # Parse arguments from sys.argv
    # marimo ensures sys.argv[1:] contains only the args after the '--'
    args, unknown = parser.parse_known_args(sys.argv[1:])

    # Use the parsed values in your notebook
    mo.md(f"# Report for {args.name}")
    mo.md(f"Input path for {args.name} of data cp pods : {args.input_pods_cp}; metadata pods : {args.input_pods_metadata}")
    return (args,)


@app.cell
def _(mo):
    mo.md(r"""# Input""")
    return


@app.cell
def _(args, pd):
    # TODO Add df_level6 at the compound level
    df_level = pd.read_parquet(args.input)
    len_compounds = df_level[df_level["Metadata_compound"] != args.control]["Metadata_compound"].nunique()
    len_compounds
    return (len_compounds,)


@app.cell
def _(args, pd):
    df_cp_pods_all = pd.read_parquet(args.input_pods_cp_all)
    if "cp_all_POD_um" not in df_cp_pods_all.columns and "cp_POD_um" in df_cp_pods_all.columns:
        df_cp_pods_all["cp_all_POD_um"] = df_cp_pods_all["cp_POD_um"]
    df_cp_pods_all.head()
    return (df_cp_pods_all,)


@app.cell
def _(args, pd):
    df_cp_pods = pd.read_parquet(args.input_pods_cp)
    df_cp_pods.head()
    return (df_cp_pods,)


@app.cell
def _(args, pd):
    # Load all files into a dictionary: {filename_base: DataFrame}
    dict_df_meta_pods = {pf.split('/')[-1].replace('.parquet',''): pd.read_parquet(pf) for pf in args.input_pods_metadata}
    return (dict_df_meta_pods,)


@app.cell
def _(dict_df_meta_pods):
    dict_df_meta_pods
    return


@app.cell
def _(mo):
    mo.md(r"""# Table number of active""")
    return


@app.cell
def _(df_cp_pods, dict_df_meta_pods, pd):
    dict_all_dfs_pods = {'CP_pods': df_cp_pods, **dict_df_meta_pods}

    # Prepare summary table
    _summary_rows = []

    for _name, _df in dict_all_dfs_pods.items():
        _n_rows = len(_df)
        _n_all_pass = _df['all.pass'].sum()

        # check if 'adv.incr' exists in the df
        if 'adv.incr' in _df.columns:
            # Count all.pass & adv.incr == True
            _n_allpass_adv_true = _df[_df['all.pass'] & (_df['adv.incr'] == True)].shape[0]
            # Count all.pass & adv.incr == False
            _n_allpass_adv_false = _df[_df['all.pass'] & (_df['adv.incr'] == False)].shape[0]

        _summary_rows.append({
            'dataset': _name,
            'n_rows': _n_rows,
            'n_all_pass': _n_all_pass,
            'n_allpass_adv_true': _n_allpass_adv_true,
            'n_allpass_adv_false': _n_allpass_adv_false
        })

    summary_table_pods = pd.DataFrame(_summary_rows)
    summary_table_pods
    return (dict_all_dfs_pods,)


@app.cell
def _(mo):
    mo.md(r"""# Barplot number of actives""")
    return


@app.cell
def _(dict_all_dfs_pods, len_compounds, plt):
    labels = []
    counts = []
    percentages = []

    for _name, _df in dict_all_dfs_pods.items():
        _n_pass = _df[_df['all.pass'] == True].shape[0]
        labels.append(_name)
        counts.append(_n_pass)
        percentages.append(100 * _n_pass / len_compounds)

    # Plot
    plt.figure(figsize=(6, 8))
    bars = plt.bar(labels, counts)

    plt.ylabel("Number of active compounds (all.pass = True)")
    plt.xlabel("POD type")
    plt.title("Passing compounds per POD")
    plt.xticks(rotation=45, ha='right')

    # Add % on top of bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{pct:.1f}% ({bar.get_height()} Actives)",
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Boxplot PODs""")
    return


@app.cell
def _(args, dict_all_dfs_pods, pd):
    df_all_pods = pd.concat(
        [
            df.assign(source=name)
            for name, df in dict_all_dfs_pods.items()
        ],
        ignore_index=True
    )
    df_all_pods_allpass = df_all_pods[df_all_pods['all.pass'] == True]
    pod_columns = ['cp_POD_um'] + [
        f"{m}_POD_um"
        for m in args.metadata_cell_viability
        if f"{m}_POD_um" in df_all_pods_allpass.columns
    ]
    df_plot_all_pods_allpass = df_all_pods_allpass[pod_columns].dropna(how='all')
    return df_all_pods_allpass, df_plot_all_pods_allpass, pod_columns


@app.cell
def _(df_all_pods_allpass, pod_columns):
    desc_by_source = (
        df_all_pods_allpass
        .melt(
            id_vars="source",
            value_vars=pod_columns,
            value_name="POD"
        )
        .dropna(subset=["POD"])
        .groupby("source")["POD"]
        .describe()
    )

    desc_by_source
    return


@app.cell
def _(df_plot_all_pods_allpass, np, plt, pod_columns):
    plt.figure(figsize=(10, 5))

    data_plot_all_pods_allpass = [df_plot_all_pods_allpass[col].dropna() for col in pod_columns]
    labels_plot_all_pods_allpass = [
        f"{col}\n(n={df_plot_all_pods_allpass[col].notna().sum()})"
        for col in pod_columns
    ]

    plt.figure(figsize=(10, 5))

    # Boxplot
    plt.boxplot(
        data_plot_all_pods_allpass,
        labels=labels_plot_all_pods_allpass,
        showfliers=False
    )

    # Jitter overlay
    for _i, values in enumerate(data_plot_all_pods_allpass, start=1):
        x_jitter = np.random.normal(loc=_i, scale=0.04, size=len(values))
        plt.scatter(
            x_jitter,
            values,
            alpha=0.4,
            s=15
        )

    plt.ylabel("POD (µM)")
    plt.title("POD distributions (all.pass = True)")
    plt.xticks(rotation=45, ha='right')
    plt.yscale("log")   # strongly recommended for PODs

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Table number of active grouped by channels""")
    return


@app.cell
def _(df_cp_pods_all, pd):
    _summary_rows = []

    for gene, _df in df_cp_pods_all.groupby("gene.id"):

        _n_rows = len(_df)
        _n_all_pass = _df["all.pass"].sum()

        if "adv.incr" in _df.columns:
            _n_allpass_adv_true = _df[
                (_df["all.pass"]) & (_df["adv.incr"] == True)
            ].shape[0]

            _n_allpass_adv_false = _df[
                (_df["all.pass"]) & (_df["adv.incr"] == False)
            ].shape[0]
        else:
            _n_allpass_adv_true = None
            _n_allpass_adv_false = None

        _summary_rows.append({
            "dataset": gene,
            "n_rows": _n_rows,
            "n_all_pass": _n_all_pass,
            "n_allpass_adv_true": _n_allpass_adv_true,
            "n_allpass_adv_false": _n_allpass_adv_false
        })

    summary_table_pods_channels = pd.DataFrame(_summary_rows)
    summary_table_pods_channels
    return


@app.cell
def _(mo):
    mo.md(r"""# Boxplot PODs grouped by channels""")
    return


@app.cell
def _(df_cp_pods_all, np, plt):
    # Filter your data: keep all.pass True and finite POD values > 0
    df_plot_cp_pods_all = df_cp_pods_all[
        (df_cp_pods_all["all.pass"] == True) &
        (np.isfinite(df_cp_pods_all["cp_all_POD_um"])) &
        (df_cp_pods_all["cp_all_POD_um"] < 100000000) &
        (df_cp_pods_all["cp_all_POD_um"] > 0)  # log scale requires > 0
    ]

    # Group by gene.id
    _groups = df_plot_cp_pods_all.groupby("gene.id")

    data_plot = [g["cp_all_POD_um"].values for _, g in _groups]
    _labels = [name for name, _ in _groups]

    plt.figure(figsize=(max(6, len(_labels) * 0.8), 6))

    # Boxplot
    plt.boxplot(data_plot, labels=_labels, showfliers=False)

    # Jitter points
    for _i, _values in enumerate(data_plot, start=1):
        if len(_values) == 0:
            continue  # skip empty groups
        _x_jitter = np.random.normal(_i, 0.04, size=len(_values))
        plt.scatter(_x_jitter, _values, alpha=0.5)

    plt.ylabel("POD (µM)")
    plt.title("CP POD distributions by gene (all.pass = True)")
    plt.yscale("log")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
