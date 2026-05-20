# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# Notebook modify botanical extracts µg/mL to µM (test on level 5)""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Input""")
    return


@app.cell
def _(pd):
    # load BSC level 5
    df_level5_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level5_bsc_filtered.parquet")
    return (df_level5_bsc,)


@app.cell
def _(pd):
    # load botanical extracts constiteunts infos
    df_bot_const_metadata = pd.read_csv("modules/03_postprocessing/inputs/Botanical_Constituents_and_Quantification_08192024/Quant Constituents-Table 1.tsv", sep="\t")
    return (df_bot_const_metadata,)


@app.cell
def _(df_bot_const_metadata):
    df_bot_const_metadata
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Compute average MW of the constiteunts per botanicals""")
    return


@app.cell
def _(df_bot_const_metadata):
    avg_mw = (
        df_bot_const_metadata
        .groupby('Plant scode', as_index=False)['MW (g/mol)']
        .mean()
        .rename(columns={'Plant scode': 'Metadata_compound', 'MW (g/mol)': 'avg_mw'})
    )
    return (avg_mw,)


@app.cell
def _(avg_mw):
    avg_mw
    return


@app.cell
def _(df_bot_const_metadata):
    list_botanical_scode = df_bot_const_metadata["Plant scode"].unique()
    return (list_botanical_scode,)


@app.cell
def _(avg_mw, df_level5_bsc, list_botanical_scode):
    # Formula to get uM : μM = (μg/mL×1000)/MW
    # average MW per compound
    # attach avg_mw to df_level5_bsc
    df_level5_bsc_avg_mw = df_level5_bsc.merge(avg_mw, on='Metadata_compound', how='left')

    # convert ug/mL to uM
    # keep only rows whose compound is in that list
    _mask = df_level5_bsc_avg_mw["Metadata_compound"].isin(list_botanical_scode)

    # convert ug/mL -> uM using avg_mw (g/mol)
    df_level5_bsc_avg_mw.loc[_mask, "Metadata_concentration"] = (
        df_level5_bsc_avg_mw.loc[_mask, "Metadata_concentration"] * 1000 / df_level5_bsc_avg_mw.loc[_mask, "avg_mw"]
    )
    return (df_level5_bsc_avg_mw,)


@app.cell
def _(df_level5_bsc_avg_mw):
    df_level5_bsc_avg_mw[["Metadata_compound","Metadata_concentration"]]
    return


@app.cell
def _(df_level5_bsc_avg_mw, list_botanical_scode):
    df_level5_bsc_avg_mw[df_level5_bsc_avg_mw["Metadata_compound"].isin(list_botanical_scode)][["Metadata_compound","Metadata_concentration"]]
    return


if __name__ == "__main__":
    app.run()
