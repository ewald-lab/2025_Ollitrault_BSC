import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    return (pd,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(pd):
    # Load Data
    df_level3 = pd.read_parquet("modules/02_processing/outputs/level3_bsc.parquet")
    return (df_level3,)


@app.cell
def _(df_level3):
    for feat_name in df_level3.columns:
        if "Feature" not in feat_name:
            print(feat_name)
    return


@app.cell
def _(df_level3):
    df_level3[[
                "Metadata_plate",
                "Metadata_compound_name",
                "Metadata_mtt_lumi",
                "Metadata_ldh_abs_signal",
                "Metadata_ldh_abs_background",
                "Metadata_ldh_abs",
                "Metadata_mtt_normalized",
                "Metadata_ldh_normalized",
                "Metadata_mtt_ridge_norm",
                "Metadata_ldh_ridge_norm",
                "Metadata_mtt_ridge_norm_inv",
                ]].drop_duplicates()
    return


@app.cell
def _(df_level3):
    df_level3[["Metadata_well_id",
                "Metadata_plate",
                "Metadata_compound_name",
                "Metadata_concentration",
                "Metadata_concentration",
                "Metadata_compound_smiles",
                "Metadata_compound_inchi",
                "Metadata_source",
                "Metadata_compound",
                "Metadata_compound_target",
                "Metadata_compound_pathway",
                "Metadata_compound_biological_activity",
                "Metadata_compound_saltdata",
                "Metadata_compound_formula",
                "Metadata_compound_solubility",
                "Metadata_compound_url",
                "Metadata_compound_research_area",
                "Metadata_compound_clinical_information",
                "Metadata_library",
                "Metadata_well",
                "Metadata_row",
                "Metadata_col",
                "Metadata_mtt_lumi",
                "Metadata_ldh_abs_signal",
                "Metadata_ldh_abs_background",
                "Metadata_ldh_abs",
                "Metadata_mtt_normalized",
                "Metadata_ldh_normalized",
                "Metadata_mtt_ridge_norm",
                "Metadata_ldh_ridge_norm",
                "Metadata_compound_solvent",
                "Metadata_compound_solvent_concentration",
                "Metadata_mtt_ridge_norm_inv",
                "Metadata_perturbation",
                ]].shape
    return


@app.cell
def _(df_level3):
    df_level3[["Metadata_well_id",
                "Metadata_plate",
                "Metadata_compound_name",
                "Metadata_concentration",
                "Metadata_concentration",
                "Metadata_compound_smiles",
                "Metadata_compound_inchi",
                "Metadata_source",
                "Metadata_compound",
                "Metadata_compound_target",
                "Metadata_compound_pathway",
                "Metadata_compound_biological_activity",
                "Metadata_compound_saltdata",
                "Metadata_compound_formula",
                "Metadata_compound_solubility",
                "Metadata_compound_url",
                "Metadata_compound_research_area",
                "Metadata_compound_clinical_information",
                "Metadata_library",
                "Metadata_well",
                "Metadata_row",
                "Metadata_col",
                "Metadata_mtt_lumi",
                "Metadata_ldh_abs_signal",
                "Metadata_ldh_abs_background",
                "Metadata_ldh_abs",
                "Metadata_mtt_normalized",
                "Metadata_ldh_normalized",
                "Metadata_mtt_ridge_norm",
                "Metadata_ldh_ridge_norm",
                "Metadata_compound_solvent",
                "Metadata_compound_solvent_concentration",
                "Metadata_mtt_ridge_norm_inv",
                "Metadata_perturbation",
                ]].to_csv("modules/02_processing/outputs/bsc_raw_metadata_MT_LDH_assays.tsv", sep="\t")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
