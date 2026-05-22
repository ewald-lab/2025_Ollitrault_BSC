import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import sys
    sys.path.append("/Users/ollitrault/Documents/projects/utils")
    from util_standardizer import StandardizeMolecule
    return StandardizeMolecule, mo, pd


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# Compare Cell Painting chemicals data and BSC annotation data""")
    return


@app.cell
def _(pd):
    # Load cell painting bsc data
    df_level5 = pd.read_parquet("modules/03_postprocessing/outputs/level5_bsc_filtered.parquet")
    meta_cols_compounds = [c for c in df_level5.columns if c.startswith("Metadata_compound")]
    # Get only unique chemicals annotations 
    df_level5_compounds = df_level5[meta_cols_compounds].drop_duplicates()
    return (df_level5_compounds,)


@app.cell
def _(pd):
    # Load bsc chemical annotation data
    df_bsc_annotations = pd.read_csv("00.first_analysis_module/00_inputs/Botanical_Constituents_and_Quantification_08192024/Quant Constituents-Table 1.tsv", sep="\t")
    return (df_bsc_annotations,)


@app.cell
def _(StandardizeMolecule, df_bsc_annotations, df_level5_compounds):
    # standardize
    df_level5_compounds_standardized = StandardizeMolecule(df_level5_compounds, input_column="Metadata_compound_smiles").run()
    df_bsc_annotations_standardized = StandardizeMolecule(df_bsc_annotations, input_column="SMILES").run()
    return df_bsc_annotations_standardized, df_level5_compounds_standardized


@app.cell
def _(df_level5_compounds_standardized):
    df_level5_compounds_standardized[df_level5_compounds_standardized["Metadata_compound"] == "s0654806"]
    return


@app.cell
def _():
    "Cn1c(=O)c2[nH]cnc2n(C)c1=O" == "Cn1c(=O)c2nc[nH]c2n(C)c1=O"
    return


@app.cell
def _(df_bsc_annotations_standardized):
    df_bsc_annotations_standardized
    return


@app.cell
def _(df_bsc_annotations_standardized, df_level5_compounds_standardized):
    df_merge_cp_cp_annotations_iso = df_level5_compounds_standardized.merge(
        df_bsc_annotations_standardized,
        on="SMILES_standardized",
        how="inner"
    )
    df_merge_cp_cp_annotations_iso
    return (df_merge_cp_cp_annotations_iso,)


@app.cell
def _(df_merge_cp_cp_annotations_iso):
    df_merge_cp_cp_annotations_iso["Metadata_compound"].nunique()
    return


@app.cell
def _(df_bsc_annotations_standardized, df_level5_compounds_standardized):
    df_merge_cp_cp_annotations_noiso = df_level5_compounds_standardized.merge(
        df_bsc_annotations_standardized,
        on="SMILES_standardized_nonisomeric",
        how="inner"
    )
    df_merge_cp_cp_annotations_noiso
    return (df_merge_cp_cp_annotations_noiso,)


@app.cell
def _(df_merge_cp_cp_annotations_iso):
    df_merge_cp_cp_annotations_iso["SMILES_standardized"].nunique()
    return


@app.cell
def _(df_merge_cp_cp_annotations_noiso):
    df_merge_cp_cp_annotations_noiso["SMILES_standardized_nonisomeric"].nunique()
    return


@app.cell
def _(df_merge_cp_cp_annotations_iso):
    df_merge_cp_cp_annotations_iso[
        df_merge_cp_cp_annotations_iso["SMILES_standardized"].duplicated(keep=False)
    ].sort_values("SMILES_standardized")
    return


@app.cell
def _(df_merge_cp_cp_annotations_iso):
    df_merge_cp_cp_annotations_iso["Plant scode"].unique().tolist()
    return


@app.cell
def _(df_merge_cp_cp_annotations_iso):
    df_merge_cp_cp_annotations_iso
    return


@app.cell
def _(df_merge_cp_cp_annotations_iso):
    df_merge_cp_cp_annotations_iso[['Metadata_compound_name', 'Metadata_compound_smiles',
           'Metadata_compound_inchi', 'Metadata_compound',
           'Metadata_compound_target', 'Metadata_compound_pathway',
           'Metadata_compound_biological_activity', 'Metadata_compound_saltdata',
           'Metadata_compound_formula', 'Metadata_compound_solubility',
           'Metadata_compound_url', 'Metadata_compound_research_area',
           'Metadata_compound_clinical_information', 'Metadata_compound_solvent',
           'Metadata_compound_solvent_concentration',
           'SMILES_standardized',
           'Plant name', 'Scientific name', 'Constituent', 'Plant scode', 'DTXID',
           'MW (g/mol)', 'Concentration (mg/g) ',
           'Concentration notes\n* denotes the sum of constituent quantification from isomers\n**there are multiple constituents that eluted at the same time, assuming maximum concencentration for each',
           'Lab', 'IUPAC name', 'Constituent CAS', 'INCHI Key', 'PubChem ID',
           'Canonical SMILES', 'SMILES']].to_csv("modules/01_preprocessing/outputs/cellpainting_bsc_annotations_merged.tsv", sep="\t")
    return


if __name__ == "__main__":
    app.run()
