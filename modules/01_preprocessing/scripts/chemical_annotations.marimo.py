import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import sys
    import argparse
    return argparse, mo, pd, sys


@app.cell
def _(argparse, mo, sys):
    # Define the parser
    parser = argparse.ArgumentParser(description="My Marimo Report")
    parser.add_argument("--name", type=str, default="Default")
    parser.add_argument("--input_bsc_cp", type=str, default="modules/01_preprocessing/outputs/level2_bsc_log10conc.parquet")
    parser.add_argument("--input_bsc_annotations", type=str, default="modules/01_preprocessing/inputs/Botanical_Constituents_and_Quantification_08192024/Quant Constituents-Table 1.tsv")
    parser.add_argument("--output_annotations", type=str, default="modules/01_preprocessing/outputs/cellpainting_bsc_annotations_merged.tsv")
    parser.add_argument("--output_compounds", type=str, default="modules/01_preprocessing/outputs/compounds_bsc.tsv")
    parser.add_argument("--path_utils_standardizer", type=str, default="/Users/ollitrault/Documents/projects/utils")

    # Parse arguments from sys.argv
    # marimo ensures sys.argv[1:] contains only the args after the '--'
    args, unknown = parser.parse_known_args(sys.argv[1:])

    # Use the parsed values in your notebook
    mo.md(f"# Compare Cell Painting chemicals data and BSC annotation data")
    mo.md(f"Create file cellpainting_bsc_annotations_merged")
    return (args,)


@app.cell
def _(args, sys):
    # add path script utils
    sys.path.append(args.path_utils_standardizer)
    from util_standardizer import StandardizeMolecule
    return (StandardizeMolecule,)


@app.cell
def _(mo):
    mo.md(r"""# Compare Cell Painting chemicals data and BSC annotation data""")
    return


@app.cell
def _(args, pd):
    # Load cell painting bsc data
    df_level = pd.read_parquet(args.input_bsc_cp)
    meta_cols_compounds = ['Metadata_compound_name', 'Metadata_compound_smiles',
           'Metadata_compound_inchi', 'Metadata_compound',
           'Metadata_compound_target', 'Metadata_compound_pathway',
           'Metadata_compound_biological_activity', 'Metadata_compound_saltdata',
           'Metadata_compound_formula', 'Metadata_compound_solubility',
           'Metadata_compound_url', 'Metadata_compound_research_area',
           'Metadata_compound_clinical_information', 'Metadata_compound_solvent',
           'Metadata_compound_solvent_concentration']
    # Get only unique chemicals annotations 
    df_level_compounds = df_level[meta_cols_compounds].drop_duplicates()
    return (df_level_compounds,)


@app.cell
def _(args, pd):
    # Load bsc chemical annotation data
    df_bsc_annotations = pd.read_csv(args.input_bsc_annotations, sep="\t")
    return (df_bsc_annotations,)


@app.cell
def _(StandardizeMolecule, df_bsc_annotations, df_level_compounds):
    # standardize
    df_level_compounds_standardized = StandardizeMolecule(df_level_compounds, input_column="Metadata_compound_smiles").run()
    df_bsc_annotations_standardized = StandardizeMolecule(df_bsc_annotations, input_column="SMILES").run()
    return df_bsc_annotations_standardized, df_level_compounds_standardized


@app.cell
def _(df_level_compounds_standardized):
    df_level_compounds_standardized[df_level_compounds_standardized["Metadata_compound"] == "s0654806"]
    return


@app.cell
def _(df_bsc_annotations_standardized, df_level_compounds_standardized):
    df_merge_cp_cp_annotations_iso = df_level_compounds_standardized.merge(
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
def _(df_bsc_annotations_standardized, df_level_compounds_standardized):
    df_merge_cp_cp_annotations_noiso = df_level_compounds_standardized.merge(
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
def _(args, df_merge_cp_cp_annotations_iso):
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
           'Canonical SMILES', 'SMILES']].to_csv(args.output_annotations, sep="\t")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Write file all botanicals and constituents scode""")
    return


@app.cell
def _(df_merge_cp_cp_annotations_iso):
    len(df_merge_cp_cp_annotations_iso["Metadata_compound"].unique())
    return


@app.cell
def _(df_merge_cp_cp_annotations_iso):
    len(df_merge_cp_cp_annotations_iso["Plant scode"].unique())
    return


@app.cell
def _(df_merge_cp_cp_annotations_iso):
    list_scode_botanicals_constituents = list(df_merge_cp_cp_annotations_iso["Metadata_compound"].unique()) + list(df_merge_cp_cp_annotations_iso["Plant scode"].unique())
    list_scode_botanicals_constituents
    return (list_scode_botanicals_constituents,)


@app.cell
def _(args, list_scode_botanicals_constituents):
    with open(args.output_compounds, "w", encoding="utf-8") as file:
        for item in list_scode_botanicals_constituents:
            file.write(f"{item}\n")
    return


if __name__ == "__main__":
    app.run()
