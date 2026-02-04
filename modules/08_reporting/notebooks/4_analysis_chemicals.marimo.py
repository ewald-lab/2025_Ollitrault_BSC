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
    import numpy as np
    from rdkit import Chem
    return Chem, np, pd


@app.cell
def _(pd):
    df_level5b = pd.read_parquet("02_processed_data/level5_bsc.parquet")
    return (df_level5b,)


@app.cell
def _(pd):
    df_level4b = pd.read_parquet("02_processed_data/level4b_bsc.parquet")
    meta_compound_cols = [c for c in df_level4b.columns if c.startswith("Metadata_compound")]
    df_metadata = df_level4b[meta_compound_cols].drop_duplicates()
    return (df_metadata,)


@app.cell
def _(pd):
    df_annotations_composition = pd.read_csv("00_inputs/Botanical_Constituents_and_Quantification_08192024/Quant Constituents-Table 1.tsv", sep="\t")
    return (df_annotations_composition,)


@app.cell
def _(df_annotations_composition):
    df_annotations_composition.head()
    return


@app.cell
def _(df_level5b):
    col_metadata = []
    for _col in df_level5b.columns:
        if "Metadata" in _col:
            col_metadata.append(_col)
            print(_col)
    return


@app.cell
def _(mo):
    mo.md(r"""# Analysis chemicals mixtures""")
    return


@app.cell
def _(df_level5b):
    df_chem_infos = df_level5b[['Metadata_compound',
                                'Metadata_compound_name',
                                'Metadata_compound_smiles',
                                'Metadata_compound_inchi',
                                #'Metadata_compound_scode',
                                'Metadata_compound_target',
                                'Metadata_compound_pathway',
                                'Metadata_compound_biological_activity',
                                'Metadata_compound_saltdata',
                                'Metadata_compound_formula',
                                'Metadata_compound_solubility',
                                'Metadata_compound_url',
                                'Metadata_compound_research_area',
                                'Metadata_compound_clinical_information',
                                'Metadata_compound_solvent',
                                'Metadata_compound_solvent_concentration'
                               ]].drop_duplicates()
    return (df_chem_infos,)


@app.cell
def _(df_chem_infos):
    df_chem_infos.shape
    return


@app.cell
def _(df_chem_infos):
    df_chem_infos["Metadata_compound"].nunique()
    return


@app.cell
def _(df_chem_infos):

    df_chem_infos["Metadata_compound_inchi"].nunique()
    return


@app.cell
def _(df_chem_infos):
    df_chem_infos["Metadata_compound_smiles"].nunique()
    return


@app.cell
def _(df_chem_infos):
    df_chem_infos["Metadata_compound_name"].nunique()
    return


@app.cell
def _(Chem, df_chem_infos, np, pd):
    def count_fragments(smiles):
        """Return number of molecular fragments in a SMILES string using RDKit."""
        if pd.isna(smiles) or not smiles.strip():
            return np.nan
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan  # invalid SMILES
        return len(Chem.GetMolFrags(mol))

    # Apply to your dataframe
    df_chem_infos["Metadata_n_fragments"] = df_chem_infos["Metadata_compound_smiles"].apply(count_fragments)
    return


@app.cell
def _(df_chem_infos):
    df_chem_infos["Metadata_n_fragments"].value_counts(dropna=False)
    return


@app.cell
def _(df_chem_infos):
    df_chem_infos[df_chem_infos["Metadata_n_fragments"] != 1]
    return


@app.cell
def _(df_chem_infos):
    df_chem_infos[df_chem_infos["Metadata_compound_name"].str.contains("dole", na=False)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis UMAP [to extand to all CP data]""")
    return


@app.cell
def _():
    import umap.umap_ as umap
    import matplotlib.pyplot as plt
    from rdkit.Chem import AllChem
    import sys
    sys.path.append("/Users/ollitrault/Documents/projects/utils")
    from util_standardizer import StandardizeMolecule
    return AllChem, StandardizeMolecule, plt, umap


@app.cell
def _(AllChem, Chem, np):
    def smiles_to_morgan(smiles, radius=2, n_bits=2048):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    #def canonicalize(smiles):
    #    mol = Chem.MolFromSmiles(smiles)
    #    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else None
    return (smiles_to_morgan,)


@app.cell
def _(StandardizeMolecule, df_annotations_composition, df_metadata):
    df_metadata_standardized = StandardizeMolecule(df_metadata, input_column="Metadata_compound_smiles").run()
    df_annotations_composition_standardized = StandardizeMolecule(df_annotations_composition, input_column="Canonical SMILES").run()
    return df_annotations_composition_standardized, df_metadata_standardized


@app.cell
def _(df_annotations_composition_standardized, df_metadata_standardized):
    # drop na
    df_metadata_standardized_nona = df_metadata_standardized[
        df_metadata_standardized["SMILES_standardized_nonisomeric"].notna()
    ]
    df_annotations_composition_standardized_nona = (
        df_annotations_composition_standardized[
            df_annotations_composition_standardized["SMILES_standardized_nonisomeric"].notna()
        ]
    )
    return (
        df_annotations_composition_standardized_nona,
        df_metadata_standardized_nona,
    )


@app.cell
def _(
    df_annotations_composition_standardized_nona,
    df_metadata_standardized_nona,
):
    df_metadata_annotations_composition = df_metadata_standardized_nona.merge(
        df_annotations_composition_standardized_nona,
        on="SMILES_standardized_nonisomeric",
        how="left"
    )
    return (df_metadata_annotations_composition,)


@app.cell
def _(df_metadata_annotations_composition):
    df_metadata_annotations_composition
    return


@app.cell
def _(df_annotations_composition_standardized, df_metadata_standardized):
    len(set(df_annotations_composition_standardized["SMILES_standardized_nonisomeric"]) & set(df_metadata_standardized["SMILES_standardized_nonisomeric"]))
    return


@app.cell
def _(df_metadata_annotations_composition):
    df_metadata_annotations_composition
    return


@app.cell
def _(df_metadata_annotations_composition, np, smiles_to_morgan):
    # Convert SMILES to fingerprints
    fps = df_metadata_annotations_composition["SMILES_standardized_nonisomeric"].apply(smiles_to_morgan)

    # Drop invalid SMILES
    mask = fps.notnull()
    fps = np.vstack(fps[mask])
    labels = df_metadata_annotations_composition.loc[mask, "Plant name"]
    labels = (
        df_metadata_annotations_composition
        .loc[mask, "Plant name"]
        .fillna("unkown")
    )
    return fps, labels


@app.cell
def _(fps, umap):
    reducer = umap.UMAP(
        n_neighbors=25,        # larger = smoother chemical space
        min_dist=0.05,         # tighter clusters
        n_components=2,
        metric="jaccard",      # IMPORTANT for binary fingerprints
        random_state=42
    )

    embedding = reducer.fit_transform(fps)
    return (embedding,)


@app.cell
def _(labels):
    distinct_colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#42d4f4', '#f032e6', '#fabed4', '#469990', '#dcbeff',
        '#9A6324', '#fffac8', '#800000', '#aaffc3', '#000075',
        '#a9a9a9', '#ffffff', '#000000'
    ]
    plants = labels.unique()

    if len(plants) > len(distinct_colors):
        raise ValueError("Not enough colors for the number of Plant names")

    color_map = dict(zip(plants, distinct_colors))
    return color_map, plants


@app.cell
def _(color_map, embedding, labels, plants, plt):
    plt.figure(figsize=(10, 8))

    for plant in plants:
        idx = labels == plant
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            label=plant,
            color=color_map[plant],
            s=20,
            alpha=0.8,
            edgecolors="none"
        )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP of Molecular Composition (colored by Plant name)")
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    # upset plot
    return


@app.cell
def _():
    from upsetplot import UpSet, from_indicators
    return UpSet, from_indicators


@app.cell
def _(df_metadata_annotations_composition):
    # Keep only needed columns and drop duplicates
    df = (
        df_metadata_annotations_composition[["SMILES_standardized_nonisomeric", "Plant name"]]
        .dropna()
        .drop_duplicates()
    )

    # Create a presence/absence table:
    # rows = SMILES_standardized_nonisomeric, columns = Plant name
    presence = (
        df.assign(value=True)
          .pivot_table(
              index="SMILES_standardized_nonisomeric",
              columns="Plant name",
              values="value",
              fill_value=False,
              aggfunc="any"
          )
    )
    return (presence,)


@app.cell
def _(UpSet, from_indicators, plt, presence):
    upset_data = from_indicators(presence.columns, presence)
    UpSet(
        upset_data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality"
    ).plot()

    plt.suptitle("UpSet plot of Canonical SMILES grouped by Plant name")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
