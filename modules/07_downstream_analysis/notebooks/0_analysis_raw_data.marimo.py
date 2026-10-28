import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Welcome to the Analysis workflow of BSC Cell Painting dataset""")
    return


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    return np, os, pd, plt, px


@app.cell
def _(os):
    os.getcwd()
    return


@app.cell
def _(pd):
    # Load Data
    dftest = pd.read_parquet("/Users/ollitrault/Documents/projects/2025_04_13_OASIS_CellPainting/data/external/annotations/rat_invivo/binary_rat_invivo_tox_noendpointnegatives_v2.parquet")
    dftest
    return


@app.cell
def _(pd):
    df = pd.read_parquet("modules/01_preprocessing/inputs/bsc_raw_dinov2_biochem.parquet")
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""## Explore columns""")
    return


@app.cell
def _(df):
    # Column names
    df.columns
    return


@app.cell
def _(df):
    df["library"].value_counts()
    return


@app.cell
def _(df):
    df[df["library"] == "botanical"]['compound_concentration_um']
    return


@app.cell
def _(df):
    # Number of rows
    len(df)
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""## Explore chemical informations""")
    return


@app.cell
def _(df, mo):
    # Get all columns that contain 'compound_'
    compound_columns = [col for col in df.columns if "compound_" in col]

    # Create and display value counts for each compound column
    results_chemicals = []
    for col in compound_columns:
        results_chemicals.append(mo.md(f"### {col}"))
        counts = df[col].value_counts().reset_index()
        counts.columns = ["Value", "Count"]
        results_chemicals.append(mo.ui.table(counts))

    mo.vstack(results_chemicals)
    return


@app.cell
def _(df, mo, px):
    # Get unique combinations of compound_smiles and compound_concentration_um
    unique_compound_smiles_and_concentration = df[["compound_smiles", "compound_concentration_um"]].drop_duplicates()

    # Create histogram plot
    fig1 = px.histogram(
        unique_compound_smiles_and_concentration,
        x="compound_concentration_um",
        title="Distribution of Unique Compound Concentrations (μM)",
        labels={
            "compound_concentration_um": "Concentration (μM)",
            "count": "Frequency",
        },
        nbins=75,
    )

    fig1.update_layout(xaxis_tickangle=-45)

    mo.ui.plotly(fig1)
    return


@app.cell
def _(mo):
    mo.md(r"""## Explore chemical space [ Other analysis module ?]""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Explore chemical physiological parameter [ Other analysis module ?]""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Explore chemical biological annotations [ Other analysis module ?]""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Explore experimental set ups""")
    return


@app.cell
def _(df, mo):
    # Create and display value counts for each experimental column
    results_experimental = []
    for col_experimental in [
        "plate",
        "plate_type",
        "plate_number",
        "image_id",
        "well_id",
        "well",
        "row",
        "col",
        "workspace",
    ]:
        results_experimental.append(mo.md(f"### {col_experimental}"))
        counts_experimental = df[col_experimental].value_counts().reset_index()
        counts_experimental.columns = ["Value", "Count"]
        results_experimental.append(mo.ui.table(counts_experimental))

    mo.vstack(results_experimental)
    return


@app.cell
def _(mo):
    mo.md(r"""## Explore experimental values""")
    return


@app.cell
def _(df):
    print("len", "hoechst_33342", len(df["hoechst_33342"].iloc[0]))
    print("len", "phenovue_fluor_488", len(df["phenovue_fluor_488"].iloc[0]))
    print("len", "wga555/phalloidin568", len(df["wga555/phalloidin568"].iloc[0]))
    print("len", "cp_nuclei_acid_512", len(df["cp_nuclei_acid_512"].iloc[0]))
    print("len", "mito_641_narrow", len(df["mito_641_narrow"].iloc[0]))
    print("len", "brightfield", len(df["brightfield"].iloc[0]))
    return


@app.cell
def _(df, mo, np, plt):
    columns_cp_values = [
        "hoechst_33342",
        "phenovue_fluor_488",
        "wga555/phalloidin568",
        "cp_nuclei_acid_512",
        "mito_641_narrow",
        "brightfield",
    ]

    plots = []

    for col_cp in columns_cp_values:
        # Randomly sample 100 arrays/lists from this column
        sampled = df[col_cp].sample(n=100, random_state=42).tolist()

        # Flatten arrays into one list of values
        values = np.concatenate(sampled)

        # Create histogram (same structure as your snippet)
        fig_cp, ax_cp = plt.subplots(figsize=(6, 3))
        ax_cp.hist(values, bins=50, color="skyblue", edgecolor="black")
        ax_cp.set_title(f"Histogram of {col_cp} (100 random samples)")
        ax_cp.set_xlabel("Intensity")
        ax_cp.set_ylabel("Frequency")

        # Append the figure with a title markdown
        plots.append(fig_cp)

    mo.vstack(plots)
    return


@app.cell
def _(mo):
    mo.md(r"""## Experimental check""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Check experimental replicates""")
    return


@app.cell
def _(df):
    df[["compound_concentration_um", "compound_smiles"]].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""### Check concentration grouped by chemicals (smiles)""")
    return


@app.cell
def _(df):
    df[["compound_concentration_um", "compound_smiles"]].drop_duplicates()["compound_concentration_um"].value_counts()
    return


@app.cell
def _(df):
    df[["compound_concentration_um", "compound_smiles", "well_id"]].value_counts()
    return


@app.cell
def _(df):
    df[["compound_concentration_um", "compound_smiles", "well_id"]].value_counts()
    return


@app.cell
def _(df):
    df[["compound_concentration_um", "compound_smiles", "well_id"]].drop_duplicates()[
        ["compound_concentration_um", "compound_smiles"]
    ].value_counts()
    return


@app.cell
def _(df):
    replicate_distribution = (
        df[["compound_concentration_um", "compound_smiles", "well_id"]]
        .drop_duplicates()
        .value_counts(subset=["compound_concentration_um", "compound_smiles"])
        .reset_index(name="count")["count"]
        .value_counts()
        .sort_index()
    )

    print(replicate_distribution)
    return


@app.cell
def _(df):
    df[["compound_concentration_um", "compound_smiles"]].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""### Check annotation grouped by chemicals""")
    return


@app.cell
def _(df, mo):
    results_chemicals_annotations_counts = []
    for col_chamical_annotation in [
        "compound_target",
        "compound_pathway",
        "compound_biological_activity",
        "compound_saltdata",
        "compound_research_area",
        "compound_clinical_information",
    ]:
        results_chemicals_annotations_counts.append(
            df[[col_chamical_annotation, "compound_smiles"]].drop_duplicates()[col_chamical_annotation].value_counts()
        )

    mo.vstack(results_chemicals_annotations_counts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Explore raw data axiom""")
    return


@app.cell
def _(pd):
    # raw cell painting data
    df_raw_plate = pd.read_parquet("/Users/ollitrault/Documents/projects/2024_09_09_Axiom_OASIS/1_snakemake/inputs/profiles/dino//plates/plate_41002687.parquet")
    print(df_raw_plate.shape)
    df_raw_plate.head()
    return


@app.cell
def _(pd):
    # Processed data
    df_raw = pd.read_parquet("/Users/ollitrault/Documents/projects/2024_09_09_Axiom_OASIS/1_snakemake/inputs/profiles/dino/raw.parquet")
    print(df_raw.shape)
    df_raw.head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
