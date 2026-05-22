import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import argparse
    import marimo as mo
    import sys
    import pandas as pd
    import numpy as np
    import umap
    import plotly.express as px
    import plotly.graph_objects as go
    return argparse, go, mo, pd, px, sys, umap


@app.cell
def _(argparse, mo, sys):
    # Define the parser
    parser = argparse.ArgumentParser(description="My Marimo Report")
    parser.add_argument("--name", type=str, default="Default")
    parser.add_argument("--input_path", type=str, default="02_processed_data/level3_bsc_filtered.parquet")
    parser.add_argument("--level", type=str, default="3")
    parser.add_argument("--control", type=str, default="s0617825") # control DMSO

    # Parse arguments from sys.argv
    # marimo ensures sys.argv[1:] contains only the args after the '--'
    args, unknown = parser.parse_known_args(sys.argv[1:])

    # Use the parsed values in your notebook
    mo.md(f"# Report for {args.name}")
    mo.md(f"Input path of {args.name} data : {args.input_path}")
    return (args,)


@app.cell
def _(args, mo):
    mo.md(f"""# 1. Analysis of Cell Painting dataset Level {args.level}""")
    return


@app.cell
def _(args, pd):
    # Load Data
    df_level = pd.read_parquet(args.input_path)
    return (df_level,)


@app.cell
def _(df_level):
    ## Metadata
    feature_cols = [c for c in df_level.columns if c.startswith("Feature_")]
    meta_cols = [c for c in df_level.columns if c.startswith("Metadata_")]
    return feature_cols, meta_cols


@app.cell
def _(feature_cols):
    len([c for c in feature_cols if "DNA" in c])
    return


@app.cell
def _(mo):
    mo.md(r"""## Metadata columns""")
    return


@app.cell
def _(meta_cols):
    for _col in meta_cols:
        print(_col)
    return


@app.cell
def _(df_level, meta_cols, mo):
    # Create and display value counts for each compound column
    results_metadata = []
    for _col in meta_cols:
        results_metadata.append(mo.md(f"### {_col}"))
        counts = df_level[_col].value_counts().reset_index()
        counts.columns = ["Value", "Count"]
        results_metadata.append(mo.ui.table(counts))

    mo.vstack(results_metadata)
    return


@app.cell
def _(mo):
    mo.md(r"""## Number of concentration""")
    return


@app.cell
def _(df_level, go, mo):
    # Get unique combinations of compound_smiles and compound_concentration_um
    unique_compound_smiles_and_concentration = df_level[["Metadata_compound", "Metadata_concentration"]]

    concentration_counts = unique_compound_smiles_and_concentration["Metadata_concentration"].value_counts().sort_index()

    fig1 = go.Figure(data=[
        go.Bar(
            x=concentration_counts.index,
            y=concentration_counts.values,
            width=concentration_counts.index * 0.05,  # Width proportional to x-value for log scale
        )
    ])

    fig1.update_layout(
        title="Distribution of Unique Compound Concentrations (μM)",
        xaxis_title="Concentration (μM)",
        yaxis_title="Count",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_tickangle=-45,
        bargap=0,  # Reduce gap between bars
    )

    mo.ui.plotly(fig1)
    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP Control""")
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP Control by plates""")
    return


@app.cell
def _(args, df_level, feature_cols, pd, umap):
    # --- 0. Filter to control compound only
    df_control = df_level[df_level["Metadata_compound"] == args.control].copy()

    # --- 1. Extract features and metadata
    X_DMSO = df_control[feature_cols].values
    _plates = df_control["Metadata_plate"].astype(str)
    _concentration = df_control["Metadata_concentration"].astype(float)
    _well = df_control["Metadata_well"].astype(str)
    # --- 2. Run UMAP for 2D projection
    reducer_dmso = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    embedding_dmso = reducer_dmso.fit_transform(X_DMSO)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_dmso = pd.DataFrame(embedding_dmso, columns=["UMAP_1", "UMAP_2"])
    df_umap_dmso["Metadata_plate"] = _plates.values
    df_umap_dmso["Metadata_well"] = _well.values
    df_umap_dmso["Metadata_concentration"] = _concentration.values
    df_umap_dmso["Metadata_log10conc"] = df_control["Metadata_log10conc"].astype(float).values
    return (df_umap_dmso,)


@app.cell
def _(args, df_umap_dmso, px):
    # --- 4. Create interactive Plotly scatter plot
    _fig = px.scatter(
        df_umap_dmso.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="Metadata_plate",
        title=f"UMAP of Level {args.level} Cell Painting Profiles of Control Colored by Plate",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_sequence=px.colors.qualitative.Safe  # nice, distinct palette
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="Plate ID",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP Control by well position""")
    return


@app.cell
def _(args, df_umap_dmso, px):
    # --- 4. Create interactive Plotly scatter plot
    _fig = px.scatter(
        df_umap_dmso.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="Metadata_well",
        title=f"UMAP of Level {args.level} Cell Painting Profiles of Control Colored by Well",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_sequence=px.colors.qualitative.Safe  # nice, distinct palette
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="Well",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


@app.cell
def _(args, df_umap_dmso, px):
    # --- 4. Create interactive Plotly scatter plot

    # Extract the row letter from Metadata_well (first character)
    df_umap_dmso['Plate_Row'] = df_umap_dmso['Metadata_well'].str[0]

    # Define a color map for rows (A-H for 96-well, A-P for 384-well plates)
    # Using a qualitative color palette with enough distinct colors
    row_colors = {
        'A': '#1f77b4',  # blue
        'B': '#ff7f0e',  # orange
        'C': '#2ca02c',  # green
        'D': '#d62728',  # red
        'E': '#9467bd',  # purple
        'F': '#8c564b',  # brown
        'G': '#e377c2',  # pink
        'H': '#7f7f7f',  # gray
        'I': '#bcbd22',  # olive
        'J': '#17becf',  # cyan
        'K': '#aec7e8',  # light blue
        'L': '#ffbb78',  # light orange
        'M': '#98df8a',  # light green
        'N': '#ff9896',  # light red
        'O': '#c5b0d5',  # light purple
        'P': '#c49c94',  # light brown
    }

    _fig = px.scatter(
        df_umap_dmso.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="Plate_Row",
        title=f"UMAP of Level {args.level} Cell Painting Profiles of Control Colored by Plate Row",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_map=row_colors,  # Use custom color mapping
        category_orders={"Plate_Row": sorted(df_umap_dmso['Plate_Row'].unique())}  # Order rows alphabetically
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="Plate Row",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


@app.cell
def _(args, df_umap_dmso, px):
    # --- 4. Create interactive Plotly scatter plot

    # Extract the column number from Metadata_well (characters after the first letter)
    df_umap_dmso['Plate_Column'] = df_umap_dmso['Metadata_well'].str[1:].astype(int)

    # Convert to string to ensure discrete color treatment
    df_umap_dmso['Plate_Column'] = df_umap_dmso['Plate_Column'].astype(str)

    # Use Plotly's Alphabet palette - 26 highly distinct qualitative colors
    _fig = px.scatter(
        df_umap_dmso.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="Plate_Column",
        title=f"UMAP of Level {args.level} Cell Painting Profiles of Control Colored by Plate Column",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_sequence=px.colors.qualitative.Alphabet,  # Distinct qualitative colors
        category_orders={"Plate_Column": sorted(df_umap_dmso['Plate_Column'].unique(), key=int)}  # Order numerically
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="Plate Column",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP Profiles""")
    return


@app.cell
def _(df_level, feature_cols, pd, umap):
    # --- 1. Extract features and metadata
    X = df_level[feature_cols].values
    _compound = df_level["Metadata_compound"].astype(str)
    _plates = df_level["Metadata_plate"].astype(str)
    _concentration = df_level["Metadata_concentration"].astype(float)
    # --- 2. Run UMAP for 2D projection
    reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    embedding = reducer.fit_transform(X)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])
    df_umap["Metadata_compound"] = _compound
    df_umap["Metadata_plate"] = _plates
    df_umap["Metadata_concentration"] = _concentration
    df_umap["Metadata_log10conc"] = df_level["Metadata_log10conc"].astype(float)
    return (df_umap,)


@app.cell
def _(args, df_umap, px):
    # --- 4. Create interactive Plotly scatter plot
    _fig = px.scatter(
        df_umap.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="Metadata_plate",
        title=f"UMAP of Level {args.level} Cell Painting Profiles Colored by Plate",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_sequence=px.colors.qualitative.Safe  # nice, distinct palette
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="Plate ID",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


@app.cell
def _(args, df_umap, px):
    # --- 4. Create interactive Plotly scatter plot
    #df_umap["log_concentration"] = np.log10(
    #    df_umap["Metadata_concentration"].replace(0, np.nan)
    #)
    _fig = px.scatter(
        df_umap.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color= "Metadata_log10conc", #"log_concentration",
        title=f"UMAP of Level {args.level} Cell Painting Profiles Colored by Concentration",
        opacity=0.8,
        width=850,
        height=650,
        color_continuous_scale="Viridis",  # continuous gradient
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="Plate ID",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Control profiles umap""")
    return


@app.cell
def _(args, df_umap, px):
    df_umap["is_control"] = df_umap["Metadata_compound"] == args.control

    _fig = px.scatter(
        df_umap.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="is_control",
        title="UMAP of Level {args.level} Cell Painting Profiles",
        opacity=0.6,
        width=850,
        height=650,
        color_discrete_map={
            False: "lightblue",
            True: "red"
        }
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="Control",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
