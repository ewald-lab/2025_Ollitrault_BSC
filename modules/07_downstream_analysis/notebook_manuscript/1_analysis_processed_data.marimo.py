# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.6",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import argparse
    import marimo as mo
    import sys
    import pandas as pd
    import umap
    import plotly.express as px
    return argparse, mo, pd, px, sys, umap


@app.cell
def _(mo):
    mo.md(r"""# 1. Analysis of Cell Painting dataset processed Level 4""")
    return


@app.cell
def _(argparse, mo, sys):
    # Define the parser
    parser = argparse.ArgumentParser(description="My Marimo Report")
    parser.add_argument("--name", type=str, default="Default")
    parser.add_argument("--input_path", type=str, default="modules/03_postprocessing/outputs/level4b_bsc_filtered.parquet")
    parser.add_argument("--control", type=str, default="s0617825")
    parser.add_argument("--metadata_cell_viability", nargs="+", default=["Metadata_mtt_normalized", "Metadata_ldh_normalized"])

    # Parse arguments from sys.argv
    # marimo ensures sys.argv[1:] contains only the args after the '--'
    args, unknown = parser.parse_known_args(sys.argv[1:])

    # Use the parsed values in your notebook
    mo.md(f"# Report for {args.name}")
    mo.md(f"Input path of {args.name} data : {args.input_path}")
    return (args,)


@app.cell
def _(args, pd):
    df_level4 = pd.read_parquet(args.input_path)
    if True:
        df_level4 = df_level4.groupby("Metadata_plate").filter(
            lambda g: ~(g["Metadata_compound_name"] == "DMSO").all()
        )
    return (df_level4,)


@app.cell
def _(df_level4):
    meta_cols = [c for c in df_level4.columns if c.startswith("Metadata_")]
    all_features = [c for c in df_level4.columns if c.startswith("Feature_")]
    return all_features, meta_cols


@app.cell
def _(all_features, args, meta_cols):
    print("There are ", len(all_features)," features in ", args.input_path) 
    print("There are ", len(meta_cols)," metadata columns in ", args.input_path) 
    return


@app.cell
def _(mo):
    mo.md(r"""# Experimental set up""")
    return


@app.cell
def _(df_level4):
    n_chem_rep = sum(df_level4[["Metadata_compound", "Metadata_concentration"]].value_counts().to_frame()['count'] > 1)
    n_chem_no_rep = sum(df_level4[["Metadata_compound", "Metadata_concentration"]].value_counts().to_frame()['count'] == 1)
    print("There are :", n_chem_rep, "chemicals-concentration assays that have replicates")
    print("There are :", n_chem_no_rep, "chemicals-concentration assays that don't have replicates")

    n_chem_rep_unique_chem = (
                                df_level4[["Metadata_compound", "Metadata_concentration"]]
                                .value_counts()
                                .loc[lambda x: x > 1]
                                .index.get_level_values("Metadata_compound")
                                .nunique()
                            )
    n_chem_no_rep_unique_chem = (
                                df_level4[["Metadata_compound", "Metadata_concentration"]]
                                .value_counts()
                                .loc[lambda x: x == 1]
                                .index.get_level_values("Metadata_compound")
                                .nunique()
                            )
    print("There are :", n_chem_rep_unique_chem, "chemicals tested at different concentration that have replicates")
    print("There are :", n_chem_no_rep_unique_chem, "chemicals tested at different concentration that don't have replicates")

    n_chem_rep_unique_conc = (
                                df_level4[["Metadata_compound", "Metadata_concentration"]]
                                .value_counts()
                                .loc[lambda x: x > 1]
                                .index.get_level_values("Metadata_compound")
                                .value_counts()
                                .eq(1)
                                .sum()
                            )
    n_chem_no_rep_unique_conc = (
                                    df_level4[["Metadata_compound", "Metadata_concentration"]]
                                    .value_counts()
                                    .loc[lambda x: x == 1]
                                    .index.get_level_values("Metadata_compound")
                                    .value_counts()
                                    .eq(1)
                                    .sum()
                                )
    print("There are :", n_chem_rep_unique_conc, "chemicals tested at one concentration that have replicates (DMSO)")
    print("There are :", n_chem_no_rep_unique_conc, "chemicals tested at one concentration that don't have replicates")
    return


@app.cell
def _(df_level4):
    # Replicates per plates 
    plates = df_level4["Metadata_plate"].unique()
    for _plate in plates:
        n_chem_rep_plate = sum(df_level4[df_level4["Metadata_plate"] == _plate][["Metadata_compound", "Metadata_concentration"]].value_counts().to_frame()['count'] > 1)
        n_chem_no_rep_plate = sum(df_level4[df_level4["Metadata_plate"] == _plate][["Metadata_compound", "Metadata_concentration"]].value_counts().to_frame()['count'] == 1)
        print("In plate ", _plate,", there are :", n_chem_rep_plate, "chemicals tested at different concentration that have replicates")
        print("In plate ", _plate,", there are :", n_chem_no_rep_plate, "chemicals tested at different concentration that don't have replicates")
    return (plates,)


@app.cell
def _(args, df_level4, plates):
    for _plate in plates:
        n_dmso = sum(df_level4[df_level4["Metadata_plate"] == _plate]["Metadata_compound"] == args.control)
        print("In plate ", _plate,", there are :", n_dmso, "DMSO out of 384 wells")
    return


@app.cell
def _(df_level4):
    # Check replicates accross different plates
    replicates_same_conc = (
        df_level4.groupby(["Metadata_compound", "Metadata_concentration"])
        ["Metadata_plate"]
        .nunique()
        .reset_index(name="n_plates")
    )
    print(replicates_same_conc.sort_values("n_plates", ascending=False).head())
    print("Replicates with the same concentration accross different plates")
    return


@app.cell
def _(df_level4, plates):
    # Check concentration tested per plates
    for _plate in plates:
        df_conc_per_plate = df_level4[df_level4["Metadata_plate"] == _plate]["Metadata_concentration"].value_counts().to_frame().sort_values("count", ascending=False).head()
        print(_plate)
        print(df_conc_per_plate.head())
    return


@app.cell
def _(df_level4, plates):
    # Check number of concentration tested per plates
    for _plate in plates:
        n_unique_concentration = df_level4.loc[
            df_level4["Metadata_plate"] == _plate,
            "Metadata_concentration"
        ].nunique()

        print(f"{_plate}: {n_unique_concentration} unique concentrations")
    return


@app.cell
def _(df_level4, plates):
    # Check number of chemicals tested per plate
    for _plate in plates:
        n_unique_chemical = df_level4.loc[
            df_level4["Metadata_plate"] == _plate,
            "Metadata_compound"
        ].nunique()

        print(f"{_plate}: {n_unique_chemical} unique chemicals")
    return


@app.cell
def _(mo):
    mo.md(r"""# Analysis plates experiments""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Plates positions""")
    return


@app.cell
def _():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
    import math
    return BoundaryNorm, ListedColormap, Normalize, Patch, math, np, plt, sns


@app.cell
def _(
    BoundaryNorm,
    ListedColormap,
    Normalize,
    Patch,
    args,
    df_level4,
    math,
    np,
    plt,
    sns,
):
    _plates = df_level4["Metadata_plate"].unique()
    n_plates = len(_plates)

    n_cols = 4
    n_rows = math.ceil(n_plates / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False
    )

    for ax, (_plate, df_plate) in zip(
        axes.flat,
        df_level4.groupby("Metadata_plate", sort=False)
    ):
        df_plate = df_plate.copy()

        df_plate["is_DMSO"] = df_plate["Metadata_compound"] == args.control
        df_plate["conc_for_plot"] = np.where(
            df_plate["is_DMSO"],
            np.nan,
            df_plate["Metadata_concentration"]
        )

        unique_concs = np.sort(
            df_plate.loc[~df_plate["is_DMSO"], "Metadata_concentration"]
            .dropna()
            .unique()
        )
        n_concs = len(unique_concs)

        plate_map = (
            df_plate.pivot(
                index="Metadata_row",
                columns="Metadata_col",
                values="conc_for_plot"
            )
            .sort_index()
            .sort_index(axis=1)
        )

        legend_elements = []

        # --- Case 1: DMSO-only plate ---
        if n_concs == 0:
            df_plate["conc_for_plot"] = np.where(
                df_plate["is_DMSO"],
                0.0,
                np.nan
            )

            plate_map = (
                df_plate.pivot(
                    index="Metadata_row",
                    columns="Metadata_col",
                    values="conc_for_plot"
                )
                .sort_index()
                .sort_index(axis=1)
            )

            sns.heatmap(
                plate_map,
                cmap=ListedColormap(["lightgray"]),
                cbar=False,
                linewidths=0.5,
                linecolor="gray",
                ax=ax
            )

            ax.legend(
                handles=[Patch(facecolor="lightgray", label="DMSO")],
                title="Concentration",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0.0
            )

            ax.set_title(f"Plate {_plate} (DMSO only)")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            continue

        # --- Case 2: Discrete concentrations, per plate ---
        if n_concs < 20:
            cmap = ListedColormap(
                sns.color_palette("tab20", n_concs)
            )

            # BoundaryNorm needs n_concs + 1 boundaries.
            # These boundaries are computed per plate from the concentrations
            # present on that plate only.
            if n_concs == 1:
                bounds = [
                    unique_concs[0] - 0.5,
                    unique_concs[0] + 0.5
                ]
            else:
                bounds = np.concatenate([
                    [unique_concs[0] - 0.5],
                    (unique_concs[:-1] + unique_concs[1:]) / 2,
                    [unique_concs[-1] + 0.5]
                ])

            norm = BoundaryNorm(bounds, cmap.N)

            sns.heatmap(
                plate_map,
                cmap=cmap,
                norm=norm,
                cbar=False,
                linewidths=0.5,
                linecolor="gray",
                ax=ax
            )

            legend_elements.extend(
                Patch(facecolor=cmap(i), label=f"{conc:.2f} µM")
                for i, conc in enumerate(unique_concs)
            )

        # --- Case 3: Continuous mapping for many concentrations ---
        else:
            cmap = sns.color_palette("hls", as_cmap=True)

            norm = Normalize(
                vmin=np.nanmin(plate_map.values),
                vmax=np.nanmax(plate_map.values)
            )

            sns.heatmap(
                plate_map,
                cmap=cmap,
                norm=norm,
                cbar=True,
                cbar_kws={"label": "Compound concentration (µM)"},
                linewidths=0.5,
                linecolor="gray",
                ax=ax
            )

        # --- Overlay DMSO wells ---
        for _, row in df_plate.loc[df_plate["is_DMSO"]].iterrows():
            if (
                row["Metadata_row"] in plate_map.index
                and row["Metadata_col"] in plate_map.columns
            ):
                r = plate_map.index.get_loc(row["Metadata_row"])
                c = plate_map.columns.get_loc(row["Metadata_col"])

                ax.add_patch(
                    plt.Rectangle(
                        (c, r),
                        1,
                        1,
                        facecolor="lightgray",
                        edgecolor="gray",
                        linewidth=0.5
                    )
                )

        legend_elements.append(
            Patch(facecolor="lightgray", edgecolor="gray", label="DMSO")
        )

        ax.legend(
            handles=legend_elements,
            title="Concentration",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0
        )

        ax.set_title(f"Plate {_plate}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    # Remove unused axes
    for ax in axes.flat[n_plates:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("modules/07_downstream_analysis/figures/Figure_S2_A", dpi=300, bbox_inches="tight")
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""## Plates profiles umap""")
    return


@app.cell
def _(all_features, df_level4, pd, umap):
    # --- 1. Extract features and metadata
    X = df_level4[all_features].values
    _compound = df_level4["Metadata_compound"].astype(str)
    _plates = df_level4["Metadata_plate"].astype(str)
    _concentration = df_level4["Metadata_concentration"].astype(float)
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
    df_umap["Metadata_log10conc"] = df_level4["Metadata_log10conc"].astype(float)
    return (df_umap,)


@app.cell
def _(df_umap, px):
    # --- 4. Create interactive Plotly scatter plot
    _fig = px.scatter(
        df_umap.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="Metadata_plate",
        title="UMAP of Level 4b Cell Painting Profiles Colored by Plate",
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
    mo.md(r"""## Plates profiles control umap""")
    return


@app.cell
def _(args, df_umap, px):
    # --- 4. Create interactive Plotly scatter plot for Control
    df_umap_control = df_umap[df_umap["Metadata_compound"] == args.control]
    _fig = px.scatter(
        df_umap_control.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color="Metadata_plate",
        title="UMAP of Level 4b Cell Painting Profiles filtered oon control Colored by Plate",
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
    _fig.write_image(
        "modules/07_downstream_analysis/figures/Figure_S3_A.png",
        scale=3
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
        title="UMAP of Level 4 Cell Painting Profiles",
        opacity=0.8,
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
def _(mo):
    mo.md(r"""## Concentration profiles umap""")
    return


@app.cell
def _(df_umap, px):
    # --- 4. Create interactive Plotly scatter plot
    #df_umap["log_concentration"] = np.log10(
    #    df_umap["Metadata_concentration"].replace(0, np.nan)
    #)
    _fig = px.scatter(
        df_umap.sample(frac=1, random_state=42).reset_index(drop=True),
        x="UMAP_1",
        y="UMAP_2",
        color= "Metadata_log10conc", #"log_concentration",
        title="UMAP of Level 4 Cell Painting Profiles Colored by Concentration",
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
    mo.md(r"""## Plates by metadata (Cell Count, LDH, MTT)""")
    return


@app.cell
def _(args):
    zip(["A", "B"], args.metadata_cell_viability)
    return


@app.cell
def _(args, df_level4, math, np, plt, sns):
    for supp_fig_letter, _column_metadata in zip(["B", "C"], args.metadata_cell_viability):
        _cmap = sns.color_palette("coolwarm", as_cmap=True)
        _vmin = df_level4[_column_metadata].min()
        _vmax = df_level4[_column_metadata].max()

        _plates = sorted(df_level4["Metadata_plate"].unique())
        _n_plates = len(_plates)
        _ncols = 4
        _nrows = math.ceil(_n_plates / _ncols)

        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 6, _nrows * 4))
        _axes = np.array(_axes).flatten()

        for _i, _plate in enumerate(_plates):
            _df_plate = df_level4[df_level4["Metadata_plate"] == _plate].copy()
            _df_plate["viability_values"] = _df_plate[_column_metadata]
            plate_map_metadata = (
                _df_plate.pivot(
                    index="Metadata_row",
                    columns="Metadata_col",
                    values="viability_values",
                )
                .sort_index()
                .sort_index(axis=1)
            )
            sns.heatmap(
                plate_map_metadata,
                ax=_axes[_i],
                cmap=_cmap,
                vmin=_vmin,
                vmax=_vmax,
                cbar=(_i == _n_plates - 1),   # only last plate gets colorbar
                cbar_kws={"label": _column_metadata},
                linewidths=0.5,
                linecolor="gray",
            )
            _axes[_i].set_title(f"Plate {_plate}", fontsize=10, fontweight="bold")
            _axes[_i].set_xlabel("Column", fontsize=8)
            _axes[_i].set_ylabel("Row", fontsize=8)
            _axes[_i].tick_params(labelsize=7)

        # Hide unused axes
        for _j in range(_i + 1, len(_axes)):
            _axes[_j].set_visible(False)

        _fig.suptitle(_column_metadata, fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(f"modules/07_downstream_analysis/figures/Figure_S2_{supp_fig_letter}.png", dpi=300, bbox_inches="tight")
        plt.show()
    return


if __name__ == "__main__":
    app.run()
