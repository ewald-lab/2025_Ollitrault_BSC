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
    df_annotations_composition = pd.read_csv("00_inputs/Botanical_Constituents_and_Quantification_08192024/Quant Constituents-Table 1.tsv", sep="\t")
    return (df_annotations_composition,)


@app.cell
def _(df_annotations_composition):
    df_annotations_composition.head()
    return


@app.cell
def _(pd):
    df_level4b = pd.read_parquet("02_processed_data/level4b_bsc.parquet")
    meta_compound_cols = [c for c in df_level4b.columns if c.startswith("Metadata_compound")]
    df_metadata = df_level4b[meta_compound_cols].drop_duplicates()
    return (df_metadata,)


@app.cell
def _(df_metadata):
    df_metadata.shape
    return


@app.cell
def _(df_metadata):
    df_metadata.head()
    return


@app.cell
def _(df_annotations_composition):
    df_annotations_composition.head()
    return


@app.cell
def _(pd):
    df_cp_pod = pd.read_parquet("03_tables/pods_bsc_cp_original_conc.parquet")
    #df_meta_pod_mtt = pd.read_parquet("03_tables/meta_pods_Metadata_mtt_normalized.parquet")
    df_meta_pod_mtt = pd.read_parquet("03_tables/pods_bsc_Metadata_mtt_normalized_original_conc.parquet")
    #df_meta_pod_ldh = pd.read_parquet("03_tables/meta_pods_Metadata_ldh_normalized.parquet")
    df_meta_pod_ldh = pd.read_parquet("03_tables/pods_bsc_Metadata_ldh_normalized_original_conc.parquet")
    return df_cp_pod, df_meta_pod_ldh, df_meta_pod_mtt


@app.cell
def _(df_cp_pod):
    df_cp_pod
    return


@app.cell
def _(df_meta_pod_mtt):
    df_meta_pod_mtt
    return


@app.cell
def _(df_meta_pod_ldh):
    df_meta_pod_ldh
    return


@app.cell
def _(df_cp_pod, df_meta_pod_ldh, df_meta_pod_mtt):
    print("Shape CP:", df_cp_pod.shape)
    print("Shape mtt:", df_meta_pod_mtt.shape)
    print("Shape ldh:", df_meta_pod_ldh.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis Cell Painting POD VS Metadata POD""")
    return


@app.cell
def _():
    # Boxplot
    return


@app.cell
def _(df_cp_pod, df_meta_pod_ldh, df_meta_pod_mtt, pd):
    import matplotlib.pyplot as plt

    _data = [
        df_cp_pod["cp_POD_um"].dropna(),
        df_meta_pod_mtt["Metadata_mtt_normalized_POD_um"].replace(9999, pd.NA).dropna(),
        df_meta_pod_ldh["Metadata_ldh_normalized_POD_um"].replace(9999, pd.NA).dropna()
    ]

    _labels_x = ["CP", "MTT", "LDH"]

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        _data,
        labels=_labels_x,
        showfliers=True,
        widths=0.6
    )
    plt.yscale("log")
    plt.ylim(0.01, 10000000000000000000)
    plt.ylabel("Value")
    plt.title("Comparison of POD Metrics (all)")

    plt.tight_layout()
    plt.show()
    for _label, _d in zip(_labels_x, _data):
        print(f"{_label}: n = {len(_d)}")
    return (plt,)


@app.cell
def _(df_cp_pod, df_meta_pod_ldh, df_meta_pod_mtt, pd, plt):
    _data = [
        df_cp_pod.loc[df_cp_pod["all.pass"] == True, "cp_POD_um"].dropna(),
        df_meta_pod_mtt.loc[df_meta_pod_mtt["all.pass"] == True, "Metadata_mtt_normalized_POD_um"]
            .replace(9999, pd.NA)
            .dropna(),
        df_meta_pod_ldh.loc[df_meta_pod_ldh["all.pass"] == True, "Metadata_ldh_normalized_POD_um"]
            .replace(9999, pd.NA)
            .dropna()
    ]

    _labels_x = ["CP", "MTT", "LDH"]

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        _data,
        labels=_labels_x,
        showfliers=True,
        widths=0.6
    )
    plt.yscale("log")
    plt.ylim(0.01, 1500)
    plt.ylabel("Value")
    plt.title("Comparison of POD Metrics (filter all.pass)")

    plt.tight_layout()
    plt.show()
    for _label, _d in zip(_labels_x, _data):
        print(f"{_label}: n = {len(_d)}")
    return


@app.cell
def _(df_cp_pod, df_meta_pod_ldh, df_meta_pod_mtt, pd, plt):
    _data = [
        df_cp_pod.loc[
            (df_cp_pod["all.pass"] == True) & (df_cp_pod["adv.incr"] == True),
            "cp_POD_um"
        ].dropna(),

        df_meta_pod_mtt.loc[
            (df_meta_pod_mtt["all.pass"] == True) & (df_meta_pod_mtt["adv.incr"] == False),
            "Metadata_mtt_normalized_POD_um"
        ]
        .replace(9999, pd.NA)
        .dropna(),

        df_meta_pod_ldh.loc[
            (df_meta_pod_ldh["all.pass"] == True) & (df_meta_pod_ldh["adv.incr"] == True),
            "Metadata_ldh_normalized_POD_um"
        ]
        .replace(9999, pd.NA)
        .dropna()
    ]

    _labels_x = ["CP", "MTT", "LDH"]

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        _data,
        labels=_labels_x,
        showfliers=True,
        widths=0.6
    )
    plt.yscale("log")
    plt.ylim(0.01, 1500)
    plt.ylabel("Value")
    plt.title("Comparison of POD Metrics (filter all.pass and adv.incr)")

    plt.tight_layout()
    plt.show()
    for _label, _d in zip(_labels_x, _data):
        print(f"{_label}: n = {len(_d)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis Cell Painting POD BSC VS chemicals""")
    return


@app.cell
def _():
    import umap.umap_ as umap
    from rdkit.Chem import AllChem
    import sys
    sys.path.append("/Users/ollitrault/Documents/projects/utils")
    from util_standardizer import StandardizeMolecule
    return AllChem, StandardizeMolecule, umap


@app.cell
def _(AllChem, Chem, np):
    def smiles_to_morgan(smiles, radius=2, n_bits=2048):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    return (smiles_to_morgan,)


@app.cell
def _(df_annotations_composition_standardized_nona):
    df_annotations_composition_standardized_nona["Scode"].unique()
    return


@app.cell
def _(StandardizeMolecule, df_annotations_composition, df_metadata):
    df_metadata_standardized = StandardizeMolecule(df_metadata, input_column="Metadata_compound_smiles").run()
    df_annotations_composition_standardized = StandardizeMolecule(df_annotations_composition, input_column="Canonical SMILES").run()
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
def _(df_annotations_composition):
    df_annotations_composition
    return


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
def _(
    df_cp_pod,
    df_meta_pod_ldh,
    df_meta_pod_mtt,
    df_metadata_annotations_composition,
):
    # 1. Join BMD
    df_metadata_annotations_composition_cp = df_metadata_annotations_composition.merge(
        df_cp_pod[["Metadata_compound", "cp_POD", "cp_POD_um"]],
        on="Metadata_compound",
        how="left"
    )

    # 2. Join MTT POD
    df_metadata_annotations_composition_cp_mtt = df_metadata_annotations_composition_cp.merge(
        df_meta_pod_mtt[
            ["Metadata_compound", "Metadata_mtt_normalized_POD", "Metadata_mtt_normalized_POD_um"]
        ],
        on="Metadata_compound",
        how="left"
    )

    # 3. Join LDH POD
    df_metadata_annotations_composition_cp_mtt_ldh = df_metadata_annotations_composition_cp_mtt.merge(
        df_meta_pod_ldh[
            ["Metadata_compound", "Metadata_ldh_normalized_POD", "Metadata_ldh_normalized_POD_um"]
        ],
        on="Metadata_compound",
        how="left"
    )
    return (df_metadata_annotations_composition_cp_mtt_ldh,)


@app.cell
def _(
    df_annotations_composition,
    df_metadata,
    df_metadata_annotations_composition_cp_mtt_ldh,
):
    print("df_metadata:", len(df_metadata))
    print("df_annotations_composition:", len(df_annotations_composition))
    print("df_merged:", len(df_metadata_annotations_composition_cp_mtt_ldh))
    return


@app.cell
def _():
    # UMAP
    return


@app.cell
def _(df_metadata_annotations_composition_cp_mtt_ldh):
    df_metadata_annotations_composition_cp_mtt_ldh
    return


@app.cell
def _(df_metadata_annotations_composition_cp_mtt_ldh, np, smiles_to_morgan):
    # Convert SMILES to fingerprints
    fps_series = df_metadata_annotations_composition_cp_mtt_ldh["SMILES_standardized_nonisomeric"].apply(smiles_to_morgan)

    # Drop invalid SMILES
    mask = fps_series.notna()
    fps = np.vstack(fps_series[mask])
    bmd = df_metadata_annotations_composition_cp_mtt_ldh.loc[mask, "bmd"]
    return bmd, fps


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
def _(bmd, embedding, plt):
    plt.figure(figsize=(8, 7))

    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=bmd,
        cmap="viridis",
        s=25,
        alpha=0.8
    )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP of Morgan Fingerprints colored by BMD")

    cbar = plt.colorbar(sc)
    cbar.set_label("BMD")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    # Boxplot
    return


@app.cell
def _(df_metadata_annotations_composition_cp_mtt_ldh):
    df_metadata_annotations_composition_cp_mtt_ldh_box_plot = df_metadata_annotations_composition_cp_mtt_ldh[
        ["Plant name", "bmd"]
    ].dropna()
    return (df_metadata_annotations_composition_cp_mtt_ldh_box_plot,)


@app.cell
def _(df_metadata_annotations_composition_cp_mtt_ldh_box_plot, np, plt):
    df_ = df_metadata_annotations_composition_cp_mtt_ldh_box_plot.copy()

    plt.figure(figsize=(10, 5))

    ax = df_.boxplot(
        column="bmd",
        by="Plant name",
        grid=False,
        rot=45
    )

    # 🔑 Use the x-tick labels (true box order)
    plants = [tick.get_text() for tick in ax.get_xticklabels()]

    for i, plant in enumerate(plants, start=1):
        y = df_.loc[df_["Plant name"] == plant, "bmd"]
        x = np.random.normal(i, 0.04, size=len(y))
        plt.plot(x, y, "k.", alpha=0.4)

    plt.ylabel("POD")
    plt.xlabel("Plant name")
    plt.title("POD distribution grouped by Plant name")
    plt.suptitle("")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    # boxplot bsc vs all actives
    return


@app.cell
def _(
    df_annotations_composition,
    df_cp_pod,
    df_meta_pod_ldh,
    df_meta_pod_mtt,
    df_metadata,
):
    # 1. Join BMD
    df_metadata_cp = df_metadata.merge(
        df_cp_pod[["Metadata_compound", "bmd"]],
        on="Metadata_compound",
        how="left"
    )

    # 2. Join MTT POD
    df_metadata_cp_mtt = df_metadata_cp.merge(
        df_meta_pod_mtt[
            ["Metadata_compound", "Metadata_mtt_normalized_POD"]
        ],
        on="Metadata_compound",
        how="left"
    )

    # 3. Join LDH POD
    df_metadata_cp_mtt_ldh = df_metadata_cp_mtt.merge(
        df_meta_pod_ldh[
            ["Metadata_compound", "Metadata_ldh_normalized_POD"]
        ],
        on="Metadata_compound",
        how="left"
    )
    # 3. Join BSC annotations plants
    df_annotations_unique = (
        df_annotations_composition[["Scode", "Plant name"]]
        .drop_duplicates(subset=["Scode"])
    )
    df_metadata_cp_mtt_ldh_annotated = df_metadata_cp_mtt_ldh.merge(
        df_annotations_unique[
            ["Plant name", "Scode"]
        ],
        left_on="Metadata_compound",
        right_on="Scode",
        how="left"
    )
    return (df_metadata_cp_mtt_ldh_annotated,)


@app.cell
def _(df_metadata_cp_mtt_ldh_annotated):
    df_metadata_cp_mtt_ldh_annotated
    return


@app.cell
def _(df_annotations_composition, df_metadata_cp_mtt_ldh_annotated, np):
    # Set of Scodes
    scode_set = set(
        df_annotations_composition["Scode"]
        .dropna()
        .unique()
    )

    # Create binary grouping
    df_metadata_cp_mtt_ldh_annotated["Group"] = np.where(
        df_metadata_cp_mtt_ldh_annotated["Metadata_compound"].isin(scode_set),
        "Plant",
        "Constituent components"
    )
    return (scode_set,)


@app.cell
def _(scode_set):
    scode_set
    return


@app.cell
def _(df_metadata_cp_mtt_ldh_annotated, np, pd, plt):
    plt.figure(figsize=(6, 5))

    ax_ = df_metadata_cp_mtt_ldh_annotated.boxplot(
        column="bmd",
        by="Group",
        grid=False,
        showfliers=False
    )

    # True box order
    groups_ = [tick.get_text() for tick in ax_.get_xticklabels()]

    for i_, group_ in enumerate(groups_, start=1):

        df_sub = df_metadata_cp_mtt_ldh_annotated[
            df_metadata_cp_mtt_ldh_annotated["Group"] == group_
        ]

        # Generate jitter ONCE per row
        x_ = np.random.normal(i_, 0.04, size=len(df_sub))

        # Plot points
        plt.plot(x_, df_sub["bmd"], "k.", alpha=0.4)

        # Add plant names next to points (when available)
        for xi, yi, plant_ in zip(
            x_,
            df_sub["bmd"],
            df_sub["Plant name"]
        ):
            if pd.notna(plant_) and plant_ != "NA":
                plt.text(
                    xi + 0.01,
                    yi,
                    plant_,
                    fontsize=7,
                    alpha=0.7,
                    verticalalignment="center"
                )

    plt.ylabel("POD")
    plt.xlabel("")
    plt.title("POD distribution: Plant vs Active chemicals")
    plt.suptitle("")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(df_metadata_cp_mtt_ldh_annotated):
    df_metadata_cp_mtt_ldh_annotated
    return


@app.cell
def _():
    # UMAP on CP features
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
