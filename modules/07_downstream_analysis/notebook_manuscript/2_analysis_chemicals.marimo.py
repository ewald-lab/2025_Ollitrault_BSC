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
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib.patches as mpatches
    from adjustText import adjust_text
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba
    return Chem, adjust_text, dendrogram, linkage, mpatches, np, pd, pdist


@app.cell
def _(pd):
    #df_annotations_composition = pd.read_csv("modules/03_postprocessing/inputs/Botanical_Constituents_and_Quantification_08192024/Quant Constituents-Table 1.tsv", sep="\t")
    df_annotations_composition = pd.read_csv("modules/01_preprocessing/outputs/cellpainting_bsc_annotations_merged.tsv", sep="\t")
    return (df_annotations_composition,)


@app.cell
def _(df_annotations_composition):
    df_annotations_composition.head()
    return


@app.cell
def _(df_annotations_composition):
    df_annotations_composition.shape
    return


@app.cell
def _(mo):
    mo.md(r"""# Analysis chemicals mixtures""")
    return


@app.cell
def _(Chem, df_annotations_composition, np, pd):
    def count_fragments(smiles):
        """Return number of molecular fragments in a SMILES string using RDKit."""
        if pd.isna(smiles) or not smiles.strip():
            return np.nan
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan  # invalid SMILES
        return len(Chem.GetMolFrags(mol))

    # Apply to your dataframe
    df_annotations_composition["Metadata_n_fragments"] = df_annotations_composition["SMILES"].apply(count_fragments)
    return


@app.cell
def _(df_annotations_composition):
    df_annotations_composition["Metadata_n_fragments"].value_counts(dropna=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis UMAP Chemical structure""")
    return


@app.cell
def _():
    import umap.umap_ as umap
    import matplotlib.pyplot as plt
    from rdkit.Chem import AllChem
    import sys
    sys.path.append("/Users/ollitrault/Documents/projects/utils")
    from util_standardizer import StandardizeMolecule
    return AllChem, plt, umap


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
def _(df_annotations_composition):
    #df_annotations_composition_standardized = StandardizeMolecule(df_annotations_composition.drop(columns=["SMILES_standardized"], errors="ignore"), input_column="Canonical SMILES").run()
    df_annotations_composition_standardized = df_annotations_composition
    return (df_annotations_composition_standardized,)


@app.cell
def _(df_annotations_composition_standardized):
    # drop na
    df_annotations_composition_standardized_nona = (
        df_annotations_composition_standardized[
            df_annotations_composition_standardized["SMILES_standardized"].notna()
        ]
    )
    # drop duplicates
    df_annotations_composition_standardized_nona = (
        df_annotations_composition_standardized_nona
        .groupby("SMILES_standardized", as_index=False)
        .agg({
            "Constituent": "first",
            "Plant name": list,
            "Plant scode": list,
        })
    )
    return (df_annotations_composition_standardized_nona,)


@app.cell
def _(df_annotations_composition_standardized_nona):
    df_annotations_composition_standardized_nona["SMILES_standardized"]
    return


@app.cell
def _(df_annotations_composition_standardized_nona):
    df_annotations_composition_standardized_nona["Constituent"]
    return


@app.cell
def _(df_annotations_composition_standardized_nona, np, smiles_to_morgan):
    # Convert SMILES to fingerprints
    fps = df_annotations_composition_standardized_nona["SMILES_standardized"].apply(smiles_to_morgan)

    # Drop invalid SMILES
    mask = fps.notnull()
    fps = np.vstack(fps[mask])
    labels = df_annotations_composition_standardized_nona.loc[mask, "Plant name"]
    labels = (
        df_annotations_composition_standardized_nona
        .loc[mask, "Plant name"]
        .fillna("unkown")
    )
    return fps, labels, mask


@app.cell
def _(fps, umap):
    reducer = umap.UMAP(
        n_neighbors=40,        # larger = smoother chemical space
        min_dist=0.05,         # tighter clusters
        n_components=2,
        metric="jaccard",      # IMPORTANT for binary fingerprints
        random_state=42
    )

    embedding = reducer.fit_transform(fps)
    return (embedding,)


@app.cell
def _(df_annotations_composition_standardized_nona, mask):
    distinct_colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#42d4f4', '#f032e6', '#fabed4', '#469990', '#dcbeff',
        '#9A6324', '#fffac8', '#800000', '#aaffc3', '#000075',
        '#a9a9a9', '#ffffff', '#000000'
    ]
    # Flatten all plant lists to get unique plants
    all_plants = list(dict.fromkeys(p for sublist in df_annotations_composition_standardized_nona.loc[mask, "Plant name"] for p in sublist))
    if len(all_plants) > len(distinct_colors):
        raise ValueError("Not enough colors")
    color_map = dict(zip(all_plants, distinct_colors))
    color_map
    return all_plants, color_map


@app.cell
def _(
    adjust_text,
    all_plants,
    color_map,
    df_annotations_composition_standardized_nona,
    embedding,
    mask,
    mpatches,
    plt,
):
    from matplotlib.patches import Wedge


    df_masked = df_annotations_composition_standardized_nona.loc[mask].copy()
    df_masked["_x"] = embedding[:, 0]
    df_masked["_y"] = embedding[:, 1]

    # No groupby needed — already deduplicated
    df_grouped = df_masked.rename(columns={"_x": "x", "_y": "y", "Constituent": "constituent", "Plant name": "plants"})

    fig, ax = plt.subplots(figsize=(14, 11))

    # ── Invisible scatter just for adjustText anchoring ───────────────────────────
    ax.scatter(df_grouped["x"], df_grouped["y"], s=0, alpha=0, zorder=0)

    # ── Labels + adjustText FIRST, before any wedges ─────────────────────────────
    texts = []
    for _, row in df_grouped.iterrows():
        t = ax.text(
            row["x"], row["y"],
            row["constituent"],
            fontsize=11,
            fontstyle="italic",
            color="#1a1a1a",
            zorder=5,
        )
        texts.append(t)

    adjust_text(
        texts,
        x=df_grouped["x"].values,
        y=df_grouped["y"].values,
        ax=ax,
        add_objects=[],         # more breathing room around each label
        force_text=(1, 1),     # stronger repulsion between labels
        force_points=(0.2, 0.2),   # stronger repulsion from points
        max_move=8.0,              # allow labels to move further
        arrowprops=dict(
            arrowstyle="-",
            color="#888888",
            lw=0.6,
            alpha=0.8,
            shrinkA=0,
            shrinkB=4,
        ),
    )

    # ── Wedges AFTER adjustText ───────────────────────────────────────────────────
    radius = 0.08
    for _, row in df_grouped.iterrows():
        unique_plants = list(dict.fromkeys(row["plants"]))  # already a list, just dedupe
        colors = [color_map[p] for p in unique_plants]
        n = len(colors)
        angle = 90
        for color in colors:
            slice_angle = 360 / n
            ax.add_patch(Wedge(
                center=(row["x"], row["y"]),
                r=radius,
                theta1=angle - slice_angle,
                theta2=angle,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            ))
            angle -= slice_angle

    # ── rest of your styling code unchanged ───────────────────────────────────────

    # ── Legend ────────────────────────────────────────────────────────────────────
    # Legend uses all_plants
    legend_patches = [mpatches.Patch(color=color_map[p], label=p) for p in all_plants]
    ax.legend(
        handles=legend_patches,
        loc="upper right",        # or "best" to let matplotlib decide
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=14,
        borderpad=0.8,
        title="Botanical extracts",
        title_fontsize=16,
    ).get_title().set_fontweight("bold")

    # ── Style ─────────────────────────────────────────────────────────────────────
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    #ax.set_title("UMAP of Molecular Composition", fontsize=14, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_xlim(df_grouped["x"].min() - 1, df_grouped["x"].max() + 1)
    ax.set_ylim(df_grouped["y"].min() - 1, df_grouped["y"].max() + 1)
    ax.set_aspect("equal")  # critical — keeps wedges circular

    plt.tight_layout()
    plt.savefig(f"modules/09_downstream_analysis/figures/Figure_1_B.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# upset plot""")
    return


@app.cell
def _():
    from upsetplot import UpSet, from_indicators
    return UpSet, from_indicators


@app.cell
def _(df_annotations_composition_standardized_nona):
    df_annotations_composition_standardized_nona
    return


@app.cell
def _(df_annotations_composition_standardized):
    # Keep only needed columns and drop duplicates
    df = (
        df_annotations_composition_standardized[["SMILES_standardized", "Plant name"]]
        .dropna()
        .drop_duplicates()
    )

    # Create a presence/absence table:
    # rows = SMILES_standardized_nonisomeric, columns = Plant name
    presence = (
        df.assign(value=True)
          .pivot_table(
              index="SMILES_standardized",
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
    plt.savefig(f"modules/09_downstream_analysis/figures/Figure_S1_A.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Hierarchical clustering""")
    return


@app.cell
def _(labels):
    labels
    return


@app.cell
def _(
    dendrogram,
    df_annotations_composition_standardized_nona,
    fps,
    linkage,
    mask,
    mpatches,
    np,
    pdist,
    plt,
):
    # ── 1. Constituent names aligned with fps/labels ───────────────────────────────
    _constituent_names = df_annotations_composition_standardized_nona.loc[mask, "Constituent"].values

    # ── 2. Compute Jaccard distance matrix ────────────────────────────────────────
    _jaccard_dist = pdist(fps, metric="jaccard")

    # ── 3. Ward linkage ────────────────────────────────────────────────────────────
    _linkage_matrix = linkage(_jaccard_dist, method="average")

    # ── 4. Color map by plant ─────────────────────────────────────────────────────
    _distinct_colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#42d4f4', '#f032e6', '#fabed4', '#469990', '#dcbeff',
        '#9A6324', '#fffac8', '#800000', '#aaffc3', '#000075',
        '#a9a9a9', '#ffffff', '#000000'
    ]

    _plants = list(dict.fromkeys(
        p
        for plant_list in df_annotations_composition_standardized_nona.loc[mask, "Plant name"]
        for p in plant_list
    ))

    if len(_plants) > len(_distinct_colors):
        raise ValueError("Not enough colors for the number of Plant names")

    _color_map = dict(zip(_plants, _distinct_colors))

    # Per-leaf list of unique plants (handles multi-botanical constituents)
    _leaf_plants = {
        constituent: list(dict.fromkeys(
            p
            for plant_list in grp["Plant name"]
            for p in plant_list
        ))
        for constituent, grp in df_annotations_composition_standardized_nona.loc[mask].groupby("Constituent")
    }

    # ── 5. Plot ────────────────────────────────────────────────────────────────────
    _fig, _ax = plt.subplots(figsize=(10, 28))
    _dend = dendrogram(
        _linkage_matrix,
        ax=_ax,
        labels=_constituent_names,
        color_threshold=0,
        above_threshold_color="grey",
        leaf_rotation=0,
        leaf_font_size=16,
        orientation="left",
    )

    # ── Wedge slices per leaf ─────────────────────────────────────────────────────
    _y_coords = np.arange(5, 10 * len(_dend["leaves"]) + 5, 10)

    # Get x position of the leftmost dendrogram point (where labels sit)
    _x_label = _ax.get_xlim()[0]

    for _y, _leaf_idx in zip(_y_coords, _dend["leaves"]):
        _constituent   = _constituent_names[_leaf_idx]
        _unique_plants = _leaf_plants.get(_constituent, [])
        _colors        = [_color_map.get(p, "#aaaaaa") for p in _unique_plants]
        _n             = len(_colors)
        if _n == 0:
            continue

        # Draw a simple colored rectangle strip per plant next to the y label
        _rect_total_width = 0.06
        for _y, _leaf_idx in zip(_y_coords, _dend["leaves"]):
            _constituent   = _constituent_names[_leaf_idx]
            _unique_plants = _leaf_plants.get(_constituent, [])
            _colors        = [_color_map.get(p, "#aaaaaa") for p in _unique_plants]
            _n             = len(_colors)
            if _n == 0:
                continue
            _slice_width = _rect_total_width / _n
            for _pi, _color in enumerate(_colors):
                _ax.add_patch(plt.Rectangle(
                    xy=(_x_label - _rect_total_width + _pi * _slice_width - 0.96, _y - 3),
                    width=_slice_width,
                    height=10,
                    facecolor=_color,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=4,
                    clip_on=False,
                ))

    # ── Black labels ──────────────────────────────────────────────────────────────
    for _tick in _ax.get_yticklabels():
        _tick.set_color("black")

    _ax.set_ylabel("Constituents", fontsize=11)
    _ax.set_xlabel("Average Distance (Jaccard)", fontsize=11)
    _ax.spines[["top", "right", "bottom"]].set_visible(False)
    _ax.tick_params(axis="y", pad=12)

    _legend_patches = [mpatches.Patch(color=_color_map[p], label=p) for p in _plants]
    _ax.legend(
        handles=_legend_patches,
        bbox_to_anchor=(0, 1),
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.95,
        fontsize=18,
        borderpad=0.8,
    )

    plt.tight_layout()
    plt.savefig(f"modules/09_downstream_analysis/figures/Figure_S1_B.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
