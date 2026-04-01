import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import glob
    import pickle
    import pandas as pd
    import numpy as np
    import umap
    from harmonypy import run_harmony
    import matplotlib.pyplot as plt
    from pycytominer import aggregate, consensus
    return consensus, mo, np, pd, plt, run_harmony, umap


@app.cell
def _(mo):
    mo.md(r"""# Batch correction harmony : Axiom & BSC""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Input""")
    return


@app.cell
def _(pd):
    df_level4b_bsc = pd.read_parquet("modules/03_postprocessing/outputs/level4b_bsc_filtered.parquet")
    if True:
        df_level4b_bsc = df_level4b_bsc.groupby("Metadata_plate").filter(
            lambda g: ~(g["Metadata_compound_name"] == "DMSO").all()
        )
    df_level4b_axiom = pd.read_parquet("modules/02_processing/outputs/level4a_axiom.parquet")
    return df_level4b_axiom, df_level4b_bsc


@app.cell
def _(df_level4b_bsc):
    features_all_bsc = [c for c in df_level4b_bsc.columns if c.startswith("Feature_")]
    features_all_bsc
    return (features_all_bsc,)


@app.cell
def _(df_level4b_axiom, df_level4b_bsc, np):
    meta_all_bsc_axiom = [c for c in df_level4b_bsc.columns if c.startswith("Metadata_")] + \
                            [c for c in df_level4b_axiom.columns if c.startswith("Metadata_")]
    meta_all_bsc_axiom = np.unique(meta_all_bsc_axiom)
    meta_all_bsc_axiom
    return (meta_all_bsc_axiom,)


@app.cell
def _(
    df_level4b_axiom,
    df_level4b_bsc,
    features_all_bsc,
    meta_all_bsc_axiom,
    pd,
):
    # bind
    df_level4b_bsc["Metadata_batch"] = "bsc"
    df_level4b_axiom["Metadata_batch"] = "axiom"

    # Row bind (concatenate)
    df_level4b_bsc_axiom = pd.concat([df_level4b_bsc, df_level4b_axiom], ignore_index=True)
    # selct feats and metadata
    feats_level4b_bsc_axiom = df_level4b_bsc_axiom[features_all_bsc]
    # meta
    meta_level4b_bsc_axiom = df_level4b_bsc_axiom[meta_all_bsc_axiom]
    return (
        df_level4b_bsc_axiom,
        feats_level4b_bsc_axiom,
        meta_level4b_bsc_axiom,
    )


@app.cell
def _(feats_level4b_bsc_axiom):
    feats_level4b_bsc_axiom
    return


@app.cell
def _(meta_level4b_bsc_axiom):
    meta_level4b_bsc_axiom
    return


@app.cell
def _(mo):
    mo.md(r"""# Batch correction Harmony""")
    return


@app.cell
def _(feats_level4b_bsc_axiom, meta_level4b_bsc_axiom, run_harmony):
    harmony_out_level4b_bsc_axiom = run_harmony(feats_level4b_bsc_axiom,
                              meta_level4b_bsc_axiom,
                              "Metadata_batch",
                              max_iter_harmony=20,
                              nclust=300)  # Number of compounds
    #features_all_harmony = [f'harmony_{i}' for i in range(feats_level4b_bsc_axiom_harmony.shape[1])]
    return (harmony_out_level4b_bsc_axiom,)


@app.cell
def _(features_all_bsc, harmony_out_level4b_bsc_axiom, pd):
    feats_level4b_bsc_axiom_harmony = pd.DataFrame(harmony_out_level4b_bsc_axiom.Z_corr)
    feats_level4b_bsc_axiom_harmony.columns = features_all_bsc
    return (feats_level4b_bsc_axiom_harmony,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# Analysis level 4""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## UMAP level 4 batch corrected""")
    return


@app.cell
def _(feats_level4b_bsc_axiom_harmony, umap):
    # Run UMAP
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )
    embedding_harmony = _reducer.fit_transform(feats_level4b_bsc_axiom_harmony)
    return (embedding_harmony,)


@app.cell
def _(embedding_harmony, meta_level4b_bsc_axiom, np, plt):
    # Prepare batch colors
    _batch_categories = meta_level4b_bsc_axiom["Metadata_batch"].astype("category")
    _batch_codes = _batch_categories.cat.codes.values
    _batch_labels = _batch_categories.cat.categories.tolist()

    # Create colormap
    _cmap = plt.cm.tab20(np.linspace(0, 1, len(_batch_labels)))

    # Plot
    _fig, _ax = plt.subplots(figsize=(10, 8))
    for _i, _label in enumerate(_batch_labels):
        _mask = _batch_codes == _i
        _ax.scatter(
            embedding_harmony[_mask, 0],
            embedding_harmony[_mask, 1],
            c=[_cmap[_i]],
            label=_label,
            alpha=0.7,
            s=30
        )

    _ax.set_xlabel("UMAP1", fontsize=12)
    _ax.set_ylabel("UMAP2", fontsize=12)
    _ax.set_title("UMAP of Cell Painting Profiles", fontsize=14)
    _ax.legend(title="Batch", bbox_to_anchor=(1.05, 1), loc="upper left")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## UMAP level 4 NOT batch corrected""")
    return


@app.cell
def _(feats_level4b_bsc_axiom, umap):
    # Run UMAP
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )
    embedding_raw = _reducer.fit_transform(feats_level4b_bsc_axiom)
    return (embedding_raw,)


@app.cell
def _(embedding_raw, meta_level4b_bsc_axiom, np, plt):
    # Prepare batch colors
    _batch_categories = meta_level4b_bsc_axiom["Metadata_batch"].astype("category")
    _batch_codes = _batch_categories.cat.codes.values
    _batch_labels = _batch_categories.cat.categories.tolist()

    # Create colormap
    _cmap = plt.cm.tab20(np.linspace(0, 1, len(_batch_labels)))

    # Plot
    _fig, _ax = plt.subplots(figsize=(10, 8))
    for _i, _label in enumerate(_batch_labels):
        _mask = _batch_codes == _i
        _ax.scatter(
            embedding_raw[_mask, 0],
            embedding_raw[_mask, 1],
            c=[_cmap[_i]],
            label=_label,
            alpha=0.7,
            s=30
        )

    _ax.set_xlabel("UMAP1", fontsize=12)
    _ax.set_ylabel("UMAP2", fontsize=12)
    _ax.set_title("UMAP of Cell Painting Profiles", fontsize=14)
    _ax.legend(title="Batch", bbox_to_anchor=(1.05, 1), loc="upper left")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP level 4 batch corrected filter DMSO""")
    return


@app.cell
def _(
    feats_level4b_bsc_axiom,
    feats_level4b_bsc_axiom_harmony,
    meta_level4b_bsc_axiom,
):
    # Create mask
    mask = ~meta_level4b_bsc_axiom["Metadata_compound"].isin(["DMSO", "s0617825"])

    # Apply to both
    meta_level4b_bsc_axiom_dmso = meta_level4b_bsc_axiom[mask].reset_index(drop=True)
    feats_level4b_bsc_axiom_dmso = feats_level4b_bsc_axiom[mask].reset_index(drop=True)
    feats_level4b_bsc_axiom_harmony_dmso = feats_level4b_bsc_axiom_harmony[mask].reset_index(drop=True)
    return (
        feats_level4b_bsc_axiom_dmso,
        feats_level4b_bsc_axiom_harmony_dmso,
        meta_level4b_bsc_axiom_dmso,
    )


@app.cell
def _(feats_level4b_bsc_axiom_harmony_dmso, umap):
    # Run UMAP
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )
    embedding_harmony_dmso = _reducer.fit_transform(feats_level4b_bsc_axiom_harmony_dmso)
    return (embedding_harmony_dmso,)


@app.cell
def _(embedding_harmony_dmso, meta_level4b_bsc_axiom_dmso, np, plt):
    # Prepare batch colors
    _batch_categories = meta_level4b_bsc_axiom_dmso["Metadata_batch"].astype("category")
    _batch_codes = _batch_categories.cat.codes.values
    _batch_labels = _batch_categories.cat.categories.tolist()

    # Create colormap
    _cmap = plt.cm.tab20(np.linspace(0, 1, len(_batch_labels)))

    # Plot
    _fig, _ax = plt.subplots(figsize=(10, 8))
    for _i, _label in enumerate(_batch_labels):
        _mask = _batch_codes == _i
        _ax.scatter(
            embedding_harmony_dmso[_mask, 0],
            embedding_harmony_dmso[_mask, 1],
            c=[_cmap[_i]],
            label=_label,
            alpha=0.7,
            s=30
        )

    _ax.set_xlabel("UMAP1", fontsize=12)
    _ax.set_ylabel("UMAP2", fontsize=12)
    _ax.set_title("UMAP of Cell Painting Profiles", fontsize=14)
    _ax.legend(title="Batch", bbox_to_anchor=(1.05, 1), loc="upper left")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## UMAP level 4 NOT batch corrected filter DMSO""")
    return


@app.cell
def _(feats_level4b_bsc_axiom_dmso, umap):
    # Run UMAP
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )
    embedding_raw_dmso = _reducer.fit_transform(feats_level4b_bsc_axiom_dmso)
    return (embedding_raw_dmso,)


@app.cell
def _(embedding_raw_dmso, meta_level4b_bsc_axiom_dmso, np, plt):
    # Prepare batch colors
    _batch_categories = meta_level4b_bsc_axiom_dmso["Metadata_batch"].astype("category")
    _batch_codes = _batch_categories.cat.codes.values
    _batch_labels = _batch_categories.cat.categories.tolist()

    # Create colormap
    _cmap = plt.cm.tab20(np.linspace(0, 1, len(_batch_labels)))

    # Plot
    _fig, _ax = plt.subplots(figsize=(10, 8))
    for _i, _label in enumerate(_batch_labels):
        _mask = _batch_codes == _i
        _ax.scatter(
            embedding_raw_dmso[_mask, 0],
            embedding_raw_dmso[_mask, 1],
            c=[_cmap[_i]],
            label=_label,
            alpha=0.7,
            s=30
        )

    _ax.set_xlabel("UMAP1", fontsize=12)
    _ax.set_ylabel("UMAP2", fontsize=12)
    _ax.set_title("UMAP of Cell Painting Profiles", fontsize=14)
    _ax.legend(title="Batch", bbox_to_anchor=(1.05, 1), loc="upper left")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis level 5""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Compute level batch corrected""")
    return


@app.cell
def _(feats_level4b_bsc_axiom_harmony, meta_level4b_bsc_axiom, pd):
    df_feats_level4b_bsc_axiom_harmony = pd.concat(
        [meta_level4b_bsc_axiom, feats_level4b_bsc_axiom_harmony],
        axis=1
    )
    df_feats_level4b_bsc_axiom_harmony
    return (df_feats_level4b_bsc_axiom_harmony,)


@app.cell
def _(consensus, df_feats_level4b_bsc_axiom_harmony, features_all_bsc):
    feats_level5_bsc_axiom_harmony = consensus(
            df_feats_level4b_bsc_axiom_harmony,
            replicate_columns="Metadata_perturbation",     
            features=features_all_bsc,
            operation="modz",
        )
    return (feats_level5_bsc_axiom_harmony,)


@app.cell
def _(df_feats_level4b_bsc_axiom_harmony, feats_level5_bsc_axiom_harmony, pd):
    meta_cols = [c for c in df_feats_level4b_bsc_axiom_harmony.columns if c.startswith("Metadata_compound")] + ["Metadata_batch", "Metadata_perturbation"]
    df_feats_level5_bsc_axiom_harmony = pd.merge(
        feats_level5_bsc_axiom_harmony,
        df_feats_level4b_bsc_axiom_harmony[meta_cols].drop_duplicates(subset=["Metadata_perturbation"]),
        on="Metadata_perturbation",  # join key (common to both DataFrames)
        how="left",  # or 'inner' if you only want matching wells
    )
    return df_feats_level5_bsc_axiom_harmony, meta_cols


@app.cell
def _(mo):
    mo.md(r"""## Compute level NOT batch corrected""")
    return


@app.cell
def _(consensus, df_level4b_bsc_axiom, features_all_bsc):
    feats_level5_bsc_axiom = consensus(
            df_level4b_bsc_axiom,
            replicate_columns="Metadata_perturbation",     
            features=features_all_bsc,
            operation="modz",
        )
    return (feats_level5_bsc_axiom,)


@app.cell
def _(df_level4b_bsc_axiom, feats_level5_bsc_axiom, meta_cols, pd):
    df_feats_level5_bsc_axiom = pd.merge(
        feats_level5_bsc_axiom,
        df_level4b_bsc_axiom[meta_cols].drop_duplicates(subset=["Metadata_perturbation"]),
        on="Metadata_perturbation",  # join key (common to both DataFrames)
        how="left",  # or 'inner' if you only want matching wells
    )
    return (df_feats_level5_bsc_axiom,)


@app.cell
def _(mo):
    mo.md(r"""## UMAP level 5 batch corrected""")
    return


@app.cell
def _(df_feats_level5_bsc_axiom_harmony, features_all_bsc, umap):
    # Run UMAP
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )
    embedding_harmony_lvl5 = _reducer.fit_transform(df_feats_level5_bsc_axiom_harmony[features_all_bsc])
    return (embedding_harmony_lvl5,)


@app.cell
def _(df_feats_level5_bsc_axiom_harmony, embedding_harmony_lvl5, np, plt):
    # Prepare batch colors
    _batch_categories = df_feats_level5_bsc_axiom_harmony["Metadata_batch"].astype("category")
    _batch_codes = _batch_categories.cat.codes.values
    _batch_labels = _batch_categories.cat.categories.tolist()

    # Create colormap
    _cmap = plt.cm.tab20(np.linspace(0, 1, len(_batch_labels)))

    # Plot
    _fig, _ax = plt.subplots(figsize=(10, 8))
    for _i, _label in enumerate(_batch_labels):
        _mask = _batch_codes == _i
        _ax.scatter(
            embedding_harmony_lvl5[_mask, 0],
            embedding_harmony_lvl5[_mask, 1],
            c=[_cmap[_i]],
            label=_label,
            alpha=0.7,
            s=30
        )

    _ax.set_xlabel("UMAP1", fontsize=12)
    _ax.set_ylabel("UMAP2", fontsize=12)
    _ax.set_title("UMAP of Cell Painting Profiles", fontsize=14)
    _ax.legend(title="Batch", bbox_to_anchor=(1.05, 1), loc="upper left")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP level 5 NOT batch corrected""")
    return


@app.cell
def _():
    return


@app.cell
def _(df_feats_level5_bsc_axiom, features_all_bsc, umap):
    # Run UMAP
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )
    embedding_raw_lvl5 = _reducer.fit_transform(df_feats_level5_bsc_axiom[features_all_bsc])
    return (embedding_raw_lvl5,)


@app.cell
def _(df_feats_level5_bsc_axiom, embedding_raw_lvl5, np, plt):
    # Prepare batch colors
    _batch_categories = df_feats_level5_bsc_axiom["Metadata_batch"].astype("category")
    _batch_codes = _batch_categories.cat.codes.values
    _batch_labels = _batch_categories.cat.categories.tolist()

    # Create colormap
    _cmap = plt.cm.tab20(np.linspace(0, 1, len(_batch_labels)))

    # Plot
    _fig, _ax = plt.subplots(figsize=(10, 8))
    for _i, _label in enumerate(_batch_labels):
        _mask = _batch_codes == _i
        _ax.scatter(
            embedding_raw_lvl5[_mask, 0],
            embedding_raw_lvl5[_mask, 1],
            c=[_cmap[_i]],
            label=_label,
            alpha=0.7,
            s=30
        )

    _ax.set_xlabel("UMAP1", fontsize=12)
    _ax.set_ylabel("UMAP2", fontsize=12)
    _ax.set_title("UMAP of Cell Painting Profiles", fontsize=14)
    _ax.legend(title="Batch", bbox_to_anchor=(1.05, 1), loc="upper left")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
