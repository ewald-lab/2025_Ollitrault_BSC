import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import umap
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from matplotlib.patches import Patch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from sklearn.metrics.pairwise import cosine_similarity
    return AllChem, Chem, DataStructs, Path, go, mo, np, pd, pio, px, umap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis Axiom VS BSC""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Input data and labels""")
    return


@app.cell
def _(Path):
    figure_dir = Path("modules/07_downstream_analysis/figures")
    return (figure_dir,)


@app.cell
def _(pd):
    # BSC annotations are used only to label botanicals and constituents in the
    # Axiom vs BSC comparison plots.
    df_bsc_annotations = pd.read_csv(
        "modules/01_preprocessing/outputs/cellpainting_bsc_annotations_merged.tsv",
        sep="\t",
    )
    list_botanicals = df_bsc_annotations["Plant scode"].dropna().unique()
    bsc_control = "s0617825"
    print(
        "BSC annotation metadata: "
        f"{df_bsc_annotations['Metadata_compound'].nunique()} compounds, "
        f"{len(list_botanicals)} botanicals"
    )
    return (df_bsc_annotations,)


@app.cell
def _(df_bsc_annotations):
    dic_botanicals_scode_name = (
        df_bsc_annotations[["Plant scode", "Plant name"]]
        .drop_duplicates()
        .dropna()
        .set_index("Plant scode")["Plant name"]
        .to_dict()
    )
    print(f"Botanical label lookup entries: {len(dic_botanicals_scode_name)}")
    return (dic_botanicals_scode_name,)


@app.cell
def _(pd):
    df_metadata_compound_oasis = pd.read_csv("modules/01_preprocessing/inputs/v5_oasis_03Sept2024_simple.csv")
    return (df_metadata_compound_oasis,)


@app.cell
def _(pd):
    df_level4a_axiom = pd.read_parquet("modules/05_batch_correction_bsc_axiom/outputs/level4b_axiom_batch.parquet")
    df_level5_axiom_all = pd.read_parquet("modules/05_batch_correction_bsc_axiom/outputs/level5_axiom_selected_features_batch.parquet")
    df_level6_axiom_all = pd.read_parquet("modules/05_batch_correction_bsc_axiom/outputs/level6_axiom_selected_features_batch.parquet")
    return df_level4a_axiom, df_level5_axiom_all, df_level6_axiom_all


@app.cell
def _(pd):
    # Cell Painting profiles
    df_level4_bsc_batch = pd.read_parquet("modules/05_batch_correction_bsc_axiom/outputs/level4b_bsc_batch.parquet")
    df_level4_bsc_batch = df_level4_bsc_batch.groupby("Metadata_plate").filter(
        lambda g: ~(g["Metadata_compound_name"] == "DMSO").all()
    )
    df_level5_bsc_batch = pd.read_parquet("modules/05_batch_correction_bsc_axiom/outputs/level5_bsc_selected_features_batch.parquet")
    df_level6_bsc_batch = pd.read_parquet("modules/05_batch_correction_bsc_axiom/outputs/level6_bsc_selected_features_batch.parquet")
    return df_level4_bsc_batch, df_level5_bsc_batch, df_level6_bsc_batch


@app.cell
def _(df_level4_bsc_batch, df_level4a_axiom, pd):
    # Add source column
    _df_level4a_axiom = df_level4a_axiom.copy()
    _df_level4_bsc_batch = df_level4_bsc_batch.copy()
    _df_level4a_axiom["Dataset"] = "AXIOM"
    _df_level4_bsc_batch["Dataset"] = "BSC"

    # Bind rows
    df_level4 = pd.concat(
        [_df_level4a_axiom, _df_level4_bsc_batch],
        axis=0,
        ignore_index=True
    )
    return (df_level4,)


@app.cell
def _(df_level5_axiom_all, df_level5_bsc_batch, pd):
    # Add source column
    _df_level5_axiom_all = df_level5_axiom_all.copy()
    _df_level5_bsc_batch = df_level5_bsc_batch.copy()
    _df_level5_axiom_all["Dataset"] = "AXIOM"
    _df_level5_bsc_batch["Dataset"] = "BSC"

    # Bind rows
    df_level5 = pd.concat(
        [_df_level5_axiom_all, _df_level5_bsc_batch],
        axis=0,
        ignore_index=True
    )
    return (df_level5,)


@app.cell
def _(df_level6_axiom_all, df_level6_bsc_batch, pd):
    # Add source column
    _df_level6_axiom_all = df_level6_axiom_all.copy()
    _df_level6_bsc_batch = df_level6_bsc_batch.copy()
    _df_level6_axiom_all["Dataset"] = "AXIOM"
    _df_level6_bsc_batch["Dataset"] = "BSC"

    # Bind rows
    df_level6 = pd.concat(
        [_df_level6_axiom_all, _df_level6_bsc_batch],
        axis=0,
        ignore_index=True
    )
    print(_df_level6_axiom_all.shape, _df_level6_bsc_batch.shape, df_level6.shape)
    return (df_level6,)


@app.cell
def _(df_level4, df_level6):
    all_selected_features = [
        _col for _col in df_level4.columns if _col.startswith("Feature_")
    ]
    all_features_bsc_axiom = [
        _col for _col in df_level6.columns if _col.startswith("Feature_")
    ]
    print(f"Level 4 Axiom/BSC features: {len(all_selected_features)}")
    print(f"Level 6 Axiom/BSC features: {len(all_features_bsc_axiom)}")
    return all_features_bsc_axiom, all_selected_features


@app.cell
def _(df_level4):
    df_level4
    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP Chemical Space""")
    return


@app.cell
def _(
    AllChem,
    Chem,
    DataStructs,
    df_level4_bsc_batch,
    df_level4a_axiom,
    df_metadata_compound_oasis,
    dic_botanicals_scode_name,
    np,
    pd,
    umap,
):


    # ── 1. Build SMILES lookup for Axiom ─────────────────────────────────────────
    _axiom_oasis_ids = df_level4a_axiom["Metadata_OASIS_ID"].unique()
    _df_axiom_smiles = (
        df_metadata_compound_oasis[
            df_metadata_compound_oasis["OASIS_ID"].isin(_axiom_oasis_ids)
        ][["OASIS_ID", "SMILES"]]
        .drop_duplicates()
        .rename(columns={"OASIS_ID": "Metadata_compound", "SMILES": "SMILES"})
    )
    _df_axiom_smiles["Dataset"] = "AXIOM"

    # ── 2. Build SMILES lookup for BSC ────────────────────────────────────────────
    _df_bsc_smiles = (
        df_level4_bsc_batch[["Metadata_compound", "Metadata_compound_smiles"]]
        .drop_duplicates()
        .dropna(subset=["Metadata_compound_smiles"])
        .rename(columns={"Metadata_compound_smiles": "SMILES"})
    )
    _df_bsc_smiles["Dataset"] = "BSC"

    # ── 3. Combine ────────────────────────────────────────────────────────────────
    df_smiles = (
        pd.concat([_df_axiom_smiles, _df_bsc_smiles], axis=0, ignore_index=True)
        .drop_duplicates(subset=["Metadata_compound"])
        .dropna(subset=["SMILES"])
        .reset_index(drop=True)
    )

    # ── 4. Label point type ───────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _label_point_type(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    df_smiles["PointType"] = df_smiles["Metadata_compound"].apply(_label_point_type)
    df_smiles["DisplayName"] = df_smiles["Metadata_compound"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c, _c) if _c in _botanical_scodes else _c
    )

    # ── 5. Compute Morgan fingerprints (radius=2, nBits=2048, binary) ─────────────
    def _smiles_to_fp(_smi):
        try:
            _mol = Chem.MolFromSmiles(_smi)
            if _mol is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(_mol, radius=2, nBits=2048)
        except:
            return None

    df_smiles["FP"] = df_smiles["SMILES"].apply(_smiles_to_fp)
    df_smiles = df_smiles[df_smiles["FP"].notna()].reset_index(drop=True)

    print(f"Valid structures: {len(df_smiles)}")

    # ── 6. Tanimoto distance matrix ───────────────────────────────────────────────
    _fps = df_smiles["FP"].tolist()
    _n = len(_fps)
    _dist_matrix = np.zeros((_n, _n))

    for _i in range(_n):
        for _j in range(_i + 1, _n):
            _sim = DataStructs.TanimotoSimilarity(_fps[_i], _fps[_j])
            _dist_matrix[_i, _j] = 1 - _sim
            _dist_matrix[_j, _i] = 1 - _sim

    # ── 7. UMAP on precomputed Tanimoto distance ──────────────────────────────────
    _reducer = umap.UMAP(
        n_components=2,
        metric="precomputed",
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    _embedding = _reducer.fit_transform(_dist_matrix)
    df_smiles["UMAP_1"] = _embedding[:, 0]
    df_smiles["UMAP_2"] = _embedding[:, 1]
    return (df_smiles,)


@app.cell
def _(df_smiles, go):
    # ── 8. Plot ───────────────────────────────────────────────────────────────────
    _dataset_colors = {
        _ds: _col for _ds, _col in zip(
            df_smiles["Dataset"].unique(),
            ["#1f77b4", "#ff7f0e"]
        )
    }

    _marker_config = {
        "Compound":  dict(symbol="circle",  size=6,  opacity=0.7),
        "Botanical": dict(symbol="diamond", size=10, opacity=0.9),
        "DMSO":      dict(symbol="star",    size=14, opacity=1.0),
    }

    _fig = go.Figure()

    for _dataset in df_smiles["Dataset"].unique():
        _df_ds = df_smiles[df_smiles["Dataset"] == _dataset]
        _color = _dataset_colors[_dataset]

        for _ptype, _mconf in _marker_config.items():
            _df_sub = _df_ds[_df_ds["PointType"] == _ptype]
            if _df_sub.empty:
                continue

            _fig.add_trace(go.Scatter(
                x=_df_sub["UMAP_1"],
                y=_df_sub["UMAP_2"],
                mode="markers",
                name=f"{_dataset} — {_ptype}",
                marker=dict(
                    symbol=_mconf["symbol"],
                    size=_mconf["size"],
                    opacity=_mconf["opacity"],
                    color=_color,
                    line=dict(width=0.8, color="black") if _ptype != "Compound" else dict(width=0),
                ),
                customdata=_df_sub[["Metadata_compound", "DisplayName", "SMILES"]].values,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "ID: %{customdata[0]}<br>"
                    f"Dataset: {_dataset}<br>"
                    "SMILES: %{customdata[2]}<br>"
                    "UMAP1: %{x:.2f} | UMAP2: %{y:.2f}"
                    "<extra></extra>"
                ),
            ))

    # Botanical text labels
    _df_bot = df_smiles[df_smiles["PointType"] == "Botanical"]
    _fig.add_trace(go.Scatter(
        x=_df_bot["UMAP_1"],
        y=_df_bot["UMAP_2"],
        mode="text",
        text=_df_bot["DisplayName"],
        textposition="top center",
        textfont=dict(size=8, color="black"),
        showlegend=False,
        hoverinfo="skip",
    ))

    _fig.update_layout(
        template="plotly_white",
        title="Chemical space UMAP — Morgan fingerprints (Tanimoto)<br><sup>◆ Botanicals · ★ DMSO · ● Compounds</sup>",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title_text="Dataset — Type",
        width=950,
        height=700,
        legend=dict(itemsizing="constant", font=dict(size=10)),
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## UMAP Cell Painting Space""")
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP Cell Painting level 4 Controls""")
    return


@app.cell
def _(df_level4_bsc_batch, df_level4a_axiom, pd):
    # Add source column
    _df_level4a_axiom = df_level4a_axiom.copy()
    _df_level4_bsc_batch = df_level4_bsc_batch.copy()
    _df_level4a_axiom["Dataset"] = "AXIOM"
    _df_level4_bsc_batch["Dataset"] = "BSC"

    # Bind rows
    df_level4_control = pd.concat(
        [
            _df_level4a_axiom[_df_level4a_axiom["Metadata_compound"] == "DMSO"],
            _df_level4_bsc_batch[_df_level4_bsc_batch["Metadata_compound"] == "s0617825"],
        ],
        axis=0,
        ignore_index=True
    )
    return (df_level4_control,)


@app.cell
def _(all_selected_features, df_level4_control, pd, umap):
    # --- 1. Extract features and metadata
    _X = df_level4_control[all_selected_features].values

    _plates = df_level4_control["Metadata_plate"].astype(str).values
    _concentration = df_level4_control["Metadata_concentration"].astype(float).values
    _log10conc = df_level4_control["Metadata_log10conc"].astype(float).values
    _dataset = df_level4_control["Dataset"].astype(str).values

    # --- 2. Run UMAP for 2D projection
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    _embedding = _reducer.fit_transform(_X)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_level4_control = pd.DataFrame(_embedding, columns=["UMAP_1", "UMAP_2"])
    df_umap_level4_control["Metadata_plate"] = _plates
    df_umap_level4_control["Metadata_concentration"] = _concentration
    df_umap_level4_control["Metadata_log10conc"] = _log10conc
    df_umap_level4_control["Dataset"] = _dataset
    return (df_umap_level4_control,)


@app.cell
def _(df_umap_level4_control, figure_dir, pio, px):
    # --- 4. Create interactive Plotly scatter plot (Figure S6)
    _fig = px.scatter(
        df_umap_level4_control,
        x="UMAP_1",
        y="UMAP_2",
        color="Dataset",
        title="UMAP of Level 4 Cell Painting Profiles of CONTROLS Colored by Dataset (axiom vs BSC)",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_sequence=px.colors.qualitative.Safe  # nice, distinct palette
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="Dataset",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    pio.write_image(
        _fig,
        str(figure_dir / "Figure_S6.png"),
        width=850, height=650, scale=4
    )
    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP Cell Painting level 4""")
    return


@app.cell
def _(all_selected_features, df_level4, pd, umap):
    # --- 1. Extract features and metadata
    _X = df_level4[all_selected_features].values

    _plates = df_level4["Metadata_plate"].astype(str).values
    _concentration = df_level4["Metadata_concentration"].astype(float).values
    _log10conc = df_level4["Metadata_log10conc"].astype(float).values
    _dataset = df_level4["Dataset"].astype(str).values

    # --- 2. Run UMAP for 2D projection
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    _embedding = _reducer.fit_transform(_X)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_level4 = pd.DataFrame(_embedding, columns=["UMAP_1", "UMAP_2"])
    df_umap_level4["Metadata_plate"] = _plates
    df_umap_level4["Metadata_concentration"] = _concentration
    df_umap_level4["Metadata_log10conc"] = _log10conc
    df_umap_level4["Dataset"] = _dataset
    return (df_umap_level4,)


@app.cell
def _(df_umap_level4, px):
    # --- 4. Create interactive Plotly scatter plot
    _fig = px.scatter(
        df_umap_level4,
        x="UMAP_1",
        y="UMAP_2",
        color="Dataset",
        title="UMAP of Level 4 Cell Painting Profiles Colored by Dataset (axiom vs BSC)",
        opacity=0.8,
        width=850,
        height=650,
        color_discrete_sequence=px.colors.qualitative.Safe  # nice, distinct palette
    )

    _fig.update_layout(
        template="plotly_white",
        legend_title_text="Dataset",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2"
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP Cell Painting level 5""")
    return


@app.cell
def _(all_features_bsc_axiom, df_level5, pd, umap):
    _X = df_level5[all_features_bsc_axiom].values

    _dataset = df_level5["Dataset"].astype(str).values
    _compounds = df_level5["Metadata_compound"].astype(str).values
    # --- 2. Run UMAP for 2D projection
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    _embedding = _reducer.fit_transform(_X)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_level5 = pd.DataFrame(_embedding, columns=["UMAP_1", "UMAP_2"])
    df_umap_level5["Dataset"] = _dataset
    df_umap_level5["Metadata_compound"] = _compounds
    return (df_umap_level5,)


@app.cell
def _(df_umap_level5):
    df_umap_level5
    return


@app.cell
def _(Path, df_umap_level5, dic_botanicals_scode_name, go, pio, px):
    # ── Label point type ──────────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _label_point_type(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    df_umap_level5["PointType"] = df_umap_level5["Metadata_compound"].apply(_label_point_type)

    df_umap_level5["DisplayName"] = df_umap_level5["Metadata_compound"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c, _c) if _c in _botanical_scodes else _c
    )

    # ── Color per dataset ─────────────────────────────────────────────────────────
    _dataset_colors = {
        _ds: _col for _ds, _col in zip(
            df_umap_level5["Dataset"].unique(),
            px.colors.qualitative.Safe
        )
    }

    # ── Marker config per point type ──────────────────────────────────────────────
    _marker_config = {
        "Compound":  dict(symbol="circle",       size=6,  opacity=0.7),
        "Botanical": dict(symbol="diamond",      size=10, opacity=0.9),
        "DMSO":      dict(symbol="star",         size=14, opacity=1.0),
    }

    # ── Build figure manually with go.Scatter ─────────────────────────────────────
    _fig = go.Figure()

    for _dataset in df_umap_level5["Dataset"].unique():
        _df_ds = df_umap_level5[df_umap_level5["Dataset"] == _dataset]
        _color = _dataset_colors[_dataset]

        for _ptype, _mconf in _marker_config.items():
            _df_sub = _df_ds[_df_ds["PointType"] == _ptype]
            if _df_sub.empty:
                continue

            _fig.add_trace(go.Scatter(
                x=_df_sub["UMAP_1"],
                y=_df_sub["UMAP_2"],
                mode="markers",
                name=f"{_dataset} — {_ptype}",
                marker=dict(
                    symbol=_mconf["symbol"],
                    size=_mconf["size"],
                    opacity=_mconf["opacity"],
                    color=_color,
                    line=dict(width=0.8, color="black") if _ptype != "Compound" else dict(width=0),
                ),
                customdata=_df_sub[["Metadata_compound", "DisplayName"]].values,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "ID: %{customdata[0]}<br>"
                    f"Dataset: {_dataset}<br>"
                    "UMAP1: %{x:.2f} | UMAP2: %{y:.2f}"
                    "<extra></extra>"
                ),
            ))

    # ── Botanical text labels ─────────────────────────────────────────────────────
    _df_bot_labels = df_umap_level5[df_umap_level5["PointType"] == "Botanical"]
    _fig.add_trace(go.Scatter(
        x=_df_bot_labels["UMAP_1"],
        y=_df_bot_labels["UMAP_2"],
        mode="text",
        text=_df_bot_labels["DisplayName"],
        textposition="top center",
        textfont=dict(size=8, color="black"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────────
    _fig.update_layout(
        template="plotly_white",
        title="UMAP of Level 5 Cell Painting Profiles<br><sup>◆ Botanicals · ★ DMSO · ● Compounds</sup>",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title_text="Dataset — Type",
        width=950,
        height=700,
        legend=dict(
            itemsizing="constant",
            font=dict(size=10),
        )
    )
    # ── Save high-resolution image ────────────────────────────────────────────────
    pio.write_image(_fig, Path.home() / "Downloads" / "umap_level5_axiom_bsc.png", width=950, height=700, scale=6)
    _fig.show()
    return


@app.cell
def _(df_level4, df_umap_level5):
    _ldh_means = (
        df_level4
        .groupby(["Dataset", "Metadata_compound"], as_index=False)
        ["Metadata_ldh_ridge_norm"].mean()
    )

    _merged = df_umap_level5.merge(
        _ldh_means,
        on=["Dataset", "Metadata_compound"],
        how="left",
    )

    df_umap_level5["Metadata_ldh_ridge_norm"] = _merged["Metadata_ldh_ridge_norm"].values
    return


@app.cell
def _(Path, df_umap_level5, dic_botanicals_scode_name, go, pio):
    # ── Label point type (recomputed here in case this is a fresh cell) ──────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _label_point_type(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    df_umap_level5["PointType"] = df_umap_level5["Metadata_compound"].apply(_label_point_type)

    df_umap_level5["DisplayName"] = df_umap_level5["Metadata_compound"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c, _c) if _c in _botanical_scodes else _c
    )

    # ── Marker config per point type (shape only — color now encodes LDH) ────────
    _marker_config = {
        "Compound":  dict(symbol="circle",  size=7,  opacity=0.85),
        "Botanical": dict(symbol="diamond", size=11, opacity=0.95),
        "DMSO":      dict(symbol="star",    size=15, opacity=1.0),
    }

    # ── Symmetric color range so 0 sits at the midpoint of a diverging scale ─────
    _ldh_abs_max = df_umap_level5["Metadata_ldh_ridge_norm"].abs().max()

    _fig = go.Figure()

    for _ptype, _mconf in _marker_config.items():
        _df_sub = df_umap_level5[df_umap_level5["PointType"] == _ptype]
        if _df_sub.empty:
            continue

        _fig.add_trace(go.Scatter(
            x=_df_sub["UMAP_1"],
            y=_df_sub["UMAP_2"],
            mode="markers",
            name=_ptype,
            marker=dict(
                symbol=_mconf["symbol"],
                size=_mconf["size"],
                opacity=_mconf["opacity"],
                color=_df_sub["Metadata_ldh_ridge_norm"],
                colorscale="RdBu_r",
                cmin=-_ldh_abs_max,
                cmax=_ldh_abs_max,
                cmid=0,
                colorbar=dict(title="LDH ridge<br>norm") if _ptype == "Compound" else None,
                showscale=(_ptype == "Compound"),  # only draw one shared colorbar
                line=dict(width=0.8, color="black") if _ptype != "Compound" else dict(width=0),
            ),
            customdata=_df_sub[["Metadata_compound", "DisplayName", "Dataset", "Metadata_ldh_ridge_norm"]].values,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "ID: %{customdata[0]}<br>"
                "Dataset: %{customdata[2]}<br>"
                "LDH ridge norm: %{customdata[3]:.3f}<br>"
                "UMAP1: %{x:.2f} | UMAP2: %{y:.2f}"
                "<extra></extra>"
            ),
        ))

    # ── Botanical text labels ─────────────────────────────────────────────────────
    _df_bot_labels = df_umap_level5[df_umap_level5["PointType"] == "Botanical"]
    _fig.add_trace(go.Scatter(
        x=_df_bot_labels["UMAP_1"],
        y=_df_bot_labels["UMAP_2"],
        mode="text",
        text=_df_bot_labels["DisplayName"],
        textposition="top center",
        textfont=dict(size=8, color="black"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────────
    _fig.update_layout(
        template="plotly_white",
        title="UMAP of Level 5 Cell Painting Profiles — colored by LDH ridge norm<br><sup>◆ Botanicals · ★ DMSO · ● Compounds</sup>",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title_text="Point type",
        width=950,
        height=700,
        legend=dict(itemsizing="constant", font=dict(size=10)),
    )

    # ── Save high-resolution image ────────────────────────────────────────────────
    pio.write_image(_fig, Path.home() / "Downloads" / "umap_level5_ldh_ridge_norm.png", width=950, height=700, scale=6)
    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP Cell Painting level 6""")
    return


@app.cell
def _(all_features_bsc_axiom, df_level6, pd, umap):
    _X = df_level6[all_features_bsc_axiom].values

    _dataset = df_level6["Dataset"].astype(str)
    _compounds = df_level6["Metadata_compound"].astype(str)
    # --- 2. Run UMAP for 2D projection
    _reducer = umap.UMAP(
        n_neighbors=15,          # local neighborhood size
        min_dist=0.1,            # controls cluster tightness
        metric="cosine",      # or 'cosine' for morphology data
        random_state=42
    )

    _embedding = _reducer.fit_transform(_X)

    # --- 3. Store UMAP coordinates in dataframe
    df_umap_level6 = pd.DataFrame(_embedding, columns=["UMAP_1", "UMAP_2"])
    df_umap_level6["Dataset"] = _dataset
    df_umap_level6["Metadata_compound"] = _compounds
    return (df_umap_level6,)


@app.cell
def _(df_umap_level6, dic_botanicals_scode_name, go, px):
    # ── Label point type ──────────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _label_point_type(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    df_umap_level6["PointType"] = df_umap_level6["Metadata_compound"].apply(_label_point_type)

    df_umap_level6["DisplayName"] = df_umap_level6["Metadata_compound"].apply(
        lambda _c: dic_botanicals_scode_name.get(_c, _c) if _c in _botanical_scodes else _c
    )

    # ── Color per dataset ─────────────────────────────────────────────────────────
    _dataset_colors = {
        _ds: _col for _ds, _col in zip(
            df_umap_level6["Dataset"].unique(),
            px.colors.qualitative.Safe
        )
    }

    # ── Marker config per point type ──────────────────────────────────────────────
    _marker_config = {
        "Compound":  dict(symbol="circle",       size=6,  opacity=0.7),
        "Botanical": dict(symbol="diamond",      size=10, opacity=0.9),
        "DMSO":      dict(symbol="star",         size=14, opacity=1.0),
    }

    # ── Build figure manually with go.Scatter ─────────────────────────────────────
    _fig = go.Figure()

    for _dataset in df_umap_level6["Dataset"].unique():
        _df_ds = df_umap_level6[df_umap_level6["Dataset"] == _dataset]
        _color = _dataset_colors[_dataset]

        for _ptype, _mconf in _marker_config.items():
            _df_sub = _df_ds[_df_ds["PointType"] == _ptype]
            if _df_sub.empty:
                continue

            _fig.add_trace(go.Scatter(
                x=_df_sub["UMAP_1"],
                y=_df_sub["UMAP_2"],
                mode="markers",
                name=f"{_dataset} — {_ptype}",
                marker=dict(
                    symbol=_mconf["symbol"],
                    size=_mconf["size"],
                    opacity=_mconf["opacity"],
                    color=_color,
                    line=dict(width=0.8, color="black") if _ptype != "Compound" else dict(width=0),
                ),
                customdata=_df_sub[["Metadata_compound", "DisplayName"]].values,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "ID: %{customdata[0]}<br>"
                    f"Dataset: {_dataset}<br>"
                    "UMAP1: %{x:.2f} | UMAP2: %{y:.2f}"
                    "<extra></extra>"
                ),
            ))

    # ── Botanical text labels ─────────────────────────────────────────────────────
    _df_bot_labels = df_umap_level6[df_umap_level6["PointType"] == "Botanical"]
    _fig.add_trace(go.Scatter(
        x=_df_bot_labels["UMAP_1"],
        y=_df_bot_labels["UMAP_2"],
        mode="text",
        text=_df_bot_labels["DisplayName"],
        textposition="top center",
        textfont=dict(size=8, color="black"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────────
    _fig.update_layout(
        template="plotly_white",
        title="UMAP of Level 6 Cell Painting Profiles<br><sup>◆ Botanicals · ★ DMSO · ● Compounds</sup>",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title_text="Dataset — Type",
        width=950,
        height=700,
        legend=dict(
            itemsizing="constant",
            font=dict(size=10),
        )
    )

    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### UMAP Cell Painting level 5 (Figure 4A) & level 6 (Figure S9)""")
    return


@app.cell
def _(
    df_umap_level5,
    df_umap_level6,
    dic_botanicals_scode_name,
    figure_dir,
    go,
    mo,
    pio,
):
    # ── Shared config ─────────────────────────────────────────────────────────────
    _botanical_scodes = set(dic_botanicals_scode_name.keys())

    def _label_point_type(_compound):
        if _compound in ("s0617825", "DMSO"):
            return "DMSO"
        elif _compound in _botanical_scodes:
            return "Botanical"
        else:
            return "Compound"

    # Scientific color palette — colorblind-safe
    _dataset_palette = [
        "#2166AC", "#D6604D", "#4DAC26", "#8073AC",
        "#E08214", "#1A9641", "#D01C8B", "#4393C3",
    ]

    _marker_config = {
        "Compound":  dict(symbol="circle",  size=5,  opacity=0.55, line_width=0),
        "Botanical": dict(symbol="diamond", size=11, opacity=0.95, line_width=0.8),
        "DMSO":      dict(symbol="star",    size=13, opacity=1.0,  line_width=0.8),
    }

    _scientific_axis = dict(
        showgrid=True,
        gridcolor="#e8e8e8",
        gridwidth=0.5,
        zeroline=False,
        showline=True,
        linecolor="#333333",
        linewidth=1.2,
        tickfont=dict(size=11, family="Arial"),
        title_font=dict(size=13, family="Arial"),
        mirror=True,
    )

    # ── Process both dataframes ───────────────────────────────────────────────────
    _panels = [
        (df_umap_level5, "Level 5 profiles"),
        (df_umap_level6, "Level 6 profiles"),
    ]

    for _df, _ in _panels:
        _df["PointType"]   = _df["Metadata_compound"].apply(_label_point_type)
        _df["DisplayName"] = _df["Metadata_compound"].apply(
            lambda _c: dic_botanicals_scode_name.get(_c, _c) if _c in _botanical_scodes else _c
        )

    # Unified dataset color mapping across both panels
    _all_datasets = sorted(set(
        df_umap_level5["Dataset"].unique().tolist() +
        df_umap_level6["Dataset"].unique().tolist()
    ))
    _dataset_colors = {
        _ds: _dataset_palette[_i % len(_dataset_palette)]
        for _i, _ds in enumerate(_all_datasets)
    }

    # ── Build one single-panel UMAP figure ────────────────────────────────────────
    def _make_umap_panel(_df, _legend_inside):
        _fig = go.Figure()
        _legend_shown = set()  # one legend entry per (dataset, ptype) combo

        # ── Compound / Botanical / DMSO traces ───────────────────────────────────
        for _ptype in ["Compound", "Botanical", "DMSO"]:   # draw order: compounds first
            _mconf = _marker_config[_ptype]
            for _dataset in _all_datasets:
                _df_sub = _df[
                    (_df["Dataset"]    == _dataset) &
                    (_df["PointType"]  == _ptype)
                ]
                if _df_sub.empty:
                    continue

                _legend_key  = (_dataset, _ptype)
                _show_legend = _legend_key not in _legend_shown
                if _show_legend:
                    _legend_shown.add(_legend_key)

                _fig.add_trace(go.Scatter(
                    x=_df_sub["UMAP_1"],
                    y=_df_sub["UMAP_2"],
                    mode="markers",
                    name=f"{_dataset} — {_ptype}",
                    legendgroup=_legend_key[0],
                    showlegend=_show_legend,
                    marker=dict(
                        symbol=_mconf["symbol"],
                        size=_mconf["size"],
                        opacity=_mconf["opacity"],
                        color=_dataset_colors[_dataset],
                        line=dict(width=_mconf["line_width"], color="#222222"),
                    ),
                    customdata=_df_sub[["Metadata_compound", "DisplayName", "Dataset"]].values,
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "ID: %{customdata[0]}<br>"
                        "Dataset: %{customdata[2]}<br>"
                        "UMAP1: %{x:.2f} | UMAP2: %{y:.2f}"
                        "<extra></extra>"
                    ),
                ))

        # ── Botanical text labels ─────────────────────────────────────────────────
        _df_bot = _df[_df["PointType"] == "Botanical"].copy()
        # deduplicate labels at identical positions
        _df_bot = _df_bot.drop_duplicates(subset=["Metadata_compound"])

        _fig.add_trace(go.Scatter(
            x=_df_bot["UMAP_1"],
            y=_df_bot["UMAP_2"],
            mode="text",
            text=_df_bot["DisplayName"],
            textposition="top center",
            textfont=dict(size=8, family="Arial", color="#111111"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # ── Legend: show point type via symbol ───────────────────────────────────
        # Add invisible dummy traces to show symbol legend
        for _ptype, _symbol, _label in [
            ("Compound",  "circle",  "Compound  ●"),
            ("Botanical", "diamond", "Botanical  ◆"),
            ("DMSO",      "star",    "DMSO  ★"),
        ]:
            _fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol=_symbol, size=9, color="#555555",
                            line=dict(width=0.8, color="#222222")),
                name=f"<i>{_label}</i>",
                legendgroup="__type__",
                showlegend=True,
            ))

        # ── Layout ────────────────────────────────────────────────────────────────
        _legend = dict(
            title=dict(text="<b>Dataset / Type</b>", font=dict(size=11)),
            font=dict(size=10, family="Arial"),
            itemsizing="constant",
            bordercolor="#cccccc",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            tracegroupgap=4,
        )
        if _legend_inside:
            # Upper right corner, inside the plot area
            _legend.update(x=0.99, y=0.99, xanchor="right", yanchor="top")
            _width, _height, _margin_r = 650, 520, 40
        else:
            _width, _height, _margin_r = 1000, 650, 180

        _fig.update_layout(
            template="plotly_white",
            xaxis=dict(**_scientific_axis, title=dict(text="UMAP 1")),
            yaxis=dict(**_scientific_axis, title=dict(text="UMAP 2")),
            width=_width,
            height=_height,
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=_legend,
            margin=dict(l=60, r=_margin_r, t=40, b=60),
        )
        return _fig

    # ── Figure 4A: profiles aggregated at the concentration level (Level 5) ──────
    _fig_4a = _make_umap_panel(df_umap_level5, _legend_inside=True)
    pio.write_image(
        _fig_4a,
        str(figure_dir / "Figure_4_A.png"),
        width=_fig_4a.layout.width, height=_fig_4a.layout.height, scale=4
    )

    # ── Figure S9: profiles aggregated at the compound level (Level 6) ───────────
    _fig_s9 = _make_umap_panel(df_umap_level6, _legend_inside=False)
    pio.write_image(
        _fig_s9,
        str(figure_dir / "Figure_S9.png"),
        width=_fig_s9.layout.width, height=_fig_s9.layout.height, scale=4
    )

    mo.vstack([_fig_4a, _fig_s9])
    return


if __name__ == "__main__":
    app.run()
