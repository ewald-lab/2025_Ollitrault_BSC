import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import spearmanr
    from adjustText import adjust_text

    # Re-use the exact MW function from preprocessing, so the botanical MW
    # here is the same one that was used to convert µg/mL → µM.
    sys.path.insert(0, "modules/01_preprocessing/scripts")
    from format_bsc_dino import compute_botanical_mw
    return (
        Line2D,
        Path,
        adjust_text,
        compute_botanical_mw,
        json,
        mo,
        np,
        pd,
        pl,
        plt,
        spearmanr,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Botanical PoD vs constituent PoD, adjusted for abundance and MW

    **Question:** can the Cell Painting (CP) PoD of a whole botanical be
    explained by its constituents, given how much of each one the extract
    contains?

    ## Adjusted constituent PoD

    A constituent tested alone has a PoD in µM *of that molecule*, while the
    botanical PoD is in µM *of extract*. To compare them we ask: *at which
    extract dose does constituent i, inside the mix, reach its own PoD?*

    Worked through for aristolochic acid I (AAI) in Aristolochia, at an
    extract dose E = 1 µM on the botanical MW basis (MW_bot = 339.9):

    | Step | Formula | AAI |
    |---|---|---|
    | extract mass in the well | E × MW_bot / 1000 | 0.340 µg/mL |
    | the constituent's share (cᵢ = 76 mg/g) | × cᵢ / 1000 | 0.0258 µg/mL |
    | back to µM (MWᵢ = 341.3) | × 1000 / MWᵢ | 0.0757 µM |

    AAI's own PoD is 0.149 µM, so at 1 µM of extract it sits at about half
    of it; the extract dose that gets it there is 1.97 µM. In one line:

    ```
    adjusted PoDᵢ = PoDᵢ × (1000 / cᵢ) × (MWᵢ / MW_bot)
    ```

    The dilution term (1000 / cᵢ) does most of the work; the MW term is
    usually close to 1. The result is an **extract dose**, so it compares
    directly with the measured botanical PoD.

    ## Predicted mixture PoD (concentration addition)

    Each constituent contributes a **toxic unit** TUᵢ(E) = E / adjusted
    PoDᵢ, equal to 1 when that constituent alone is at its PoD. Additivity
    means the mixture is at its PoD once the toxic units sum to 1:

    ```
    Σᵢ E / adjusted PoDᵢ = 1   →   predicted mixture PoD = 1 / Σᵢ (1 / adjusted PoDᵢ)
    ```

    For Aristolochia at 1 µM of extract the toxic units are 0.5066 (AAI),
    0.0005 (AAII) and 0.0001 (AAC). They sum to 0.507, so the predicted PoD
    is 1 / 0.507 = 1.97 µM, against 1.01 µM measured. Each constituent's
    share of that sum is its **contribution %** — 99.9% for AAI.

    So the prediction is always below the smallest adjusted PoD, and is
    driven by whichever constituent is both potent *and* abundant.

    **Assumptions:** constituents act by the same mechanism and do not
    interact; constituents without a PoD, and all unquantified mass, count
    as zero, so anything missing can only make the extract come out more
    potent than predicted; contents are means with the SD dropped.

    **How to read the figure** (y = measured botanical PoD,
    x = adjusted constituent PoD):

    - **on the 1:1 line**: the constituent alone explains the botanical PoD
    - **right of the line** (x > y): the constituent is too dilute to
      explain the botanical effect alone; other compounds also contribute
    - **left of the line** (x < y): the constituent should already be active
      at a lower botanical dose than observed; the extract is less potent
      than expected (matrix effects, antagonism, or overestimated content)

    Same logic as the "MW adjusted PoD" column in
    `docs/Ashwin_data/BSC_CP_botanical_pod_table_v5.xlsx`.
    """
    )
    return


@app.cell
def _(Path, json):
    # ── Settings ──────────────────────────────────────────────────────────
    # The conversion mode and constituent file are read from the BSC config,
    # so they match what the pipeline used to compute the botanical PoDs.
    with open("configs/config_base_bsc.json") as _f:
        _config = json.load(_f)

    conversion_mode = _config.get("bsc_conversion_mode", "mw_mean")
    constituent_table_path = _config["post_processing"]["bsc_annotation"]

    featurewise_pod_path = "modules/04_concresponse/outputs/pods/pods_bsc_cp_feature_wise_all.parquet"
    featurewise_threshold_path = "modules/04_concresponse/outputs/pods/feature_wise/feature_wise_median_permutation_threshold_bsc.txt"
    annotations_path = "modules/01_preprocessing/outputs/cellpainting_bsc_annotations_merged.tsv"

    figure_dir = Path("modules/07_downstream_analysis/figures")
    table_dir = Path("modules/07_downstream_analysis/tables")
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Same colours as 6_bsc_analysis.marimo.py
    botanical_colors = {
        "Kratom": "#e6194B",
        "Ashwagandha": "#3cb44b",
        "Comfrey": "#ffe119",
        "Usnea": "#4363d8",
        "Asian ginseng": "#f58231",
        "Blue cohosh": "#42d4f4",
        "Yohimbe": "#f032e6",
        "Kava": "#fabed4",
        "Milk thistle": "#469990",
        "Goldenseal": "#dcbeff",
        "Aristolochia": "#9A6324",
        "Green tea extract": "#fffac8",
        "Ephedra": "#800000",
    }

    print(f"Conversion mode: {conversion_mode}")
    print(f"Constituent table: {constituent_table_path}")
    return (
        annotations_path,
        botanical_colors,
        constituent_table_path,
        conversion_mode,
        featurewise_pod_path,
        featurewise_threshold_path,
        figure_dir,
        table_dir,
    )


@app.cell
def _(featurewise_pod_path, featurewise_threshold_path, pd):
    # ── CP PoD per compound ───────────────────────────────────────────────
    # Same rule as notebooks 4 and 6: keep features that pass all fit checks,
    # keep compounds with enough active features, and take the 5th
    # percentile of their feature-wise PoDs.
    _df_featurewise = pd.read_parquet(featurewise_pod_path)
    with open(featurewise_threshold_path) as _f:
        _min_active_features = float(_f.read().strip())

    _df_passed = _df_featurewise[_df_featurewise["all.pass"] == True]
    _n_active = _df_passed.groupby("Metadata_compound").size()
    _active_compounds = _n_active[_n_active >= _min_active_features].index

    cp_pod_um = (
        _df_passed[_df_passed["Metadata_compound"].isin(_active_compounds)]
        .groupby("Metadata_compound")["cp_POD_um"]
        .quantile(0.05)
    )
    print(f"Compounds with a CP PoD: {len(cp_pod_um)} "
          f"(threshold: {_min_active_features:g} active features)")
    return (cp_pod_um,)


@app.cell
def _(compute_botanical_mw, constituent_table_path, conversion_mode, pl):
    # ── MW used for each botanical ────────────────────────────────────────
    _df_constituent_table = pl.read_csv(constituent_table_path, separator="\t")
    botanical_mw = (
        compute_botanical_mw(_df_constituent_table, conversion_mode)
        .to_pandas()
        .set_index("Metadata_compound")["avg_mw"]
    )
    return (botanical_mw,)


@app.cell
def _(annotations_path, botanical_mw, cp_pod_um, pd):
    # ── One row per (constituent, botanical) pair ─────────────────────────
    # A constituent can belong to two botanicals (e.g. epicatechin in green
    # tea and kratom), so it gets one row for each.
    _df_annotations = pd.read_csv(annotations_path, sep="\t")
    _content_col = next(
        c for c in _df_annotations.columns if c.strip() == "Concentration (mg/g)"
    )

    df_constituents = pd.DataFrame({
        "constituent_scode": _df_annotations["Metadata_compound"],
        "constituent": _df_annotations["Constituent"],
        "botanical_scode": _df_annotations["Plant scode"],
        "botanical": _df_annotations["Plant name"],
        "mw_constituent": _df_annotations["MW (g/mol)"],
        # "5.2 ± 0.8" → 5.2 (keep the mean, drop the SD)
        "content_mg_g": (
            _df_annotations[_content_col]
            .astype(str)
            .str.extract(r"^\s*(\d*\.?\d+)")[0]
            .astype(float)
        ),
    })

    df_constituents["mw_botanical"] = df_constituents["botanical_scode"].map(botanical_mw)
    df_constituents["pod_constituent_um"] = df_constituents["constituent_scode"].map(cp_pod_um)
    df_constituents["pod_botanical_um"] = df_constituents["botanical_scode"].map(cp_pod_um)

    # Botanical dose (µM) at which this constituent reaches its own PoD
    df_constituents["pod_constituent_adjusted_um"] = (
        df_constituents["pod_constituent_um"]
        * (1000 / df_constituents["content_mg_g"])
        * (df_constituents["mw_constituent"] / df_constituents["mw_botanical"])
    )

    # 1 = explains alone; >1 = too dilute; <1 = extract less potent than expected
    df_constituents["ratio_adjusted_vs_botanical"] = (
        df_constituents["pod_constituent_adjusted_um"]
        / df_constituents["pod_botanical_um"]
    )

    # Share of the predicted mixture effect (concentration addition)
    _inverse_pod = 1 / df_constituents["pod_constituent_adjusted_um"]
    df_constituents["share_of_mixture_pct"] = (
        100 * _inverse_pod / _inverse_pod.groupby(df_constituents["botanical"]).transform("sum")
    )

    df_constituents = df_constituents.sort_values(
        ["botanical", "pod_constituent_adjusted_um"]
    ).reset_index(drop=True)

    print(f"Constituent-botanical pairs: {len(df_constituents)}")
    print(f"  with a constituent CP PoD: {df_constituents['pod_constituent_um'].notna().sum()}")
    print(f"  with both PoDs (plotted): "
          f"{df_constituents[['pod_constituent_um', 'pod_botanical_um']].notna().all(axis=1).sum()}")
    df_constituents
    return (df_constituents,)


@app.cell
def _(df_constituents, pd):
    # ── Summary per botanical ─────────────────────────────────────────────
    # Top driver = constituent with the lowest adjusted PoD (largest share).
    _rows = []
    for _botanical, _df in df_constituents.groupby("botanical"):
        _df_with_pod = _df.dropna(subset=["pod_constituent_adjusted_um"])
        _pod_botanical = _df["pod_botanical_um"].iloc[0]

        if _df_with_pod.empty:
            _top = None
            _mixture_pod = float("nan")
        else:
            _top = _df_with_pod.iloc[0]
            _mixture_pod = 1 / (1 / _df_with_pod["pod_constituent_adjusted_um"]).sum()

        _rows.append({
            "botanical": _botanical,
            "botanical_pod_um": _pod_botanical,
            "n_constituents_tested": len(_df),
            "n_constituents_with_pod": len(_df_with_pod),
            "top_driver": None if _top is None else _top["constituent"],
            "top_driver_share_pct": None if _top is None else _top["share_of_mixture_pct"],
            "top_driver_adjusted_pod_um": None if _top is None else _top["pod_constituent_adjusted_um"],
            "top_driver_ratio": None if _top is None else _top["ratio_adjusted_vs_botanical"],
            "mixture_predicted_pod_um": _mixture_pod,
            "mixture_ratio": _mixture_pod / _pod_botanical,
        })

    df_botanical_summary = pd.DataFrame(_rows).sort_values("mixture_ratio")
    df_botanical_summary
    return (df_botanical_summary,)


@app.cell
def _(df_botanical_summary, df_constituents, np, spearmanr):
    # ── Does the adjusted constituent PoD track the botanical PoD? ────────
    # (1) all constituent points; (2) one point per botanical (top driver).
    # Note: in (1), constituents of the same botanical share the same y.
    _pairs = df_constituents.dropna(subset=["pod_constituent_adjusted_um", "pod_botanical_um"])
    _rho_all, _p_all = spearmanr(
        np.log10(_pairs["pod_constituent_adjusted_um"]),
        np.log10(_pairs["pod_botanical_um"]),
    )

    _top = df_botanical_summary.dropna(subset=["top_driver_adjusted_pod_um", "botanical_pod_um"])
    _rho_top, _p_top = spearmanr(
        _top["top_driver_adjusted_pod_um"].astype(float),
        _top["botanical_pod_um"],
    )

    # How many top drivers fall within 10-fold of the 1:1 line?
    _within_10_top = (np.abs(np.log10(_top["top_driver_ratio"].astype(float))) <= 1).sum()

    correlation_text = (
        f"Spearman ρ, all constituents: {_rho_all:.2f} (p = {_p_all:.2g}, n = {len(_pairs)})\n"
        f"Spearman ρ, top driver per botanical: {_rho_top:.2f} (p = {_p_top:.2g}, n = {len(_top)})\n"
        f"Top driver within 10-fold of 1:1: {_within_10_top}/{len(_top)}"
    )
    print(correlation_text)
    return (correlation_text,)


@app.cell
def _(
    Line2D,
    adjust_text,
    botanical_colors,
    correlation_text,
    df_botanical_summary,
    df_constituents,
    figure_dir,
    np,
    plt,
):
    # ── Figure: botanical PoD (y) vs adjusted constituent PoD (x) ─────────
    def marker_area(content_mg_g):
        """Dot area, so that the radius scales with the content in mg/g."""
        return 20 + 30 * np.sqrt(content_mg_g)

    def make_pod_scatter(label_all=False, file_stem="BSC_constituent_abundance_pod"):
        """
        label_all=False : label only the top driver of each botanical
        label_all=True  : label every constituent that is plotted
        """
        pairs = df_constituents.dropna(subset=["pod_constituent_adjusted_um", "pod_botanical_um"])

        fig, ax = plt.subplots(figsize=(12, 10) if label_all else (9, 7.5))

        # Axis range covering every point, in whole decades
        all_values = np.concatenate([
            pairs["pod_constituent_adjusted_um"],
            pairs["pod_botanical_um"],
        ])
        lo = 10 ** np.floor(np.log10(all_values.min()))
        hi = 10 ** np.ceil(np.log10(all_values.max()))
        line = np.array([lo, hi])

        # 1:1 agreement line and a light 10-fold band around it
        ax.fill_between(line, line / 10, line * 10, color="grey", alpha=0.12, lw=0)
        ax.plot(line, line, color="black", lw=1.2, ls="--")

        # One dot per constituent, size = content in the extract (mg/g)
        for botanical, df in pairs.groupby("botanical"):
            ax.scatter(
                df["pod_constituent_adjusted_um"],
                df["pod_botanical_um"],
                s=marker_area(df["content_mg_g"]),
                color=botanical_colors.get(botanical, "grey"),
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
                zorder=3,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # Labels, spread apart by adjustText
        if label_all:
            labels = [
                ax.text(row["pod_constituent_adjusted_um"], row["pod_botanical_um"],
                        row["constituent"], fontsize=6.5, zorder=5)
                for _, row in pairs.iterrows()
            ]
        else:
            labels = [
                ax.text(row["top_driver_adjusted_pod_um"], row["botanical_pod_um"],
                        row["top_driver"], fontsize=8, zorder=5)
                for _, row in df_botanical_summary.dropna(
                    subset=["top_driver_adjusted_pod_um", "botanical_pod_um"]
                ).iterrows()
            ]
        adjust_text(
            labels,
            ax=ax,
            expand=(1.3, 1.6) if label_all else (1.4, 1.8),
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        )

        ax.set_aspect("equal")
        ax.set_xlabel("Constituent CP PoD, adjusted for abundance and MW\n(botanical µM at which the constituent reaches its own PoD)")
        ax.set_ylabel("Whole botanical CP PoD (µM, weighted MW)")
        ax.set_title("Can one constituent explain the botanical PoD?")
        ax.text(0.02, 0.98, correlation_text, transform=ax.transAxes, fontsize=7.5,
                va="top", ha="left", bbox=dict(facecolor="white", edgecolor="lightgrey"))
        ax.text(0.97, 0.03, "too dilute to explain alone →", transform=ax.transAxes,
                fontsize=8, ha="right", color="dimgrey")
        ax.text(0.03, 0.75, "← extract less potent\n    than expected", transform=ax.transAxes,
                fontsize=8, ha="left", color="dimgrey")

        # Legend: botanicals, then marker meaning, then marker size scale
        legend_botanicals = [
            Line2D([], [], marker="o", ls="", markersize=8, markeredgecolor="black",
                   color=botanical_colors.get(b, "grey"), label=b)
            for b in sorted(pairs["botanical"].unique())
        ]
        legend_markers = [
            Line2D([], [], marker="o", ls="", markersize=8, color="white",
                   markeredgecolor="black", label="constituent"),
            Line2D([], [], color="black", ls="--", label="1:1"),
        ]
        legend_sizes = [
            Line2D([], [], marker="o", ls="", markersize=np.sqrt(marker_area(c)),
                   color="white", markeredgecolor="black", label=f"{c} mg/g")
            for c in (1, 10, 100)
        ]
        ax.legend(handles=legend_botanicals + legend_markers + legend_sizes,
                  bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False,
                  labelspacing=1.1)

        fig.tight_layout()
        fig.savefig(figure_dir / f"{file_stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figure_dir / f"{file_stem}.svg", bbox_inches="tight")
        return fig
    return (make_pod_scatter,)


@app.cell
def _(make_pod_scatter):
    # Top driver of each botanical labelled
    make_pod_scatter()
    return


@app.cell
def _(make_pod_scatter):
    # Same plot, every constituent labelled
    make_pod_scatter(label_all=True, file_stem="BSC_constituent_abundance_pod_all_labels")
    return


@app.cell
def _(mo):
    mo.md(r"""## Tables""")
    return


@app.cell
def _(df_botanical_summary, df_constituents, table_dir):
    # Save both tables for the manuscript / supplementary material
    df_constituents.to_csv(table_dir / "BSC_constituent_abundance_pod.csv", index=False)
    df_botanical_summary.to_csv(table_dir / "BSC_botanical_driver_summary.csv", index=False)
    print(f"Saved tables to {table_dir}")
    return


@app.cell
def _(df_constituents, mo):
    # Constituents that could not be plotted, and why
    def _reason(row):
        reasons = []
        if row["pod_botanical_um"] != row["pod_botanical_um"]:  # NaN check
            reasons.append("botanical has no CP PoD")
        if row["pod_constituent_um"] != row["pod_constituent_um"]:
            reasons.append("constituent has no CP PoD")
        return "; ".join(reasons)

    _missing = df_constituents[["pod_constituent_um", "pod_botanical_um"]].isna().any(axis=1)
    df_not_plotted = df_constituents[_missing].copy()
    df_not_plotted["reason"] = df_not_plotted.apply(_reason, axis=1)
    df_not_plotted = df_not_plotted[["botanical", "constituent", "content_mg_g", "reason"]]

    mo.vstack([
        mo.md(f"**Not plotted:** {len(df_not_plotted)} constituent-botanical pairs"),
        df_not_plotted,
    ])
    return


if __name__ == "__main__":
    app.run()
