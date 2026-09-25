import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    from functools import lru_cache
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image
    return (
        Image,
        Path,
        ThreadPoolExecutor,
        lru_cache,
        mo,
        np,
        pd,
        plt,
        re,
        subprocess,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Plot Cell Painting images of botanicals

    Select treatments, dose levels and a channel. Only the TIFFs needed for the
    current selection are downloaded (anonymously) from the public Cell Painting Gallery
    (`cpg0048-bsc`) and cached locally, so switching back to a previous selection is instant.

    Each botanical plate holds one dose level, so DMSO images are taken from the same plate as the treatment.
    """
    )
    return


@app.cell
def _(Path, np):
    METADATA_PATH = "modules/01_preprocessing/inputs/bsc_raw_dinov2_biochem.parquet"
    S3_BASE = "s3://cellpainting-gallery/cpg0048-bsc/axiom/images/2024_09_20_BSC_Batch1/images"
    TIFF_DIR = Path("modules/07_downstream_analysis/outputs/images/tiff")
    PNG_DIR = Path("modules/07_downstream_analysis/outputs/images/png")
    TIFF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    # Botanical plates published in the gallery (one plate per dose level)
    PUBLIC_PLATES = [f"plate_{p}" for p in range(41005680, 41005688)]

    # Channel IDs from the plate Index.xml; colors follow the training notebook
    CHANNELS = {
        "Mito": {"id": 5, "dye": "Mito 641", "color": [1.0, 0.0, 0.0]},
        "DNA": {"id": 1, "dye": "Hoechst 33342", "color": [0.0, 0.0, 1.0]},
        "ER": {"id": 2, "dye": "PhenoVue Fluor 488", "color": [0.0, 1.0, 0.0]},
        "AGP": {"id": 3, "dye": "WGA555/Phalloidin568", "color": [1.0, 1.0, 0.0]},
        "RNA": {"id": 4, "dye": "Nucleic acid 512", "color": [0.6, 0.0, 0.8]},
        "Brightfield": {"id": 6, "dye": "Brightfield", "color": [1.0, 1.0, 1.0]},
    }
    for _ch in CHANNELS.values():
        _ch["color"] = np.array(_ch["color"], dtype=np.float32)
    FLUO_CHANNELS = [c for c in CHANNELS if c != "Brightfield"]
    ALL_CHANNELS = "All channels"
    return (
        ALL_CHANNELS,
        CHANNELS,
        FLUO_CHANNELS,
        METADATA_PATH,
        PNG_DIR,
        PUBLIC_PLATES,
        S3_BASE,
        TIFF_DIR,
    )


@app.cell
def _(METADATA_PATH, PUBLIC_PLATES, pd):
    _str_cols = ["plate", "image_id", "well", "compound_name", "library"]
    meta = pd.read_parquet(METADATA_PATH, columns=_str_cols + ["compound_concentration_um"])
    meta = meta.astype({c: str for c in _str_cols})
    meta = meta[meta.plate.isin(PUBLIC_PLATES)].reset_index(drop=True)

    # image_id ends with _r=<row>_c=<col>_f=<field> (0-based); image files are 1-based
    meta[["row", "col", "field"]] = meta.image_id.str.extract(r"_r=(\d+)_c=(\d+)_f=(\d+)$").astype(int)
    meta["plate_number"] = meta.plate.str.removeprefix("plate_")

    # Rank plates by dose: mean concentration rank of botanicals across plates -> 1 (lowest) .. 8 (highest)
    _bot = meta[meta.library == "botanical"]
    _conc_rank = _bot.groupby("compound_name").compound_concentration_um.rank(method="dense")
    dose_level = _conc_rank.groupby(_bot.plate).mean().rank(method="dense").astype(int)
    meta["dose_level"] = meta.plate.map(dose_level)
    return dose_level, meta


@app.cell
def _(ALL_CHANNELS, CHANNELS, dose_level, meta, mo):
    _botanicals = sorted(meta.loc[meta.library == "botanical", "compound_name"].unique())
    treatments = mo.ui.multiselect(
        options=["DMSO"] + _botanicals,
        value=["DMSO", "Aristolochia fangchi extract", "Ephedra extract"],
        label="Treatments",
    )
    channel = mo.ui.dropdown(options=list(CHANNELS) + [ALL_CHANNELS], value="Mito", label="Channel")
    _dose_options = {f"D{d} ({p})": d for p, d in dose_level.sort_values().items()}
    doses = mo.ui.multiselect(options=_dose_options, value=list(_dose_options), label="Dose levels")
    n_fields = mo.ui.slider(1, 4, value=1, label="Fields per condition", show_value=True)
    crop = mo.ui.dropdown(
        options={"512 px": 512, "1024 px": 1024, "1536 px": 1536, "Full (2160 px)": 2160},
        value="1024 px",
        label="Center crop",
    )
    seed = mo.ui.number(start=0, stop=10_000, value=0, label="Random seed")
    shared_contrast = mo.ui.checkbox(value=True, label="Shared contrast (set on DMSO)")
    download = mo.ui.run_button(label="Download missing images")

    mo.vstack(
        [
            mo.hstack([treatments, channel], justify="start"),
            doses,
            mo.hstack([n_fields, crop, seed, shared_contrast], justify="start"),
        ]
    )
    return (
        channel,
        crop,
        doses,
        download,
        n_fields,
        seed,
        shared_contrast,
        treatments,
    )


@app.cell
def _(doses, meta, n_fields, np, pd, seed, treatments):
    # Randomly pick fields for each treatment x dose level (DMSO comes from the same plate)
    _parts = []
    for _t in treatments.value:
        for _d in sorted(doses.value):
            _pool = meta[(meta.compound_name == _t) & (meta.dose_level == _d)]
            if _pool.empty:
                continue
            _pick = _pool.sample(min(n_fields.value, len(_pool)), random_state=seed.value)
            _parts.append(_pick.assign(treatment=_t, replicate=np.arange(len(_pick))))
    selection = (
        pd.concat(_parts, ignore_index=True)
        if _parts
        else pd.DataFrame(columns=[*meta.columns, "treatment", "replicate"])
    )
    selection[["treatment", "dose_level", "plate", "well", "field", "compound_concentration_um"]]
    return (selection,)


@app.cell
def _(
    ALL_CHANNELS,
    CHANNELS,
    S3_BASE,
    TIFF_DIR,
    ThreadPoolExecutor,
    channel,
    download,
    mo,
    selection,
    subprocess,
):
    mo.stop(selection.empty, mo.md("No image matches the current selection."))

    shown_channels = list(CHANNELS) if channel.value == ALL_CHANNELS else [channel.value]

    # (selection index, channel) -> local TIFF path
    image_paths = {}
    _missing = []
    for _i, _r in selection.iterrows():
        for _ch in shown_channels:
            _name = f"r{_r.row + 1:02d}c{_r.col + 1:02d}f{_r.field + 1:02d}p01-ch{CHANNELS[_ch]['id']}sk1fk1fl1.tiff"
            _local = TIFF_DIR / _r.plate_number / _name
            image_paths[(_i, _ch)] = _local
            if not _local.exists():
                _missing.append((f"{S3_BASE}/{_r.plate_number}/{_name}", _local))

    mo.stop(
        bool(_missing) and not download.value,
        mo.vstack(
            [
                mo.md(f"**{len(_missing)} / {len(image_paths)} files** to download (~{len(_missing) * 8} MB)."),
                download,
            ]
        ),
    )


    def _fetch(item):
        _url, _dest = item
        _dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["aws", "s3", "cp", _url, str(_dest), "--no-sign-request", "--only-show-errors"], check=True
        )


    with ThreadPoolExecutor(max_workers=8) as _pool:
        for _ in mo.status.progress_bar(
            _pool.map(_fetch, _missing), total=len(_missing), title="Downloading images"
        ):
            pass
    return image_paths, shown_channels


@app.cell
def _(Image, lru_cache, np):
    @lru_cache(maxsize=64)
    def load_tiff(path):
        return np.asarray(Image.open(path))


    def center_crop(img, size):
        h, w = img.shape
        size = min(size, h, w)
        y0, x0 = (h - size) // 2, (w - size) // 2
        return img[y0 : y0 + size, x0 : x0 + size].astype(np.float32)


    def contrast_limits(imgs, p_lo=0.5, p_hi=99.5):
        pooled = np.concatenate([img.ravel()[::7] for img in imgs])
        return np.percentile(pooled, [p_lo, p_hi])


    def rescale(img, lo, hi):
        return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return center_crop, contrast_limits, load_tiff, rescale


@app.cell
def _(
    center_crop,
    contrast_limits,
    crop,
    image_paths,
    load_tiff,
    selection,
    shared_contrast,
    shown_channels,
):
    # Shared contrast limits per channel, computed on DMSO fields so treatment effects stay visible
    if shared_contrast.value:
        _ref = selection[selection.treatment == "DMSO"]
        if _ref.empty:
            _ref = selection
        limits = {
            _ch: contrast_limits([center_crop(load_tiff(image_paths[(_i, _ch)]), crop.value) for _i in _ref.index])
            for _ch in shown_channels
        }
    else:
        limits = None
    return (limits,)


@app.cell
def _(
    CHANNELS,
    FLUO_CHANNELS,
    center_crop,
    contrast_limits,
    crop,
    image_paths,
    load_tiff,
    np,
    rescale,
):
    def render(idx, channels, limits):
        """Colorize and sum channels of one field into an RGB image."""
        rgb = None
        for ch in channels:
            img = center_crop(load_tiff(image_paths[(idx, ch)]), crop.value)
            lo, hi = limits[ch] if limits else contrast_limits([img])
            layer = rescale(img, lo, hi)[..., None] * CHANNELS[ch]["color"]
            rgb = layer if rgb is None else rgb + layer
        return np.clip(rgb, 0, 1)


    def tile_channels(channel_name):
        # The merged view excludes brightfield
        return FLUO_CHANNELS if channel_name not in CHANNELS else [channel_name]
    return render, tile_channels


@app.cell
def _(
    CHANNELS,
    PNG_DIR,
    channel,
    crop,
    doses,
    limits,
    n_fields,
    plt,
    re,
    render,
    selection,
    tile_channels,
    treatments,
):
    _doses = sorted(doses.value)
    _rows = [(_t, _k) for _t in treatments.value for _k in range(n_fields.value)]
    _lookup = {(_r.treatment, _r.dose_level, _r.replicate): _i for _i, _r in selection.iterrows()}

    fig_grid, _axes = plt.subplots(
        len(_rows), len(_doses), figsize=(2.2 * len(_doses), 2.2 * len(_rows)), squeeze=False
    )
    for _ri, (_t, _k) in enumerate(_rows):
        for _ci, _d in enumerate(_doses):
            _ax = _axes[_ri, _ci]
            _ax.set_xticks([])
            _ax.set_yticks([])
            for _s in _ax.spines.values():
                _s.set_visible(False)
            if _ri == 0:
                _ax.set_title(f"D{_d}", fontsize=10)
            if _ci == 0:
                _ax.set_ylabel(_t if n_fields.value == 1 else f"{_t}\nfield {_k + 1}", fontsize=8)
            _idx = _lookup.get((_t, _d, _k))
            if _idx is None:
                continue
            _r = selection.loc[_idx]
            _ax.imshow(render(_idx, tile_channels(channel.value), limits))
            _ax.text(0.03, 0.97, f"{_r.well} f{_r.field + 1}", color="w", fontsize=6, va="top", transform=_ax.transAxes)
            if _t != "DMSO":
                _ax.text(
                    0.03, 0.03, f"{_r.compound_concentration_um:.3g} µM", color="w", fontsize=6, transform=_ax.transAxes
                )

    _label = channel.value if channel.value in CHANNELS else "Merged"
    fig_grid.suptitle(f"{_label} ({CHANNELS[_label]['dye']})" if _label in CHANNELS else "Merged fluorescent channels")
    fig_grid.tight_layout()

    _slug = "_".join(re.sub(r"\W+", "", _t.split()[0]) for _t in treatments.value)
    fig_grid.savefig(
        PNG_DIR / f"grid_{_label}_{_slug}_crop{crop.value}_n{n_fields.value}.png", dpi=300, bbox_inches="tight"
    )
    fig_grid
    return


@app.cell
def _(ALL_CHANNELS, channel, mo, selection):
    mo.stop(channel.value != ALL_CHANNELS)
    _options = {
        f"{_r.treatment} | D{_r.dose_level} | {_r.well} f{_r.field + 1}": _i for _i, _r in selection.iterrows()
    }
    detail_image = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="Image for channel breakdown")
    detail_image
    return (detail_image,)


@app.cell
def _(
    CHANNELS,
    FLUO_CHANNELS,
    PNG_DIR,
    crop,
    detail_image,
    limits,
    plt,
    re,
    render,
    selection,
):
    # Each channel on its own plus the merge, as in the training notebook
    _idx = detail_image.value
    _r = selection.loc[_idx]
    fig_detail, _axes = plt.subplots(2, 4, figsize=(13, 6.4))
    _axes = _axes.ravel()
    for _ax, _ch in zip(_axes, CHANNELS):
        _ax.imshow(render(_idx, [_ch], limits))
        _ax.set_title(f"{_ch} ({CHANNELS[_ch]['dye']})", fontsize=9)
    _axes[len(CHANNELS)].imshow(render(_idx, FLUO_CHANNELS, limits))
    _axes[len(CHANNELS)].set_title("Merged", fontsize=9)
    for _ax in _axes:
        _ax.axis("off")
    fig_detail.suptitle(f"{_r.treatment} - D{_r.dose_level} ({_r.plate}) - {_r.well} field {_r.field + 1}")
    fig_detail.tight_layout()

    _slug = re.sub(r"\W+", "", _r.treatment.split()[0])
    fig_detail.savefig(
        PNG_DIR / f"channels_{_slug}_D{_r.dose_level}_{_r.well}_f{_r.field + 1}_crop{crop.value}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig_detail
    return


if __name__ == "__main__":
    app.run()
