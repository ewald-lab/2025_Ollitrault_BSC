from __future__ import annotations

import argparse
import os
from contextlib import nullcontext

import joblib
import numpy as np
import pandas as pd
import polars as pl

try:
    import cupy as cp
except ImportError:
    cp = None

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def log_status(message: str) -> None:
    """Print progress immediately in local and Slurm logs."""
    print(message, flush=True)


def empty_prediction_frame() -> pl.DataFrame:
    """Return an empty prediction table with the expected output schema."""
    return pl.DataFrame(
        schema={
            "Metadata_compound": pl.Utf8,
            "y_prob": pl.Float64,
            "y_pred": pl.Int64,
            "y_actual": pl.Int64,
            "k_fold": pl.Int64,
            "repeat": pl.Int64,
            "Metadata_Label": pl.Utf8,
            "Metadata_Count_0": pl.Int64,
            "Metadata_Count_1": pl.Int64,
            "Model_type": pl.Utf8,
        },
    )


def normalize_device(device: str) -> str:
    """Normalize user-facing device names."""
    device = device.lower()
    if device == "cuda":
        return "gpu"
    if device not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported device '{device}'. Use 'cpu' or 'gpu'.")
    return device


def count_available_gpus() -> int:
    """Count visible CUDA devices."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        visible_devices = [device.strip() for device in cuda_visible.split(",") if device.strip()]
        if visible_devices == ["NoDevFiles"]:
            return 0
        return len(visible_devices)

    if cp is None:
        return 0

    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def to_model_array(values: np.ndarray, device: str):
    """Move arrays to the requested compute backend."""
    if device == "gpu" and cp is not None:
        return cp.asarray(values)
    return np.asarray(values)


def to_numpy(values, device: str) -> np.ndarray:
    """Return model outputs as NumPy arrays for stable serialization."""
    if device == "gpu" and cp is not None and isinstance(values, cp.ndarray):
        return cp.asnumpy(values)
    return np.asarray(values)


def scale_pos_weight(y_values: np.ndarray) -> float:
    """Compute the XGBoost class-imbalance weight."""
    n_negative = np.sum(y_values == 0)
    n_positive = np.sum(y_values == 1)
    if n_positive == 0:
        return 1.0
    return float(n_negative / n_positive)


def make_xgb_classifier(device: str, gpu_id: int, n_jobs: int, y_values: np.ndarray) -> XGBClassifier:
    """Build an XGBoost binary classifier for the requested backend."""
    xgb_device = f"cuda:{gpu_id}" if device == "gpu" else "cpu"
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=150,
        tree_method="hist",
        device=xgb_device,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight(y_values),
        n_jobs=n_jobs,
    )


def binary_classifier(
    dat: pd.DataFrame,
    meta: pd.DataFrame,
    n_splits: int,
    *,
    device: str = "cpu",
    gpu_id: int = 0,
    shuffle: bool = False,
    cc: bool = False,
    n_repeats: int = 1,
    fit_final_model: bool = True,
    n_jobs: int = 1,
    label_name: str = "label",
    model_type: str = "Actual",
) -> tuple[XGBClassifier | None, pl.DataFrame]:
    """Perform a binary XGBoost classification.

    Parameters
    ----------
    dat : pd.DataFrame
        The input dataframe containing features and labels.
    meta : pd.DataFrame
        Metadata associated with the input data.
    n_splits : int
        Number of folds for cross-validation.
    shuffle : bool, optional
        Whether to shuffle the data before splitting (default is False).
    cc : bool, optional
        Whether to use only cell count as a feature (default is False).
    n_repeats : int, optional
        Number of times to repeat the cross-validation (default is 1).
    fit_final_model : bool, optional
        Whether to train and return a final model fitted on all samples.

    Returns
    -------
    tuple[XGBClassifier, pl.DataFrame]
        A tuple containing:
        - The final model trained on all data
        - A Polars DataFrame with predictions from all folds and iterations

    """
    device = normalize_device(device)
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    labeled_rows = dat["Label"].notna()
    dat = dat.loc[labeled_rows].copy()
    meta = meta.loc[labeled_rows].reset_index(drop=True)

    dat["Label"] = dat["Label"].astype(int)
    x = dat.drop(columns=["Label"]).reset_index(drop=True)
    y = dat["Label"].reset_index(drop=True)

    if cc:
        x = x[["Cell_Count"]]
    else:
        x = x.drop(columns=["Cell_Count"])

    if shuffle:
        y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    total_folds = n_splits * n_repeats

    log_status(
        f"[CV] {label_name} | {model_type}: starting {n_repeats} repeat(s) x {n_splits} fold(s) "
        f"({total_folds} folds); samples={len(y)}, features={x.shape[1]}, "
        f"inactive={int((y == 0).sum())}, active={int((y == 1).sum())}, device={device.upper()}",
    )

    pred_df = []
    device_context = cp.cuda.Device(gpu_id) if device == "gpu" and cp is not None else nullcontext()
    with device_context:
        for split_number, (train_index, val_index) in enumerate(rskf.split(x, y), start=1):
            repeat_num = (split_number - 1) // n_splits + 1
            fold_num = (split_number - 1) % n_splits + 1
            global_fold = (repeat_num - 1) * n_splits + fold_num
            if fold_num == 1:
                log_status(f"[CV] {label_name} | {model_type}: starting repeat {repeat_num}/{n_repeats}")

            x_fold_train = to_model_array(x.iloc[train_index].to_numpy(), device)
            x_fold_val = to_model_array(x.iloc[val_index].to_numpy(), device)
            y_fold_train, y_fold_val = y.iloc[train_index], y.iloc[val_index]
            train_active = int((y_fold_train == 1).sum())
            val_active = int((y_fold_val == 1).sum())

            log_status(
                f"[CV] {label_name} | {model_type}: fold {global_fold}/{total_folds} "
                f"(repeat {repeat_num}/{n_repeats}, fold {fold_num}/{n_splits}); "
                f"train={len(train_index)} inactive={len(train_index) - train_active} active={train_active}; "
                f"validation={len(val_index)} inactive={len(val_index) - val_active} active={val_active}",
            )

            le = LabelEncoder()
            y_fold_train = le.fit_transform(y_fold_train)
            y_fold_val = le.transform(y_fold_val)
            y_fold_train_model = to_model_array(y_fold_train, device)

            meta_fold_val = meta.iloc[val_index]

            model = make_xgb_classifier(device, gpu_id, n_jobs, y_fold_train)

            # Train the model on the fold training set
            model.fit(x_fold_train, y_fold_train_model)

            # Validate the model on the fold validation set
            y_fold_prob = to_numpy(model.predict_proba(x_fold_val)[:, 1], device)
            y_fold_pred = to_numpy(model.predict(x_fold_val), device)
            log_status(f"[CV] {label_name} | {model_type}: finished fold {global_fold}/{total_folds}")

            pred_df.append(
                pl.DataFrame({
                    "Metadata_compound": list(meta_fold_val["Metadata_compound"]),
                    "y_prob": list(y_fold_prob),
                    "y_pred": list(y_fold_pred),
                    "y_actual": list(y_fold_val),
                    "k_fold": global_fold,
                    "repeat": repeat_num,
                }),
            )

        final_model = None
        if fit_final_model:
            log_status(f"[CV] {label_name} | {model_type}: fitting final full-data model")
            le = LabelEncoder()
            y_all = le.fit_transform(y)
            final_model = make_xgb_classifier(device, gpu_id, n_jobs, y_all)
            final_model.fit(to_model_array(x.to_numpy(), device), to_model_array(y_all, device))

    class_results = pl.concat(pred_df, how="vertical") if pred_df else empty_prediction_frame().drop("Model_type")
    log_status(f"[CV] {label_name} | {model_type}: complete; prediction_rows={class_results.height}")
    return final_model, class_results


def process_label(
    dat,
    label_column,
    n_splits,
    labels,
    *,
    device: str = "cpu",
    gpu_id: int = 0,
    shuffle: bool = False,
    cc: bool = False,
    n_repeats: int = 1,
    output_dir: str | None = None,
    save_model: bool = False,
    n_jobs: int = 1,
):
    """Process a single label_column."""
    prof = dat.rename({label_column: "Label"})
    model_type = "Random_baseline" if shuffle else ("Cellcount_baseline" if cc else "Actual")

    num_0 = prof.filter(pl.col("Label") == 0).height
    num_1 = prof.filter(pl.col("Label") == 1).height
    if num_0 < n_splits or num_1 < n_splits:
        raise ValueError(
            f"Label '{label_column}' has inactive={num_0}, active={num_1}; "
            f"cannot run {n_splits}-fold cross-validation. "
            "Filter labels first with filter_classification_labels.py.",
        )

    meta_cols = [i for i in prof.columns if "Metadata_" in i]
    all_meta_cols = [i for i in prof.columns if i in labels] + meta_cols

    prof_meta = prof.select(meta_cols)
    prof = prof.drop(all_meta_cols)
    log_status(
        f"[Label] {label_column} | {model_type}: inactive={num_0}, active={num_1}, "
        f"n_splits={n_splits}, n_repeats={n_repeats}",
    )

    model, class_res = binary_classifier(
        prof.to_pandas(),
        prof_meta.to_pandas(),
        n_splits=n_splits,
        device=device,
        gpu_id=gpu_id,
        shuffle=shuffle,
        cc=cc,
        n_repeats=n_repeats,
        fit_final_model=save_model,
        n_jobs=n_jobs,
        label_name=label_column,
        model_type=model_type,
    )

    if output_dir and save_model and model is not None:
        model_file_type = "random" if shuffle else ("cellcount" if cc else "actual")
        safe_label = label_column.replace("/", "_").replace("\\", "_").replace(" ", "_")

        model_filename = f"{safe_label}_{model_file_type}_model.pkl"
        model_path = os.path.join(output_dir, model_filename)

        joblib.dump(model, model_path)
        log_status(f"Saved model to {model_path}")

    class_res = class_res.with_columns(
        pl.lit(label_column).alias("Metadata_Label"),
        pl.lit(num_0).alias("Metadata_Count_0"),
        pl.lit(num_1).alias("Metadata_Count_1"),
    )

    return class_res


def predict_binary(
    input_path: str | None = None,
    label_path: str | None = None,
    output_path: str | None = None,
    dat: pl.DataFrame | None = None,
    n_repeats: int = 1,
    label_column: str = "OASIS_ID",
    device: str = "cpu",
    save_models: bool = True,
    n_jobs: int = 1,
    include_labels: list[str] | None = None,
) -> None:
    """Build classifier for binary outcomes.

    Parameters
    ----------
    input_path : str
        Filepath for input profiles.
    label_path : str
        Filepath for input binary labels.
    n_repeats : int, optional
        Number of times to repeat cross-validation (default is 1).
    label_column : str, optional
        Column in label_path to use as label (default is "OASIS_ID"). If using "ID", will use Metadata_compound for join.
    output_path : str
        Filepath for model classification results.

    """
    device = normalize_device(device)
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    if output_path is None:
        raise ValueError("output_path must be provided.")

    num_gpus = count_available_gpus()
    if device == "gpu" and num_gpus == 0:
        raise RuntimeError("GPU execution requested, but no CUDA devices are visible.")

    model_n_jobs = max(1, n_jobs)

    log_status(
        f"Running classification on {device.upper()} for {n_repeats} repeat(s); "
        f"visible GPUs={num_gpus}, xgboost_n_jobs={model_n_jobs}",
    )

    n_splits = 5

    if dat is None:
        if input_path is None:
            raise ValueError("Either 'dat' or 'input_path' must be provided.")
        dat = pl.read_parquet(input_path)

    if label_column == "OASIS_ID":
        meta = pl.read_parquet(label_path).rename({"OASIS_ID": "Metadata_OASIS_ID"})
        log_status(f"Data shape before join: {dat.shape}")
        log_status(f"Metadata shape before join: {meta.shape}")
        dat = dat.join(meta, on="Metadata_OASIS_ID", how="left")
    elif label_column == "ID":
        meta = pl.read_parquet(label_path).rename({"ID": "Metadata_compound"})
        log_status(f"Data shape before join: {dat.shape}")
        log_status(f"Metadata shape before join: {meta.shape}")
        dat = dat.join(meta, on="Metadata_compound", how="left")
    else:
        raise ValueError(f"Unsupported label_column '{label_column}'. Use 'OASIS_ID' or 'ID'.")

    all_labels = [i for i in meta.columns if "Metadata_" not in i]
    labels = all_labels
    if include_labels:
        requested_labels = set(include_labels)
        missing_labels = sorted(requested_labels.difference(all_labels))
        if missing_labels:
            raise ValueError(f"Labels not found in {label_path}: {missing_labels}")
        labels = [label for label in all_labels if label in requested_labels]

    log_status(f"Data shape after join: {dat.shape}; processing {len(labels)}/{len(all_labels)} label(s)")

    model_output_dir = None
    if save_models:
        model_output_dir = os.path.join(os.path.dirname(output_path), "models")
        os.makedirs(model_output_dir, exist_ok=True)

    def _run_label(current_label_column: str, *, shuffle: bool = False, cc: bool = False, save_model: bool = False):
        return process_label(
            dat,
            current_label_column,
            n_splits,
            all_labels,
            device=device,
            gpu_id=0,
            shuffle=shuffle,
            cc=cc,
            n_repeats=n_repeats,
            output_dir=model_output_dir,
            save_model=save_model,
            n_jobs=model_n_jobs,
        )

    def _concat_results(results, model_type: str) -> pl.DataFrame | None:
        results = [res for res in results if res is not None]
        if not results:
            return None
        return pl.concat(results, how="vertical").with_columns(pl.lit(model_type).alias("Model_type"))

    log_status(f"Starting Actual models for {len(labels)} label(s)")
    pred_results = [_run_label(label, shuffle=False, save_model=save_models) for label in labels]
    pred_df = _concat_results(pred_results, "Actual")

    log_status(f"Starting Random_baseline models for {len(labels)} label(s)")
    null_results = [_run_label(label, shuffle=True) for label in labels]
    null_df = _concat_results(null_results, "Random_baseline")

    log_status(f"Starting Cellcount_baseline models for {len(labels)} label(s)")
    cc_results = [_run_label(label, cc=True) for label in labels]
    cc_df = _concat_results(cc_results, "Cellcount_baseline")

    prediction_frames = [frame for frame in [pred_df, null_df, cc_df] if frame is not None and not frame.is_empty()]
    if prediction_frames:
        predictions = pl.concat(prediction_frames, how="vertical")
        log_status(f"Writing {predictions.height} prediction rows to {output_path}")
        predictions.write_parquet(output_path)
    else:
        log_status(f"No predictions generated; writing empty prediction table to {output_path}")
        empty_prediction_frame().write_parquet(output_path)


def preprocess_axiom_data_features(
    input_path: str,
) -> pl.DataFrame:
    """Read aggregated Axiom features and remove controls."""
    profiles = pl.read_parquet(input_path)
    return profiles.filter(~pl.col("Metadata_compound").is_in(["DMSO"]))


def preprocess_axiom_data_cell_count(
    input_path: str,
) -> pl.DataFrame:
    """Read cell-count profiles and aggregate one median count per compound/OASIS ID."""
    profiles = (
        pl.read_parquet(input_path)
        .rename({"Metadata_Count_Cells": "Cell_Count"})
        .filter(~pl.col("Metadata_compound").is_in(["DMSO"]))
    )

    return profiles.group_by(["Metadata_compound", "Metadata_OASIS_ID"]).agg(pl.col("Cell_Count").median())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute prediction.")
    parser.add_argument("--input_path_level6", required=True, help="Path to input aggregated profiles")
    parser.add_argument("--input_path_level4", required=True, help="Path to input level-4 profiles with cell counts")
    parser.add_argument("--label_path", required=True, help="Path to input matrix of binary to predict")
    parser.add_argument("--label_column", required=False, default="OASIS_ID", help="Column in label_path to use as label (default: OASIS_ID). If using 'ID', will use Metadata_compound for join.")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--n_repeats", required=False, default=1, type=int, help="Number of times to repeat cross-validation (default: 1)")
    parser.add_argument("--device", required=False, default="cpu", choices=["cpu", "gpu", "cuda"], help="Run XGBoost on CPU or GPU/CUDA (default: cpu)")
    parser.add_argument("--n_jobs", required=False, default=1, type=int, help="Number of CPU threads available to XGBoost")
    parser.add_argument("--include_label", action="append", default=None, help="Label column to process; may be provided more than once")
    parser.add_argument("--include_labels", required=False, default=None, help="Comma-separated label columns to process")
    parser.add_argument("--save_models", dest="save_models", action="store_true", help="Save final full-data models")
    parser.add_argument("--no_save_models", dest="save_models", action="store_false", help="Skip saving final full-data models")
    parser.set_defaults(save_models=True)
    args = parser.parse_args()

    include_labels = []
    if args.include_label:
        include_labels.extend(args.include_label)
    if args.include_labels:
        include_labels.extend(label for label in args.include_labels.split(",") if label)
    include_labels = list(dict.fromkeys(include_labels)) or None

    dat_aggregated_features = preprocess_axiom_data_features(
        input_path=args.input_path_level6,
    )
    dat_cell_count = preprocess_axiom_data_cell_count(
        input_path=args.input_path_level4,
    )

    dat_features_cell_count = dat_aggregated_features.join(dat_cell_count, on="Metadata_compound", how="left")

    predict_binary(
        dat=dat_features_cell_count,
        label_path=args.label_path,
        output_path=args.output,
        label_column=args.label_column,
        n_repeats=args.n_repeats,
        device=args.device,
        save_models=args.save_models,
        n_jobs=args.n_jobs,
        include_labels=include_labels,
    )
