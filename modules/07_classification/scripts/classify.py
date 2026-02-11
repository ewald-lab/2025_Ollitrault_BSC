import traceback  # noqa: D100

try:
    import cupy as cp
except ImportError:
    import numpy as cp  # fallback for macOS

import argparse

import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm.contrib.concurrent import thread_map
from xgboost import XGBClassifier


def binary_classifier(
    dat: pd.DataFrame,
    meta: pd.DataFrame,
    n_splits: int,
    gpu_id: int,
    *,
    shuffle: bool = False,
    cc: bool = False,
) -> pl.DataFrame:
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

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the predicted labels and probabilities
        for each sample in the validation sets across all folds.

    """
    dat["Label"] = dat["Label"].astype(int)
    x = dat.drop(columns=["Label"])
    y = dat["Label"]

    if cc:
        x = x[["Cell_Count"]]
    else:
        x = x.drop(columns=["Cell_Count"])

    if shuffle:
        y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=n_splits)

    pred_df = []
    fold = 1
    with cp.cuda.Device(gpu_id):
        for train_index, val_index in kf.split(x, y):
            x_fold_train, x_fold_val = cp.array(x.iloc[train_index].to_numpy()), cp.array(x.iloc[val_index].to_numpy())
            y_fold_train, y_fold_val = y.iloc[train_index], y.iloc[val_index]

            le = LabelEncoder()
            y_fold_train = cp.array(le.fit_transform(y_fold_train))
            y_fold_val = cp.array(le.fit_transform(y_fold_val))

            meta_fold_val = meta.iloc[val_index]

            # Initialize the model
            model = XGBClassifier(
                objective="binary:logistic",
                n_estimators=150,
                tree_method="hist",
                device=f"cuda:{gpu_id}",
                learning_rate=0.05,
                scale_pos_weight=(y_fold_train == 0).sum() / (y_fold_train == 1).sum(),
            )

            # Train the model on the fold training set
            model.fit(x_fold_train, y_fold_train)

            # Validate the model on the fold validation set
            y_fold_prob = model.predict_proba(x_fold_val)[:, 1]
            y_fold_pred = model.predict(x_fold_val)

            pred_df.append(
                pl.DataFrame({
                    "Metadata_compound": list(meta_fold_val["Metadata_compound"]),
                    "y_prob": list(y_fold_prob),
                    "y_pred": list(y_fold_pred),
                    "y_actual": list(y_fold_val),
                    "k_fold": fold,
                }),
            )
            fold += 1

    return pl.concat(pred_df, how="vertical")

def process_label(dat, label_column, n_splits, labels, gpu_id, *, shuffle: bool = False, cc: bool = False):
    """Process a single label_column."""
    try:
        prof = dat.rename({label_column: "Label"})

        num_0 = prof.filter(pl.col("Label") == 0).height
        num_1 = prof.filter(pl.col("Label") == 1).height

        if (num_0 >= n_splits) & (num_1 >= n_splits):
            meta_cols = [i for i in prof.columns if "Metadata_" in i]
            all_meta_cols = [i for i in prof.columns if i in labels] + meta_cols

            prof_meta = prof.select(meta_cols)
            prof = prof.drop(all_meta_cols)

            # Call your classifier
            class_res = binary_classifier(
                prof.to_pandas(),
                prof_meta.to_pandas(),
                n_splits=n_splits,
                gpu_id=gpu_id,
                shuffle=shuffle,
                cc=cc,
            )

            # Add the metadata columns
            class_res = class_res.with_columns(
                pl.lit(label_column).alias("Metadata_Label"),
                pl.lit(num_0).alias("Metadata_Count_0"),
                pl.lit(num_1).alias("Metadata_Count_1"),
            )

            return class_res

    except Exception as e:
        print(f"An error occurred for label '{label_column}':")
        print(traceback.format_exc())
        return None


def predict_binary(
    input_path: str,
    label_path: str,
    output_path: str,
) -> None:
    """Build classifier for binary outcomes.

    Parameters
    ----------
    input_path : str
        Filepath for input profiles.
    label_path : str
        Filepath for input binary labels.
    output_path : str
        Filepath for model classification results.

    """
    n_splits = 5
    num_gpus = 4

    dat = pl.read_parquet(input_path)
    meta = pl.read_parquet(label_path).rename({"OASIS_ID": "Metadata_compound"})
    labels = [i for i in meta.columns if "Metadata_" not in i]

    dat = dat.join(meta, on="Metadata_compound", how="left")

    tasks = [
        (dat, label_column, n_splits, labels, i % num_gpus)
        for i, (label_column) in enumerate(
            [(label_column) for label_column in labels]
        )
    ]

    # Train actual models
    pred_results = thread_map(
        lambda args: process_label(*args, shuffle=False),
        tasks,
        max_workers=num_gpus,
        desc="Processing labels",
    )

    pred_results = [res for res in pred_results if res is not None]
    if pred_results:
        pred_df = pl.concat(pred_results, how="vertical")
        pred_df = pred_df.with_columns(
            pl.lit("Actual").alias("Model_type")
        )

    # Random baseline
    null_results = thread_map(
        lambda args: process_label(*args, shuffle=True),
        tasks,
        max_workers=num_gpus,
        desc="Processing labels for random baseline",
    )

    null_results = [res for res in null_results if res is not None]
    if null_results:
        null_df = pl.concat(null_results, how="vertical")
        null_df = null_df.with_columns(
            pl.lit("Random_baseline").alias("Model_type")
        )

    # Cell count baseline
    cc_results = thread_map(
        lambda args: process_label(*args, cc=True),
        tasks,
        max_workers=num_gpus,
        desc="Processing labels for cell count baseline",
    )

    cc_results = [res for res in cc_results if res is not None]
    if cc_results:
        cc_df = pl.concat(cc_results, how="vertical")
        cc_df = cc_df.with_columns(
            pl.lit("Cellcount_baseline").alias("Model_type")
        )

    # write out results
    if not pred_df.is_empty() and not null_df.is_empty() and not cc_df.is_empty():
        pl.concat([pred_df, null_df, cc_df], how="vertical").write_parquet(output_path)

# python 01_analysis/7_script_classifier/7_data_axiom_hitcalls.py --input_pod_mtt 03_tables/pods_axiom_Metadata_mtt_normalized_original_conc.parquet --input_pod_ldh 03_tables/pods_axiom_Metadata_ldh_normalized_original_conc.parquet --output 03_tables/test.parquet
# I have replaced all Metadata_OASIS_ID to Metadata_compound
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute prediction.")
    parser.add_argument("--input_path", required=True, help="Path to input aggregated profiles")
    parser.add_argument("--label_path", required=True, help="Path to input matrix of binary to predict") 
    parser.add_argument("--output", required=True, help="Directory for output Parquet file")
    args = parser.parse_args()
    predict_binary(
                    input_path=args.input_path,
                    label_path=args.label_path,
                    output_path=args.output,
                )