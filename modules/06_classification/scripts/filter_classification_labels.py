from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

CONTROL_COMPOUNDS = ["DMSO"]
JOIN_COLUMNS = {
    "OASIS_ID": "Metadata_OASIS_ID",
    "ID": "Metadata_compound",
}


def label_columns_from_path(label_path: str, label_column: str) -> list[str]:
    """Return assay label columns from a label parquet file."""
    if label_column not in JOIN_COLUMNS:
        raise ValueError(f"Unsupported label_column '{label_column}'. Use 'OASIS_ID' or 'ID'.")

    schema = pl.scan_parquet(label_path).collect_schema()
    return [
        column
        for column in schema.names()
        if column != label_column and not column.startswith("Metadata_")
    ]


def preprocess_axiom_data_features(input_path: str) -> pl.DataFrame:
    """Read aggregated Axiom features and remove controls."""
    return pl.read_parquet(input_path).filter(~pl.col("Metadata_compound").is_in(CONTROL_COMPOUNDS))


def preprocess_axiom_data_cell_count(input_path: str) -> pl.DataFrame:
    """Read cell-count profiles and aggregate one median count per compound/OASIS ID."""
    profiles = (
        pl.read_parquet(input_path)
        .rename({"Metadata_Count_Cells": "Cell_Count"})
        .filter(~pl.col("Metadata_compound").is_in(CONTROL_COMPOUNDS))
    )
    return profiles.group_by(["Metadata_compound", "Metadata_OASIS_ID"]).agg(pl.col("Cell_Count").median())


def classification_input_frame(input_path_level6: str, input_path_level4: str) -> pl.DataFrame:
    """Build the same joined feature/cell-count frame used by classify.py."""
    data_features = preprocess_axiom_data_features(input_path_level6)
    data_cell_count = preprocess_axiom_data_cell_count(input_path_level4)
    return data_features.join(data_cell_count, on="Metadata_compound", how="left")


def join_labels(data: pl.DataFrame, label_path: str, label_column: str) -> pl.DataFrame:
    """Join labels to the classification input frame."""
    join_column = JOIN_COLUMNS[label_column]
    labels = pl.read_parquet(label_path).rename({label_column: join_column})
    return data.join(labels, on=join_column, how="left")


def label_counts(data: pl.DataFrame, label: str) -> dict[str, int]:
    """Count labeled, inactive, and active examples for a single assay label."""
    row = data.select(
        pl.col(label).is_not_null().sum().alias("n_labeled"),
        (pl.col(label) == 0).sum().alias("n_inactive"),
        (pl.col(label) == 1).sum().alias("n_active"),
    ).row(0, named=True)
    return {key: int(value) for key, value in row.items()}


def skip_reason(counts: dict[str, int], n_splits: int, min_labeled: int, min_active: int) -> str:
    """Explain why an assay label failed filtering, or return an empty reason."""
    reasons = []
    if counts["n_inactive"] < n_splits:
        reasons.append(f"inactive<{n_splits}")
    if counts["n_active"] < n_splits:
        reasons.append(f"active<{n_splits}")
    if counts["n_labeled"] < min_labeled:
        reasons.append(f"labeled_total<{min_labeled}")
    if counts["n_active"] < min_active:
        reasons.append(f"active<{min_active}")
    return ";".join(reasons)


def filter_labels(
    input_path_level6: str,
    input_path_level4: str,
    label_path: str,
    label_column: str,
    output_path: str,
    audit_output_path: str,
    label_group_name: str,
    n_splits: int,
    min_labeled: int,
    min_active: int,
) -> None:
    """Filter assay columns and write a filtered label matrix plus an audit table."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if min_labeled < 0:
        raise ValueError("min_labeled must be >= 0")
    if min_active < 0:
        raise ValueError("min_active must be >= 0")

    labels = label_columns_from_path(label_path, label_column)
    data = classification_input_frame(input_path_level6, input_path_level4)
    joined = join_labels(data, label_path, label_column)

    audit_rows = []
    kept_labels = []
    for label in labels:
        counts = label_counts(joined, label)
        reason = skip_reason(counts, n_splits, min_labeled, min_active)
        keep_label = reason == ""
        if keep_label:
            kept_labels.append(label)

        audit_rows.append(
            {
                "label_group": label_group_name,
                "label": label,
                "n_labeled": counts["n_labeled"],
                "n_inactive": counts["n_inactive"],
                "n_active": counts["n_active"],
                "keep_label": keep_label,
                "skip_reason": reason,
            },
        )

    original_labels = pl.read_parquet(label_path)
    original_labels.select([label_column, *kept_labels]).write_parquet(output_path)
    pl.DataFrame(audit_rows).write_csv(audit_output_path, separator="\t")

    print(
        f"{label_group_name}: kept {len(kept_labels)}/{len(labels)} labels "
        f"(n_splits={n_splits}, min_labeled={min_labeled}, min_active={min_active})",
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Filter binary classification labels before model training.")
    parser.add_argument("--input_path_level6", required=True, help="Path to input aggregated profiles")
    parser.add_argument("--input_path_level4", required=True, help="Path to input level-4 profiles with cell counts")
    parser.add_argument("--label_path", required=True, help="Path to input binary label matrix")
    parser.add_argument("--label_column", default="OASIS_ID", choices=JOIN_COLUMNS.keys())
    parser.add_argument("--output", required=True, help="Output filtered label Parquet path")
    parser.add_argument("--audit_output", required=True, help="Output TSV audit path")
    parser.add_argument("--label_group_name", default=None, help="Name to use in the audit table")
    parser.add_argument("--n_splits", default=5, type=int, help="CV folds; requires this many inactive and active labels")
    parser.add_argument("--filter_label_size", default=50, type=int, help="Minimum labeled examples required")
    parser.add_argument("--filter_label_size_active", default=20, type=int, help="Minimum active examples required")
    args = parser.parse_args()

    label_group_name = args.label_group_name or Path(args.label_path).stem
    filter_labels(
        input_path_level6=args.input_path_level6,
        input_path_level4=args.input_path_level4,
        label_path=args.label_path,
        label_column=args.label_column,
        output_path=args.output,
        audit_output_path=args.audit_output,
        label_group_name=label_group_name,
        n_splits=args.n_splits,
        min_labeled=args.filter_label_size,
        min_active=args.filter_label_size_active,
    )


if __name__ == "__main__":
    main()
