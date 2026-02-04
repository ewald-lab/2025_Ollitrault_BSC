#!/usr/bin/env python3
"""
2B_data_processing_level3.py
----------------------------
Aggregates Level-2 expanded features into Level-3 well-level summaries.

Usage:
    python 2B_data_processing_level3.py \
        --input 02_processed_data/level2_expanded.parquet \
        --output 02_processed_data/level3_aggregated_annotate.parquet \
        --operation median \
        --strata Metadata_well_id

Description:
    This script takes Level-2 expanded features (where each image has
    independent numeric feature columns) and performs aggregation across the
    specified strata (e.g., wells). The resulting Level-3 table
    summarizes features per well using a chosen operation such as
    median, mean, or sum.

Arguments:
    --input       Path to input Parquet file (Level-2 expanded data)
    --output      Path to output Parquet file (aggregated Level-3 data)
    --operation   Aggregation operation: median | mean (default: median)
    --strata      Column(s) to group by, e.g. Metadata_well_id or plate_id (default: Metadata_well_id)
"""

import argparse
from pathlib import Path

import pandas as pd
from pycytominer import aggregate


def main():
    parser = argparse.ArgumentParser(description="Aggregate Level-2 expanded DINOv2 features into Level-3 summaries.")
    parser.add_argument("--input", required=True, help="Path to input Parquet file (Level-2 expanded data)")
    parser.add_argument("--output", required=True, help="Path to output Parquet file (Level-3 aggregated data)")
    parser.add_argument("--operation", default="median", choices=["median", "mean"], help="Aggregation operation")
    parser.add_argument(
        "--strata", nargs="+", default=["Metadata_well_id"], help="Columns to group by (default: Metadata_well_id)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading Level-2 data from: {input_path}")
    df_level2_expanded = pd.read_parquet(input_path)

    # Select only numeric columns that belong to those feature families
    feature_cols = [c for c in df_level2_expanded.columns if c.startswith("Feature_")]

    meta_cols = [c for c in df_level2_expanded.columns if c.startswith("Metadata_")]

    print(f"🔢 Found {len(feature_cols)} numeric feature columns.")
    print(f"🧩 Aggregating by {args.strata} using '{args.operation}'.")

    df_level3_aggregate = aggregate(
        df_level2_expanded,
        strata=args.strata,     
        features=feature_cols,
        operation=args.operation,
    )

    print(df_level2_expanded[meta_cols].drop_duplicates(subset=args.strata))
    df_level3_aggregate_annotate = pd.merge(
        df_level3_aggregate,
        df_level2_expanded[meta_cols].drop_duplicates(subset=args.strata),
        on=args.strata[0],  # join key (common to both DataFrames)
        how="left",  # or 'inner' if you only want matching wells
    )

    print(f"💾 Saving Level-3 aggregated and annotated data to: {output_file}")
    df_level3_aggregate_annotate.to_parquet(output_file, index=False)

    print("✅ Done.")


if __name__ == "__main__":
    main()
