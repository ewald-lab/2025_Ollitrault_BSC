#!/usr/bin/env python3
"""
7_axiom_aggregation_compounds.py
----------------------------
Aggregates Level-5 consensus profiles at perturbation level perturbation into Level-6 compound aggregated profiles for axiom dataset.

Usage:
    python 7_axiom_aggregation_compounds.py \
        --input 02_processed_data/level5_consensus.parquet \
        --input 02_processed_data/level5_consensus.parquet \
        --output 02_processed_data/level6_aggregated_compound.parquet \
        --operation median \
        --strata Metadata_compound

Description:
    This script takes Level-5 consensus profiles and performs aggregation at the compound level.
    The resulting Level-6 table
    summarizes features per compound accross the differeent perturbation using a chosen operation such as
    median, mean.

Arguments:
    --input       Path to input Parquet file (Level-5 consensus data)
    --output      Path to output Parquet file (aggregated Level-6 data)
    --operation   Aggregation operation: median | mean (default: median)
    --strata      Column(s) to group by, e.g. Metadata_compound (default: Metadata_compound)
"""

import argparse
from pathlib import Path

import pandas as pd
from pycytominer import aggregate


def main():
    parser = argparse.ArgumentParser(description="Aggregate Level-5 consensus DINOv2 features into Level-6 compound aggregated.")
    parser.add_argument("--input", required=True, help="Path to input Parquet file (Level-5 consensus data)")
    parser.add_argument("--metadata", required=True, help="Path to input Parquet file (axiom metadata)")
    parser.add_argument("--output", required=True, help="Path to output Parquet file (Level-6 aggregated data)")
    parser.add_argument("--operation", default="median", choices=["median", "mean"], help="Aggregation operation")
    parser.add_argument(
        "--strata", nargs="+", default=["Metadata_OASIS_ID"], help="Columns to group by (default: Metadata_OASIS_ID)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    metadata_path = Path(args.metadata)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading Level-5 data from: {input_path}")
    df_level5_consensus = pd.read_parquet(input_path)

    print(f"📂 Loading Axiom metadata from: {metadata_path}")
    df_metadata = pd.read_parquet(metadata_path)

    print(df_level5_consensus.shape)
    print(df_metadata.columns)
    df_level5_consensus = pd.merge(
        df_level5_consensus,
        df_metadata[["Metadata_Compound","Metadata_OASIS_ID"]].drop_duplicates(subset=["Metadata_Compound"]),
        left_on="Metadata_compound",
        right_on="Metadata_Compound",
        how="left",
        validate="m:1"
    )
    print(df_level5_consensus.shape)

    # Select only numeric columns that belong to those feature families
    feature_cols = [c for c in df_level5_consensus.columns if c.startswith("Feature_")]

    meta_cols = [c for c in df_level5_consensus.columns if c.startswith("Metadata_compound")] + ["Metadata_OASIS_ID"]

    print(f"🔢 Found {len(feature_cols)} numeric feature columns.")
    print(f"🧩 Aggregating by {args.strata} using '{args.operation}'.")

    df_level6_aggregate = aggregate(
        df_level5_consensus,
        strata=args.strata,     
        features=feature_cols,
        operation=args.operation,
    )

    print(df_level5_consensus[meta_cols].drop_duplicates(subset=args.strata))
    df_level6_aggregate_annotate = pd.merge(
        df_level6_aggregate,
        df_level5_consensus[meta_cols].drop_duplicates(subset=args.strata),
        on=args.strata[0],  # join key (common to both DataFrames)
        how="left",  # or 'inner' if you only want matching wells
    )

    df_level6_aggregate_annotate["Metadata_AggType"] = "all"

    print(f"💾 Saving Level-6 aggregated and annotated data to: {output_file}")
    df_level6_aggregate_annotate.to_parquet(output_file, index=False)

    print("✅ Done.")

if __name__ == "__main__":
    main()
