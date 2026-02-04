#!/usr/bin/env python3
"""
2E_data_processing_level5_consensus.py
----------------------------------
Computes consensus profiles for each chemical (compound_scode) from feature-selected data using pycytominer,
and merges chemical-level metadata annotations.

Usage:
    python 2E_data_processing_level5_consensus.py \
        --input 02_processed_data/level4b.parquet \
        --output 02_processed_data/level5.parquet
"""

import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*DataFrameGroupBy.apply operated on the grouping columns.*"
)
import argparse
import warnings
from pathlib import Path

import pandas as pd
from pycytominer import consensus


def main():
    parser = argparse.ArgumentParser(description="Compute consensus chemical-level profiles using pycytominer.")
    parser.add_argument("--input", required=True, help="Path to feature-selected input Parquet file (Level 4b data)")
    parser.add_argument("--output", required=True, help="Directory for output Parquet file")
    parser.add_argument("--replicate_column", nargs="+", default="Metadata_perturbation", help="Columns identifying replicates")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {input_path}")
    df_level4 = pd.read_parquet(input_path)

    # Separate metadata and feature columns
    meta_cols = [c for c in df_level4.columns if c.startswith("Metadata_")]
    features_cols = [c for c in df_level4.columns if c.startswith("Feature_")]
    print(f"🧬 {len(features_cols)} feature columns and {len(meta_cols)} metadata columns detected.")

    # -------------------------------
    # 🧠 Compute consensus profiles
    # -------------------------------
    print("⚙️ Computing consensus profiles using MODZ...")

    df_level5 = consensus(
        profiles=df_level4,
        replicate_columns=args.replicate_column,  # replicate identifier
        features=features_cols,
        operation="modz",
    )
    print(f"✅ Consensus profiles computed with shape: {df_level5.shape}")

    # -------------------------------
    # 🔗 Merge chemical annotations
    # -------------------------------
    meta_cols_chemicals = [args.replicate_column, "Metadata_concentration"] + [c for c in df_level4.columns if c.startswith("Metadata_") and "compound" in c.lower()]
    print(f"🧩 Merging chemical metadata columns: {meta_cols_chemicals}")
    print("🔗 Joining consensus profiles with compound metadata...")
    df_level5_annotate = pd.merge(
        df_level5,
        df_level4[meta_cols_chemicals].drop_duplicates(subset=[args.replicate_column]),
        on=args.replicate_column,
        how="left",
    )
    # -------------------------------
    # 💾 Save output
    # -------------------------------
    print(f"💾 Saving annotated consensus profiles to {output_file}")
    df_level5_annotate.to_parquet(output_file, index=False)

    print("🎉 Level 5 consensus chemical profile generation complete.")


if __name__ == "__main__":
    main()
