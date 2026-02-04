#!/usr/bin/env python3
"""
2C_data_processing_level4a_normalize.py
--------------------------------------
Normalizes Level-3 aggregated DINOv2 features per plate using mad_robustize scaling.

Usage:
    python 2C_data_processing_level4a_normalize.py \
        --input 02_processed_data/level3.parquet \ 
        --output 02_processed_data/level4a.parquet

Description:
    This script reads Level-3 aggregated data, normalizes feature columns
    separately for each plate using the "mad_robustize" method other DMSO control, and writes the
    combined Level-4 normalized dataset.

Arguments:
    --input       Path to input Parquet file (Level-3 aggregated data)
    --output      Path to output Parquet file (Level-4 normalized data)
"""

import argparse
from pathlib import Path

import pandas as pd
from pycytominer import normalize


def main():
    parser = argparse.ArgumentParser(description="Normalize Level-3 aggregated features per plate.")
    parser.add_argument("--input", required=True, help="Input Parquet file path")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    parser.add_argument("--control", required=False, default="DMSO", help="Control compound name for normalization")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading Level-3 data from {input_path}")
    df = pd.read_parquet(input_path)

    # Select only numeric columns that belong to those feature families
    feature_cols = [c for c in df.columns if c.startswith("Feature_")]

    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]

    print(f"🔍 Found {len(feature_cols)} numeric feature columns.")
    print(f"🔍 Found {len(meta_cols)} metadata columns.")

    # Normalize each plate separately
    plates = df["Metadata_plate"].unique()
    print(f"🧪 Normalizing {len(plates)} plates...")

    normalized_list = []
    for plate in plates:
        plate_df = df[df["Metadata_plate"] == plate]
        norm_df = normalize(
            profiles=plate_df,
            features=feature_cols,
            meta_features=meta_cols,
            method="mad_robustize",
            samples=f"Metadata_compound == '{args.control}'",  # Control compound
        )
        normalized_list.append(norm_df)
        print("Plate", plate, "normalized")

    df_level4a = pd.concat(normalized_list).reset_index(drop=True)
    print(f"✅ Normalized data shape: {df_level4a.shape}")

    print(f"💾 Writing normalized data to {output_path}")
    df_level4a.to_parquet(output_path, index=False)

    print("✅ Done.")


if __name__ == "__main__":
    main()
