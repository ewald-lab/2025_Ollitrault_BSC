#!/usr/bin/env python3
"""
2D_data_processing_level4b_feature_selection.py
----------------------------------
Performs feature selection on normalized Level 4a data using various selection strategies.

Usage:
    python 2D_data_processing_level4b_feature_selection.py \
        --input 02_processed_data/level4a_normalized.parquet \
        --output 02_processed_data/level4b_feature_selected.parquet \
        --feature_select_categories DNA ER AGP RNA Mito Brightfield
"""

import argparse
from pathlib import Path
from pyexpat import features

import pandas as pd
from pycytominer import feature_select


def main():
    parser = argparse.ArgumentParser(description="Perform feature selection on normalized profiles.")
    parser.add_argument("--input", required=True, help="Path to input Parquet file (Level 4a normalized data)")
    parser.add_argument("--output", required=True, help="Directory for output Parquet file")
    parser.add_argument("--feature_select_categories", nargs="+", default='all', help="Feature selection categories to apply : DNA, ER, AGP, RNA, Mito, Brightfield")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {input_path}")
    df_level4a = pd.read_parquet(input_path)

    # 🧠 Identify metadata and feature columns
    meta_cols = [c for c in df_level4a.columns if c.startswith("Metadata_")]
    features_cols = [c for c in df_level4a.columns if c.startswith("Feature_")]

    # Filter features by selected categories
    all_features_filtered = []
    if args.feature_select_categories != 'all':
        for category in args.feature_select_categories:
            category_features = [c for c in features_cols if f"_{category}_" in c]
            all_features_filtered.extend(category_features)
        features_cols = all_features_filtered

    print(f"🔍 Found {len(features_cols)} feature columns and {len(meta_cols)} metadata columns.")

    # ⚙️ Feature selection options
    feature_select_opts = [
        "variance_threshold",
        "drop_na_columns",
        "correlation_threshold",
        "blocklist",
        "drop_outliers",
    ]

    # 🚀 Apply feature selection
    print("✨ Running feature selection...")
    df_level4b = feature_select(
        profiles=df_level4a, features=features_cols, samples="all", operation=feature_select_opts
    )

    print(f"💾 Saving feature-selected data to {output_path}")
    df_level4b.to_parquet(output_path, index=False)

    print("✅ Feature selection complete.")


if __name__ == "__main__":
    main()
