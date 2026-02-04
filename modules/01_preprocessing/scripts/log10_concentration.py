#!/usr/bin/env python3
"""
1C_data_log10concentration.py
------------------
Log10 transform concentration column.

Usage:
    python 01_analysis/1C_data_log10concentration.py --input 02_processed_data/level2_bsc.parquet --output 02_processed_data/level2_test.parquet --control s0617825
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import polars as pl
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Expand DINOv2 feature arrays into individual columns.")
    parser.add_argument("--input", required=True, help="Path to input Parquet file")
    parser.add_argument("--output", required=True, help="Directory for output Parquet file")
    parser.add_argument("--control", required=True, help="Name of control compound")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {input_path}")
    df = pl.read_parquet(input_path)

    print("🔍 Applying log10 transformation to concentration column 'Metadata_concentration'")
    # Compute shift value per compound (excluding control)
    shift_df = (
        df
        .filter(pl.col("Metadata_compound") != args.control)
        .group_by("Metadata_compound")
        .agg(
            pl.col("Metadata_concentration")
            .min()
            .alias("min_conc")
        )
        .with_columns(
            pl.when(pl.col("min_conc") > 0)
            .then((pl.col("min_conc") / 3).log10().abs())
            .otherwise(None)
            .alias("shift_val")
        )
        .select("Metadata_compound", "shift_val")
    )

    # Join shift values back to main dataframe
    df_log10 = (
        df
        .join(shift_df, on="Metadata_compound", how="left")
        .with_columns(
            pl.when(pl.col("Metadata_compound") == args.control)
            .then(pl.lit(0.0))
            .otherwise(pl.col("Metadata_concentration").log10() + pl.col("shift_val"))
            .alias("Metadata_log10conc")
        )
        .drop("shift_val")
    )

    if False:
        compounds = df.select("Metadata_compound").to_series().unique().to_list()
        compounds = [i for i in compounds if i != args.control]
        df_log10 = df.filter(pl.col("Metadata_compound") == args.control).with_columns(
            pl.lit(0).cast(pl.Float64).alias("Metadata_log10conc"),
        )

        for compound in tqdm(compounds):
            temp = df.filter(pl.col("Metadata_compound") == compound)
            concs = temp.select(pl.col("Metadata_concentration")).to_series().sort().unique().to_list()
            shift_val = np.abs(np.log10(concs[0] / 3))

            temp = temp.with_columns(
                (pl.col("Metadata_concentration").log10() + shift_val).alias("Metadata_log10conc"),
            )

            df_log10 = pl.concat([df_log10, temp], how="vertical")              

    print(f"💾 Writing log10 transformed data to {output_file}")
    df_log10.to_pandas().to_parquet(output_file)
    print("✅ Done.")

if __name__ == "__main__":
    main()