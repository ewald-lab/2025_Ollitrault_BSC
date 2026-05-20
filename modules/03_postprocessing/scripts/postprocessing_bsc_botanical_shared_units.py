#!/usr/bin/env python3
"""
postprocessing_bsc_botanical_shared_units.py
----------------------------------
Post-processing to filter BSC relevant data for all data levels.
Usage:
    python modules/03_postprocessing/scripts/postprocessing_bsc_botanical_shared_units.py \
        --input modules/03_postprocessing/outputs/level4b_bsc_filtered.parquet \
        --output modules/03_postprocessing/outputs/level4b_bsc_filtered_converted.parquet \
        --bsc_annotation "modules/03_postprocessing/inputs/Botanical_Constituents_and_Quantification_08192024/Quant Constituents-Table 1.tsv"
"""
import warnings
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*DataFrameGroupBy.apply operated on the grouping columns.*"
)
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Post-process BSC data to filter relevant information.")
    parser.add_argument("--input", required=True, help="Path to processed input Parquet file (Level X data)")
    parser.add_argument("--output", required=True, help="Path for output Parquet file")
    parser.add_argument("--bsc_annotation", required=True,
                        default="modules/03_postprocessing/inputs/Botanical_Constituents_and_Quantification_08192024/Quant Constituents-Table 1.tsv",
                        help="Path to the BSC annotation file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {input_path}")
    df_level_bsc = pd.read_parquet(input_path)

    print(f"📂 Loading BSC annotation from {args.bsc_annotation}")
    df_bsc_annotations = pd.read_csv(args.bsc_annotation, sep="\t")

    # ── Convert botanical concentrations from µg/mL → µM ────────────────────
    # Formula: µM = (µg/mL × 1000) / MW (g/mol)
    # Use the average MW per botanical plant scode derived from the annotation file.

    list_botanical_scode = df_bsc_annotations["Plant scode"].unique()

    avg_mw = (
        df_bsc_annotations
        .groupby("Plant scode", as_index=False)["MW (g/mol)"]
        .mean()
        .rename(columns={"Plant scode": "Metadata_compound", "MW (g/mol)": "avg_mw"})
    )

    # Attach avg_mw to filtered level data
    df_level_bsc = df_level_bsc.merge(avg_mw, on="Metadata_compound", how="left")

    # Apply conversion only to rows whose compound is a botanical plant scode
    mask = df_level_bsc["Metadata_compound"].isin(list_botanical_scode)
    df_level_bsc.loc[mask, "Metadata_concentration"] = (
        df_level_bsc.loc[mask, "Metadata_concentration"] * 1000
        / df_level_bsc.loc[mask, "avg_mw"]
    )

    # Drop the helper column now that conversion is done
    df_level_bsc = df_level_bsc.drop(columns=["avg_mw"])

    print(f"💾 Saving BSC post-processed profiles to {output_file}")
    df_level_bsc.to_parquet(output_file, index=False)


if __name__ == "__main__":
    main()