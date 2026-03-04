#!/usr/bin/env python3
"""
3_post_data_postprocessing_bsc.py
----------------------------------
Post-processing to filter BSC relevvant data for all data levels.

Usage:
    python 3_post_data_postprocessing_bsc.py \
        --input 02_processed_data/level5.parquet \
        --output 02_processed_data/level5.parquet \
        --bsc_annotation 00_inputs/Botanical_Constituents_and_Quantification_08192024/Quant Constituents-Table 1.tsv
"""

import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*DataFrameGroupBy.apply operated on the grouping columns.*"
)
import argparse
import warnings
from pathlib import Path

import pandas as pd
from common.utils.util_standardizer import StandardizeMolecule

def main():
    parser = argparse.ArgumentParser(description="Post-process BSC data to filter relevant information.")
    parser.add_argument("--input", required=True, help="Path to processed input Parquet file (Level X data)")
    parser.add_argument("--output", required=True, help="Path for output Parquet file")
    parser.add_argument("--bsc_annotation", required=True, default="00_inputs/Botanical_Constituents_and_Quantification_08192024/Quant Constituents-Table 1.tsv", help="Path to the BSC annotation file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {input_path}")
    df_level = pd.read_parquet(input_path)

    print(f"📂 Loading BSC annotation from {args.bsc_annotation}")
    df_bsc_annotations = pd.read_csv(args.bsc_annotation, sep="\t")

    # Filter to only BSC relevant compounds
    meta_cols_compounds = [c for c in df_level.columns if c.startswith("Metadata_compound")]
    # Get only unique chemicals annotations 
    df_level_compounds = df_level[meta_cols_compounds].drop_duplicates()

    # standardize
    df_level_compounds_standardized = StandardizeMolecule(df_level_compounds, input_column="Metadata_compound_smiles").run()
    df_bsc_annotations_standardized = StandardizeMolecule(df_bsc_annotations, input_column="SMILES").run()

    # merge considering stereoisomers
    df_merge_cp_cp_annotations_iso = df_level_compounds_standardized.merge(
        df_bsc_annotations_standardized,
        on="SMILES_standardized",
        how="inner"
    )
    # filter level data to only BSC relevant compounds
    # adding botanicals and DMSO control
    list_botanical_compounds = (
        df_merge_cp_cp_annotations_iso["Metadata_compound"].unique().tolist() +
        df_merge_cp_cp_annotations_iso["Plant scode"].unique().tolist() +
        ["s0617825"]
    )
    df_level_bsc = df_level[df_level["Metadata_compound"].isin(list_botanical_compounds)]

    # filter out plates that contain only DMSO (no compound wells)
    plates_with_compounds = (
        df_level_bsc[df_level_bsc["Metadata_compound"] != "s0617825"]
        ["Metadata_plate"]
        .unique()
    )
    df_level_bsc = df_level_bsc[df_level_bsc["Metadata_plate"].isin(plates_with_compounds)]

    print(f"💾 Saving BSC post processed profiles to {output_file}")
    df_level_bsc.to_parquet(output_file, index=False)


if __name__ == "__main__":
    main()
