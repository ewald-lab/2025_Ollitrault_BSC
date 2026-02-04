#!/usr/bin/env python3
"""
1B_data_format_axiom_oasis_dino.py
------------------
Concatenate and expands array-based DINOv2 feature columns into independent numeric columns.

Usage:
    python 01_analysis/1B_data_format_axiom_oasis_dino.py --input /1_snakemake/inputs/profiles/dino/plates \
                                              --metadata /1_snakemake/inputs/profiles/dino/metadata.parquet \
                                              --output 02_processed_data/

    python 01_analysis/1B_data_format_axiom_oasis_dino.py --input /Users/ollitrault/Documents/projects/2024_09_09_Axiom_OASIS/1_snakemake/inputs/profiles/dino/plates \
                                              --metadata /Users/ollitrault/Documents/projects/2024_09_09_Axiom_OASIS/1_snakemake/inputs/metadata/metadata.parquet \
                                              --output 02_processed_data/
"""

import argparse
import os
from pathlib import Path
from sys import meta_path

import pandas as pd
from tqdm import tqdm

def expand_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Expand each array column into individual numeric columns."""
    df_expanded = df.copy()
    all_features = []

    for col in feature_cols:
        # Ensure the column exists
        if col not in df.columns:
            print(f"⚠️  Column {col} not found in input data. Skipping.")
            continue

        # Convert each list/array element to DataFrame
        expanded = pd.DataFrame(df[col].tolist())
        expanded.columns = [f"Feature_{col}_{i + 1}" for i in range(expanded.shape[1])]

        # Append names
        all_features += expanded.columns.tolist()

        # Merge back
        df_expanded = pd.concat([df_expanded, expanded], axis=1)

    # Drop original columns
    df_expanded = df_expanded.drop(columns=feature_cols, errors="ignore")
    return df_expanded

def main():
    parser = argparse.ArgumentParser(description="Expand DINOv2 feature arrays into individual columns.")
    parser.add_argument("--input", required=True, help="Path to input Parquet file")
    parser.add_argument("--metadata", required=True, help="Path to metadata Parquet file")
    parser.add_argument("--output", required=True, help="Directory for output Parquet file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    features_rename_map = {
        "hoechst_33342": "DNA",
        "phenovue_fluor_488": "ER",
        "wga555/phalloidin568": "AGP",
        "cp_nuclei_acid_512": "RNA",
        "mito_641_narrow": "Mito",
        "brightfield": "Brightfield",
    }

    meta_cols = [
        "well_id",
        "plate",
        "compound_name",
        "compound_concentration_um",
        "compound_smiles",
        "compound_inchi",
        "source",
        "compound_scode",
        "compound_target",
        "compound_pathway",
        "compound_biological_activity",
        "compound_saltdata",
        "compound_formula",
        "compound_solubility",
        "compound_url",
        "compound_research_area",
        "compound_clinical_information",
        "library",
        "compound_solvent",
        "compound_solvent_concentration",
        "well",
        "row",
        "col",
        # mtt and ldh
        "mtt_lumi",
        "mtt_normalized",
        "mtt_ridge_norm",
        "mtt_ridge_norm_inv",
        "ldh_abs",
        "ldh_abs_signal",
        "ldh_abs_background",
        "ldh_normalized",
        "ldh_ridge_norm",
    ]

    print(f"📂 Loading metadata from {args.metadata}")
    meta = pd.read_parquet(args.metadata)



    profiles = []
    plates = os.listdir(args.input)
    print(f"📂 Loading data from {args.input}")
    for plate in tqdm(plates):
        prof_path = f"{args.input}/{plate}"
        #print(f"📂 Loading data from {prof_path}")
        df = pd.read_parquet(prof_path)
        # Rename all feature names
        # print(f"🔍 Rename features columns: {meta_cols}")
        df = df.rename(columns=features_rename_map)
        # print(f"🔍 Expanding feature columns: {list(features_rename_map.values())}")
        df_expanded = expand_features(df, list(features_rename_map.values()))

        profiles.append(df_expanded)

    # Rename all metadata names
    rename_map_metadata = {c: f"Metadata_{c}" for c in df.columns if c in [m.replace("Metadata_", "") for m in meta_cols]}
    meta = meta.rename(columns=rename_map_metadata)
    meta = meta.rename(columns={
        "Metadata_Plate": "Metadata_plate",
        "Metadata_Well": "Metadata_well",
        "Metadata_Compound": "Metadata_compound",
        "Metadata_Concentration": "Metadata_concentration",
        "Metadata_Perturbation": "Metadata_perturbation",
    })
    #meta["Metadata_perturbation"] = (
    #    meta["Metadata_compound"].astype(str) + "_" + meta["Metadata_concentration"].astype(str)
    #)
    
    df_profiles = pd.concat(profiles, ignore_index=True)
    df_profiles.rename(columns={"well_id": "Metadata_well_id",
                                "image_id": "Metadata_image_id"}, inplace=True)

    df_profiles = df_profiles.merge(meta, on="Metadata_well_id")
    print(f"💾 Writing expanded data to {output_file}")
    df_profiles.to_parquet(output_file, index=False)
    print("✅ Done.")


if __name__ == "__main__":
    main()
