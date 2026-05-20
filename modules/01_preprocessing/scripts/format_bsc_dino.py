#!/usr/bin/env python3
"""
1A_data_format_dino.py
------------------
Expands array-based DINOv2 feature columns into independent numeric columns.
"""

import argparse
from pathlib import Path

import polars as pl


def expand_features(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    expanded_cols = []

    for col in feature_cols:
        if col not in df.columns:
            print(f"⚠️  Column {col} not found in input data. Skipping.")
            continue

        # Infer feature length safely (max length)
        feat_len = (
            df.select(pl.col(col).list.len().max())
              .item()
        )

        for i in range(feat_len):
            expanded_cols.append(
                pl.col(col)
                  .list.get(i)
                  .alias(f"Feature_{col}_{i + 1}")
            )

    return (
        df
        .with_columns(expanded_cols)
        .drop(feature_cols)
    )


def main():
    parser = argparse.ArgumentParser(description="Expand DINOv2 feature arrays into individual columns.")
    parser.add_argument("--input", required=True, help="Path to input Parquet file")
    parser.add_argument("--output", required=True, help="Path to output Parquet file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {input_path}")
    df = pl.read_parquet(input_path)

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

    print("🔍 Rename feature columns")
    df = df.rename(features_rename_map)

    print("🔍 Rename metadata columns")
    rename_map_metadata = {
        c: f"Metadata_{c}"
        for c in df.columns
        if c in meta_cols
    }

    df = (
        df.rename(rename_map_metadata)
          .rename({
              "Metadata_compound_scode": "Metadata_compound",
              "Metadata_compound_concentration_um": "Metadata_concentration",
          })
    )

    df = df.with_columns(
        (
            pl.col("Metadata_compound").cast(pl.Utf8)
            + "_"
            + pl.col("Metadata_concentration").cast(pl.Utf8)
        ).alias("Metadata_perturbation")
    )

    print(f"🔍 Expanding feature columns: {list(features_rename_map.values())}")
    df_expanded = expand_features(df, list(features_rename_map.values()))

    print(f"🔍 Expanding feature columns: {list(features_rename_map.values())}")
    df_expanded = expand_features(df, list(features_rename_map.values()))

    print(f"💾 Writing expanded data to {output_path}")
    df_expanded.write_parquet(output_path)
    print("✅ Done.")


if __name__ == "__main__":
    main()
