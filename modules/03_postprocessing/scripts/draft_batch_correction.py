#!/usr/bin/env python3

import argparse
import pandas as pd
from harmonypy import run_harmony
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Harmony batch correction")
    parser.add_argument("--bsc", required=True)
    parser.add_argument("--axiom", required=True)
    parser.add_argument("--out-bsc", required=True)
    parser.add_argument("--out-axiom", required=True)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--nclust", type=int, default=300)
    parser.add_argument("--filter-dmso", action="store_true", default=True)
    args = parser.parse_args()

    # --- Load ---
    df_bsc = pd.read_parquet(args.bsc)
    df_axiom = pd.read_parquet(args.axiom)

    # --- Optional filter ---
    if args.filter_dmso:
        df_bsc = df_bsc.groupby("Metadata_plate").filter(
            lambda g: ~(g["Metadata_compound_name"] == "DMSO").all()
        )

    # --- Add batch labels ---
    df_bsc["Metadata_batch"] = "bsc"
    df_axiom["Metadata_batch"] = "axiom"

    df = pd.concat([df_bsc, df_axiom], ignore_index=True)

    # --- Columns ---
    features = [c for c in df.columns if c.startswith("Feature_")]
    metadata = [c for c in df.columns if c.startswith("Metadata_")]

    # --- Run Harmony ---
    harmony = run_harmony(
        df[features].values,
        df[metadata],
        "Metadata_batch",
        max_iter_harmony=args.max_iter,
        nclust=args.nclust,
        verbose=True
    )

    # --- Replace features ---
    df_corrected = pd.concat(
        [
            df.drop(columns=features),
            pd.DataFrame(harmony.Z_corr, columns=features)
        ],
        axis=1
    )

    # --- Split back ---
    df_bsc_corr = df_corrected[df_corrected["Metadata_batch"] == "bsc"].drop(columns=["Metadata_batch"])
    df_axiom_corr = df_corrected[df_corrected["Metadata_batch"] == "axiom"].drop(columns=["Metadata_batch"])

    # --- Save ---
    Path(args.out_bsc).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_axiom).parent.mkdir(parents=True, exist_ok=True)

    df_bsc_corr.to_parquet(args.out_bsc, index=False)
    df_axiom_corr.to_parquet(args.out_axiom, index=False)

    print(f"✓ Done")
    print(f"BSC shape: {df_bsc_corr.shape}")
    print(f"Axiom shape: {df_axiom_corr.shape}")


if __name__ == "__main__":
    main()