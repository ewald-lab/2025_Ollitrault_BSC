import pandas as pd
import polars as pl
import argparse
from pathlib import Path

def conv_pod_um(df_pods, df_metadata, pod_col, conv_pod_nm):
    df = df_pods.join(df_metadata, on="Metadata_compound")
    df = df.with_columns(
        (10 ** (pl.col(pod_col) - pl.col("Metadata_conc_shift"))).alias(conv_pod_nm)
    ).select(df_pods.columns + [conv_pod_nm])#.drop(["Metadata_min_conc", "Metadata_conc_shift"])

    return df

def main():
    parser = argparse.ArgumentParser(description="Transform back log10 concentration to original concentration.")
    parser.add_argument("--input_pods", required=True, help="Path to input Parquet file with PODs")
    parser.add_argument("--input_metadata", required=True, help="Path to input Parquet file with metadata")
    parser.add_argument("--output", required=True, help="Path to output Parquet file with transformed concentration to uM")
    parser.add_argument("--control", required=True, default="DMSO", help="Name of control compound")
    parser.add_argument("--pod_col", required=True, help="Name of POD column to transform back")
    parser.add_argument("--is_metadata_pods", action="store_true", help="Whether the input pods are metadata pods")

    args = parser.parse_args()

    input_pods_path = Path(args.input_pods)
    input_metadata_path = Path(args.input_metadata)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {input_pods_path} and {input_metadata_path}")
    if args.is_metadata_pods:
        #df_pods = pl.read_parquet(input_pods_path).filter(pl.col("all.pass") == True).filter(pl.col('adv.incr') == args.adv_incr).select(["Metadata_compound", "bmd"]).rename({"bmd": args.pod_col+"_POD"})
        df_pods = pl.read_parquet(input_pods_path).select(["Metadata_compound", "gene.id", "bmd", "all.pass", 'adv.incr']).rename({"bmd": args.pod_col+"_POD"})
    else:
        df_pods = pl.read_parquet(input_pods_path).select(["Metadata_compound", "gene.id", "bmd", "all.pass", 'adv.incr']).rename({"bmd": args.pod_col+"_POD"})

    df_metadata = pl.read_parquet(input_metadata_path)

    print("🔍 Transforming back log10 concentration to original concentration")
    # Get info to convert PODs
    df_metadata = df_metadata.filter(pl.col("Metadata_compound") != args.control).select(
        ["Metadata_compound", "Metadata_concentration"]
    ).group_by("Metadata_compound").agg([
        pl.col("Metadata_concentration").min().alias("Metadata_min_conc")
    ])

    df_metadata = df_metadata.with_columns(
        (pl.col("Metadata_min_conc")/3).log10().abs().alias("Metadata_conc_shift")
    )

    df = conv_pod_um(df_pods, df_metadata, args.pod_col+"_POD", args.pod_col + "_POD_um")
    print(f"💾 Writing transformed data to {output_path}")
    print(df.head())
    df.to_pandas().to_parquet(output_path)
    print("✅ Done.")

if __name__ == "__main__":
    main()