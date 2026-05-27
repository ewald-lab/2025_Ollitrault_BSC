import argparse
import polars as pl
import pandas as pd

def compile_dist(input_files: list, transform: str, output_path: str) -> None:
    dfs = []
    for fp in input_files:
        dat = pl.read_parquet(fp)
        dfs.append(dat.unique())

    dfs = pl.concat(dfs, how="vertical")
    meta_cols = [i for i in dfs.columns if "Metadata" in i and i != "Metadata_distance"]
    df_wide = dfs.pivot(
        values="Distance",
        index=meta_cols,
        on="Metadata_distance",
        aggregate_function="median",

    )

    # apply transform if specified
    if transform == "log10":
        dist_cols = [i for i in df_wide.columns if "Metadata" not in i]
        df_wide = df_wide.with_columns([pl.col(col).log10().alias(col) for col in dist_cols])

        # shift by min_val to ensure all log10 transformed values are > 0
        min_val = df_wide.select([pl.col(col).min() for col in dist_cols]).transpose().min().item()
        df_wide = df_wide.with_columns([(pl.col(col) - min_val + 1e-6).alias(col) for col in dist_cols])
    
    # Ensure it is a pandas DataFrame
    # Now this line should work
    df_wide = df_wide.with_columns(
        pl.col(pl.Categorical, pl.Enum).cast(pl.Utf8)
    )
    df_wide.write_parquet(output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Compile distance parquet files into a single transformed parquet table."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of input .parquet distance files"
    )
    parser.add_argument(
        "--transform",
        choices=["none", "log10"],
        default="none",
        help="Distance transform to apply (default: none)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output csv file"
    )

    args = parser.parse_args()

    transform = None if args.transform == "none" else args.transform
    print(args.inputs, transform, args.output)
    compile_dist(args.inputs, transform, args.output)


if __name__ == "__main__":
    main()