import logging  # noqa: CPY001, D100

import polars as pl
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def call_hits(
    mtt_path: str,
    ldh_path: str,
    cc_path: str,
    hits_path: str,
) -> None:
    """Binarize assay curves."""
    # 1. Read in data
    mtt = pl.read_parquet(mtt_path)
    ldh = pl.read_parquet(ldh_path)
    cc = pl.read_parquet(cc_path)

    # 2. Get hits
    hits = mtt.select("Metadata_OASIS_ID") 

    mtt_hits = mtt.filter(pl.col("all.pass") == True).select("Metadata_OASIS_ID").to_series().to_list()
    ldh_hits = ldh.filter(pl.col("all.pass") == True).select("Metadata_OASIS_ID").to_series().to_list()
    cc_hits = cc.filter(pl.col("all.pass") == True).select("Metadata_OASIS_ID").to_series().to_list()

    hits = hits.with_columns(
        pl.when(pl.col("Metadata_OASIS_ID").is_in(mtt_hits)).then(pl.lit(1)).otherwise(pl.lit(0)).alias("MTT"),
        pl.when(pl.col("Metadata_OASIS_ID").is_in(ldh_hits)).then(pl.lit(1)).otherwise(pl.lit(0)).alias("LDH"),
        pl.when(pl.col("Metadata_OASIS_ID").is_in(cc_hits)).then(pl.lit(1)).otherwise(pl.lit(0)).alias("cell_count"),
    )
    print(hits)
    hits = hits.rename({"Metadata_OASIS_ID": "OASIS_ID"})

    # 3. Write out results
    hits.write_parquet(hits_path)

# python 01_analysis/7_script_classifier/7_data_axiom_hitcalls.py --input_pod_mtt 03_tables/pods_axiom_Metadata_mtt_normalized_original_conc.parquet --input_pod_ldh 03_tables/pods_axiom_Metadata_ldh_normalized_original_conc.parquet --output 03_tables/test.parquet
# I have replaced all Metadata_OASIS_ID to Metadata_compound
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate Level-5 consensus DINOv2 features into Level-6 compound aggregated.")
    parser.add_argument("--input_pod_mtt", required=True, help="Path to input POD Parquet file of MTT")
    parser.add_argument("--input_pod_ldh", required=True, help="Path to input POD Parquet file of LDH") # !! recompute cc
    parser.add_argument("--input_pod_cc", required=False, help="Path to output POD Parquet file Cell Count")
    parser.add_argument("--output", required=True, help="Directory for output Parquet file")
    args = parser.parse_args()
    call_hits(
                mtt_path=args.input_pod_mtt,
                ldh_path=args.input_pod_ldh,
                cc_path=args.input_pod_cc,
                hits_path=args.output
            )