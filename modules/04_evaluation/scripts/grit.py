
import argparse
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*not hasattr.*"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*DataFrameGroupBy.apply operated on the grouping columns.*"
)

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from cytominer_eval import evaluate
# Compatibility for older libraries expecting np.float etc.
import numpy as np
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "str"):
    np.str = str
import plotnine as gg


def main():
    parser = argparse.ArgumentParser(description="Compute GRiT using cytominer-eval.")
    parser.add_argument("--input", required=True, help="Path to input")
    parser.add_argument("--output", required=True, help="Path to ouput")
    parser.add_argument("--grouping", required=False, default="Metadata_perturbation", help="grouping for chemicals [chemical_concentration, chemical]")
    parser.add_argument("--control", required=False, default="DMSO", help="control for GRiT calculation [DMSO]")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ##### PROCESS #####
    df = pd.read_parquet(input_file)
    # Get compunds with replicates
    list_perturbation_replicates = (
        df.groupby(args.grouping)["Metadata_well_id"].nunique().sort_values(ascending=False).index.tolist()
    )
    df = df[df[args.grouping].isin(list_perturbation_replicates)].copy()

    # Separate metadata and feature columns
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    features_cols = [c for c in df.columns if c.startswith("Feature_")]

    if False :
        plates_grit = df["Metadata_plate"].unique()
        grit_scores = []

        grit_replicate_groups = {
            "profile_col": "Metadata_well_id",
            "replicate_group_col": args.grouping,
        }
        for plate_grit in plates_grit:
            control_wells_grit = df[df["Metadata_plate"] == plate_grit].query(f"Metadata_compound == '{args.control}'").Metadata_well_id.unique().tolist()
            grit_results = evaluate(
                                    profiles=df[df["Metadata_plate"] == plate_grit],
                                    features=features_cols,
                                    meta_features=meta_cols,
                                    replicate_groups=grit_replicate_groups,
                                    operation="grit",
                                    similarity_metric="pearson",
                                    grit_replicate_summary_method="median",
                                    grit_control_perts=control_wells_grit).assign(plate=plate_grit)
            grit_scores.append(grit_results)
        grit_scores_df = pd.concat(grit_scores).reset_index(drop=True)
        grit_scores_df.to_csv(args.output + '.csv', index=False)
    grit_replicate_groups = {
        "profile_col": "Metadata_well_id",
        "replicate_group_col": args.grouping,
    }

    # all control wells across *all* plates
    control_wells_grit = (
        df.query(f"Metadata_compound == '{args.control}'")
        .Metadata_well_id
        .unique()
        .tolist()
    )

    grit_scores_df = evaluate(
        profiles=df,
        features=features_cols,
        meta_features=meta_cols,
        replicate_groups=grit_replicate_groups,
        operation="grit",
        similarity_metric="pearson",
        grit_replicate_summary_method="median",
        grit_control_perts=control_wells_grit
    )

    grit_scores_df.to_csv(args.output + ".csv", index=False)


if __name__ == "__main__":
    main()