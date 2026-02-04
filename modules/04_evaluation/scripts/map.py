
import argparse
import logging
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
from copairs import map
from copairs.matching import assign_reference_index
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description="Compute mAP using copairs.")
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
    #df = df[df["Metadata_plate"] == "plate_41005688"].copy()
    # Get compunds with replicates
    list_perturbation_replicates = (
        df.groupby(args.grouping)["Metadata_well_id"].nunique().sort_values(ascending=False).index.tolist()
    )
    df = df[df[args.grouping].isin(list_perturbation_replicates)].copy()

    # Separate metadata and feature columns
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    features_cols = [c for c in df.columns if c.startswith("Feature_")]
    # 
    activity_ap_list = []
    activity_map_list = []
    reference_col = "Metadata_reference_index"

    condition_control = df[df["Metadata_compound"] == args.control][args.grouping].unique()
    print('Control condition:', condition_control, condition_control[0])
    df_activity = assign_reference_index(
        df,
        f"{args.grouping} == '{condition_control[0]}'",  # condition to get reference profiles (neg controls)
        reference_col=reference_col,
        default_value=-1,
    )
    # positive pairs are replicates of the same treatment
    pos_sameby = [args.grouping, reference_col]
    pos_diffby = []
    
    neg_sameby = []
    # negative pairs are replicates of different treatments
    neg_diffby = [args.grouping, reference_col]

    metadata = df_activity.filter(regex="^Metadata")
    profiles = df_activity.filter(regex="^Feature").values
    
    print(   len(pos_sameby), pos_sameby,
             len(pos_diffby), pos_diffby,
             len(neg_sameby), neg_sameby,
             len(neg_diffby), neg_diffby)
    print(metadata.shape, profiles.shape)

    #quit()
    activity_ap = map.average_precision(
        metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
    )
    activity_ap = activity_ap.query(f"{args.grouping} != '{condition_control[0]}'")  # remove control

    activity_map = map.mean_average_precision(
    activity_ap, pos_sameby, null_size=1000000, threshold=0.05, seed=0
    )
    
    print('Done.')
    activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(np.log10)
    activity_ap.to_csv(args.output + '_ap.csv', index=False)
    activity_map.to_csv(args.output + '_map.csv', index=False)

if __name__ == "__main__":
    main()