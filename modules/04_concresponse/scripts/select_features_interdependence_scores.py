import argparse
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm
import ids
from statsmodels.stats.multitest import multipletests
import torch

def well_to_coord(well):
    row_letter = well[0]
    col = int(well[1:])

    row_num = ord(row_letter) - ord("A") + 1  # A=1, B=2, ...

    return pd.Series({"row": row_num, "col": col})
def get_closest_dmso(comp_fit_coords, dat_ctrl):

    closest_rows = []

    for _, this in comp_fit_coords.iterrows():

        ctrl_same_plate = dat_ctrl[
            dat_ctrl["Metadata_plate"] == this["Metadata_plate"]
        ]

        if ctrl_same_plate.empty:
            continue

        # compute Euclidean distance
        dist = np.sqrt(
            (ctrl_same_plate["row"] - this["row"])**2 +
            (ctrl_same_plate["col"] - this["col"])**2
        )

        ctrl_same_plate = ctrl_same_plate.assign(dist=dist)
        min_dist = dist.min()

        closest = ctrl_same_plate[ctrl_same_plate["dist"] == min_dist]
        closest_rows.append(closest)

    if len(closest_rows) == 0:
        return pd.DataFrame()

    return pd.concat(closest_rows).drop_duplicates()

def process_compound(compound, dat, dat_ctrl, dat_ctrl_by_plate,
                     dmso_mode, features_cols, output_path, num_tests=10000, p_value_correction=True):
    # Subset compound
    dat_comp = dat.loc[dat["Metadata_compound"] == compound].copy()
    if dat_comp.empty:
        print(f"No data for {compound}")
        return None

    # Control selection
    if dmso_mode == "closest":
        ctrl_fit = get_closest_dmso(dat_comp, dat_ctrl)
    else:
        plates = dat_comp["Metadata_plate"].unique()
        ctrl_fit = pd.concat(
            [dat_ctrl_by_plate[p] for p in plates if p in dat_ctrl_by_plate],
            ignore_index=True
        )

    print(
        f"Fitting IDS for {compound} "
        f"with {len(dat_comp)} compound and "
        f"{len(ctrl_fit)} DMSO points."
    )

    # Combine
    dat_fit = pd.concat([ctrl_fit, dat_comp]).sort_values("Metadata_log10conc")

    # Prepare IDS inputs
    X = dat_fit[features_cols].values
    Y = dat_fit[["Metadata_log10conc"]].values  # must be 2D

    # ── Convert to tensors ────────────────────────────────────────────────────
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # Compute IDS
    C, p_vals = ids.compute_IDS(
        X,
        Y,
        p_val=True,
        num_tests=num_tests
    )

    p_vals_flat = p_vals.detach().cpu().numpy().flatten()
    C_flat      = C.detach().cpu().numpy().flatten()

    # ── Benjamini-Hochberg correction ─────────────────────────────────────
    sig_mask, p_vals_corrected, _, _ = multipletests(
        p_vals_flat, alpha=0.05, method="fdr_bh"
    )
    # ──────────────────────────────────────────────────────────────────────

    if not p_value_correction:
        sig_mask = p_vals_flat < 0.05

    n_significant = np.sum(sig_mask)
    print(f"{compound}: {np.sum(p_vals_flat < 0.05)} features significant before BH correction")
    print(f"{compound}: {n_significant} features significant after BH correction")
    print(f"  Raw p-values:       {p_vals_flat}")
    print(f"  Corrected p-values: {p_vals_corrected}")

    # Write significant features to output file
    sig_features = np.array(features_cols)[sig_mask]
    sig_scores   = C_flat[sig_mask]
    sig_pvals    = p_vals_flat[sig_mask]
    sig_pvals_bh = p_vals_corrected[sig_mask]

    sig_df = pd.DataFrame({
        "feature":          sig_features,
        "ids_score":        sig_scores,
        "p_value_raw":      sig_pvals,
        "p_value_bh":       sig_pvals_bh        # BH-adjusted p-value
    }).sort_values("p_value_bh")

    sig_df.to_csv(output_path, index=False)
    print(f"Significant features written to {output_path}")

    return {
        "compound":          compound,
        "n_significant":     int(n_significant),
        "ids_scores":        C_flat,
        "p_values":          p_vals_flat,
        "p_values_bh":       p_vals_corrected    # full corrected array
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dose-response BMD pipeline"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="modules/03_postprocessing/outputs/level4b_bsc_filtered.parquet",
        help="Path to input parquet file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="modules/06_concresponse/outputs/pods/test.csv",
        help="Path to output parquet file"
    )
    parser.add_argument(
        "--ctrl",
        type=str,
        default="s0617825",
        help="Control compound name (e.g., DMSO id)"
    )
    parser.add_argument(
        "--dmso_mode",
        type=str,
        default="closest",
        choices=["closest", "plate"],
        help="How to select DMSO controls"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="DNA,ER,AGP,RNA,Mito,Brightfield",
        help="Comma-separated list of categories"
    )
    parser.add_argument(
        "--log10_dose",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use log10 dose (true/false)"
    )
    parser.add_argument(
        "--compounds",
        type=str,
        default="s0654867",
        help="'all' or comma-separated compound list"
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=1000,
        help="Number of random tests for IDS p-value estimation"
    )
    parser.add_argument(
        "--p_value_correction",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to apply Benjamini-Hochberg correction (true/false)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    threads_per_fit = 2

    available = os.cpu_count()
    n_workers = max(1, (available - 1) // threads_per_fit)

    print(
        f"Detected {available} cores → "
        f"{n_workers} workers × {threads_per_fit} threads/fit"
    )

    args = parse_args()

    # Assign variables
    input_path = args.input_path
    output_path = args.output_path
    ctrl = args.ctrl
    dmso_mode = args.dmso_mode
    #category = args.category.split(",")
    #log10_dose = args.log10_dose
    compounds = [args.compounds]

    # Compute IDS
    dat = pd.read_parquet(input_path)
    features_cols = [c for c in dat.columns if c.startswith("Feature_")]
    dat[["row", "col"]] = dat["Metadata_well"].apply(well_to_coord)
    dat_ctrl = dat[dat["Metadata_compound"] == ctrl].copy()
    if compounds == ["all"]:
        compounds = dat["Metadata_compound"].unique().tolist()
        compounds = [c for c in compounds if c != ctrl]
    dat_ctrl_by_plate = None
    if dmso_mode == "plate":
        dat_ctrl_by_plate = {
            plate: df
            for plate, df in dat_ctrl.groupby("Metadata_plate")
        }
    elif dmso_mode != "closest":
        raise ValueError("dmso_mode must be 'closest' or 'plate'")
    results = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        
        futures = {
            executor.submit(
                process_compound,
                cmpd,
                dat,
                dat_ctrl,
                dat_ctrl_by_plate,
                dmso_mode,
                features_cols,
                output_path,
                num_tests=args.num_tests,
                p_value_correction=args.p_value_correction
            ): cmpd
            for cmpd in compounds
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            compound = futures[future]
            try:
                results[compound] = future.result()
            except Exception as e:
                print(f"Error processing {compound}: {e}")

