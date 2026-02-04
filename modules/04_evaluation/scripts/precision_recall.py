
import argparse
import warnings

import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*not hasattr.*"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*DataFrameGroupBy.apply operated on the grouping columns.*"
)
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Compute precision recall using cytominer-eval.")
    parser.add_argument("--input", required=True, help="Path to input")
    parser.add_argument("--output", required=True, help="Path to ouput")
    parser.add_argument("--grouping", required=True, help="grouping for chemicals [chemical_concentration, chemical]")
    parser.add_argument(
                        "--prefix_feature",
                        required=True,
                        help=(
                            "Prefix of features to select, e.g.: "
                            "'all', "
                            "'Feature_hoechst_33342', "
                            "'Feature_phenovue_fluor_488', "
                            "'Feature_wga555/phalloidin568', "
                            "'Feature_cp_nuclei_acid_512', "
                            "'Feature_mito_641_narrow', "
                            "'Feature_brightfield'"
                        )
                    )
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ##### PROCESS #####
    df = pd.read_parquet(input_file)
    # Get compunds with replicates
    list_smiles_replicates = (
        df.groupby("Metadata_compound_scode")["Metadata_well_id"].nunique().sort_values(ascending=False).index.tolist()
    )
    df = df[df["Metadata_compound_scode"].isin(list_smiles_replicates)].copy()
    # set features
    meta_cols = [
        "Metadata_well_id",
        "Metadata_plate",
        "Metadata_compound_name",
        "Metadata_compound_concentration_um",
        "Metadata_compound_smiles",
        "Metadata_compound_inchi",
        "Metadata_source",
        "Metadata_compound_scode",
        "Metadata_compound_target",
        "Metadata_compound_pathway",
        "Metadata_compound_biological_activity",
        "Metadata_compound_saltdata",
        "Metadata_compound_formula",
        "Metadata_compound_solubility",
        "Metadata_compound_url",
        "Metadata_compound_research_area",
        "Metadata_compound_clinical_information",
        "Metadata_library",
        "Metadata_compound_solvent",
        "Metadata_compound_solvent_concentration",
        "Metadata_well",
        "Metadata_row",
        "Metadata_col",
        # mtt and ldh
        "Metadata_mtt_lumi",
        "Metadata_mtt_normalized",
        "Metadata_mtt_ridge_norm",
        "Metadata_mtt_ridge_norm_inv",
        "Metadata_ldh_abs",
        "Metadata_ldh_abs_signal",
        "Metadata_ldh_abs_background",
        "Metadata_ldh_normalized",
        "Metadata_ldh_ridge_norm",
    ]
    prefixes_features = [
        "Feature_hoechst_33342",
        "Feature_phenovue_fluor_488",
        "Feature_wga555_phalloidin568",
        "Feature_cp_nuclei_acid_512",
        "Feature_mito_641_narrow",
        "Feature_brightfield",
    ]
    if args.prefix_feature == "all":
        prefixes_features = [
            "Feature_hoechst_33342",
            "Feature_phenovue_fluor_488",
            "Feature_wga555/phalloidin568",
            "Feature_cp_nuclei_acid_512",
            "Feature_mito_641_narrow",
            "Feature_brightfield",
        ]
    else:
        prefixes_features = args.prefix_feature

    # Get all column names that start with any of the prefixes
    all_features = [col for col in df.columns if col.startswith(tuple(prefixes_features))]
    
    k_list = [2, 3, 5, 10, 15, 25]
    if args.grouping  == "chemical_concentration":
        precision_recall_results_list_chem_conc = []
        for k_chem_conc in k_list:
            precision_recall_results_chem_conc = evaluate(
                profiles=df,
                features=all_features,
                meta_features=meta_cols,
                replicate_groups=["Metadata_compound_scode", "Metadata_compound_concentration_um"],
                operation="precision_recall",
                similarity_metric="pearson",
                precision_recall_k=k_chem_conc,
            )
            precision_recall_results_chem_conc["k"] = k_chem_conc
            precision_recall_results_list_chem_conc.append(precision_recall_results_chem_conc)
        df_precision_recall_results_chem_conc = pd.concat(
            precision_recall_results_list_chem_conc, ignore_index=True
        )
        # Save table
        df_precision_recall_results_chem_conc.to_parquet(args.output + '.parquet', index=False)
        
        p = gg.ggplot(df_precision_recall_results_chem_conc, gg.aes(x="precision", y="recall")) + gg.geom_point(alpha=0.6) + gg.facet_grid("k")
        p.save(args.output + '.png')
    
    if args.grouping == "chemical":
        precision_recall_results_list_chem = []
        for k_chem_conc in k_list:
            precision_recall_results_chem = evaluate(
                profiles=df,
                features=all_features,
                meta_features=meta_cols,
                replicate_groups=["Metadata_compound_scode"],
                operation="precision_recall",
                similarity_metric="pearson",
                precision_recall_k=k_chem_conc,
            )
            precision_recall_results_chem["k"] = k_chem_conc
            precision_recall_results_list_chem.append(precision_recall_results_chem)
        df_precision_recall_results_chem = pd.concat(
            precision_recall_results_list_chem, ignore_index=True
        )
        # Save table
        df_precision_recall_results_chem.to_parquet(args.output + '.parquet', index=False)

        p = gg.ggplot(df_precision_recall_results_chem, gg.aes(x="precision", y="recall")) + gg.geom_point(alpha=0.6) + gg.facet_grid("k")
        p.save(args.output + '.png')

if __name__ == "__main__":
    main()