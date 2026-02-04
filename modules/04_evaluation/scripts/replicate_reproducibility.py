
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
    parser = argparse.ArgumentParser(description="Compute Replicate reproducibility using cytominer-eval.")
    parser.add_argument("--input", required=True, help="Path to input")
    parser.add_argument("--output", required=True, help="Path to ouput")
    parser.add_argument("--grouping", required=False, default="Metadata_perturbation", help="grouping for chemicals [perturbation]")
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
    print(list_perturbation_replicates)
    df = df[df[args.grouping].isin(list_perturbation_replicates)].copy()

    # Separate metadata and feature columns
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    features_cols = [c for c in df.columns if c.startswith("Feature_")]

    percent_matching_chem_conc, corr_df_chem_conc = evaluate(
                                                            profiles=df,
                                                            features=features_cols,
                                                            meta_features=meta_cols,
                                                            replicate_groups=[args.grouping],
                                                            operation="replicate_reproducibility",
                                                            similarity_metric="pearson",
                                                            replicate_reproducibility_quantile=0.95,
                                                            replicate_reproducibility_return_median_cor=True,
                                                            )
    # Save table
    corr_df_chem_conc.to_csv(args.output + '.csv', index=False)
    print(corr_df_chem_conc)
    # Build the plot
    p = (
        gg.ggplot(corr_df_chem_conc, gg.aes(x="similarity_metric"))
        + gg.geom_density(alpha=0.6)
        + gg.theme_bw()
        + gg.ggtitle("Replicate Reproducibility")
    )

    # Save the plot safely
    p.save(args.output + '.png', verbose=False)

    # Close the figure to avoid duplicated axes or memory buildup
    plt.close("all")

if __name__ == "__main__":
    main()