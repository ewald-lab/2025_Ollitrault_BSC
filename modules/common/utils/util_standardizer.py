from typing import Union
import pandas as pd
import logging
from chembl_structure_pipeline import standardizer
from multiprocessing import Pool, cpu_count
from rdkit import RDLogger
from rdkit.rdBase import BlockLogs
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO)

class StandardizeMolecule:
    def __init__(
        self,
        input: Union[str, pd.DataFrame],
        output: str = None,
        num_cpu: int = None,
        input_column: str = "SMILES",
        input_format: str = "SMILES",
        limit_rows: int = None,
        largest_fragment: bool = True,
        augment: bool = True,
        random_seed: int = 42,
    ):
        """
        Initialize the class.

        :param input: Input file name (TSV/TXT/CSV) or a pandas dataframe containing the SMILES
        :param output: Output file name (optional)
        :param num_cpu: Number of CPUs to use (default: 1)
        :param input_column: Column name containing SMILES/InChI (default: SMILES)
        :param input_format: Format of input molecules: SMILES or InChI (default: SMILES)
        :param limit_rows: Limit the number of rows to be processed (optional)
        :param largest_fragment: If True, selects the largest fragment before standardization (default: True)
        :param augment: If True, returns input file augmented with standardized outputs
        :param random_seed: Random seed for reproducibility
        """
        self.input = input
        self.output = output
        self.num_cpu = num_cpu or max(cpu_count() - 1, 1)  # Use all CPUs except 1
        self.input_column = input_column
        self.input_format = input_format
        self.limit_rows = limit_rows
        self.largest_fragment = largest_fragment
        self.augment = augment
        self.random_seed = random_seed

    def _standardize_mol(self, args) -> dict:
        """Standardize a single molecule using ChEMBL protocol."""
        idx, mol_raw_string = args
        # Disable RDKit logging
        block = BlockLogs()  # noqa: F841
        # Convert raw string to RDKit molecule
        if self.input_format == "SMILES":
            mol_rdkit = Chem.MolFromSmiles(mol_raw_string)
        elif self.input_format == "InChI":
            mol_rdkit = Chem.MolFromInchi(mol_raw_string)
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")
        # Handle invalid molecule
        if mol_rdkit is None:
            logging.error(f"Reading Error: '{mol_raw_string}' could not be parsed.")
            return {
                "index": idx,
                "Mol_original": mol_raw_string,
                "SMILES_standardized": pd.NA,
                "SMILES_standardized_nonisomeric": pd.NA,
                "InChI_standardized": pd.NA,
                "InChI_standardized_nonisomeric": pd.NA,
                "InChIKey_standardized": pd.NA,
                "InChIKey_standardized_nonisomeric": pd.NA,
            }

        # Standardize structure
        if self.largest_fragment:
            mol_rdkit = rdMolStandardize.FragmentParent(mol_rdkit)
        mol_chembl = standardizer.standardize_mol(mol_rdkit)
        try:
            # select biggest fragment and standardize
            if self.largest_fragment:
                mol_rdkit = rdMolStandardize.FragmentParent(mol_rdkit)
            mol_chembl = standardizer.standardize_mol(mol_rdkit)
        except Exception as e:
            logging.error(f"Standardization failed for {mol_rdkit}: {e}")
            return {
                "index": idx,
                "Mol_original": mol_raw_string,
                "SMILES_standardized": pd.NA,
                "SMILES_standardized_nonisomeric": pd.NA,
                "InChI_standardized": pd.NA,
                "InChI_standardized_nonisomeric": pd.NA,
                "InChIKey_standardized": pd.NA,
                "InChIKey_standardized_nonisomeric": pd.NA,
            }
        # Generate standardized outputs
        smiles_std = Chem.MolToSmiles(mol_chembl, isomericSmiles=True)
        inchi_std = MolToInchi(mol_chembl)
        inchikey_std = MolToInchiKey(mol_chembl)

        Chem.RemoveStereochemistry(mol_chembl)
        smiles_std_noiso = Chem.MolToSmiles(mol_chembl, isomericSmiles=False)
        inchi_std_noiso = MolToInchi(mol_chembl)
        inchikey_std_noiso = MolToInchiKey(mol_chembl)

        return {
            "index": idx,
            "Mol_original": mol_raw_string,
            "SMILES_standardized": smiles_std,
            "SMILES_standardized_nonisomeric": smiles_std_noiso,
            "InChI_standardized": inchi_std,
            "InChI_standardized_nonisomeric": inchi_std_noiso,
            "InChIKey_standardized": inchikey_std,
            "InChIKey_standardized_nonisomeric": inchikey_std_noiso,
        }

    def _process_input(self) -> pd.DataFrame:
        """Load input file or use DataFrame directly."""
        if isinstance(self.input, pd.DataFrame):
            df = self.input.copy()
        else:
            try:
                df = pd.read_csv(self.input)
            except Exception:
                df = pd.read_csv(self.input, sep='\t')

        if self.limit_rows:
            df = df.head(self.limit_rows)

        if self.input_column not in df.columns:
            raise KeyError(f"Input column '{self.input_column}' not found in data.")

        return df

    def run(self) -> pd.DataFrame:
        """Run the standardization using multiprocessing."""
        df = self._process_input()
        mol_list_idx = list(zip(df.index, df[self.input_column].astype(str)))

        logging.info(f"Starting multiprocessing with {self.num_cpu} CPUs...")
        results = []
        with Pool(self.num_cpu) as pool:
            for res in tqdm(
                pool.imap_unordered(self._standardize_mol, mol_list_idx),
                total=len(mol_list_idx),
                desc="Standardizing molecules",
                unit="mol"
            ):
                results.append(res)

        result_df = pd.DataFrame(results).set_index("index")

        if self.augment:
            result_df = df.join(result_df, how="left")

        if self.output:
            result_df.to_csv(self.output, index=False)
            logging.info(f"Results saved to {self.output}")

        return result_df

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standardize molecular structures using the ChEMBL standardization pipeline."
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Input file path (CSV/TSV/TXT) containing molecular data.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path to save standardized results.")
    parser.add_argument("--num_cpu", type=int, default=None,
                        help="Number of CPUs to use (default: all available minus one).")
    parser.add_argument("--input_column", type=str, default="SMILES",
                        help="Name of the input column containing SMILES or InChI (default: SMILES).")
    parser.add_argument("--input_format", type=str, choices=["SMILES", "InChI"], default="SMILES",
                        help="Format of input molecules (default: SMILES).")
    parser.add_argument("--limit_rows", type=int, default=None,
                        help="Limit the number of rows to process (optional).")
    parser.add_argument("--augment", action="store_true",
                        help="Include original data in output (default: True).")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Return only standardized data.")
    parser.set_defaults(augment=True)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()

    standardizer_instance = StandardizeMolecule(
        input=args.input,
        output=args.output,
        num_cpu=args.num_cpu,
        input_column=args.input_column,
        input_format=args.input_format,
        limit_rows=args.limit_rows,
        augment=args.augment,
        random_seed=args.random_seed,
    )

    standardized_df = standardizer_instance.run()
    print(standardized_df.head())
