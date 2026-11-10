BSC Botanical_Constituents_and_Quantification_03122026.xlsx annotation data were given by the collaborators as an xlsx file.
It replaces Botanical_Constituents_and_Quantification_08192024.xlsx.

Differences with the 08192024 xlsx (checked cell by cell):
- "Supplier & Prep" sheet: identical.
- "Quant Constituents" sheet: the 107 previous rows are identical. Two rows were added for Ashwagandha (after Palmitamide):
    Tropine        DTXSID2049399    MW 141.214  0.97 mg/g  note "***"  CAS 120-29-6  CYHOMWAPJJPNMW-JIGDXULJSA-N  no PubChem ID
    Pseudotropine  DTXSID501026533  MW 141.214  0.97 mg/g  note "***"  CAS 135-97-7  CYHOMWAPJJPNMW-RNLVFQAGSA-N  no PubChem ID
  "***" is not defined in the header (only * and ** are). To confirm with the collaborators.
- Header typo fixed in the notes column: "concencentration" -> "concentration".

How the tsv files were made:
- "Supplier & Prep-Table 1.tsv": copied unchanged from the 08192024 folder (sheet is identical).
- "Quant Constituents-Table 1.tsv": copy of the 08192024 tsv (all manual edits kept: Plant scode column, PubChem SMILES column,
  Theophylline SMILES fix, see ../Botanical_Constituents_and_Quantification_08192024/readme.txt), with the 2 new rows inserted after Palmitamide.
    - Values copied exactly from the xlsx (0.97 mg/g kept as given).
    - Plant scode = s0654791 (Ashwagandha).
    - SMILES = xlsx "Canonical SMILES". PubChem has no entry for these two InChIKeys (PubChem tropine records, e.g. CID 449293,
      have no stereo at C3, so tropine and pseudotropine cannot be told apart there). The xlsx SMILES match the xlsx InChIKeys (checked with RDKit).
    - The old header (with the "concencentration" typo) is kept, because chemical_annotations.marimo.py uses that exact column name.

Effect: neither tropine nor pseudotropine is in the Cell Painting data, so the matched constituents are unchanged (85 constituents, 13 botanicals).
Only the Ashwagandha (s0654791) MW used for ug/mL -> uM conversion changes:
    mw_mean      619.2 -> 575.7
    mw_weighted  567.9 -> 499.6
