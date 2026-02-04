BSC Botanical_Constituents_and_Quantification_08192024.xlsx annotation data were given by the colaborators as an xlsx file.

Processing Botanical_Constituents_and_Quantification_08192024.xlsx:
- I transformed the file into two tsv files : "Quant Constituents-Table 1.tsv"; "Supplier & Prep-Table 1.tsv"
- I have manually added a column to "Quant Constituent-Table 1.tsv" for the corresponding Cell Painting Plant scode identifiers
- I have manually added a new 'SMILES' column using pubchem exchange using the inchikey column

BSC t0002-10.1080-19390211.2024.2417679.tsv is a table extracted from the article https://doi.org/10.1080/19390211.2024.2417679

Processing t0002-10.1080-19390211.2024.2417679.tsv:
- I have manually added a column to "Quant Constituent-Table 1.tsv" for the corresponding Cell Painting Plant scode identifiers

Chemical Theophylline:
In the annotation file : This chemical is written as   CN1C2=C(C(=O)N(C1=O)C)NC=N2 (using pubchem) and is standardized to : Cn1c(=O)c2[nH]cnc2n(C)c1=O
In the Cell Painting data set : Chemical is written as N1(C)C2NC=NC=2C(N(C)C1=O)=O and is standardized to :                 Cn1c(=O)c2nc[nH]c2n(C)c1=O
The hydrogen is added in the standardization protocol but not at the same place... So I couldn't match the anntotation and the cell painting data for this chemical
Done : Modified manually CN1C2=C(C(=O)N(C1=O)C)NC=N2 in the annotation file to Cn1c(=O)c2nc[nH]c2n(C)c1=O
