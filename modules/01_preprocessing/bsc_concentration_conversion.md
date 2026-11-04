# Botanical concentration conversion: `mw_mean` vs `mw_weighted`

Botanical extracts are dosed in µg/mL. Preprocessing ([scripts/format_bsc_dino.py](scripts/format_bsc_dino.py)) converts them to µM so they can be compared with single chemicals. The MW used for each botanical is set by `bsc_conversion_mode` in [configs/config_base_bsc.json](../../configs/config_base_bsc.json):

| Mode | MW used | Notes |
|---|---|---|
| `mw_mean` (default) | unweighted mean MW of the quantified constituents | same behaviour as before this option existed |
| `mw_weighted` | content-weighted MW (formula below) | uses the constituent content (mg/g) |
| `none` | – | botanicals stay in µg/mL |

Constituent data: `inputs/Botanical_Constituents_and_Quantification_08192024/Quant Constituents-Table 1.tsv`. Content values such as `5.2 ± 0.8` are read as the mean (5.2) and the SD is dropped.

## Formulas

For an extract dosed at C (µg/mL):

```
µM = C × 1000 / MW_b
```

**`mw_mean`**: each of the n constituents counts equally, whatever its content:

```
MW_mean = (1/n) × Σ MWᵢ
```

**`mw_weighted`**: each constituent is weighted by its share of the quantified mass, wᵢ = cᵢ / Σc, where cᵢ is the content in mg/g:

```
MW_weighted = Σcᵢ / Σ(cᵢ / MWᵢ) = 1 / Σ(wᵢ / MWᵢ)
```

Constituent i is present at C·wᵢ µg/mL, which is C·wᵢ·1000/MWᵢ µM. Summed over all constituents, this gives C·1000/MW_weighted. So `mw_weighted` gives the exact total molar concentration of the constituents, assuming the whole extract has the same composition as its quantified part. `mw_mean` gives the same result only when all constituents have the same content.

For a given botanical, the two modes always differ by the same factor:

```
µM_weighted / µM_mean = MW_mean / MW_weighted
```

On a log₁₀ concentration axis this shifts the curve without changing its shape, so fitted PoDs move by exactly the same factor.

## Comparison on the BSC data

Data: 2024 TSV, 8 doses per botanical (3-fold spacing), 360 wells per botanical.

| Botanical | % mass quantified | MW mean | MW weighted | µM per 1 µg/mL (mean → weighted) | Change | Top dose µM (mean → weighted) |
|---|---|---|---|---|---|---|
| Green tea extract | 77% | 279.1 | 386.5 | 3.58 → 2.59 | **−27.8%** | 4478 → 3234 |
| Usnea | 40% | 271.0 | 299.6 | 3.69 → 3.34 | −9.5% | 3063 → 2770 |
| Asian ginseng | 3.8% | 950.2 | 1012.0 | 1.05 → 0.99 | −6.1% | 116 → 109 |
| Aristolochia | 8.0% | 326.6 | 339.9 | 3.06 → 2.94 | −3.9% | 2541 → 2442 |
| Goldenseal | 1.7% | 346.4 | 356.2 | 2.89 → 2.81 | −2.8% | 3609 → 3510 |
| Comfrey | 3.9% | 331.9 | 339.6 | 3.01 → 2.94 | −2.3% | 331 → 324 |
| Blue cohosh | 2.3% | 285.2 | 289.2 | 3.51 → 3.46 | −1.4% | 4383 → 4322 |
| Milk thistle | 49% | 456.7 | 462.5 | 2.19 → 2.16 | −1.3% | 3657 → 3610 |
| Yohimbe | 6.4% | 351.2 | 353.7 | 2.85 → 2.83 | −0.7% | 1780 → 1767 |
| Kava | 50% | 262.3 | 246.3 | 3.81 → 4.06 | +6.5% | 229 → 244 |
| Ephedra | 2.4% | 190.2 | 174.5 | 5.26 → 5.73 | +9.0% | 3285 → 3581 |
| Ashwagandha | 4.1% | 619.2 | 567.9 | 1.62 → 1.76 | +9.0% | 81 → 88 |
| Kratom | 11% | 408.5 | 373.3 | 2.45 → 2.68 | +9.4% | 514 → 563 |

## Findings

- **Small effect overall.** 12 of 13 botanicals change by less than 10% (median 6%). The average shift is 0.03 on a log₁₀ scale, while one dose step is 0.48 on a log₁₀ scale (3-fold). Green tea is the exception: −28%, about 0.3 of a dose step.
- **Direction depends on which constituents dominate the mass.**
  - The µM value goes *down* when the most abundant constituents are heavier than average. In green tea, EGCG (458 g/mol) is 57% of the quantified mass. The unweighted mean is pulled down by small molecules present in tiny amounts: theophylline (0.15 mg/g), caffeine (1.1 mg/g) and theobromine (1.4 mg/g), all about 180–195 g/mol, and gallic acid (6.2 mg/g, 170 g/mol).
  - The µM value goes *up* when the abundant constituents are lighter than average, as in Kratom (mitragynine), Ashwagandha, Ephedra and Kava.
- **Ranking of botanicals by µM per µg/mL** stays mostly the same (Spearman ρ = 0.86). Green tea drops from 4th to 10th; the others move by 2 places at most.
- **The shift is not related to the fraction of mass quantified** (Spearman ρ = −0.10).

## Caveats

- `mw_weighted` only describes the quantified part of the extract, which ranges from 1.7% of the mass (Goldenseal) to 77% (Green tea). For poorly quantified extracts, assuming the rest has the same composition is weak.
- The quantification notes in the source file (`*` = sum of isomers, `**` = co-eluting constituents, maximum concentration assumed) are not corrected for.
- Two TSV rows (Aconite, Oleander) have no plant scode. They match no wells and are skipped.
- `docs/Ashwin_data/BSC_CP_botanical_pod_table_v5.xlsx` uses a newer constituent file (03122026), so its MW values differ for some botanicals. For example, Ashwagandha there has 22 constituents and a mean MW of 575.7, against 20 constituents and 619.2 in the 2024 TSV.
