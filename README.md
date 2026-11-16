# Exploratory Data Analytics of the Manchester Housing Market

COM7024 Programming for Data Analytics (Arden University) · Student ID: 25350515

Exploratory data analysis for **Estate Management Plc**, focusing on:

1. **Waterfront properties**: size of the price premium (Mann-Whitney U test)
2. **Floor space**: correlation of living space, living area and plot size with price
3. **Year built vs price**: Pearson/Spearman correlation, decade trends and partial correlation controlling for size

## Contents

| File / folder | Description |
|---|---|
| `Manchester_Housing_EDA.ipynb` | Executed Jupyter notebook with all code, outputs and charts |
| `analysis_code.py` | The same code as a plain Python script |
| `figures/` | The charts used in the report (PNG), exported from the notebook |
| `tables/` | The result tables used in the report (CSV), exported from the notebook |
| `requirements.txt` | Python packages needed |

## How to reproduce

The dataset is Arden University course material, so it is **not** included in this repository.

```bash
pip install -r requirements.txt
# place "Manchester_house_Dataset[3678].csv" in this folder, then:
jupyter notebook Manchester_Housing_EDA.ipynb   # Kernel -> Restart & Run All
```

## Key results

| Question | Result |
|---|---|
| Waterfront premium | Median £1,365,000 vs £446,350 (+206%); Mann-Whitney p < 0.001, r = 0.77 |
| Floor space vs price | sqft_living r = 0.70; sqft_total r = 0.09 |
| Year built vs price | r = 0.04 overall; U-shaped by decade; partial r = −0.28 controlling for size |

### Additional insights (beyond the brief)

| Insight | Result |
|---|---|
| Bedrooms at the same floor space (1,800–2,200 sqft) | More bedrooms → lower price: 2-bed £527k vs 4-bed £420k (Spearman ρ = −0.11) |
| Renovation by age group | Largest uplift for 1940–1969 homes: +53% (vs +19% pre-1940) |
| View without waterfront | Excellent view £1.06m vs no view £430k (about 2.5×) |
