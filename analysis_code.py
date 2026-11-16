# Manchester Housing EDA - COM7024 Programming for Data Analytics
# Full Python code from Manchester_Housing_EDA.ipynb

# ==================================================
# Section 1 - Import libraries
# ==================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==================================================
# Section 2 - Import dataset
# ==================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# utf-8-sig removes a hidden character at the start of the file
RAW_FILE = "Manchester_house_Dataset[3678].csv"
raw = pd.read_csv(RAW_FILE, encoding="utf-8-sig")

print("Rows, columns:", raw.shape)
print(raw.head())

# ==================================================
# Section 3 - Check the data types
# ==================================================
raw.info()

# the type each column should really be
expected = {
    "Unnamed: 0": "redundant row index -> drop",
    "id": "identifier -> string (not a quantity)",
    "price": "int (GBP)",
    "bedrooms": "int (count)",
    "bathrooms": "float (half-baths exist)",
    "sqft_living": "int",
    "sqft_total": "int",
    "floors": "float (1.5 floors exist)",
    "waterfront": "category / boolean (0/1)",
    "view": "ordinal category 0-4",
    "condition": "ordinal category 1-5",
    "grade": "ordinal category 1-13",
    "built": "int (year)",
    "renovated": "int (year, 0 = never)",
    "living_area": "int",
}
dtype_check = pd.DataFrame({"pandas_dtype": raw.dtypes.astype(str),
                            "expected_type": pd.Series(expected)})
print(dtype_check)

# ==================================================
# Section 4 - Basic statistics on all variables (before cleaning)
# ==================================================
initial_stats = raw.describe().T
initial_stats["skew"] = raw.skew(numeric_only=True)
initial_stats["missing"] = raw.isna().sum()
print(initial_stats.round(2))

# Missing values and duplicates
missing_before = raw.isna().sum()
print("Missing values per column:\n", missing_before[missing_before > 0])
print("\nTotal missing cells:", int(missing_before.sum()))

# leave out the index column, otherwise every row looks different
dup_before = raw.drop(columns="Unnamed: 0").duplicated().sum()
print("Exact duplicate rows (ignoring index):", dup_before)
print("Repeated property ids:", raw["id"].duplicated().sum())

print(raw[raw.drop(columns="Unnamed: 0").duplicated(keep=False)])

# max bedrooms is 33, so check all houses with 10+ bedrooms
print(raw.loc[raw["bedrooms"] >= 10, ["id", "price", "bedrooms", "bathrooms", "sqft_living"]])

# ==================================================
# Section 5 - Pre-processing (cleaning the data)
# ==================================================
# copy so the raw data stays unchanged for the before/after comparison
df = raw.copy()

df = df.drop(columns="Unnamed: 0")

df = df.drop_duplicates().reset_index(drop=True)
# repeated ids with different prices are kept (real re-sales)
print("Rows after removing duplicates:", len(df))

# 33 bedrooms in 1,620 sqft is a typo for 3
df.loc[df["bedrooms"] == 33, "bedrooms"] = 3

# fill the few missing values with the median, which outliers don't affect
for col in ["bedrooms", "bathrooms"]:
    df[col] = df[col].fillna(df[col].median())
print("Missing cells after imputation:", int(df.isna().sum().sum()))

# Data type corrections
df["id"] = df["id"].astype(str)
df["bedrooms"] = df["bedrooms"].astype(int)
df["waterfront"] = df["waterfront"].astype(bool)
for col in ["view", "condition", "grade"]:
    df[col] = pd.Categorical(df[col], ordered=True)

# New columns for the three questions
df["waterfront_label"] = np.where(df["waterfront"], "Waterfront", "Non-waterfront")
df["is_renovated"] = df["renovated"] > 0
df["age"] = 2015 - df["built"]
df["decade_built"] = (df["built"] // 10) * 10
df["price_per_sqft"] = df["price"] / df["sqft_living"]
df["log_price"] = np.log(df["price"])
print(df.dtypes)

# Outlier detection using IQR (Tukey, 1977)
# outliers are counted, not removed, because they are real sales
def iqr_outliers(s):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return int(((s < low) | (s > high)).sum())

rows = {}
for col in ["price", "sqft_living", "sqft_total", "living_area"]:
    n = iqr_outliers(df[col])
    rows[col] = {"outliers": n, "% of rows": 100 * n / len(df)}
outlier_table = pd.DataFrame(rows).T
print(outlier_table)

# ==================================================
# Section 6 - Basic statistics after cleaning
# ==================================================
post_stats = df.describe().T
post_stats["skew"] = df.skew(numeric_only=True)
print(post_stats.round(2))

# Before vs after cleaning
new_cols = [c for c in df.columns if c not in raw.columns]
comparison = pd.DataFrame({
    "Before": [raw.shape[0], raw.shape[1], int(raw.isna().sum().sum()), int(dup_before),
               int(raw["bedrooms"].max()), str(raw["bedrooms"].dtype),
               str(raw["waterfront"].dtype), str(raw["id"].dtype)],
    "After": [df.shape[0], raw.shape[1] - 1, int(df.isna().sum().sum()),
              int(df.drop(columns=new_cols).duplicated().sum()),
              int(df["bedrooms"].max()), str(df["bedrooms"].dtype),
              str(df["waterfront"].dtype), str(df["id"].dtype)],
}, index=["Rows", "Original columns", "Missing cells", "Duplicate rows",
          "Max bedrooms", "bedrooms dtype", "waterfront dtype", "id dtype"])
print(comparison)

# ==================================================
# Section 7 - Investigation 1: Waterfront properties
# ==================================================
wf_summary = df.groupby("waterfront_label").agg(
    count=("price", "size"),
    median_price=("price", "median"),
    mean_price=("price", "mean"),
    median_sqft_living=("sqft_living", "median"),
    median_price_per_sqft=("price_per_sqft", "median"),
)
wf_summary["share_%"] = 100 * wf_summary["count"] / len(df)
print(wf_summary.round(2))

# median used because of outliers
med = wf_summary["median_price"]
premium = med["Waterfront"] - med["Non-waterfront"]
print(f"Median premium: £{premium:,.0f} ({100 * premium / med['Non-waterfront']:.0f}% higher)")

# price is skewed, so Mann-Whitney U is used instead of a t-test
wf = df.loc[df["waterfront"], "price"]
nwf = df.loc[~df["waterfront"], "price"]
u, p = stats.mannwhitneyu(wf, nwf, alternative="two-sided")
r_rb = 1 - 2 * u / (len(wf) * len(nwf))   # effect size (rank-biserial)
print(f"Mann-Whitney U = {u:,.0f}, p = {p:.2e}, rank-biserial r = {abs(r_rb):.2f}")

# Price boxplot by waterfront status
# log scale, otherwise the non-waterfront box is squashed flat
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="waterfront_label", y="price", hue="waterfront_label",
            legend=False)
plt.yscale("log")
plt.title("Property price by waterfront status (log scale)")
plt.xlabel("")
plt.ylabel("Price (£, log scale)")
plt.tight_layout()
plt.show()

# price per sqft takes house size out of the comparison
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="waterfront_label", y="price_per_sqft", hue="waterfront_label",
               legend=False, cut=0)
plt.title("Price per square foot by waterfront status")
plt.xlabel("")
plt.ylabel("Price per sqft (£)")
plt.tight_layout()
plt.show()

# % of each group in each view rating
view_ct = pd.crosstab(df["waterfront_label"], df["view"], normalize="index") * 100
print(view_ct.round(1))

# ==================================================
# Section 8 - Investigation 2: Floor space
# ==================================================
# Floor space distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["sqft_living"], bins=60, kde=True, color="#4C72B0")
plt.axvline(df["sqft_living"].median(), color="black", ls="--", label="Median")
plt.title("Distribution of living floor space")
plt.xlabel("Living floor space (sqft)")
plt.ylabel("Number of properties")
plt.legend()
plt.tight_layout()
plt.show()

# Floor space vs price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sqft_living", y="price", hue="waterfront_label",
                alpha=0.4, s=12)
sns.regplot(data=df, x="sqft_living", y="price", scatter=False, color="black",
            line_kws={"lw": 1.5, "label": "Linear trend"})
plt.title("Living floor space vs price")
plt.xlabel("Living floor space (sqft)")
plt.ylabel("Price (£ millions)")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v / 1e6:.0f}m"))
plt.legend(title="")
plt.tight_layout()
plt.show()

# Pearson and Spearman correlation with price
rows = []
for col in ["sqft_living", "living_area", "sqft_total"]:
    pr, pp = stats.pearsonr(df[col], df["price"])
    sr, sp = stats.spearmanr(df[col], df["price"])
    rows.append({"variable": col, "pearson_r": pr, "pearson_p": pp,
                 "spearman_rho": sr, "spearman_p": sp})
space_corr = pd.DataFrame(rows).set_index("variable")
print(space_corr)

# Correlation heatmap
num_cols = ["price", "sqft_living", "living_area", "sqft_total", "bedrooms",
            "bathrooms", "floors", "built", "age"]
heat_data = df[num_cols].copy()
heat_data["grade"] = df["grade"].astype(int)
heat_data["waterfront"] = df["waterfront"].astype(int)
corr = heat_data.corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True,
            cbar_kws={"label": "Pearson r"})
plt.title("Correlation matrix of key variables")
plt.tight_layout()
plt.show()

# ==================================================
# Section 9 - Investigation 3: Is year built correlated with price?
# ==================================================
pr, pp = stats.pearsonr(df["built"], df["price"])
sr, sp = stats.spearmanr(df["built"], df["price"])
print(f"Pearson r = {pr:.3f} (p = {pp:.1e});  Spearman rho = {sr:.3f} (p = {sp:.1e})")

# group by decade because a scatter of 20,000 points is too messy
decade = df.groupby("decade_built").agg(median_price=("price", "median"),
                                        median_sqft=("sqft_living", "median"),
                                        n=("price", "size"))
decade["median_price_per_sqft"] = df.groupby("decade_built")["price_per_sqft"].median()

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(decade.index, decade["median_price"], marker="o", color="#4C72B0", label="Median price")
ax1.set_xlabel("Decade built")
ax1.set_ylabel("Median price (£)", color="#4C72B0")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
ax2 = ax1.twinx()
ax2.plot(decade.index, decade["median_price_per_sqft"], marker="s", color="#DD8452",
         label="Median price per sqft")
ax2.set_ylabel("Median price per sqft (£)", color="#DD8452")
ax2.grid(False)
plt.title("Median price and price per sqft by decade built")
plt.tight_layout()
plt.show()
print(decade)

# partial correlation: removes the effect of size, as newer houses are bigger
def residuals(y, x):
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)

r_partial, p_partial = stats.pearsonr(residuals(df["price"], df["sqft_living"]),
                                      residuals(df["built"], df["sqft_living"]))
print(f"Partial r (built, price | sqft_living) = {r_partial:.3f}, p = {p_partial:.1e}")

# Renovated vs never renovated
reno = df.groupby("is_renovated")["price"].agg(["size", "median"])
reno.index = ["Never renovated", "Renovated"]
print(reno)

# ==================================================
# Section 10 - Additional insights (beyond the brief)
# ==================================================
# Insight 1: bedrooms vs price
# only houses of similar size (1,800-2,200 sqft) so the comparison is fair
similar_size = df[(df["sqft_living"] >= 1800) & (df["sqft_living"] <= 2200)]
bed_table = similar_size.groupby("bedrooms").agg(homes=("price", "size"),
                                                  median_price=("price", "median"))
# at least 20 houses per group so the medians are reliable
bed_table = bed_table[bed_table["homes"] >= 20]

rho, p_bed = stats.spearmanr(similar_size["bedrooms"], similar_size["price"])
print(f"Spearman rho (bedrooms vs price, similar size) = {rho:.3f}, p = {p_bed:.1e}")
print(bed_table)

# Insight 2: renovation uplift by age group
df["age_group"] = pd.cut(df["built"], bins=[1899, 1939, 1969, 2015],
                         labels=["Before 1940", "1940-1969", "1970 onwards"])
reno_age = df.pivot_table(index="age_group", columns="is_renovated", values="price",
                          aggfunc="median", observed=True)
reno_age.columns = ["Never renovated", "Renovated"]
reno_age["uplift_%"] = 100 * (reno_age["Renovated"] / reno_age["Never renovated"] - 1)
reno_age["renovated_homes"] = df[df["is_renovated"]].groupby("age_group", observed=True).size()
print(reno_age)

# Insight 3: view premium without waterfront
# waterfront homes removed so water can't explain the price
no_water = df[~df["waterfront"]]
view_table = no_water.groupby("view", observed=True).agg(
    homes=("price", "size"),
    median_price=("price", "median"),
    median_price_per_sqft=("price_per_sqft", "median"))

plt.figure(figsize=(8, 5))
plt.bar(view_table.index.astype(str), view_table["median_price"], color="#4C72B0")
for i, v in enumerate(view_table["median_price"]):
    plt.text(i, v + 15000, f"£{v / 1000:,.0f}k", ha="center")
plt.title("Median price by view rating (non-waterfront homes only)")
plt.xlabel("View rating (0 = none, 4 = excellent)")
plt.ylabel("Median price (£)")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
plt.tight_layout()
plt.show()
print(view_table)

# Insight 4: building age bands
df["age_band"] = pd.cut(df["age"], bins=[-1, 10, 25, 50, 75, 115],
                        labels=["0-10", "11-25", "26-50", "51-75", "76+"])
age_table = df.groupby("age_band", observed=True).agg(
    homes=("price", "size"),
    median_price=("price", "median"),
    median_sqft=("sqft_living", "median"),
    median_price_per_sqft=("price_per_sqft", "median"))
print(age_table)

# ==================================================
# Section 11 - Summary of key results
# ==================================================
summary = pd.DataFrame({
    "Result": [
        f"Waterfront homes: {int(wf_summary.loc['Waterfront', 'count'])} "
        f"({wf_summary.loc['Waterfront', 'share_%']:.1f}% of stock)",
        f"Waterfront median premium: {100 * premium / med['Non-waterfront']:.0f}%",
        f"sqft_living vs price: Pearson r = {space_corr.loc['sqft_living', 'pearson_r']:.3f}",
        f"sqft_total vs price: Pearson r = {space_corr.loc['sqft_total', 'pearson_r']:.3f}",
        f"built vs price: Pearson r = {pr:.3f}, Spearman rho = {sr:.3f}",
        f"built vs price controlling for size: partial r = {r_partial:.3f}",
    ]})
print(summary)

