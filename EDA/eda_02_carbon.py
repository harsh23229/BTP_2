import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT = BASE_DIR / "dataset" / "processed" / "SustainableFoodDB_CarbonFootprints.csv"
OUT = BASE_DIR / "EDA" / "results" / "carbon"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

print("=" * 60)
print("SUSTAINABLEFOODDB CARBON EDA")
print("=" * 60)

print(f"\nShape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Numerical statistics
numeric = ["CarbonFootprint", "Similarity", "Count"]

print("\nNumerical statistics:")
print(df[numeric].describe().T)

print("\nSkewness:")
print(df[numeric].skew())

# Values outside expected similarity range
print("\nSimilarity values > 1:")
print(df[df["Similarity"] > 1][
    ["Ingredient", "SueatableIngredient", "Similarity"]
].head(20))

print(f"Total Similarity > 1: {(df['Similarity'] > 1).sum()}")

# Top ingredients by frequency
print("\nTop 20 ingredients by frequency:")
print(
    df.nlargest(20, "Count")[
        ["Ingredient", "SueatableIngredient", "Count", "CarbonFootprint"]
    ]
)

# Highest carbon footprint
print("\nTop 20 ingredients by carbon footprint:")
print(
    df.nlargest(20, "CarbonFootprint")[
        ["Ingredient", "SueatableIngredient", "CarbonFootprint", "Count"]
    ]
)

# Save statistics
stats = df[numeric].describe().T
stats["skewness"] = df[numeric].skew()
stats.to_csv(OUT / "carbon_statistics.csv")

# Save high-frequency and high-carbon tables
df.nlargest(20, "Count").to_csv(
    OUT / "top_frequency_ingredients.csv", index=False
)

df.nlargest(20, "CarbonFootprint").to_csv(
    OUT / "top_carbon_ingredients.csv", index=False
)

# Histograms
for col in numeric:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col].dropna(), bins=50)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(OUT / f"{col}_distribution.png", dpi=150)
    plt.close()

# Boxplots
for col in numeric:
    plt.figure(figsize=(8, 4))
    plt.boxplot(df[col].dropna(), vert=False)
    plt.xlabel(col)
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(OUT / f"{col}_boxplot.png", dpi=150)
    plt.close()

print(f"\nResults saved to: {OUT}")