#!/usr/bin/env python3
"""
07_recipe_kcal_validation.py

Corrected recipe-level kcal validation.

Compares:
  1. Sum of ingredient-level energy_kcal per recipe from recipe_ingredients.csv
  2. Recipe-level Energy (kCal)/(kcal) from recipe_nutrition.csv

Important:
- Explicitly excludes Energy (kJ) rows.
- Treats duplicate kcal labels/rows carefully: reports all matching kcal rows,
  checks whether duplicate kcal values agree, and uses the MEDIAN of valid
  kcal-labelled rows per recipe as the primary recipe-level value. This avoids
  summing duplicate representations of the same recipe energy.
- If a recipe has conflicting kcal-labelled values, it is flagged for review.
- Does not alter source files, rescale values, or generate substitution scores.

Run from BTP_2:
  /usr/local/bin/python3 GastroGraph-main/src/07_recipe_kcal_validation.py
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INGREDIENT_FILE = ROOT / "dataset" / "raw" / "RecipeDB_full" / "recipe_ingredients.csv"
NUTRITION_FILE = ROOT / "dataset" / "raw" / "RecipeDB_full" / "recipe_nutrition.csv"
OUT_DIR = ROOT / "GastroGraph-custom" / "data"

ABS_DIFF_THRESHOLD_KCAL = 100.0
REL_DIFF_THRESHOLD = 0.25
DUPLICATE_KCAL_TOLERANCE = 1.0  # kcal; only used to flag disagreement, not to modify values


def find_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.strip().lower() in lookup:
            return lookup[candidate.strip().lower()]
    return None


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def normalize_recipe_id(series):
    # Avoid joining integer IDs such as 123 with string IDs such as "123.0".
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INGREDIENT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INGREDIENT_FILE}")
    if not NUTRITION_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {NUTRITION_FILE}")

    ingredients = pd.read_csv(INGREDIENT_FILE, low_memory=False)
    nutrition = pd.read_csv(NUTRITION_FILE, low_memory=False)

    ing_recipe_col = find_column(ingredients, ["recipe_id", "recipeid", "recipe id"])
    ing_kcal_col = find_column(ingredients, ["energy_kcal", "energy (kcal)", "kcal"])
    ing_title_col = find_column(ingredients, ["recipe_title", "recipe title", "title"])

    nut_recipe_col = find_column(nutrition, ["recipe_id", "recipeid", "recipe id"])
    nut_title_col = find_column(nutrition, ["recipe_title", "recipe title", "title"])
    nut_nutrient_col = find_column(nutrition, ["nutrient", "nutrient_name", "name"])
    nut_value_col = find_column(nutrition, ["quantity", "value", "amount", "nutrient_quantity"])

    required = [
        ("recipe_ingredients recipe ID", ing_recipe_col),
        ("recipe_ingredients energy_kcal", ing_kcal_col),
        ("recipe_nutrition recipe ID", nut_recipe_col),
        ("recipe_nutrition nutrient", nut_nutrient_col),
        ("recipe_nutrition quantity", nut_value_col),
    ]
    missing = [name for name, col in required if col is None]
    if missing:
        raise ValueError(
            "Could not identify required columns: " + ", ".join(missing)
            + f"\nrecipe_ingredients columns: {list(ingredients.columns)}"
            + f"\nrecipe_nutrition columns: {list(nutrition.columns)}"
        )

    ingredients["_recipe_key"] = normalize_recipe_id(ingredients[ing_recipe_col])
    ingredients["_ingredient_kcal"] = to_numeric(ingredients[ing_kcal_col])
    nutrition["_recipe_key"] = normalize_recipe_id(nutrition[nut_recipe_col])
    nutrition["_nutrient_label"] = nutrition[nut_nutrient_col].astype("string").str.strip()
    nutrition["_nutrient_label_norm"] = (
        nutrition["_nutrient_label"].str.lower().str.replace(r"\s+", " ", regex=True)
    )
    nutrition["_nutrient_value"] = to_numeric(nutrition[nut_value_col])

    # Exact kcal labels only: allow kCal/kcal and harmless whitespace,
    # but exclude kJ and labels such as "Energy (kJ)".
    # The regex is anchored to energy labels to avoid matching unrelated nutrients.
    kcal_mask = nutrition["_nutrient_label_norm"].str.fullmatch(
        r"energy\s*\(\s*kcal\s*\)", na=False
    )
    energy_kcal_rows = nutrition.loc[kcal_mask].copy()

    if energy_kcal_rows.empty:
        labels = nutrition[nut_nutrient_col].dropna().astype(str).drop_duplicates().head(50).tolist()
        raise ValueError(
            "No exact 'Energy (kcal)'/'Energy (kCal)' rows found. "
            f"Example nutrient labels: {labels}"
        )

    # Aggregate ingredient kcal values by recipe. Missing values are not treated as zero.
    ingredient_agg = (
        ingredients.groupby("_recipe_key", dropna=False)
        .agg(
            ingredient_rows=("_ingredient_kcal", "size"),
            ingredient_kcal_nonmissing=("_ingredient_kcal", "count"),
            ingredient_kcal_sum=("_ingredient_kcal", "sum"),
            ingredient_kcal_median=("_ingredient_kcal", "median"),
            ingredient_kcal_max=("_ingredient_kcal", "max"),
        )
        .reset_index()
    )
    ingredient_agg.loc[
        ingredient_agg["ingredient_kcal_nonmissing"] == 0, "ingredient_kcal_sum"
    ] = np.nan

    # Summarize kcal rows per recipe without adding duplicate labels together.
    # Median is the primary comparison value; min/max and row count expose duplicates.
    kcal_agg = (
        energy_kcal_rows.groupby("_recipe_key", dropna=False)
        .agg(
            recipe_kcal_rows=("_nutrient_value", "size"),
            recipe_kcal_nonmissing=("_nutrient_value", "count"),
            recipe_energy_kcal=("_nutrient_value", "median"),
            recipe_energy_kcal_min=("_nutrient_value", "min"),
            recipe_energy_kcal_max=("_nutrient_value", "max"),
            recipe_energy_labels=("_nutrient_label", lambda s: " | ".join(sorted(set(s.dropna().astype(str))))),
        )
        .reset_index()
    )
    kcal_agg.loc[kcal_agg["recipe_kcal_nonmissing"] == 0, "recipe_energy_kcal"] = np.nan

    kcal_agg["duplicate_kcal_rows_flag"] = kcal_agg["recipe_kcal_rows"] > 1
    kcal_agg["conflicting_kcal_rows_flag"] = (
        (kcal_agg["recipe_energy_kcal_max"] - kcal_agg["recipe_energy_kcal_min"])
        > DUPLICATE_KCAL_TOLERANCE
    )

    # Attach one recipe title for easier inspection.
    title_parts = []
    if ing_title_col:
        title_parts.append(ingredients[["_recipe_key", ing_title_col]].rename(columns={ing_title_col: "_title"}))
    if nut_title_col:
        title_parts.append(nutrition[["_recipe_key", nut_title_col]].rename(columns={nut_title_col: "_title"}))
    if title_parts:
        titles = pd.concat(title_parts, ignore_index=True)
        titles["_title"] = titles["_title"].astype("string").str.strip()
        titles = titles[titles["_title"].notna() & (titles["_title"] != "")]
        titles = titles.drop_duplicates("_recipe_key").rename(columns={"_title": "recipe_title"})
    else:
        titles = pd.DataFrame(columns=["_recipe_key", "recipe_title"])

    comparison = ingredient_agg.merge(kcal_agg, on="_recipe_key", how="outer")
    comparison = comparison.merge(titles, on="_recipe_key", how="left")

    comparison["ingredient_minus_recipe_kcal"] = (
        comparison["ingredient_kcal_sum"] - comparison["recipe_energy_kcal"]
    )
    comparison["absolute_difference_kcal"] = comparison["ingredient_minus_recipe_kcal"].abs()
    comparison["ingredient_to_recipe_ratio"] = (
        comparison["ingredient_kcal_sum"] / comparison["recipe_energy_kcal"].replace(0, np.nan)
    )
    comparison["relative_difference"] = (
        comparison["ingredient_minus_recipe_kcal"]
        / comparison["recipe_energy_kcal"].abs().replace(0, np.nan)
    )
    comparison["absolute_relative_difference"] = comparison["relative_difference"].abs()

    has_both = comparison["ingredient_kcal_sum"].notna() & comparison["recipe_energy_kcal"].notna()
    comparison["comparison_status"] = np.select(
        [
            comparison["ingredient_kcal_sum"].isna() & comparison["recipe_energy_kcal"].notna(),
            comparison["ingredient_kcal_sum"].notna() & comparison["recipe_energy_kcal"].isna(),
            has_both,
        ],
        [
            "missing ingredient kcal total",
            "missing recipe-level kcal",
            "compared",
        ],
        default="no kcal data",
    )

    comparison["large_difference_flag"] = (
        has_both
        & (comparison["absolute_difference_kcal"] > ABS_DIFF_THRESHOLD_KCAL)
        & (comparison["absolute_relative_difference"] > REL_DIFF_THRESHOLD)
    )
    comparison["multiple_kcal_rows_flag"] = comparison["recipe_kcal_rows"].fillna(0) > 1
    comparison["conflicting_kcal_rows_flag"] = comparison["conflicting_kcal_rows_flag"].fillna(False)
    comparison["negative_ingredient_sum_flag"] = comparison["ingredient_kcal_sum"] < 0
    comparison["negative_recipe_energy_flag"] = comparison["recipe_energy_kcal"] < 0

    compared = comparison.loc[has_both].copy()
    valid_ratios = compared.loc[
        compared["ingredient_to_recipe_ratio"].notna()
        & np.isfinite(compared["ingredient_to_recipe_ratio"])
        & (compared["ingredient_to_recipe_ratio"] > 0),
        "ingredient_to_recipe_ratio",
    ]

    # Scale-factor diagnostics only. No factor is applied.
    scale_factors = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    scale_rows = []
    for factor in scale_factors:
        if valid_ratios.empty:
            break
        adjusted = valid_ratios * factor
        scale_rows.append({
            "diagnostic_factor_on_ingredient_sum": factor,
            "median_adjusted_ratio": float(adjusted.median()),
            "median_absolute_log10_ratio_error": float(
                np.median(np.abs(np.log10(adjusted.clip(lower=1e-12))))
            ),
            "fraction_within_25pct": float(((adjusted >= 0.75) & (adjusted <= 1.25)).mean()),
            "warning": "diagnostic only; do not apply without validating source semantics",
        })
    scale_check = pd.DataFrame(scale_rows)

    # Keep exact kcal rows with recipe IDs for direct review.
    detail_cols = [c for c in [nut_recipe_col, nut_title_col, nut_nutrient_col, nut_value_col] if c]
    energy_kcal_rows[detail_cols].to_csv(
        OUT_DIR / "nutrient_kcal_recipe_energy_rows.csv", index=False
    )

    comparison_cols = [
        "_recipe_key", "recipe_title", "ingredient_rows", "ingredient_kcal_nonmissing",
        "ingredient_kcal_sum", "ingredient_kcal_median", "ingredient_kcal_max",
        "recipe_kcal_rows", "recipe_kcal_nonmissing", "recipe_energy_kcal",
        "recipe_energy_kcal_min", "recipe_energy_kcal_max",
        "ingredient_minus_recipe_kcal", "absolute_difference_kcal",
        "ingredient_to_recipe_ratio", "relative_difference", "comparison_status",
        "large_difference_flag", "multiple_kcal_rows_flag", "conflicting_kcal_rows_flag",
        "negative_ingredient_sum_flag", "negative_recipe_energy_flag",
    ]
    comparison[[c for c in comparison_cols if c in comparison.columns]].to_csv(
        OUT_DIR / "nutrient_kcal_recipe_comparison.csv", index=False
    )
    comparison.loc[comparison["large_difference_flag"]].sort_values(
        "absolute_difference_kcal", ascending=False
    ).to_csv(OUT_DIR / "nutrient_kcal_recipe_large_differences.csv", index=False)
    comparison.loc[comparison["multiple_kcal_rows_flag"]].to_csv(
        OUT_DIR / "nutrient_kcal_recipe_multiple_energy_rows.csv", index=False
    )
    comparison.loc[comparison["conflicting_kcal_rows_flag"]].to_csv(
        OUT_DIR / "nutrient_kcal_recipe_conflicting_energy_rows.csv", index=False
    )
    scale_check.to_csv(OUT_DIR / "nutrient_kcal_recipe_scale_check.csv", index=False)

    # Statistics
    if not compared.empty:
        difference_median = compared["ingredient_minus_recipe_kcal"].median()
        absolute_difference_median = compared["absolute_difference_kcal"].median()
        absolute_relative_median = compared["absolute_relative_difference"].replace(
            [np.inf, -np.inf], np.nan
        ).median()
        ratio_median = valid_ratios.median() if not valid_ratios.empty else np.nan
        within_band = (
            ((valid_ratios >= 0.75) & (valid_ratios <= 1.25)).mean()
            if not valid_ratios.empty else np.nan
        )
        spearman = compared[["ingredient_kcal_sum", "recipe_energy_kcal"]].corr(
            method="spearman"
        ).iloc[0, 1]
    else:
        difference_median = absolute_difference_median = absolute_relative_median = np.nan
        ratio_median = within_band = spearman = np.nan

    lines = [
        "CORRECTED RECIPE-LEVEL KCAL VALIDATION SUMMARY",
        "=" * 48,
        f"Ingredient source rows: {len(ingredients):,}",
        f"Recipe nutrition source rows: {len(nutrition):,}",
        f"Exact Energy (kcal)/(kCal) rows selected: {len(energy_kcal_rows):,}",
        f"Energy (kJ) rows explicitly excluded: {int(nutrition['_nutrient_label_norm'].str.fullmatch(r'energy\\s*\\(\\s*kj\\s*\\)', na=False).sum()):,}",
        f"Unique recipe IDs in ingredient file: {ingredients['_recipe_key'].nunique(dropna=True):,}",
        f"Unique recipe IDs with kcal rows: {energy_kcal_rows['_recipe_key'].nunique(dropna=True):,}",
        f"Recipe IDs with both ingredient kcal sum and recipe kcal: {len(compared):,}",
        f"Recipe IDs missing ingredient kcal sum but with recipe kcal: {int((comparison['comparison_status'] == 'missing ingredient kcal total').sum()):,}",
        f"Recipe IDs with ingredient kcal sum but missing recipe kcal: {int((comparison['comparison_status'] == 'missing recipe-level kcal').sum()):,}",
        f"Recipe IDs with multiple kcal rows: {int(comparison['multiple_kcal_rows_flag'].sum()):,}",
        f"Recipe IDs with conflicting kcal rows (> {DUPLICATE_KCAL_TOLERANCE:g} kcal range): {int(comparison['conflicting_kcal_rows_flag'].sum()):,}",
        "",
        "COMPARISON (ingredient kcal sum vs recipe-level kcal median)",
        f"Median ingredient-minus-recipe difference (kcal): {float(difference_median):.4f}",
        f"Median absolute difference (kcal): {float(absolute_difference_median):.4f}",
        f"Median absolute relative difference: {float(absolute_relative_median):.4f}",
        f"Median ingredient/recipe ratio: {float(ratio_median):.6g}",
        f"Fraction of ratios within 0.75-1.25: {float(within_band):.4f}",
        f"Spearman correlation: {float(spearman):.4f}",
        f"Large-difference flags (abs diff > {ABS_DIFF_THRESHOLD_KCAL:g} AND relative diff > {REL_DIFF_THRESHOLD:.0%}): {int(comparison['large_difference_flag'].sum()):,}",
        "",
        "DIAGNOSTIC SCALE CHECK",
        "These factors multiply ingredient-level sums for diagnosis only. No factor is applied.",
    ]
    if not scale_check.empty:
        best = scale_check.sort_values("median_absolute_log10_ratio_error").iloc[0]
        lines.append(
            f"Factor with smallest median log-ratio error: {best['diagnostic_factor_on_ingredient_sum']:g} "
            f"(median adjusted ratio={best['median_adjusted_ratio']:.6g}, "
            f"fraction within 25%={best['fraction_within_25pct']:.4f})"
        )
        lines.append(scale_check.to_string(index=False))
    else:
        lines.append("Not enough valid positive ratios for scale diagnostics.")

    lines.extend([
        "",
        "INTERPRETATION WARNINGS",
        "- Only exact Energy (kcal)/(kCal) labels are selected; Energy (kJ) is excluded.",
        "- Multiple kcal rows are summarized by their median for comparison, not summed.",
        "- Conflicting kcal rows are flagged and should be investigated.",
        "- Ingredient energy sums may still differ because of missing data or source conventions.",
        "- Correlation or a plausible scale factor does not prove that scaling is appropriate.",
        "- No source values are changed, rescaled, imputed, or deleted.",
        "",
        "OUTPUTS",
        "- nutrient_kcal_recipe_validation_summary.txt",
        "- nutrient_kcal_recipe_comparison.csv",
        "- nutrient_kcal_recipe_large_differences.csv",
        "- nutrient_kcal_recipe_multiple_energy_rows.csv",
        "- nutrient_kcal_recipe_conflicting_energy_rows.csv",
        "- nutrient_kcal_recipe_energy_rows.csv",
        "- nutrient_kcal_recipe_scale_check.csv",
    ])
    summary_path = OUT_DIR / "nutrient_kcal_recipe_validation_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 76)
    print("CORRECTED RECIPE-LEVEL KCAL VALIDATION")
    print("=" * 76)
    print(f"Ingredient source rows                    : {len(ingredients):,}")
    print(f"Exact kcal rows selected                  : {len(energy_kcal_rows):,}")
    print(f"Energy (kJ) rows excluded                 : {int(nutrition['_nutrient_label_norm'].str.fullmatch(r'energy\\s*\\(\\s*kj\\s*\\)', na=False).sum()):,}")
    print(f"Recipes with both kcal totals             : {len(compared):,}")
    print(f"Recipes with multiple kcal rows           : {int(comparison['multiple_kcal_rows_flag'].sum()):,}")
    print(f"Recipes with conflicting kcal rows        : {int(comparison['conflicting_kcal_rows_flag'].sum()):,}")
    print(f"Recipes flagged for large differences     : {int(comparison['large_difference_flag'].sum()):,}")
    print(f"Median ingredient/recipe ratio            : {float(ratio_median):.6g}")
    print(f"Spearman correlation                      : {float(spearman):.4f}")
    print("\nSaved outputs:")
    for name in [
        "nutrient_kcal_recipe_validation_summary.txt",
        "nutrient_kcal_recipe_comparison.csv",
        "nutrient_kcal_recipe_large_differences.csv",
        "nutrient_kcal_recipe_multiple_energy_rows.csv",
        "nutrient_kcal_recipe_conflicting_energy_rows.csv",
        "nutrient_kcal_recipe_energy_rows.csv",
        "nutrient_kcal_recipe_scale_check.csv",
    ]:
        print(f"  {OUT_DIR / name}")
    print("\nAudit only: source values were not changed or rescaled.")


if __name__ == "__main__":
    main()
