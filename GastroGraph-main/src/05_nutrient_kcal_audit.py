"""
05_nutrient_kcal_audit.py

Extended audit for RecipeDB ingredient-level energy (kcal) and macronutrients.

Input:
  dataset/raw/RecipeDB_full/recipe_ingredients.csv

Outputs (GastroGraph-custom/data):
  nutrient_kcal_audit_summary.txt
  nutrient_kcal_audit_ingredient_summary.csv
  nutrient_kcal_audit_unit_summary.csv
  nutrient_kcal_audit_review_flags.csv
  nutrient_kcal_audit_high_variability_ingredients.csv
  nutrient_kcal_audit_extreme_records.csv
  nutrient_kcal_audit_macro_energy_consistency.csv
  nutrient_kcal_audit_duplicate_rows.csv
  nutrient_kcal_audit_recipe_duplicate_ingredients.csv

This script only audits and reports. It does not delete data, repair values,
or calculate a substitution score.
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT_FILENAME = "recipe_ingredients.csv"

REQUIRED_COLUMNS = [
    "recipe_id",
    "recipe_title",
    "ingredient_name",
    "quantity",
    "unit",
    "state",
    "energy_kcal",
    "carbohydrates",
    "protein_g",
    "fat_g",
]

NUMERIC_COLUMNS = [
    "quantity",
    "energy_kcal",
    "carbohydrates",
    "protein_g",
    "fat_g",
]

# Review thresholds only; not automatic data-cleaning rules.
EXTREME_KCAL_ABSOLUTE_THRESHOLD = 5000.0
MACRO_ENERGY_ABSOLUTE_TOLERANCE = 50.0
MACRO_ENERGY_RELATIVE_TOLERANCE = 0.25


def find_input_file(project_root: Path) -> Path:
    likely_paths = [
        project_root / "dataset" / "raw" / "RecipeDB_full" / INPUT_FILENAME,
        project_root / "dataset" / "raw" / INPUT_FILENAME,
        project_root / "dataset" / "processed" / INPUT_FILENAME,
        project_root / INPUT_FILENAME,
    ]
    for path in likely_paths:
        if path.is_file():
            return path

    matches = [
        p for p in project_root.rglob(INPUT_FILENAME)
        if not any(part.startswith(".") for part in p.parts)
    ]
    if matches:
        matches.sort(
            key=lambda p: (
                "raw" not in str(p).lower(),
                "recipedb_full" not in str(p).lower(),
                len(str(p)),
            )
        )
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {INPUT_FILENAME} beneath {project_root}.\n"
        "Expected e.g. BTP_2/dataset/raw/RecipeDB_full/recipe_ingredients.csv"
    )


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = denominator.notna() & denominator.ne(0) & numerator.notna()
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def main() -> None:
    # Script path: BTP_2/GastroGraph-main/src/05_nutrient_kcal_audit.py
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "GastroGraph-custom" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = find_input_file(project_root)

    print("=" * 76)
    print("RECIPEDB INGREDIENT-LEVEL KCAL AUDIT (EXTENDED)")
    print("=" * 76)
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")

    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    for col in ["ingredient_name", "recipe_title", "unit", "state"]:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col].eq(""), col] = pd.NA

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Base coverage
    n_rows = len(df)
    n_recipes = int(df["recipe_id"].nunique(dropna=True))
    n_ingredients = int(df["ingredient_name"].nunique(dropna=True))
    missing_by_column = df[REQUIRED_COLUMNS].isna().sum()

    # Exact duplicates are reported, not automatically removed.
    exact_duplicate_mask = df.duplicated(keep=False)
    exact_duplicate_rows = int(df.duplicated().sum())
    duplicate_rows = df.loc[exact_duplicate_mask].copy()
    duplicate_rows.to_csv(output_dir / "nutrient_kcal_audit_duplicate_rows.csv", index=False)

    # Repeated recipe+normalized ingredient names may be legitimate, so inspect them.
    df["ingredient_key"] = normalize_text(df["ingredient_name"])
    recipe_ingredient_key = df["recipe_id"].astype("string") + "||" + df["ingredient_key"].fillna("")
    repeated_pair_mask = recipe_ingredient_key.duplicated(keep=False) & df["ingredient_key"].notna()
    recipe_duplicate_ingredients = df.loc[repeated_pair_mask].copy()
    recipe_duplicate_ingredients.to_csv(
        output_dir / "nutrient_kcal_audit_recipe_duplicate_ingredients.csv",
        index=False,
    )
    repeated_pair_rows = int(repeated_pair_mask.sum())

    kcal = df["energy_kcal"]
    quantity = df["quantity"]
    carbs = df["carbohydrates"]
    protein = df["protein_g"]
    fat = df["fat_g"]

    # Energy approximation from macros: 4 kcal/g carbs + 4 kcal/g protein + 9 kcal/g fat.
    # This is a diagnostic; food-specific factors, rounding, fiber/alcohol, and data
    # conventions can explain differences.
    macros_available = carbs.notna() & protein.notna() & fat.notna()
    macro_energy = 4.0 * carbs + 4.0 * protein + 9.0 * fat
    df["macro_implied_kcal"] = macro_energy
    df["macro_energy_difference_kcal"] = kcal - macro_energy
    df["macro_energy_abs_difference_kcal"] = (kcal - macro_energy).abs()
    denominator = pd.concat([kcal.abs(), macro_energy.abs()], axis=1).max(axis=1)
    df["macro_energy_relative_difference"] = safe_divide(
        df["macro_energy_abs_difference_kcal"], denominator
    )

    comparable_macro_rows = kcal.notna() & macros_available
    macro_diff_large = (
        comparable_macro_rows
        & (
            df["macro_energy_abs_difference_kcal"].gt(MACRO_ENERGY_ABSOLUTE_TOLERANCE)
            & df["macro_energy_relative_difference"].gt(MACRO_ENERGY_RELATIVE_TOLERANCE)
        )
    )

    # Kcal and quantity validation.
    kcal_missing = int(kcal.isna().sum())
    kcal_zero = int(kcal.eq(0).sum())
    kcal_negative = int(kcal.lt(0).sum())
    kcal_nonfinite = int((~np.isfinite(kcal.dropna())).sum())
    quantity_missing = int(quantity.isna().sum())
    quantity_zero = int(quantity.eq(0).sum())
    quantity_negative = int(quantity.lt(0).sum())

    kcal_quantiles = kcal.quantile([0, .01, .25, .5, .75, .90, .95, .99, 1])
    kcal_iqr = kcal.quantile(.75) - kcal.quantile(.25)
    kcal_upper_iqr = kcal.quantile(.75) + 3 * kcal_iqr
    high_iqr_mask = kcal.gt(kcal_upper_iqr) if pd.notna(kcal_upper_iqr) else pd.Series(False, index=df.index)
    extreme_mask = kcal.gt(EXTREME_KCAL_ABSOLUTE_THRESHOLD)
    extreme_records = df.loc[extreme_mask].copy()
    extreme_records = extreme_records.sort_values("energy_kcal", ascending=False)
    extreme_records.to_csv(
        output_dir / "nutrient_kcal_audit_extreme_records.csv", index=False
    )

    # Macro-energy consistency table with source values and explicit review flag.
    macro_check_cols = [
        "recipe_id", "recipe_title", "ingredient_name", "quantity", "unit",
        "energy_kcal", "carbohydrates", "protein_g", "fat_g",
        "macro_implied_kcal", "macro_energy_difference_kcal",
        "macro_energy_abs_difference_kcal", "macro_energy_relative_difference",
    ]
    macro_check = df.loc[comparable_macro_rows, macro_check_cols].copy()
    macro_check["macro_energy_review_flag"] = np.where(
        macro_diff_large.loc[macro_check.index],
        "LARGE_KCAL_MACRO_DISAGREEMENT",
        "WITHIN_DIAGNOSTIC_TOLERANCE",
    )
    macro_check.to_csv(
        output_dir / "nutrient_kcal_audit_macro_energy_consistency.csv",
        index=False,
    )

    # Unit summary, keeping labels as provided: "cup", "cups", "tablespoon", etc.
    unit_label = df["unit"].fillna("[MISSING/UNSPECIFIED]").astype(str)
    unit_summary = (
        df.assign(unit_label=unit_label)
        .groupby("unit_label", dropna=False)
        .agg(
            rows=("recipe_id", "size"),
            unique_recipes=("recipe_id", "nunique"),
            unique_ingredients=("ingredient_name", "nunique"),
            kcal_missing=("energy_kcal", lambda s: int(s.isna().sum())),
            kcal_median=("energy_kcal", "median"),
            kcal_mean=("energy_kcal", "mean"),
            quantity_missing=("quantity", lambda s: int(s.isna().sum())),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    unit_summary.to_csv(output_dir / "nutrient_kcal_audit_unit_summary.csv", index=False)

    # Ingredient-level statistics are descriptive of recorded per-recipe contributions.
    ingredient_summary = (
        df.dropna(subset=["ingredient_name"])
        .groupby("ingredient_name", dropna=False)
        .agg(
            row_count=("recipe_id", "size"),
            recipe_count=("recipe_id", "nunique"),
            kcal_observations=("energy_kcal", "count"),
            kcal_mean=("energy_kcal", "mean"),
            kcal_median=("energy_kcal", "median"),
            kcal_std=("energy_kcal", "std"),
            kcal_min=("energy_kcal", "min"),
            kcal_max=("energy_kcal", "max"),
            quantity_observations=("quantity", "count"),
            quantity_median=("quantity", "median"),
            units_observed=("unit", lambda s: s.dropna().nunique()),
            states_observed=("state", lambda s: s.dropna().nunique()),
        )
        .reset_index()
        .sort_values(["recipe_count", "ingredient_name"], ascending=[False, True])
    )
    ingredient_summary.to_csv(
        output_dir / "nutrient_kcal_audit_ingredient_summary.csv", index=False
    )

    # Review flags are not deletion rules.
    flags = pd.Series("", index=df.index, dtype="string")
    checks = [
        (df["recipe_id"].isna(), "MISSING_RECIPE_ID"),
        (df["ingredient_name"].isna(), "MISSING_INGREDIENT_NAME"),
        (kcal.isna(), "MISSING_KCAL"),
        (kcal.lt(0), "NEGATIVE_KCAL"),
        (kcal.eq(0), "ZERO_KCAL"),
        (quantity.isna(), "MISSING_QUANTITY"),
        (quantity.lt(0), "NEGATIVE_QUANTITY"),
        (high_iqr_mask, "HIGH_KCAL_IQR_REVIEW"),
        (extreme_mask, "EXTREME_KCAL_GT_5000_REVIEW"),
        (df["unit"].isna(), "MISSING_OR_UNSPECIFIED_UNIT"),
        (macro_diff_large, "LARGE_KCAL_MACRO_DISAGREEMENT"),
    ]
    for mask, label in checks:
        mask = mask.fillna(False)
        flags.loc[mask] = flags.loc[mask] + label + ";"

    review = df.loc[flags.ne("")].copy()
    review.insert(0, "audit_flags", flags.loc[flags.ne("")])
    review.to_csv(output_dir / "nutrient_kcal_audit_review_flags.csv", index=False)

    # Ingredient variability: do not treat variability as an error; quantity varies.
    variable_ingredients = ingredient_summary[
        (ingredient_summary["recipe_count"] >= 5)
        & (ingredient_summary["kcal_observations"] >= 5)
    ].copy()
    variable_ingredients["kcal_cv"] = safe_divide(
        variable_ingredients["kcal_std"], variable_ingredients["kcal_mean"].abs()
    )
    variable_ingredients = variable_ingredients.sort_values(
        "kcal_cv", ascending=False, na_position="last"
    )
    variable_ingredients.head(100).to_csv(
        output_dir / "nutrient_kcal_audit_high_variability_ingredients.csv",
        index=False,
    )

    macro_compared = int(comparable_macro_rows.sum())
    macro_large_count = int(macro_diff_large.sum())
    extreme_count = int(extreme_mask.sum())

    lines = [
        "RECIPEDB INGREDIENT-LEVEL KCAL AUDIT (EXTENDED)",
        "=" * 52,
        f"Input file: {input_path}",
        f"Rows: {n_rows:,}",
        f"Columns: {len(df.columns) - 1:,} source columns (plus audit helper columns in memory)",
        f"Unique recipes: {n_recipes:,}",
        f"Unique ingredient names: {n_ingredients:,}",
        "",
        "MISSING VALUES",
        *[f"  {col}: {int(value):,}" for col, value in missing_by_column.items()],
        "",
        "DUPLICATES",
        f"Exact duplicate rows beyond first occurrence: {exact_duplicate_rows:,}",
        f"Rows participating in exact duplicate groups: {len(duplicate_rows):,}",
        f"Rows involved in repeated recipe + normalized ingredient names: {repeated_pair_rows:,}",
        "Repeated recipe/ingredient pairs may be legitimate; inspect before removing.",
        "",
        "KCAL VALIDATION",
        f"Missing energy_kcal: {kcal_missing:,}",
        f"Zero energy_kcal: {kcal_zero:,}",
        f"Negative energy_kcal: {kcal_negative:,}",
        f"Non-finite energy_kcal among non-missing values: {kcal_nonfinite:,}",
        f"Q3 + 3*IQR review threshold: {kcal_upper_iqr:,.4f}",
        f"Rows above Q3 + 3*IQR: {int(high_iqr_mask.sum()):,}",
        f"Absolute extreme threshold: energy_kcal > {EXTREME_KCAL_ABSOLUTE_THRESHOLD:,.0f}",
        f"Rows above absolute extreme threshold: {extreme_count:,}",
        "",
        "QUANTITY VALIDATION",
        f"Missing quantity: {quantity_missing:,}",
        f"Zero quantity: {quantity_zero:,}",
        f"Negative quantity: {quantity_negative:,}",
        "",
        "MACRO-ENERGY CONSISTENCY CHECK",
        "Approximation: 4*carbohydrates_g + 4*protein_g + 9*fat_g",
        f"Rows with kcal and all three macros available: {macro_compared:,}",
        f"Rows with large disagreement under both diagnostic thresholds: {macro_large_count:,}",
        f"Absolute difference threshold: > {MACRO_ENERGY_ABSOLUTE_TOLERANCE:.1f} kcal",
        f"Relative difference threshold: > {MACRO_ENERGY_RELATIVE_TOLERANCE:.0%}",
        "This is a diagnostic, not a strict equality requirement.",
        "",
        "ENERGY_KCAL DISTRIBUTION",
        kcal_quantiles.to_string(),
        "",
        "TOP 25 UNIT LABELS BY ROW COUNT",
        unit_summary.head(25).to_string(index=False),
        "",
        "REVIEW FLAGS",
        f"Rows with one or more review flags: {len(review):,}",
        "Flags indicate records to inspect; no rows are deleted or repaired.",
        "",
        "OUTPUT FILES",
        "nutrient_kcal_audit_summary.txt",
        "nutrient_kcal_audit_ingredient_summary.csv",
        "nutrient_kcal_audit_unit_summary.csv",
        "nutrient_kcal_audit_review_flags.csv",
        "nutrient_kcal_audit_high_variability_ingredients.csv",
        "nutrient_kcal_audit_extreme_records.csv",
        "nutrient_kcal_audit_macro_energy_consistency.csv",
        "nutrient_kcal_audit_duplicate_rows.csv",
        "nutrient_kcal_audit_recipe_duplicate_ingredients.csv",
        "",
        "INTERPRETATION NOTES",
        "1. energy_kcal appears to be recipe-specific ingredient contribution; not assumed per 100 g.",
        "2. Units are heterogeneous and many quantities are missing, so direct comparisons need care.",
        "3. Macro-derived kcal is approximate; differences can result from data conventions and other energy sources.",
        "4. Do not automatically delete extreme values or duplicate recipe/ingredient pairs.",
        "5. Recipe-level recipe_nutrition.csv has not been cross-validated in this script.",
        "6. No kcal substitution score is calculated by this audit.",
    ]
    (output_dir / "nutrient_kcal_audit_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\nAUDIT RESULTS")
    print("-" * 76)
    print(f"Rows                                      : {n_rows:,}")
    print(f"Unique recipes                            : {n_recipes:,}")
    print(f"Unique ingredient names                   : {n_ingredients:,}")
    print(f"Missing kcal                              : {kcal_missing:,}")
    print(f"Missing quantity                          : {quantity_missing:,}")
    print(f"Exact duplicate rows beyond first         : {exact_duplicate_rows:,}")
    print(f"Repeated recipe/ingredient rows           : {repeated_pair_rows:,}")
    print(f"Rows above Q3 + 3*IQR                      : {int(high_iqr_mask.sum()):,}")
    print(f"Rows with kcal + all macros               : {macro_compared:,}")
    print(f"Rows with large macro-energy disagreement : {macro_large_count:,}")
    print(f"Rows with kcal > {EXTREME_KCAL_ABSOLUTE_THRESHOLD:,.0f}                : {extreme_count:,}")
    print(f"Rows flagged for review                   : {len(review):,}")
    print("\nKcal quantiles:")
    print(kcal_quantiles.to_string())
    print("\nSaved outputs:")
    for filename in [
        "nutrient_kcal_audit_summary.txt",
        "nutrient_kcal_audit_ingredient_summary.csv",
        "nutrient_kcal_audit_unit_summary.csv",
        "nutrient_kcal_audit_review_flags.csv",
        "nutrient_kcal_audit_high_variability_ingredients.csv",
        "nutrient_kcal_audit_extreme_records.csv",
        "nutrient_kcal_audit_macro_energy_consistency.csv",
        "nutrient_kcal_audit_duplicate_rows.csv",
        "nutrient_kcal_audit_recipe_duplicate_ingredients.csv",
    ]:
        print(f"  {output_dir / filename}")
    print("\nExtended audit complete. No kcal substitution score has been calculated.")


if __name__ == "__main__":
    main()
