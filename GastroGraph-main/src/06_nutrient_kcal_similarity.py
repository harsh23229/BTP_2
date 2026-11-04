"""06_nutrient_kcal_similarity.py

Build a traceable, filtered calorie dataset and screen lower-calorie ingredients
across the full eligible ingredient vocabulary.

Expected project layout:
  BTP_2/
    dataset/raw/RecipeDB_full/recipe_ingredients.csv
    GastroGraph-main/src/06_nutrient_kcal_similarity.py
    GastroGraph-custom/data/  (audit/validation inputs and outputs)

Inputs used when available:
  - recipe_ingredients.csv
  - nutrient_kcal_audit_macro_energy_consistency.csv (Script 05)
  - nutrient_kcal_recipe_comparison.csv (Script 07; preferred)
    OR nutrient_kcal_recipe_large_differences.csv and
       nutrient_kcal_recipe_conflicting_energy_rows.csv (Script 07)
  - embedding_nearest_neighbors.csv (optional; adds similarity evidence where present)

Important methodology:
  - Only mass-based quantities with reliable unit conversions are converted to grams.
  - No volume-to-mass conversion is guessed; it needs ingredient-specific density.
  - kcal/100 g is a conditional estimate derived from energy_kcal / recorded mass.
    Recipe-level validation supports broad consistency but does not prove every row's
    semantics. Results remain conditional until the RecipeDB field definition is verified.
  - Standardized values above 950 kcal/100 g and zero-kcal observations are flagged and
    excluded from strict profiles by default to avoid implausible/ambiguous densities.
  - Macro disagreement and recipe-level mismatch are quality flags. Records with
    these flags are excluded from the STRICT trusted profiles by default.
  - Exact duplicate source records are excluded after keeping the first occurrence.
  - Calorie-only alternatives are NOT necessarily culinary substitutes. The output
    labels them as calorie-screening candidates, not validated substitutions.
  - The original source and Script 05/07 outputs are never modified.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

# ---------------------------- Configuration ----------------------------
INGREDIENTS_FILE = "recipe_ingredients.csv"
MACRO_AUDIT_FILE = "nutrient_kcal_audit_macro_energy_consistency.csv"
RECIPE_COMPARISON_FILE = "nutrient_kcal_recipe_comparison.csv"
RECIPE_LARGE_DIFF_FILE = "nutrient_kcal_recipe_large_differences.csv"
RECIPE_CONFLICT_FILE = "nutrient_kcal_recipe_conflicting_energy_rows.csv"
EMBEDDING_CANDIDATES_FILE = "embedding_nearest_neighbors.csv"

MAX_KCAL_FOR_PROFILE = 5000.0
# Conservative plausibility ceiling for standardized food energy density.
# Pure fat is roughly 900 kcal/100 g; values above this are reviewed, not trusted.
MAX_PLAUSIBLE_KCAL_PER_100G = 950.0
MACRO_ABS_DIFF_THRESHOLD = 50.0
MACRO_REL_DIFF_THRESHOLD = 0.25
MIN_PROFILE_OBSERVATIONS = 5
MAX_ALTERNATIVES_PER_INGREDIENT = 20

# Convert only units that unambiguously represent mass.
MASS_TO_GRAMS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0, "gm": 1.0, "gms": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "mg": 0.001, "milligram": 0.001, "milligrams": 0.001,
    "ug": 0.000001, "mcg": 0.000001, "microgram": 0.000001,
    "micrograms": 0.000001, "µg": 0.000001,
    "oz": 28.349523125, "ounce": 28.349523125, "ounces": 28.349523125,
    "lb": 453.59237, "lbs": 453.59237, "pound": 453.59237,
    "pounds": 453.59237,
}


def find_file(root: Path, filename: str, required: bool = True) -> Path | None:
    candidates = [
        root / "dataset" / "raw" / "RecipeDB_full" / filename,
        root / "dataset" / "raw" / filename,
        root / "GastroGraph-custom" / "data" / filename,
        root / "GastroGraph-main" / "data" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = list(root.rglob(filename))
    if matches:
        matches.sort(key=lambda p: ("raw" not in str(p).lower(), len(str(p))))
        return matches[0]
    if required:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return None


def normalize_text(series: pd.Series) -> pd.Series:
    return (series.astype("string").str.strip().str.casefold()
            .str.replace(r"\s+", " ", regex=True))


def normalize_recipe_id(series: pd.Series) -> pd.Series:
    return (series.astype("string").str.strip()
            .str.replace(r"\.0$", "", regex=True))


def normalize_unit(value) -> str | None:
    if pd.isna(value):
        return None
    unit = str(value).strip().casefold()
    unit = unit.replace(".", "").replace("_", " ")
    unit = re.sub(r"\s+", " ", unit).strip()
    aliases = {
        "gramme": "gram", "grammes": "grams", "kgs": "kg",
        "milligrams": "milligrams", "milligramme": "milligram",
        "milligrammes": "milligrams", "ounces (oz)": "oz",
        "pounds (lb)": "lb", "µg": "µg",
    }
    return aliases.get(unit, unit)


def numeric_key(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.map(lambda x: "<NA>" if pd.isna(x) else f"{float(x):.8g}")


def add_record_key(df: pd.DataFrame) -> pd.Series:
    """Create a practical matching key for raw rows and Script 05 audit rows."""
    parts = []
    for col in ["recipe_id", "recipe_title", "ingredient_name", "quantity", "unit",
                "energy_kcal", "carbohydrates", "protein_g", "fat_g"]:
        if col not in df.columns:
            parts.append(pd.Series("<MISSING_COLUMN>", index=df.index, dtype="string"))
        elif col in {"quantity", "energy_kcal", "carbohydrates", "protein_g", "fat_g"}:
            parts.append(numeric_key(df[col]).astype("string"))
        elif col == "recipe_id":
            parts.append(normalize_recipe_id(df[col]).fillna("<NA>"))
        else:
            parts.append(normalize_text(df[col]).fillna("<NA>"))
    key = parts[0].astype("string")
    for part in parts[1:]:
        key = key + "||" + part.astype("string")
    return key


def read_csv_if_exists(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        return None
    return pd.read_csv(path, low_memory=False)

def as_bool(series: pd.Series) -> pd.Series:
    """Parse CSV booleans safely, including textual True/False values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.casefold()
    return normalized.isin(["true", "1", "yes", "y"])


def main() -> None:
    # Script path: BTP_2/GastroGraph-main/src/06_nutrient_kcal_similarity.py
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "GastroGraph-custom" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    ingredient_path = find_file(project_root, INGREDIENTS_FILE)
    macro_audit_path = find_file(project_root, MACRO_AUDIT_FILE, required=False)
    recipe_comparison_path = find_file(project_root, RECIPE_COMPARISON_FILE, required=False)
    recipe_large_diff_path = find_file(project_root, RECIPE_LARGE_DIFF_FILE, required=False)
    recipe_conflict_path = find_file(project_root, RECIPE_CONFLICT_FILE, required=False)
    embedding_path = find_file(project_root, EMBEDDING_CANDIDATES_FILE, required=False)

    print("=" * 78)
    print("CALORIE DATA FILTERING, STANDARDIZATION AND FULL-VOCABULARY SCREENING")
    print("=" * 78)
    print(f"Ingredient source : {ingredient_path}")
    print(f"Output directory  : {output_dir}")

    df = pd.read_csv(ingredient_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["recipe_id", "ingredient_name", "quantity", "unit", "energy_kcal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Ingredient source missing required columns: {missing}")

    # Preserve source row number for traceability. It is not written back to source.
    df.insert(0, "source_row_number", np.arange(2, len(df) + 2))
    source_columns = [c for c in df.columns if c != "source_row_number"]
    df["ingredient_key"] = normalize_text(df["ingredient_name"])
    df["recipe_key"] = normalize_recipe_id(df["recipe_id"])
    for col in ["quantity", "energy_kcal", "carbohydrates", "protein_g", "fat_g"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["unit_normalized"] = df["unit"].map(normalize_unit) if "unit" in df.columns else pd.NA

    # ------------------ Script 05: macro-energy audit ------------------
    macro_flag_keys: set[str] = set()
    macro_audit = read_csv_if_exists(macro_audit_path)
    if macro_audit is not None and not macro_audit.empty:
        needed_for_key = ["recipe_id", "ingredient_name", "quantity", "unit", "energy_kcal",
                          "carbohydrates", "protein_g", "fat_g"]
        if all(c in macro_audit.columns for c in needed_for_key):
            macro_audit["_record_key"] = add_record_key(macro_audit)
            if "macro_energy_review_flag" in macro_audit.columns:
                flagged = macro_audit["macro_energy_review_flag"].astype("string").str.contains(
                    "LARGE_KCAL_MACRO_DISAGREEMENT", na=False
                )
                macro_flag_keys = set(macro_audit.loc[flagged, "_record_key"].astype(str))

    # Recalculate the exact Script 05 diagnostic too, so a minor CSV-key mismatch
    # cannot silently hide a macro disagreement. This mirrors its 4C + 4P + 9F rule.
    macro_cols_exist = all(c in df.columns for c in ["carbohydrates", "protein_g", "fat_g"])
    if macro_cols_exist:
        macro_available = df[["carbohydrates", "protein_g", "fat_g"]].notna().all(axis=1)
        macro_implied = 4 * df["carbohydrates"] + 4 * df["protein_g"] + 9 * df["fat_g"]
        abs_diff = (df["energy_kcal"] - macro_implied).abs()
        denom = pd.concat([df["energy_kcal"].abs(), macro_implied.abs()], axis=1).max(axis=1)
        rel_diff = abs_diff / denom.replace(0, np.nan)
        df["macro_energy_review_flag"] = (
            df["energy_kcal"].notna() & macro_available
            & abs_diff.gt(MACRO_ABS_DIFF_THRESHOLD)
            & rel_diff.gt(MACRO_REL_DIFF_THRESHOLD)
        )
        if macro_flag_keys:
            df["_record_key"] = add_record_key(df)
            df["macro_energy_review_flag"] |= df["_record_key"].astype(str).isin(macro_flag_keys)
            df.drop(columns=["_record_key"], inplace=True)
    else:
        df["macro_energy_review_flag"] = False
        warnings.warn("Macro columns unavailable; macro-energy audit flags could not be applied.")

    # ------------------ Script 07: recipe-level validation ------------------
    recipe_flag_keys: set[str] = set()
    recipe_validation = read_csv_if_exists(recipe_comparison_path)
    if recipe_validation is not None and not recipe_validation.empty:
        if "_recipe_key" not in recipe_validation.columns:
            if "recipe_id" in recipe_validation.columns:
                recipe_validation["_recipe_key"] = normalize_recipe_id(recipe_validation["recipe_id"])
        if "_recipe_key" in recipe_validation.columns:
            flag = pd.Series(False, index=recipe_validation.index)
            for flag_col in ["large_difference_flag", "conflicting_kcal_rows_flag",
                             "negative_ingredient_sum_flag", "negative_recipe_energy_flag"]:
                if flag_col in recipe_validation.columns:
                    flag |= as_bool(recipe_validation[flag_col])
            recipe_flag_keys = set(normalize_recipe_id(recipe_validation.loc[flag, "_recipe_key"]).dropna().astype(str))
    else:
        # Older Script 07 outputs may provide the two flagged subsets instead.
        for path in [recipe_large_diff_path, recipe_conflict_path]:
            flagged_df = read_csv_if_exists(path)
            if flagged_df is not None and "_recipe_key" in flagged_df.columns:
                recipe_flag_keys.update(normalize_recipe_id(flagged_df["_recipe_key"]).dropna().astype(str))

    df["recipe_level_review_flag"] = df["recipe_key"].astype("string").isin(recipe_flag_keys)

    # ------------------ Basic quality checks and standardization ------------------
    df["missing_ingredient_name_flag"] = df["ingredient_key"].isna() | df["ingredient_key"].eq("")
    df["missing_or_invalid_kcal_flag"] = df["energy_kcal"].isna() | ~np.isfinite(df["energy_kcal"].fillna(np.nan))
    df["negative_kcal_flag"] = df["energy_kcal"].lt(0).fillna(False)
    df["zero_kcal_flag"] = df["energy_kcal"].eq(0).fillna(False)
    df["extreme_kcal_flag"] = df["energy_kcal"].gt(MAX_KCAL_FOR_PROFILE).fillna(False)
    df["missing_quantity_flag"] = df["quantity"].isna() if "quantity" in df.columns else True
    df["nonpositive_quantity_flag"] = df["quantity"].le(0).fillna(False) if "quantity" in df.columns else True
    df["missing_unit_flag"] = df["unit_normalized"].isna() | df["unit_normalized"].eq("")

    df["mass_to_grams_factor"] = df["unit_normalized"].map(MASS_TO_GRAMS)
    df["quantity_g"] = df["quantity"] * df["mass_to_grams_factor"]
    df["mass_unit_standardized_flag"] = (
        df["mass_to_grams_factor"].notna() & df["quantity_g"].gt(0)
    )
    df["kcal_per_100g"] = np.where(
        df["mass_unit_standardized_flag"] & df["energy_kcal"].notna()
        & df["energy_kcal"].ge(0),
        df["energy_kcal"] / df["quantity_g"] * 100.0,
        np.nan,
    )
    # Validate the derived density too: a reasonable raw kcal value can still become
    # implausible when divided by a tiny/mismatched quantity. Keep flagged rows for audit.
    df["implausible_kcal_per_100g_flag"] = (
        df["kcal_per_100g"].notna()
        & (~np.isfinite(df["kcal_per_100g"]) | df["kcal_per_100g"].gt(MAX_PLAUSIBLE_KCAL_PER_100G))
    )

    # Exact duplicate records: keep first occurrence for analysis; preserve duplicates
    # in the excluded/traceability output with an explicit reason.
    duplicate_subset = [c for c in source_columns if c in df.columns]
    df["exact_duplicate_row_flag"] = df.duplicated(subset=duplicate_subset, keep="first")

    # One explicit reason string per row, with all applicable reasons retained.
    reason_masks = [
        (df["missing_ingredient_name_flag"], "MISSING_INGREDIENT_NAME"),
        (df["missing_or_invalid_kcal_flag"], "MISSING_OR_INVALID_KCAL"),
        (df["negative_kcal_flag"], "NEGATIVE_KCAL"),
        (df["extreme_kcal_flag"], "EXTREME_KCAL_GT_5000"),
        (df["zero_kcal_flag"], "ZERO_KCAL_REVIEW"),
        (df["implausible_kcal_per_100g_flag"], "IMPLAUSIBLE_KCAL_PER_100G_GT_950"),
        (df["missing_quantity_flag"], "MISSING_QUANTITY"),
        (df["nonpositive_quantity_flag"], "NONPOSITIVE_QUANTITY"),
        (df["missing_unit_flag"], "MISSING_OR_UNSPECIFIED_UNIT"),
        (~df["mass_unit_standardized_flag"], "UNIT_NOT_RELIABLY_CONVERTIBLE_TO_GRAMS"),
        (df["macro_energy_review_flag"], "MACRO_ENERGY_DISAGREEMENT"),
        (df["recipe_level_review_flag"], "RECIPE_LEVEL_VALIDATION_MISMATCH"),
        (df["exact_duplicate_row_flag"], "EXACT_DUPLICATE_ROW"),
    ]
    reason_series = pd.Series("", index=df.index, dtype="string")
    for mask, label in reason_masks:
        reason_series.loc[mask.fillna(False)] = reason_series.loc[mask.fillna(False)] + label + ";"
    df["quality_flags"] = reason_series.str.rstrip(";")

    # Strict trusted profile eligibility: valid mass conversion and kcal, no extreme,
    # no macro disagreement, no recipe-level mismatch, no exact duplicate.
    df["trusted_profile_eligible"] = (
        ~df["missing_ingredient_name_flag"]
        & ~df["missing_or_invalid_kcal_flag"]
        & ~df["negative_kcal_flag"]
        & ~df["extreme_kcal_flag"]
        & ~df["zero_kcal_flag"]
        & ~df["implausible_kcal_per_100g_flag"]
        & ~df["missing_quantity_flag"]
        & ~df["nonpositive_quantity_flag"]
        & df["mass_unit_standardized_flag"]
        & ~df["macro_energy_review_flag"]
        & ~df["recipe_level_review_flag"]
        & ~df["exact_duplicate_row_flag"]
        & df["kcal_per_100g"].notna()
        & np.isfinite(df["kcal_per_100g"].fillna(np.nan))
    )

    # Save a complete working dataset and an exclusion/review file. These are new
    # derived outputs; the original source is unchanged.
    cleaned_path = output_dir / "nutrient_kcal_cleaned_records.csv"
    df.to_csv(cleaned_path, index=False)
    excluded_path = output_dir / "nutrient_kcal_excluded_or_flagged_records.csv"
    df.loc[~df["trusted_profile_eligible"]].to_csv(excluded_path, index=False)
    trusted_records_path = output_dir / "nutrient_kcal_trusted_standardized_records.csv"
    trusted = df.loc[df["trusted_profile_eligible"]].copy()
    trusted.to_csv(trusted_records_path, index=False)

    # ------------------ Build profiles from trusted standardized records ------------------
    if trusted.empty:
        profiles = pd.DataFrame(columns=[
            "ingredient_key", "ingredient_name", "standardized_observations",
            "recipe_count", "kcal_per_100g_median", "kcal_per_100g_mean",
            "kcal_per_100g_q25", "kcal_per_100g_q75", "kcal_per_100g_iqr",
            "kcal_per_100g_std", "kcal_per_100g_cv", "profile_status",
        ])
    else:
        profiles = trusted.groupby("ingredient_key", dropna=False).agg(
            ingredient_name=("ingredient_name", "first"),
            standardized_observations=("kcal_per_100g", "count"),
            recipe_count=("recipe_key", "nunique"),
            kcal_per_100g_median=("kcal_per_100g", "median"),
            kcal_per_100g_mean=("kcal_per_100g", "mean"),
            kcal_per_100g_q25=("kcal_per_100g", lambda s: s.quantile(0.25)),
            kcal_per_100g_q75=("kcal_per_100g", lambda s: s.quantile(0.75)),
            kcal_per_100g_std=("kcal_per_100g", "std"),
        ).reset_index()
        profiles["kcal_per_100g_iqr"] = profiles["kcal_per_100g_q75"] - profiles["kcal_per_100g_q25"]
        profiles["kcal_per_100g_cv"] = (
            profiles["kcal_per_100g_std"] / profiles["kcal_per_100g_mean"].abs().replace(0, np.nan)
        )
        profiles["profile_status"] = np.select(
            [
                profiles["standardized_observations"] < MIN_PROFILE_OBSERVATIONS,
                profiles["kcal_per_100g_iqr"] > profiles["kcal_per_100g_median"].abs(),
            ],
            ["LOW_SAMPLE_SIZE_REVIEW", "HIGH_VARIABILITY_REVIEW"],
            default="TRUSTED_STANDARDIZED_PROFILE",
        )
    profiles_path = output_dir / "nutrient_kcal_ingredient_profiles.csv"
    profiles.to_csv(profiles_path, index=False)

    # Only profiles with enough observations and reasonable variability enter the
    # strict candidate pool. We retain all profiles in the profile output for audit.
    candidate_pool = profiles.loc[
        profiles["standardized_observations"].ge(MIN_PROFILE_OBSERVATIONS)
        & profiles["kcal_per_100g_median"].notna()
        & profiles["kcal_per_100g_median"].ge(0)
        & profiles["kcal_per_100g_iqr"].le(profiles["kcal_per_100g_median"].abs())
    ].copy()
    candidate_pool = candidate_pool.sort_values("kcal_per_100g_median").reset_index(drop=True)
    # Profiles admitted to this pool have >= minimum observations and an IQR no larger
    # than the median. These scores are still conditional on verified source semantics.

    # Optional embedding evidence is attached when available, but it does not restrict
    # the calorie screening to the old 230 pairs.
    embedding_pairs: dict[tuple[str, str], float] = {}
    embedding_df = read_csv_if_exists(embedding_path)
    if embedding_df is not None:
        expected = {"query_ingredient", "neighbor_ingredient", "cosine_similarity"}
        if expected.issubset(embedding_df.columns):
            embedding_df["query_key"] = normalize_text(embedding_df["query_ingredient"])
            embedding_df["neighbor_key"] = normalize_text(embedding_df["neighbor_ingredient"])
            embedding_df["cosine_similarity"] = pd.to_numeric(embedding_df["cosine_similarity"], errors="coerce")
            for row in embedding_df.dropna(subset=["query_key", "neighbor_key"]).itertuples(index=False):
                embedding_pairs[(str(row.query_key), str(row.neighbor_key))] = getattr(row, "cosine_similarity")

    # For each original ingredient, list up to K lower-kcal ingredients from the full
    # eligible vocabulary. This is a screening shortlist only; calorie alone cannot
    # establish culinary/functional interchangeability.
    candidate_rows = []
    pool_records = candidate_pool.to_dict("records")
    for original in pool_records:
        original_key = str(original["ingredient_key"])
        original_kcal = float(original["kcal_per_100g_median"])
        lower = candidate_pool.loc[
            (candidate_pool["ingredient_key"] != original_key)
            & (candidate_pool["kcal_per_100g_median"] < original_kcal)
        ].sort_values("kcal_per_100g_median", ascending=False).head(MAX_ALTERNATIVES_PER_INGREDIENT)
        for alt in lower.to_dict("records"):
            alt_kcal = float(alt["kcal_per_100g_median"])
            denom = max(abs(original_kcal), abs(alt_kcal))
            score = (original_kcal - alt_kcal) / denom if denom > 0 else 0.0
            pair = (original_key, str(alt["ingredient_key"]))
            candidate_rows.append({
                "original_ingredient": original["ingredient_name"],
                "original_ingredient_key": original_key,
                "substitute_candidate": alt["ingredient_name"],
                "substitute_candidate_key": str(alt["ingredient_key"]),
                "original_kcal_per_100g_median": original_kcal,
                "candidate_kcal_per_100g_median": alt_kcal,
                "kcal_difference_per_100g": original_kcal - alt_kcal,
                "kcal_reduction_percent": ((original_kcal - alt_kcal) / original_kcal * 100.0)
                    if original_kcal > 0 else np.nan,
                "kcal_reduction_score": score,
                "original_profile_observations": int(original["standardized_observations"]),
                "candidate_profile_observations": int(alt["standardized_observations"]),
                "original_profile_status": original["profile_status"],
                "candidate_profile_status": alt["profile_status"],
                "embedding_cosine_similarity_if_available": embedding_pairs.get(pair, np.nan),
                "candidate_basis": "CALORIE_ONLY_SCREENING_NOT_VALIDATED_AS_SUBSTITUTE",
                "quantity_basis": "100 g",
                "warning": "Lower calories do not establish flavour, texture, functional or dietary equivalence.",
            })

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["original_ingredient_key", "kcal_reduction_score"], ascending=[True, False]
        )
    candidates_path = output_dir / "nutrient_kcal_substitution_candidates.csv"
    candidates.to_csv(candidates_path, index=False)

    # Optional: retain old GraphSAGE candidate pairs with calorie evidence from the
    # newly standardized profiles, making it easy to compare old and expanded outputs.
    if embedding_df is not None and {"query_ingredient", "neighbor_ingredient"}.issubset(embedding_df.columns):
        profile_cols = ["ingredient_key", "ingredient_name", "standardized_observations",
                        "kcal_per_100g_median", "kcal_per_100g_iqr", "profile_status"]
        a = profiles[profile_cols].rename(columns={
            "ingredient_key": "query_key", "ingredient_name": "query_profile_name",
            "standardized_observations": "query_observations",
            "kcal_per_100g_median": "query_kcal_per_100g_median",
            "kcal_per_100g_iqr": "query_kcal_per_100g_iqr", "profile_status": "query_profile_status",
        })
        b = profiles[profile_cols].rename(columns={
            "ingredient_key": "neighbor_key", "ingredient_name": "neighbor_profile_name",
            "standardized_observations": "neighbor_observations",
            "kcal_per_100g_median": "neighbor_kcal_per_100g_median",
            "kcal_per_100g_iqr": "neighbor_kcal_per_100g_iqr", "profile_status": "neighbor_profile_status",
        })
        old = embedding_df.copy()
        old["query_key"] = normalize_text(old["query_ingredient"])
        old["neighbor_key"] = normalize_text(old["neighbor_ingredient"])
        old = old.merge(a, on="query_key", how="left").merge(b, on="neighbor_key", how="left")
        eligible_keys = set(candidate_pool["ingredient_key"].astype(str))
        old["both_profiles_eligible"] = (
            old["query_key"].astype(str).isin(eligible_keys)
            & old["neighbor_key"].astype(str).isin(eligible_keys)
        )
        denom = pd.concat([old["query_kcal_per_100g_median"].abs(),
                           old["neighbor_kcal_per_100g_median"].abs()], axis=1).max(axis=1)
        raw_score = (
            (old["query_kcal_per_100g_median"] - old["neighbor_kcal_per_100g_median"])
            / denom.replace(0, np.nan)
        )
        old["kcal_reduction_score"] = raw_score.where(old["both_profiles_eligible"])
        old["kcal_reduction_percent"] = np.where(
            old["both_profiles_eligible"] & old["query_kcal_per_100g_median"].gt(0),
            (old["query_kcal_per_100g_median"] - old["neighbor_kcal_per_100g_median"])
            / old["query_kcal_per_100g_median"] * 100.0,
            np.nan,
        )
        old["kcal_comparison_status"] = np.select(
            [~old["both_profiles_eligible"],
             old["kcal_reduction_score"].gt(0), old["kcal_reduction_score"].lt(0)],
            ["PROFILE_NOT_ELIGIBLE_OR_UNAVAILABLE", "LOWER_KCAL_NEIGHBOR", "HIGHER_KCAL_NEIGHBOR"],
            default="EQUAL_KCAL_OR_SCORE_UNAVAILABLE",
        )
        old["calorie_basis_status"] = "CONDITIONAL_PENDING_RECIPE_DB_FIELD_SEMANTICS_VERIFICATION"
        old["candidate_validity"] = "NOT_VALIDATED_BY_CALORIES_ALONE"
        old.to_csv(output_dir / "nutrient_kcal_embedding_pair_evidence.csv", index=False)

    # ------------------------------ Summary ------------------------------
    row_count = len(df)
    eligible_count = int(df["trusted_profile_eligible"].sum())
    summary_lines = [
        "CALORIE FILTERING AND FULL-VOCABULARY SCREENING SUMMARY",
        "=" * 58,
        f"Ingredient source: {ingredient_path}",
        f"Script 05 macro audit loaded: {macro_audit_path is not None}",
        f"Script 07 recipe comparison loaded: {recipe_comparison_path is not None}",
        f"Script 07 large-difference file loaded: {recipe_large_diff_path is not None}",
        f"Script 07 conflicting-energy file loaded: {recipe_conflict_path is not None}",
        f"Embedding candidates loaded (optional): {embedding_path is not None}",
        f"Source ingredient rows: {row_count:,}",
        f"Distinct normalized ingredient names in source: {df['ingredient_key'].nunique(dropna=True):,}",
        f"Rows with macro-energy disagreement: {int(df['macro_energy_review_flag'].sum()):,}",
        f"Rows in recipes flagged by Script 07: {int(df['recipe_level_review_flag'].sum()):,}",
        f"Rows with kcal > {MAX_KCAL_FOR_PROFILE:,.0f}: {int(df['extreme_kcal_flag'].sum()):,}",
        f"Rows with zero kcal (review/excluded from strict profiles): {int(df['zero_kcal_flag'].sum()):,}",
        f"Rows with derived kcal/100 g > {MAX_PLAUSIBLE_KCAL_PER_100G:,.0f}: {int(df['implausible_kcal_per_100g_flag'].sum()):,}",
        f"Rows with missing/invalid kcal: {int(df['missing_or_invalid_kcal_flag'].sum()):,}",
        f"Rows with mass units standardized to grams: {int(df['mass_unit_standardized_flag'].sum()):,}",
        f"Exact duplicate rows excluded after first occurrence: {int(df['exact_duplicate_row_flag'].sum()):,}",
        f"Trusted standardized ingredient rows: {eligible_count:,}",
        f"Ingredients with any trusted profile: {len(profiles):,}",
        f"Ingredients eligible for candidate screening (>= {MIN_PROFILE_OBSERVATIONS} observations and variability check): {len(candidate_pool):,}",
        f"Calorie-only screening candidate pairs (up to {MAX_ALTERNATIVES_PER_INGREDIENT} per original): {len(candidates):,}",
        "",
        "STANDARDIZATION AND FILTERING RULES:",
        "1. Only mass units with unambiguous conversion factors are converted to grams; no assumed cup/spoon/piece density.",
        "2. kcal per 100 g = recorded energy_kcal / quantity_g * 100; this is conditional on the source energy/quantity semantics.",
        f"3. Derived densities above {MAX_PLAUSIBLE_KCAL_PER_100G:.0f} kcal/100 g are flagged and excluded from strict profiles.",
        "4. Zero-kcal observations are retained in the audit output but excluded from strict profiles pending review.",
        "5. Macro disagreement uses Script 05 thresholds: absolute >50 kcal AND relative >25%.",
        "6. All rows from Script 07 flagged recipes are excluded from strict profiles, conservatively; this does not prove every row is wrong.",
        "7. Exact duplicate source rows are excluded after keeping the first occurrence.",
        "8. Missing/invalid kcal, negative kcal, kcal >5000, missing/nonpositive quantity, and non-mass units are not eligible for trusted profiles.",
        "9. Full-vocabulary calorie-only pairs are screening results, NOT confirmed substitutions; similarity/functional compatibility must be assessed separately.",
        "10. Recipe-level agreement supports but does not prove every ingredient-level kcal/quantity relationship; verify RecipeDB field semantics before final claims.",
        "11. Original source and Script 05/07 outputs are not modified.",
        "",
        "OUTPUTS:",
        "nutrient_kcal_cleaned_records.csv",
        "nutrient_kcal_excluded_or_flagged_records.csv",
        "nutrient_kcal_trusted_standardized_records.csv",
        "nutrient_kcal_ingredient_profiles.csv",
        "nutrient_kcal_substitution_candidates.csv",
        "nutrient_kcal_embedding_pair_evidence.csv (only if embedding file exists)",
        "nutrient_kcal_pair_summary.txt",
    ]
    (output_dir / "nutrient_kcal_pair_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\nRESULTS")
    print("-" * 78)
    print(f"Source rows                                 : {row_count:,}")
    print(f"Distinct ingredient names                   : {df['ingredient_key'].nunique(dropna=True):,}")
    print(f"Macro-energy disagreement rows              : {int(df['macro_energy_review_flag'].sum()):,}")
    print(f"Rows in Script 07 flagged recipes           : {int(df['recipe_level_review_flag'].sum()):,}")
    print(f"Zero-kcal rows flagged for review            : {int(df['zero_kcal_flag'].sum()):,}")
    print(f"Implausible derived kcal/100 g rows          : {int(df['implausible_kcal_per_100g_flag'].sum()):,}")
    print(f"Trusted standardized ingredient rows        : {eligible_count:,}")
    print(f"Ingredient profiles                         : {len(profiles):,}")
    print(f"Candidate pool                              : {len(candidate_pool):,}")
    print(f"Calorie-only screening pairs                : {len(candidates):,}")
    print("\nSaved outputs:")
    for filename in [
        "nutrient_kcal_cleaned_records.csv",
        "nutrient_kcal_excluded_or_flagged_records.csv",
        "nutrient_kcal_trusted_standardized_records.csv",
        "nutrient_kcal_ingredient_profiles.csv",
        "nutrient_kcal_substitution_candidates.csv",
        "nutrient_kcal_embedding_pair_evidence.csv",
        "nutrient_kcal_pair_summary.txt",
    ]:
        path = output_dir / filename
        if path.exists():
            print(f"  {path}")
    print("\nIMPORTANT: calorie-only candidates are not validated substitutions.")
    print("Scores are conditional estimates until RecipeDB kcal/quantity semantics are verified.")


if __name__ == "__main__":
    main()
