"""
03_carbon_audit.py

Audit the SustainableFoodDB carbon-footprint mapping file before calculating
carbon-impact scores for ingredient substitution.

This script does NOT calculate a final substitution score and does NOT
automatically accept/reject mappings. Low-similarity mappings are review flags.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

EXPECTED_FILE = "SustainableFoodDB_CarbonFootprints.csv"
REQUIRED_COLUMNS = [
    "Ingredient",
    "SueatableIngredient",
    "Similarity",
    "CarbonFootprint",
    "Count",
]
SIMILARITY_REVIEW_THRESHOLD = 0.85


def locate_input_file() -> Path:
    """Find the carbon CSV in common project locations."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / EXPECTED_FILE,
        script_dir.parent / EXPECTED_FILE,
        script_dir.parent / "data" / EXPECTED_FILE,
        script_dir.parent / "processed" / EXPECTED_FILE,
        script_dir.parent / "dataset" / "processed" / EXPECTED_FILE,
        script_dir.parent.parent / "data" / EXPECTED_FILE,
        script_dir.parent.parent / "processed" / EXPECTED_FILE,
        script_dir.parent.parent / "dataset" / "processed" / EXPECTED_FILE,
        script_dir.parent.parent / "GastroGraph-custom" / "data" / EXPECTED_FILE,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Search only within the likely BTP project tree, not the whole computer.
    search_root = script_dir.parent.parent
    try:
        matches = list(search_root.rglob(EXPECTED_FILE))
    except OSError:
        matches = []

    if matches:
        # Prefer a file in a processed directory, otherwise use the first match.
        matches.sort(key=lambda p: ("processed" not in str(p).lower(), len(str(p))))
        return matches[0]

    print(f"ERROR: Could not find {EXPECTED_FILE}.")
    print("Put the CSV in the project's processed/data folder, or update locate_input_file().")
    sys.exit(1)


def write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    input_file = locate_input_file()
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    output_dir = PROJECT_ROOT / "GastroGraph-custom" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SUSTAINABLEFOODDB CARBON DATA AUDIT")
    print("=" * 72)
    print(f"Input file : {input_file}")
    print(f"Output dir : {output_dir}")

    df = pd.read_csv(input_file, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns    : {len(df.columns)}")

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        print("\nERROR: Required columns are missing:", missing_columns)
        print("Columns found:", df.columns.tolist())
        sys.exit(1)

    # Standardize text for auditing only; preserve the source columns otherwise.
    df["Ingredient"] = df["Ingredient"].astype("string").str.strip()
    df["SueatableIngredient"] = df["SueatableIngredient"].astype("string").str.strip()

    for col in ["Similarity", "CarbonFootprint", "Count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Completeness and duplicates
    missing_by_column = df[REQUIRED_COLUMNS].isna().sum()
    blank_ingredient = df["Ingredient"].fillna("").eq("").sum()
    blank_sustainable = df["SueatableIngredient"].fillna("").eq("").sum()
    exact_duplicate_rows = int(df.duplicated().sum())
    duplicate_ingredient_rows = int(df.duplicated(subset=["Ingredient"], keep=False).sum())
    duplicate_mapping_rows = int(
        df.duplicated(subset=["Ingredient", "SueatableIngredient"], keep=False).sum()
    )

    # 2. Numeric validation
    similarity = df["Similarity"]
    carbon = df["CarbonFootprint"]
    count = df["Count"]

    similarity_outside_0_1 = int(((similarity < 0) | (similarity > 1.00001)).sum())
    negative_carbon = int((carbon < 0).sum())
    negative_count = int((count < 0).sum())
    nonfinite_numeric = {
        col: int((~np.isfinite(df[col].dropna())).sum())
        for col in ["Similarity", "CarbonFootprint", "Count"]
    }

    # Tiny floating-point excursions above 1 (e.g. 1.0000001) are reported separately.
    tiny_similarity_over_one = int(((similarity > 1) & (similarity <= 1.00001)).sum())

    # 3. Coverage and distribution
    unique_ingredients = int(df["Ingredient"].nunique(dropna=True))
    unique_sustainable = int(df["SueatableIngredient"].nunique(dropna=True))
    one_to_many = (
        df.dropna(subset=["Ingredient", "SueatableIngredient"])
        .groupby("Ingredient")["SueatableIngredient"].nunique()
    )
    many_to_one = (
        df.dropna(subset=["Ingredient", "SueatableIngredient"])
        .groupby("SueatableIngredient")["Ingredient"].nunique()
    )

    carbon_stats = carbon.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95])
    similarity_stats = similarity.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95])
    count_stats = count.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95])

    # 4. Review flags: these are candidates for manual inspection, not proof of errors.
    review = df.copy()
    review["audit_flag"] = ""
    review.loc[
        review["Ingredient"].isna() | review["Ingredient"].eq(""),
        "audit_flag"
    ] += "MISSING_INGREDIENT;"
    review.loc[
        review["SueatableIngredient"].isna() | review["SueatableIngredient"].eq(""),
        "audit_flag"
    ] += "MISSING_SUSTAINABLE_INGREDIENT;"
    review.loc[
        review["Similarity"].isna(),
        "audit_flag"
    ] += "MISSING_SIMILARITY;"
    review.loc[
        review["Similarity"] < SIMILARITY_REVIEW_THRESHOLD,
        "audit_flag"
    ] += f"LOW_SIMILARITY_LT_{SIMILARITY_REVIEW_THRESHOLD};"
    review.loc[
        review["CarbonFootprint"].isna(),
        "audit_flag"
    ] += "MISSING_CARBON_FOOTPRINT;"
    review.loc[
        review["CarbonFootprint"] < 0,
        "audit_flag"
    ] += "NEGATIVE_CARBON_FOOTPRINT;"
    review.loc[
        review["Count"].isna(),
        "audit_flag"
    ] += "MISSING_COUNT;"
    review.loc[
        review["Count"] < 0,
        "audit_flag"
    ] += "NEGATIVE_COUNT;"

    flagged = review[review["audit_flag"].ne("")].copy()
    flagged.to_csv(output_dir / "carbon_audit_review_flags.csv", index=False)

    # Per-sustainable-category summary: helps detect categories reused by many source ingredients.
    category_summary = (
        df.groupby("SueatableIngredient", dropna=False)
        .agg(
            source_ingredient_count=("Ingredient", "nunique"),
            row_count=("Ingredient", "size"),
            carbon_footprint_min=("CarbonFootprint", "min"),
            carbon_footprint_max=("CarbonFootprint", "max"),
            similarity_mean=("Similarity", "mean"),
            count_sum=("Count", "sum"),
        )
        .reset_index()
        .sort_values(["source_ingredient_count", "row_count"], ascending=False)
    )
    category_summary.to_csv(output_dir / "carbon_audit_category_summary.csv", index=False)

    # Preserve all input columns; add a transparent review flag.
    review.to_csv(output_dir / "carbon_audit_full.csv", index=False)

    lines = [
        "SUSTAINABLEFOODDB CARBON DATA AUDIT",
        "=" * 40,
        f"Input file: {input_file}",
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns)}",
        "",
        "COMPLETENESS",
        *[f"Missing {col}: {int(value):,}" for col, value in missing_by_column.items()],
        f"Blank Ingredient values: {blank_ingredient:,}",
        f"Blank SueatableIngredient values: {blank_sustainable:,}",
        "",
        "DUPLICATES",
        f"Exact duplicate rows: {exact_duplicate_rows:,}",
        f"Rows belonging to duplicated Ingredient values: {duplicate_ingredient_rows:,}",
        f"Rows belonging to duplicated Ingredient + SueatableIngredient pairs: {duplicate_mapping_rows:,}",
        "",
        "COVERAGE",
        f"Unique source ingredients: {unique_ingredients:,}",
        f"Unique SueatableIngredient categories: {unique_sustainable:,}",
        f"Source ingredients mapped to multiple categories: {int((one_to_many > 1).sum()):,}",
        f"Sustainable categories mapped from multiple source ingredients: {int((many_to_one > 1).sum()):,}",
        "",
        "NUMERIC VALIDATION",
        f"Similarity outside [0, 1.00001]: {similarity_outside_0_1:,}",
        f"Small floating-point similarity values above 1 and <= 1.00001: {tiny_similarity_over_one:,}",
        f"Negative carbon-footprint values: {negative_carbon:,}",
        f"Negative Count values: {negative_count:,}",
        *[f"Non-finite {col} values among non-missing values: {n:,}" for col, n in nonfinite_numeric.items()],
        "",
        "CARBON FOOTPRINT DISTRIBUTION",
        carbon_stats.to_string(),
        "",
        "MAPPING SIMILARITY DISTRIBUTION",
        similarity_stats.to_string(),
        "",
        "COUNT DISTRIBUTION",
        count_stats.to_string(),
        "",
        "REVIEW FLAGS",
        f"Rows flagged for review: {len(flagged):,}",
        f"Similarity review threshold: {SIMILARITY_REVIEW_THRESHOLD}",
        "Low similarity is a review signal, not proof that a mapping is incorrect.",
        "",
        "OUTPUTS",
        "carbon_audit_summary.txt",
        "carbon_audit_full.csv",
        "carbon_audit_review_flags.csv",
        "carbon_audit_category_summary.csv",
        "",
        "IMPORTANT LIMITATION",
        "The file's column names alone do not establish the carbon-footprint unit or reference basis.",
        "Confirm these from the dataset documentation before comparing or calculating carbon savings.",
        "This audit does not calculate a final substitution score.",
    ]

    write_report(output_dir / "carbon_audit_summary.txt", lines)

    print("\nAUDIT SUMMARY")
    print("-" * 72)
    print(f"Unique source ingredients            : {unique_ingredients:,}")
    print(f"Unique sustainable categories        : {unique_sustainable:,}")
    print(f"Exact duplicate rows                 : {exact_duplicate_rows:,}")
    print(f"Rows flagged for review              : {len(flagged):,}")
    print(f"Small similarity values > 1          : {tiny_similarity_over_one:,}")
    print(f"Negative carbon-footprint values     : {negative_carbon:,}")
    print("\nCarbon footprint distribution:")
    print(carbon_stats.to_string())
    print("\nOutputs written:")
    print(f"  {output_dir / 'carbon_audit_summary.txt'}")
    print(f"  {output_dir / 'carbon_audit_full.csv'}")
    print(f"  {output_dir / 'carbon_audit_review_flags.csv'}")
    print(f"  {output_dir / 'carbon_audit_category_summary.csv'}")
    print("\nAudit complete. No substitution score has been calculated.")


if __name__ == "__main__":
    main()
