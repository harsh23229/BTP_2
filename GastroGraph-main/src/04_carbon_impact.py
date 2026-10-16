"""
04_carbon_impact.py

Generate carbon-impact evidence for exploratory ingredient candidate pairs.

Inputs:
  dataset/processed/SustainableFoodDB_CarbonFootprints.csv
  GastroGraph-custom/data/embedding_nearest_neighbors.csv

Outputs (GastroGraph-custom/data):
  carbon_ingredient_lookup.csv
  carbon_impact_evidence.csv
  carbon_impact_unmatched_pairs.csv
  carbon_impact_summary.txt

Important:
- GraphSAGE nearest neighbors are candidate pairs, NOT validated substitutes.
- Carbon mapping similarity < 0.85 is flagged for review; this is a project review
  threshold, not an official SustainableFoodDB cutoff.
- The carbon-footprint unit/reference basis must be confirmed from documentation.
- Raw comparisons are retained for auditability. A separate reviewed score is
  provided only when both ingredient mappings meet the chosen review threshold.
"""

from pathlib import Path
import numpy as np
import pandas as pd

CARBON_FILENAME = "SustainableFoodDB_CarbonFootprints.csv"
NEIGHBOR_FILENAME = "embedding_nearest_neighbors.csv"
MAPPING_REVIEW_THRESHOLD = 0.85


def normalize_names(series: pd.Series) -> pd.Series:
    """Normalize case and whitespace without inventing aliases."""
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )


def find_file(project_root: Path, filename: str) -> Path:
    """Search known project locations, then search within the project root."""
    candidates = [
        project_root / "dataset" / "processed" / filename,
        project_root / "dataset" / filename,
        project_root / "GastroGraph-custom" / "data" / filename,
        project_root / "GastroGraph-custom" / "data" / "processed" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = list(project_root.rglob(filename))
    if matches:
        matches.sort(key=lambda p: ("processed" not in str(p).lower(), len(str(p))))
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {filename} under {project_root}. "
        "Check the filename and project folder."
    )


def main() -> None:
    # Expected script path: BTP_2/GastroGraph-main/src/04_carbon_impact.py
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "GastroGraph-custom" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    carbon_path = find_file(project_root, CARBON_FILENAME)
    neighbor_path = find_file(project_root, NEIGHBOR_FILENAME)

    print("=" * 72)
    print("CARBON IMPACT EVIDENCE")
    print("=" * 72)
    print(f"Carbon dataset : {carbon_path}")
    print(f"Candidate pairs: {neighbor_path}")
    print(f"Output folder  : {output_dir}")

    carbon = pd.read_csv(carbon_path, low_memory=False)
    neighbors = pd.read_csv(neighbor_path, low_memory=False)

    required_carbon = {
        "Ingredient", "SueatableIngredient", "Similarity",
        "CarbonFootprint", "Count",
    }
    required_neighbors = {
        "query_ingredient", "neighbor_ingredient", "rank", "cosine_similarity",
    }

    missing_carbon = required_carbon - set(carbon.columns)
    missing_neighbors = required_neighbors - set(neighbors.columns)
    if missing_carbon:
        raise ValueError(f"Carbon dataset missing columns: {sorted(missing_carbon)}")
    if missing_neighbors:
        raise ValueError(
            f"Nearest-neighbor file missing columns: {sorted(missing_neighbors)}"
        )

    # Parse/clean values without silently dropping rows.
    for col in ["Ingredient", "SueatableIngredient"]:
        carbon[col] = carbon[col].astype("string").str.strip()
    for col in ["Similarity", "CarbonFootprint", "Count"]:
        carbon[col] = pd.to_numeric(carbon[col], errors="coerce")

    carbon["ingredient_key"] = normalize_names(carbon["Ingredient"])
    carbon["mapping_similarity_raw"] = carbon["Similarity"]
    # Fix tiny float errors such as 1.0000001; true out-of-range values remain
    # visible in mapping_similarity_raw for inspection.
    carbon["mapping_similarity_clean"] = carbon["Similarity"].clip(lower=0, upper=1)

    carbon["mapping_review_status"] = np.select(
        [
            carbon["Ingredient"].isna() | carbon["Ingredient"].eq(""),
            carbon["SueatableIngredient"].isna() | carbon["SueatableIngredient"].eq(""),
            carbon["CarbonFootprint"].isna(),
            carbon["mapping_similarity_raw"].isna(),
            (carbon["mapping_similarity_raw"] < 0)
            | (carbon["mapping_similarity_raw"] > 1.00001),
            carbon["mapping_similarity_clean"] < MAPPING_REVIEW_THRESHOLD,
        ],
        [
            "MISSING_SOURCE_INGREDIENT",
            "MISSING_SUSTAINABLE_INGREDIENT",
            "MISSING_CARBON_VALUE",
            "MISSING_MAPPING_SIMILARITY",
            "INVALID_MAPPING_SIMILARITY",
            "REVIEW_LOW_MAPPING_SIMILARITY",
        ],
        default="MAPPING_PASSES_PROJECT_REVIEW_THRESHOLD",
    )

    # If case/whitespace normalization causes duplicate keys, do not choose
    # one mapping arbitrarily. Exclude those keys from lookup and report them.
    duplicate_key_mask = carbon["ingredient_key"].duplicated(keep=False)
    collision_keys = set(carbon.loc[duplicate_key_mask, "ingredient_key"].dropna())
    carbon["normalized_name_collision"] = carbon["ingredient_key"].isin(collision_keys)

    carbon.to_csv(output_dir / "carbon_ingredient_lookup.csv", index=False)

    lookup = carbon.loc[
        ~carbon["normalized_name_collision"]
    ].copy()
    lookup = lookup[
        [
            "ingredient_key",
            "Ingredient",
            "SueatableIngredient",
            "CarbonFootprint",
            "Similarity",
            "mapping_similarity_clean",
            "mapping_review_status",
        ]
    ]

    # Add suffixes to avoid ambiguous columns during joins.
    lookup_a = lookup.rename(columns={
        "ingredient_key": "query_key",
        "Ingredient": "carbon_source_name_a",
        "SueatableIngredient": "sustainable_category_a",
        "CarbonFootprint": "carbon_footprint_a",
        "Similarity": "mapping_similarity_raw_a",
        "mapping_similarity_clean": "mapping_similarity_a",
        "mapping_review_status": "mapping_status_a",
    })
    lookup_b = lookup.rename(columns={
        "ingredient_key": "neighbor_key",
        "Ingredient": "carbon_source_name_b",
        "SueatableIngredient": "sustainable_category_b",
        "CarbonFootprint": "carbon_footprint_b",
        "Similarity": "mapping_similarity_raw_b",
        "mapping_similarity_clean": "mapping_similarity_b",
        "mapping_review_status": "mapping_status_b",
    })

    # Prepare the proposed pairs.
    for col in ["query_ingredient", "neighbor_ingredient"]:
        neighbors[col] = neighbors[col].astype("string").str.strip()
    neighbors["rank"] = pd.to_numeric(neighbors["rank"], errors="coerce")
    neighbors["cosine_similarity"] = pd.to_numeric(
        neighbors["cosine_similarity"], errors="coerce"
    )
    neighbors["query_key"] = normalize_names(neighbors["query_ingredient"])
    neighbors["neighbor_key"] = normalize_names(neighbors["neighbor_ingredient"])

    evidence = neighbors.merge(lookup_a, on="query_key", how="left")
    evidence = evidence.merge(lookup_b, on="neighbor_key", how="left")

    cf_a = evidence["carbon_footprint_a"]
    cf_b = evidence["carbon_footprint_b"]
    both_carbon_available = cf_a.notna() & cf_b.notna()
    both_mappings_pass = (
        evidence["mapping_similarity_a"].ge(MAPPING_REVIEW_THRESHOLD)
        & evidence["mapping_similarity_b"].ge(MAPPING_REVIEW_THRESHOLD)
        & evidence["mapping_status_a"].eq("MAPPING_PASSES_PROJECT_REVIEW_THRESHOLD")
        & evidence["mapping_status_b"].eq("MAPPING_PASSES_PROJECT_REVIEW_THRESHOLD")
    )
    score_eligible = both_carbon_available & both_mappings_pass

    # Raw evidence: available for audit/review, but never enough to validate a substitute.
    evidence["carbon_saving_absolute_raw"] = np.where(
        both_carbon_available, cf_a - cf_b, np.nan
    )
    evidence["carbon_reduction_percent_raw"] = np.where(
        both_carbon_available & cf_a.gt(0),
        (cf_a - cf_b) / cf_a * 100.0,
        np.nan,
    )
    max_footprint = pd.concat([cf_a, cf_b], axis=1).max(axis=1)
    evidence["carbon_impact_score_raw"] = np.where(
        both_carbon_available & max_footprint.gt(0),
        (cf_a - cf_b) / max_footprint,
        np.nan,
    )

    # Recommended feature: only populate when both mappings pass the review threshold.
    evidence["carbon_impact_score"] = evidence["carbon_impact_score_raw"].where(
        score_eligible, np.nan
    )
    evidence["carbon_saving_absolute"] = evidence["carbon_saving_absolute_raw"].where(
        score_eligible, np.nan
    )
    evidence["carbon_reduction_percent"] = evidence[
        "carbon_reduction_percent_raw"
    ].where(score_eligible, np.nan)

    evidence["carbon_evidence_status"] = np.select(
        [
            ~both_carbon_available,
            both_carbon_available & ~both_mappings_pass,
            score_eligible & cf_a.eq(cf_b),
            score_eligible & cf_b.lt(cf_a),
            score_eligible & cf_b.gt(cf_a),
        ],
        [
            "CARBON_MAPPING_UNAVAILABLE",
            "MAPPING_REQUIRES_REVIEW_SCORE_WITHHELD",
            "EQUAL_RECORDED_FOOTPRINTS",
            "LOWER_RECORDED_FOOTPRINT_CANDIDATE_UNVALIDATED",
            "HIGHER_RECORDED_FOOTPRINT_CANDIDATE_UNVALIDATED",
        ],
        default="REVIEW",
    )

    evidence["candidate_validity"] = "NOT_VALIDATED_BY_CARBON"
    evidence["embedding_similarity_interpretation"] = (
        "Context only; existing diagnostics warn of potential embedding collapse."
    )
    evidence["carbon_unit_reference_basis"] = "NOT_CONFIRMED"

    evidence = evidence.sort_values(
        ["query_ingredient", "rank"], kind="stable"
    )
    evidence.to_csv(output_dir / "carbon_impact_evidence.csv", index=False)

    unmatched = evidence.loc[~both_carbon_available].copy()
    unmatched.to_csv(output_dir / "carbon_impact_unmatched_pairs.csv", index=False)

    score_count = int(evidence["carbon_impact_score"].notna().sum())
    raw_count = int(evidence["carbon_impact_score_raw"].notna().sum())
    summary = [
        "CARBON IMPACT EVIDENCE SUMMARY",
        "=" * 36,
        f"Carbon input: {carbon_path}",
        f"Candidate-pair input: {neighbor_path}",
        f"Candidate pairs: {len(evidence):,}",
        f"Pairs with both carbon values: {int(both_carbon_available.sum()):,}",
        f"Pairs missing at least one carbon value: {int((~both_carbon_available).sum()):,}",
        f"Pairs with both mappings passing project threshold: {int(score_eligible.sum()):,}",
        f"Pairs with raw score available (for review only): {raw_count:,}",
        f"Pairs with recommended carbon_impact_score populated: {score_count:,}",
        f"Pairs requiring mapping review: {int((both_carbon_available & ~both_mappings_pass).sum()):,}",
        f"Normalized ingredient-name collisions excluded from lookup: {len(collision_keys):,}",
        "",
        "SCORE:",
        "carbon_impact_score = (CF_A - CF_B) / max(CF_A, CF_B)",
        "Positive: B has a lower recorded footprint; zero: equal; negative: B is higher.",
        "The recommended score is withheld unless both mappings pass the project review threshold.",
        "",
        "LIMITATIONS:",
        "1. Nearest neighbors are exploratory candidates, not validated substitutes.",
        "2. Mapping threshold 0.85 is a project review rule, not an official dataset cutoff.",
        "3. Carbon-footprint unit/reference basis is not confirmed from the CSV columns alone.",
        "4. The current graph embedding diagnostics warn of potential embedding collapse.",
        "5. A carbon score measures environmental evidence only, not culinary or nutritional suitability.",
        "",
        "OUTPUTS:",
        "carbon_ingredient_lookup.csv",
        "carbon_impact_evidence.csv",
        "carbon_impact_unmatched_pairs.csv",
        "carbon_impact_summary.txt",
    ]
    (output_dir / "carbon_impact_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    print("\nRESULTS")
    print("-" * 72)
    print(f"Candidate pairs                              : {len(evidence):,}")
    print(f"Pairs with both carbon values                : {int(both_carbon_available.sum()):,}")
    print(f"Pairs missing at least one carbon value      : {int((~both_carbon_available).sum()):,}")
    print(f"Pairs passing mapping review threshold       : {int(score_eligible.sum()):,}")
    print(f"Pairs with raw score (review only)           : {raw_count:,}")
    print(f"Pairs with recommended score populated       : {score_count:,}")
    print(f"Normalized-name collision keys excluded      : {len(collision_keys):,}")
    print("\nSaved:")
    print(f"  {output_dir / 'carbon_ingredient_lookup.csv'}")
    print(f"  {output_dir / 'carbon_impact_evidence.csv'}")
    print(f"  {output_dir / 'carbon_impact_unmatched_pairs.csv'}")
    print(f"  {output_dir / 'carbon_impact_summary.txt'}")
    print("\nNo candidate is declared a valid substitute by this script.")


if __name__ == "__main__":
    main()
