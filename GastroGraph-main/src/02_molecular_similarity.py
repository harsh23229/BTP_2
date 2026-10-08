"""
02_molecular_similarity.py

Purpose
-------
Convert the audited FlavorDB pairwise molecular-overlap results into
the molecular similarity component used by the later substitution
pipeline.

Primary molecular similarity:
    IDF-weighted Jaccard similarity

Important:
    Molecular similarity is evidence of molecular-profile similarity.
    It is NOT, by itself, a substitution suitability score.

The later substitution pipeline will combine this feature with:
    - GraphSAGE / graph-context compatibility
    - carbon footprint
    - nutrition preservation
    - hard constraints

Input
-----
data/molecule_audit_pairwise_overlap.csv

Output
------
data/molecular_similarity.csv
data/molecular_similarity_summary.txt
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# PATHS
# ================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT.parent
    / "GastroGraph-custom"
    / "data"
)

INPUT_FILE = (
    DATA_DIR
    / "molecule_audit_pairwise_overlap.csv"
)

OUTPUT_FILE = (
    DATA_DIR
    / "molecular_similarity.csv"
)

SUMMARY_FILE = (
    DATA_DIR
    / "molecular_similarity_summary.txt"
)


# ================================================================
# REQUIRED COLUMNS
# ================================================================

REQUIRED_COLUMNS = [
    "entity_id_a",
    "entity_id_b",
    "entity_a",
    "entity_b",
    "molecules_a",
    "molecules_b",
    "shared_molecules",
    "jaccard",
    "weighted_jaccard_idf",
    "low_evidence_flag",
    "identical_profile_flag",
]


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def clean_boolean_column(series):
    """
    Convert boolean-like CSV values into actual Python booleans.
    """

    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
    )


def validate_input_columns(df):
    """Check that the audit file contains everything required."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Input file is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


def validate_similarity_values(df):
    """
    Validate that the IDF-weighted Jaccard values are legitimate.

    Jaccard similarity must lie in [0, 1].
    """

    values = df["weighted_jaccard_idf"]

    invalid = (
        values.isna()
        | ~np.isfinite(values)
        | (values < 0)
        | (values > 1)
    )

    invalid_count = int(invalid.sum())

    if invalid_count > 0:
        raise ValueError(
            "Found "
            f"{invalid_count:,} invalid IDF-weighted Jaccard values."
        )


def validate_pair_structure(df):
    """
    Check basic structural properties of the pairwise table.
    """

    # No self-pairs should exist.
    self_pairs = (
        df["entity_id_a"]
        == df["entity_id_b"]
    )

    self_pair_count = int(self_pairs.sum())

    if self_pair_count > 0:
        raise ValueError(
            f"Found {self_pair_count:,} self-pairs."
        )

    # A pair should have at least one shared molecule because this
    # audit table contains pairs with >= 1 shared molecule.
    invalid_shared = (
        df["shared_molecules"] < 1
    )

    invalid_shared_count = int(invalid_shared.sum())

    if invalid_shared_count > 0:
        raise ValueError(
            "Found pairs with zero shared molecules: "
            f"{invalid_shared_count:,}"
        )


# ================================================================
# LOAD AUDIT DATA
# ================================================================

def load_pairwise_audit():
    """Load and validate the audited pairwise molecular results."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Expected:\n"
            "data/molecule_audit_pairwise_overlap.csv"
        )

    print(f"Loading:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Rows loaded : {len(df):,}"
    )

    print(
        f"Columns     : {len(df.columns)}"
    )

    validate_input_columns(df)

    df["low_evidence_flag"] = clean_boolean_column(
        df["low_evidence_flag"]
    )

    df["identical_profile_flag"] = clean_boolean_column(
        df["identical_profile_flag"]
    )

    validate_similarity_values(df)

    validate_pair_structure(df)

    return df


# ================================================================
# BUILD MOLECULAR SIMILARITY TABLE
# ================================================================

def build_similarity_table(df):
    """
    Create the clean molecular-similarity table.

    weighted_jaccard_idf is retained as the original audited
    metric and is also exposed as the canonical
    molecular_similarity feature.
    """

    similarity_df = df[
        [
            "entity_id_a",
            "entity_id_b",
            "entity_a",
            "entity_b",
            "molecules_a",
            "molecules_b",
            "profile_size_ratio",
            "shared_molecules",
            "union_molecules",
            "jaccard",
            "overlap_coefficient",
            "coverage_a",
            "coverage_b",
            "symmetric_coverage",
            "weighted_shared_idf",
            "weighted_jaccard_idf",
            "weighted_coverage_a_idf",
            "weighted_coverage_b_idf",
            "low_evidence_flag",
            "identical_profile_flag",
        ]
    ].copy()

    # Canonical feature name used by the later substitution pipeline.
    similarity_df["molecular_similarity"] = (
        similarity_df["weighted_jaccard_idf"]
    )

    return similarity_df


# ================================================================
# SUMMARY STATISTICS
# ================================================================

def calculate_summary(df):
    """Calculate useful diagnostics without filtering the data."""

    similarity = df["molecular_similarity"]

    summary = {}

    summary["pairs"] = len(df)

    summary["unique_entities"] = len(
        set(df["entity_id_a"]).union(
            set(df["entity_id_b"])
        )
    )

    summary["low_evidence_pairs"] = int(
        df["low_evidence_flag"].sum()
    )

    summary["identical_profile_pairs"] = int(
        df["identical_profile_flag"].sum()
    )

    summary["similarity_min"] = float(
        similarity.min()
    )

    summary["similarity_q25"] = float(
        similarity.quantile(0.25)
    )

    summary["similarity_median"] = float(
        similarity.median()
    )

    summary["similarity_mean"] = float(
        similarity.mean()
    )

    summary["similarity_q75"] = float(
        similarity.quantile(0.75)
    )

    summary["similarity_max"] = float(
        similarity.max()
    )

    summary["similarity_zero"] = int(
        (similarity == 0).sum()
    )

    summary["similarity_one"] = int(
        (similarity == 1).sum()
    )

    return summary


# ================================================================
# HIGH-SIMILARITY DIAGNOSTICS
# ================================================================

def print_high_similarity_examples(df, n=20):
    """
    Print the highest molecular-similarity pairs.

    This is diagnostic only.

    A high molecular similarity score does NOT mean that the
    ingredients are valid substitutions.
    """

    section("TOP MOLECULAR-SIMILARITY PAIRS")

    top = (
        df
        .sort_values(
            [
                "molecular_similarity",
                "shared_molecules",
            ],
            ascending=False,
        )
        .head(n)
    )

    columns = [
        "entity_a",
        "entity_b",
        "molecules_a",
        "molecules_b",
        "shared_molecules",
        "molecular_similarity",
        "low_evidence_flag",
        "identical_profile_flag",
    ]

    print(
        top[columns].to_string(index=False)
    )


# ================================================================
# SUMMARY REPORT
# ================================================================

def write_summary(summary):
    """Write the molecular-similarity summary report."""

    lines = []

    lines.append(
        "FLAVORDB MOLECULAR SIMILARITY"
    )
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "Primary molecular similarity metric:"
    )
    lines.append(
        "IDF-weighted Jaccard similarity"
    )
    lines.append("")
    lines.append(
        "Formula:"
    )
    lines.append(
        "sum(IDF(shared molecules)) / "
        "sum(IDF(union molecules))"
    )
    lines.append("")
    lines.append(
        "This is a molecular-profile similarity feature."
    )
    lines.append(
        "It is NOT a standalone substitution score."
    )
    lines.append("")
    lines.append(
        "PAIRWISE DATA"
    )
    lines.append("-" * 70)
    lines.append(
        f"Pairs                       : {summary['pairs']:,}"
    )
    lines.append(
        f"Unique entities             : "
        f"{summary['unique_entities']:,}"
    )
    lines.append(
        f"Low-evidence pairs         : "
        f"{summary['low_evidence_pairs']:,}"
    )
    lines.append(
        f"Identical-profile pairs    : "
        f"{summary['identical_profile_pairs']:,}"
    )
    lines.append("")
    lines.append(
        "MOLECULAR SIMILARITY DISTRIBUTION"
    )
    lines.append("-" * 70)
    lines.append(
        f"Minimum                     : "
        f"{summary['similarity_min']:.6f}"
    )
    lines.append(
        f"25th percentile            : "
        f"{summary['similarity_q25']:.6f}"
    )
    lines.append(
        f"Median                      : "
        f"{summary['similarity_median']:.6f}"
    )
    lines.append(
        f"Mean                        : "
        f"{summary['similarity_mean']:.6f}"
    )
    lines.append(
        f"75th percentile            : "
        f"{summary['similarity_q75']:.6f}"
    )
    lines.append(
        f"Maximum                     : "
        f"{summary['similarity_max']:.6f}"
    )
    lines.append(
        f"Pairs with similarity = 0  : "
        f"{summary['similarity_zero']:,}"
    )
    lines.append(
        f"Pairs with similarity = 1  : "
        f"{summary['similarity_one']:,}"
    )
    lines.append("")
    lines.append(
        "INTERPRETATION"
    )
    lines.append("-" * 70)
    lines.append(
        "IDF weighting reduces the influence of molecules "
        "that occur in many FlavorDB entities."
    )
    lines.append(
        "Low-evidence profiles are retained and flagged "
        "rather than silently discarded."
    )
    lines.append(
        "Identical molecular profiles are retained and flagged."
    )
    lines.append(
        "A molecular similarity of 1 does not imply that two "
        "ingredients are valid substitutes."
    )
    lines.append("")
    lines.append(
        "The later substitution model should combine this "
        "feature with graph/context compatibility, carbon "
        "impact, nutrition preservation, and constraints."
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(lines))


# ================================================================
# MAIN
# ================================================================

def main():

    section("FLAVORDB MOLECULAR SIMILARITY")

    print(
        "Purpose: create the molecular similarity feature "
        "for the substitution pipeline."
    )

    print(
        "Primary metric: IDF-weighted Jaccard"
    )

    print(
        "No final substitution score is created here."
    )

    # ------------------------------------------------------------
    # Load audited pairwise results
    # ------------------------------------------------------------

    section("1. LOAD AUDITED PAIRWISE RESULTS")

    audit_df = load_pairwise_audit()

    print(
        "Input validation: PASSED"
    )

    # ------------------------------------------------------------
    # Build clean similarity table
    # ------------------------------------------------------------

    section("2. BUILD MOLECULAR SIMILARITY TABLE")

    similarity_df = build_similarity_table(
        audit_df
    )

    print(
        f"Similarity rows : "
        f"{len(similarity_df):,}"
    )

    print(
        "Canonical feature: molecular_similarity"
    )

    print(
        "Source metric    : weighted_jaccard_idf"
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    section("3. CALCULATE SUMMARY")

    summary = calculate_summary(
        similarity_df
    )

    print(
        f"Pairs                  : "
        f"{summary['pairs']:,}"
    )

    print(
        f"Unique entities        : "
        f"{summary['unique_entities']:,}"
    )

    print(
        f"Low-evidence pairs     : "
        f"{summary['low_evidence_pairs']:,}"
    )

    print(
        f"Identical-profile pairs: "
        f"{summary['identical_profile_pairs']:,}"
    )

    print(
        f"Median similarity      : "
        f"{summary['similarity_median']:.6f}"
    )

    print(
        f"Mean similarity        : "
        f"{summary['similarity_mean']:.6f}"
    )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    print_high_similarity_examples(
        similarity_df
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    section("4. SAVE OUTPUT")

    similarity_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    write_summary(
        summary
    )

    print(
        f"Similarity table:\n{OUTPUT_FILE}"
    )

    print(
        f"Summary report:\n{SUMMARY_FILE}"
    )

    section("DONE")

    print(
        "Molecular similarity component created."
    )

    print(
        "Next step:"
    )

    print(
        "Connect this molecular feature to the "
        "RecipeDB ingredient/candidate mapping."
    )


if __name__ == "__main__":
    main()