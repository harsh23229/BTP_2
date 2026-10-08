"""
01_molecule_audit.py

Final FlavorDB molecular-profile audit and similarity preparation.

Purpose
-------
This is the final audit step before using FlavorDB molecular similarity
inside the BTP substitution framework.

It:
1. validates FlavorDB entity/link/molecule consistency
2. builds entity -> molecule profiles
3. computes molecule frequency and smoothed IDF
4. computes raw and IDF-weighted molecular overlap
5. uses IDF-weighted Jaccard as the primary molecular similarity
6. investigates identical molecular profiles with categories and molecule metadata
7. generates candidate name matches for unresolved sanity-check names
8. saves a clean pairwise similarity table for downstream modeling

Important
---------
The molecular similarity score is NOT a complete substitution score.
It should later be combined with GraphSAGE/contextual compatibility,
carbon footprint, nutrition preservation, and hard constraints.

Run from GastroGraph-main:
    python src/01_molecule_audit.py
"""

from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import math
import re

import numpy as np
import pandas as pd


# ================================================================
# 1. PATHS
# ================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
GASTROGRAPH_ROOT = SCRIPT_DIR.parent
BTP_ROOT = GASTROGRAPH_ROOT.parent

DATASET_ROOT = BTP_ROOT / "dataset"
PROCESSED_DIR = DATASET_ROOT / "processed"

CUSTOM_ROOT = BTP_ROOT / "GastroGraph-custom"
CUSTOM_DATA_DIR = CUSTOM_ROOT / "data"
CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)

ENTITIES_FILE = PROCESSED_DIR / "FlavorDB_entities.csv"
ENTITY_MOLECULES_FILE = PROCESSED_DIR / "FlavorDB_entity_molecules.csv"
MOLECULES_FILE = PROCESSED_DIR / "FlavorDB_molecules.csv"

ENTITY_STATS_FILE = CUSTOM_DATA_DIR / "molecule_audit_entity_stats.csv"
MOLECULE_FREQ_FILE = CUSTOM_DATA_DIR / "molecule_audit_molecule_frequency.csv"
PAIR_EXAMPLES_FILE = CUSTOM_DATA_DIR / "molecule_audit_pair_examples.csv"
PAIRWISE_FILE = CUSTOM_DATA_DIR / "molecule_audit_pairwise_overlap.csv"
IDENTICAL_PROFILES_FILE = CUSTOM_DATA_DIR / "molecule_audit_identical_profiles.csv"
IDENTICAL_ENTITIES_FILE = CUSTOM_DATA_DIR / "molecule_audit_identical_profile_entities.csv"
IDENTICAL_MOLECULES_FILE = CUSTOM_DATA_DIR / "molecule_audit_identical_profile_molecules.csv"
NAME_MATCH_FILE = CUSTOM_DATA_DIR / "molecule_audit_name_matching.csv"
NAME_CANDIDATES_FILE = CUSTOM_DATA_DIR / "molecule_audit_name_candidates.csv"
SUMMARY_FILE = CUSTOM_DATA_DIR / "molecule_audit_summary.txt"


# ================================================================
# 2. SETTINGS
# ================================================================

EXAMPLE_PAIRS = [
    ("beef", "beef stock"),
    ("chicken", "chicken breast"),
    ("tomato", "tomato paste"),
    ("olive oil", "extra virgin olive oil"),
    ("black pepper", "pepper"),
    ("onion", "yellow onion"),
    ("beef", "bay leaf"),
    ("beef", "wine vinegar"),
    ("chicken", "dill seed"),
    ("lobster", "tofu"),
    ("fish", "broccoli"),
]

# Profiles below this size are retained, but flagged as low evidence.
MIN_PROFILE_SIZE = 3

# Number of unresolved-name candidates to save.
TOP_NAME_CANDIDATES = 5

# A candidate must have at least this combined name score to be shown.
# Candidates are NEVER silently accepted as mappings.
MIN_CANDIDATE_SCORE = 0.35


# ================================================================
# 3. HELPERS
# ================================================================

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_name(value):
    text = clean_text(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(value):
    return set(normalize_name(value).split())


def jaccard(set_a, set_b):
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def overlap_coefficient(set_a, set_b):
    minimum = min(len(set_a), len(set_b))
    return len(set_a & set_b) / minimum if minimum else 0.0


def symmetric_coverage(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    shared = len(set_a & set_b)
    return 0.5 * (shared / len(set_a) + shared / len(set_b))


def smoothed_idf(entity_count, molecule_frequency):
    if entity_count <= 0:
        return 0.0
    return math.log((entity_count + 1) / (molecule_frequency + 1)) + 1.0


def weighted_sum(molecules, weights):
    return sum(weights.get(m, 0.0) for m in molecules)


def profile_signature(molecule_set):
    raw = ",".join(str(x) for x in sorted(molecule_set))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def safe_join(values):
    return "|".join(str(x) for x in values)


def first_alias(aliases):
    return sorted(aliases)[0] if aliases else ""


def similarity_score(query, candidate):
    """
    Transparent name-candidate score.

    Combines:
    - exact normalized match
    - substring containment
    - token Jaccard
    - SequenceMatcher ratio

    This is ONLY for candidate generation.
    It is not used to change FlavorDB mappings automatically.
    """
    q = normalize_name(query)
    c = normalize_name(candidate)

    if not q or not c:
        return 0.0

    if q == c:
        return 1.0

    q_tokens = set(q.split())
    c_tokens = set(c.split())

    token_j = (
        len(q_tokens & c_tokens) / len(q_tokens | c_tokens)
        if q_tokens | c_tokens
        else 0.0
    )

    substring = 1.0 if (q in c or c in q) else 0.0
    sequence = SequenceMatcher(None, q, c).ratio()

    return (
        0.45 * token_j
        + 0.25 * substring
        + 0.30 * sequence
    )


# ================================================================
# 4. LOAD DATA
# ================================================================

def load_data():
    section("LOADING FLAVORDB DATA")

    required = [
        ENTITIES_FILE,
        ENTITY_MOLECULES_FILE,
        MOLECULES_FILE,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}\n\n"
                "Check that the FlavorDB files are in dataset/processed/."
            )

    entities = pd.read_csv(ENTITIES_FILE)
    entity_molecules = pd.read_csv(ENTITY_MOLECULES_FILE)
    molecules = pd.read_csv(MOLECULES_FILE)

    print(f"FlavorDB entities     : {len(entities):,}")
    print(f"Entity-molecule links : {len(entity_molecules):,}")
    print(f"FlavorDB molecules    : {len(molecules):,}")

    return entities, entity_molecules, molecules


# ================================================================
# 5. BASIC DATA AUDIT
# ================================================================

def audit_basic_structure(entities, entity_molecules, molecules, summary):
    section("1. BASIC DATA INTEGRITY")

    required_entity_cols = {"entity_id"}
    required_link_cols = {"entity_id", "pubchem_id"}
    required_molecule_cols = {"pubchem_id"}

    if not required_entity_cols.issubset(entities.columns):
        raise ValueError("FlavorDB_entities.csv is missing entity_id.")

    if not required_link_cols.issubset(entity_molecules.columns):
        raise ValueError(
            "FlavorDB_entity_molecules.csv must contain entity_id and pubchem_id."
        )

    if not required_molecule_cols.issubset(molecules.columns):
        raise ValueError("FlavorDB_molecules.csv is missing pubchem_id.")

    duplicate_entity_ids = int(entities["entity_id"].duplicated().sum())
    duplicate_links = int(
        entity_molecules.duplicated(
            subset=["entity_id", "pubchem_id"]
        ).sum()
    )
    duplicate_molecule_ids = int(molecules["pubchem_id"].duplicated().sum())

    link_ids = set(
        pd.to_numeric(
            entity_molecules["pubchem_id"],
            errors="coerce",
        ).dropna().astype(int)
    )

    molecule_table_ids = set(
        pd.to_numeric(
            molecules["pubchem_id"],
            errors="coerce",
        ).dropna().astype(int)
    )

    missing_ids = link_ids - molecule_table_ids
    unused_ids = molecule_table_ids - link_ids

    print(f"Duplicate entity_id rows          : {duplicate_entity_ids:,}")
    print(f"Duplicate entity-molecule links   : {duplicate_links:,}")
    print(f"Duplicate molecule-table IDs      : {duplicate_molecule_ids:,}")
    print(f"Referenced molecule IDs           : {len(link_ids):,}")
    print(f"Molecule-table IDs                : {len(molecule_table_ids):,}")
    print(f"Referenced IDs missing in table   : {len(missing_ids):,}")
    print(f"Unused molecule-table IDs         : {len(unused_ids):,}")

    summary.extend([
        "BASIC DATA INTEGRITY",
        "-" * 72,
        f"Entities: {len(entities):,}",
        f"Entity-molecule links: {len(entity_molecules):,}",
        f"Molecule rows: {len(molecules):,}",
        f"Duplicate entity_id rows: {duplicate_entity_ids:,}",
        f"Duplicate entity-molecule links: {duplicate_links:,}",
        f"Duplicate molecule-table IDs: {duplicate_molecule_ids:,}",
        f"Referenced molecule IDs: {len(link_ids):,}",
        f"Molecule-table IDs: {len(molecule_table_ids):,}",
        f"Referenced IDs missing in molecule table: {len(missing_ids):,}",
        f"Unused molecule-table IDs: {len(unused_ids):,}",
        "",
    ])

    return missing_ids


# ================================================================
# 6. BUILD PROFILES + ALIASES
# ================================================================

def build_profiles(entities, entity_molecules):
    section("2. BUILDING ENTITY -> MOLECULE PROFILES")

    links = entity_molecules[["entity_id", "pubchem_id"]].copy()

    links["entity_id"] = pd.to_numeric(links["entity_id"], errors="coerce")
    links["pubchem_id"] = pd.to_numeric(links["pubchem_id"], errors="coerce")
    links = links.dropna(subset=["entity_id", "pubchem_id"])

    links["entity_id"] = links["entity_id"].astype(int)
    links["pubchem_id"] = links["pubchem_id"].astype(int)
    links = links.drop_duplicates(["entity_id", "pubchem_id"])

    molecule_sets = (
        links.groupby("entity_id")["pubchem_id"]
        .apply(set)
        .to_dict()
    )

    alias_columns = [
        "entity_alias_readable",
        "entity_alias",
        "entity_alias_basket",
        "entity_alias_synonyms",
        "natural_source_name",
    ]

    available_alias_columns = [
        c for c in alias_columns if c in entities.columns
    ]

    entity_aliases = {}
    entity_meta = {}

    for _, row in entities.iterrows():
        entity_id = int(row["entity_id"])

        aliases = set()
        for column in available_alias_columns:
            value = row.get(column, "")
            if pd.isna(value):
                continue

            text = str(value).strip()
            if not text:
                continue

            aliases.add(text)

            for part in re.split(r"[|;,/]", text):
                part = part.strip()
                if part:
                    aliases.add(part)

        normalized_aliases = {
            normalize_name(x) for x in aliases if normalize_name(x)
        }

        entity_aliases[entity_id] = normalized_aliases

        meta = {}
        for column in [
            "category",
            "category_readable",
            "natural_source_name",
            "entity_alias_readable",
            "entity_alias",
            "entity_alias_basket",
            "entity_alias_synonyms",
        ]:
            if column in entities.columns:
                meta[column] = clean_text(row.get(column, ""))
            else:
                meta[column] = ""

        entity_meta[entity_id] = meta

    print(f"Entities with molecule profiles : {len(molecule_sets):,}")
    print(f"Entities without profiles       : {len(entities) - len(molecule_sets):,}")
    print(f"Alias columns used              : {', '.join(available_alias_columns)}")

    return molecule_sets, entity_aliases, entity_meta


# ================================================================
# 7. ENTITY PROFILE STATISTICS
# ================================================================

def build_entity_statistics(entities, molecule_sets, entity_aliases):
    section("3. PROFILE SIZE AUDIT")

    rows = []

    for _, row in entities.iterrows():
        entity_id = int(row["entity_id"])
        molecule_set = molecule_sets.get(entity_id, set())
        aliases = entity_aliases.get(entity_id, set())

        rows.append({
            "entity_id": entity_id,
            "entity_name": first_alias(aliases),
            "alias_count": len(aliases),
            "molecule_count": len(molecule_set),
            "small_profile_flag": len(molecule_set) < MIN_PROFILE_SIZE,
            "profile_signature": profile_signature(molecule_set),
        })

    stats = pd.DataFrame(rows)
    positive = stats.loc[stats["molecule_count"] > 0, "molecule_count"]

    print(f"Entities total                  : {len(stats):,}")
    print(f"Entities with molecules         : {len(positive):,}")
    print(f"Entities without molecules      : {(stats['molecule_count'] == 0).sum():,}")
    print(
        f"Profiles with < {MIN_PROFILE_SIZE} molecules : "
        f"{stats['small_profile_flag'].sum():,}"
    )

    if len(positive):
        print(f"Mean molecules/entity            : {positive.mean():.2f}")
        print(f"Median molecules/entity          : {positive.median():.2f}")
        print(f"Min molecules/entity             : {positive.min():.0f}")
        print(f"Max molecules/entity             : {positive.max():.0f}")
        print(f"Q25                              : {positive.quantile(.25):.2f}")
        print(f"Q75                              : {positive.quantile(.75):.2f}")

    return stats


# ================================================================
# 8. MOLECULE FREQUENCY + IDF
# ================================================================

def build_frequency_table(molecule_sets, total_entities):
    section("4. MOLECULE FREQUENCY + IDF")

    counter = Counter()

    for molecule_set in molecule_sets.values():
        for molecule_id in molecule_set:
            counter[molecule_id] += 1

    rows = []
    for molecule_id, frequency in counter.items():
        rows.append({
            "pubchem_id": molecule_id,
            "entity_frequency": frequency,
            "entity_frequency_fraction": frequency / total_entities,
            "idf": smoothed_idf(total_entities, frequency),
        })

    frequency_df = pd.DataFrame(rows)

    if frequency_df.empty:
        return frequency_df, {}

    frequency_df = frequency_df.sort_values(
        ["entity_frequency", "pubchem_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    weights = dict(
        zip(frequency_df["pubchem_id"], frequency_df["idf"])
    )

    print(f"Unique linked molecules           : {len(frequency_df):,}")
    print(
        f"Most frequent molecule frequency : "
        f"{frequency_df['entity_frequency'].max():,}"
    )
    print(f"Mean IDF                          : {frequency_df['idf'].mean():.4f}")
    print(f"Median IDF                        : {frequency_df['idf'].median():.4f}")
    print(f"Minimum IDF                       : {frequency_df['idf'].min():.4f}")
    print(f"Maximum IDF                       : {frequency_df['idf'].max():.4f}")

    print("\nTop 10 most frequent molecules:")
    print(frequency_df.head(10).to_string(index=False))

    return frequency_df, weights


# ================================================================
# 9. NAME INDEX
# ================================================================

def build_name_index(entity_aliases):
    index = defaultdict(set)
    for entity_id, aliases in entity_aliases.items():
        for alias in aliases:
            index[alias].add(entity_id)
    return index


def resolve_name(name, name_index):
    normalized = normalize_name(name)
    matches = sorted(name_index.get(normalized, set()))

    if len(matches) == 1:
        return matches[0], "EXACT_ALIAS"
    if len(matches) > 1:
        return None, "AMBIGUOUS_ALIAS"
    return None, "NOT_FOUND"


# ================================================================
# 10. PAIR METRICS
# ================================================================

def calculate_pair_metrics(
    entity_a,
    entity_b,
    set_a,
    set_b,
    weights,
):
    shared = set_a & set_b
    union = set_a | set_b

    weighted_shared = weighted_sum(shared, weights)
    weighted_union = weighted_sum(union, weights)
    weighted_a = weighted_sum(set_a, weights)
    weighted_b = weighted_sum(set_b, weights)

    return {
        "entity_id_a": entity_a,
        "entity_id_b": entity_b,
        "molecules_a": len(set_a),
        "molecules_b": len(set_b),
        "profile_size_ratio": (
            min(len(set_a), len(set_b)) / max(len(set_a), len(set_b))
            if set_a and set_b else 0.0
        ),
        "shared_molecules": len(shared),
        "union_molecules": len(union),
        "jaccard": jaccard(set_a, set_b),
        "overlap_coefficient": overlap_coefficient(set_a, set_b),
        "coverage_a": len(shared) / len(set_a) if set_a else 0.0,
        "coverage_b": len(shared) / len(set_b) if set_b else 0.0,
        "symmetric_coverage": symmetric_coverage(set_a, set_b),
        "weighted_shared_idf": weighted_shared,
        "weighted_jaccard_idf": (
            weighted_shared / weighted_union if weighted_union else 0.0
        ),
        "weighted_coverage_a_idf": (
            weighted_shared / weighted_a if weighted_a else 0.0
        ),
        "weighted_coverage_b_idf": (
            weighted_shared / weighted_b if weighted_b else 0.0
        ),
        "low_evidence_flag": (
            len(set_a) < MIN_PROFILE_SIZE
            or len(set_b) < MIN_PROFILE_SIZE
        ),
    }


# ================================================================
# 11. EXPLICIT SANITY PAIRS
# ================================================================

def audit_example_pairs(molecule_sets, entity_aliases, weights):
    section("5. EXPLICIT SANITY-PAIR AUDIT")

    name_index = build_name_index(entity_aliases)
    rows = []

    for requested_a, requested_b in EXAMPLE_PAIRS:
        id_a, status_a = resolve_name(requested_a, name_index)
        id_b, status_b = resolve_name(requested_b, name_index)

        if id_a is None or id_b is None:
            rows.append({
                "ingredient_a": requested_a,
                "ingredient_b": requested_b,
                "status": (
                    "AMBIGUOUS"
                    if "AMBIGUOUS_ALIAS" in (status_a, status_b)
                    else "NOT_FOUND"
                ),
                "match_status_a": status_a,
                "match_status_b": status_b,
            })
            print(f"{requested_a} <-> {requested_b}: {rows[-1]['status']}")
            continue

        row = calculate_pair_metrics(
            id_a,
            id_b,
            molecule_sets.get(id_a, set()),
            molecule_sets.get(id_b, set()),
            weights,
        )

        row.update({
            "ingredient_a": requested_a,
            "ingredient_b": requested_b,
            "status": "OK",
            "match_status_a": status_a,
            "match_status_b": status_b,
        })
        rows.append(row)

        print(
            f"{requested_a} <-> {requested_b}: "
            f"weighted_J={row['weighted_jaccard_idf']:.4f}, "
            f"raw_J={row['jaccard']:.4f}, "
            f"shared={row['shared_molecules']}"
        )

    return pd.DataFrame(rows)


# ================================================================
# 12. ALL-PAIR MOLECULAR SIMILARITY
# ================================================================

def build_pairwise_similarity(molecule_sets, entity_aliases, weights):
    section("6. FINAL MOLECULAR SIMILARITY TABLE")

    molecule_to_entities = defaultdict(list)

    for entity_id, molecule_set in molecule_sets.items():
        for molecule_id in molecule_set:
            molecule_to_entities[molecule_id].append(entity_id)

    # Count candidate pairs by shared molecule.
    shared_counts = Counter()

    for entity_ids in molecule_to_entities.values():
        entity_ids = sorted(set(entity_ids))

        for i, entity_a in enumerate(entity_ids):
            for entity_b in entity_ids[i + 1:]:
                shared_counts[(entity_a, entity_b)] += 1

    profile_sizes = {
        entity_id: len(profile)
        for entity_id, profile in molecule_sets.items()
    }

    weighted_sizes = {
        entity_id: weighted_sum(profile, weights)
        for entity_id, profile in molecule_sets.items()
    }

    names = {
        entity_id: first_alias(entity_aliases.get(entity_id, set()))
        for entity_id in molecule_sets
    }

    rows = []

    for (entity_a, entity_b), shared_count in shared_counts.items():
        set_a = molecule_sets[entity_a]
        set_b = molecule_sets[entity_b]

        shared = set_a & set_b
        size_a = profile_sizes[entity_a]
        size_b = profile_sizes[entity_b]

        weighted_shared = weighted_sum(shared, weights)
        weighted_union = (
            weighted_sizes[entity_a]
            + weighted_sizes[entity_b]
            - weighted_shared
        )

        union_count = size_a + size_b - shared_count

        # PRIMARY MOLECULAR SIMILARITY:
        # IDF-weighted Jaccard.
        # Rare molecules contribute more than ubiquitous molecules.
        molecular_similarity = (
            weighted_shared / weighted_union
            if weighted_union else 0.0
        )

        rows.append({
            "entity_id_a": entity_a,
            "entity_id_b": entity_b,
            "entity_a": names[entity_a],
            "entity_b": names[entity_b],
            "molecules_a": size_a,
            "molecules_b": size_b,
            "profile_size_ratio": (
                min(size_a, size_b) / max(size_a, size_b)
                if max(size_a, size_b) else 0.0
            ),
            "shared_molecules": shared_count,
            "union_molecules": union_count,
            "jaccard": (
                shared_count / union_count if union_count else 0.0
            ),
            "overlap_coefficient": (
                shared_count / min(size_a, size_b)
                if min(size_a, size_b) else 0.0
            ),
            "coverage_a": shared_count / size_a if size_a else 0.0,
            "coverage_b": shared_count / size_b if size_b else 0.0,
            "symmetric_coverage": (
                0.5 * (shared_count / size_a + shared_count / size_b)
                if size_a and size_b else 0.0
            ),
            "weighted_shared_idf": weighted_shared,
            "weighted_jaccard_idf": molecular_similarity,
            "weighted_coverage_a_idf": (
                weighted_shared / weighted_sizes[entity_a]
                if weighted_sizes[entity_a] else 0.0
            ),
            "weighted_coverage_b_idf": (
                weighted_shared / weighted_sizes[entity_b]
                if weighted_sizes[entity_b] else 0.0
            ),
            "low_evidence_flag": (
                size_a < MIN_PROFILE_SIZE
                or size_b < MIN_PROFILE_SIZE
            ),
            "identical_profile_flag": set_a == set_b,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        print("No pairs share molecules.")
        return result

    result = result.sort_values(
        [
            "weighted_jaccard_idf",
            "jaccard",
            "shared_molecules",
        ],
        ascending=False,
    ).reset_index(drop=True)

    print(f"Pairs with >= 1 shared molecule : {len(result):,}")
    print(f"Low-evidence pairs              : {result['low_evidence_flag'].sum():,}")
    print(f"Identical-profile pairs         : {result['identical_profile_flag'].sum():,}")

    print("\nTop 20 molecular-similarity pairs:")
    print(
        result[
            [
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
        ].head(20).to_string(index=False)
    )

    return result


# ================================================================
# 13. IDENTICAL PROFILE INVESTIGATION
# ================================================================

def investigate_identical_profiles(
    entities,
    molecules,
    molecule_sets,
    entity_aliases,
    entity_meta,
):
    section("7. IDENTICAL PROFILE INVESTIGATION")

    profile_groups = defaultdict(list)

    for entity_id, molecule_set in molecule_sets.items():
        profile_groups[tuple(sorted(molecule_set))].append(entity_id)

    duplicate_groups = [
        (profile, ids)
        for profile, ids in profile_groups.items()
        if len(ids) > 1
    ]

    duplicate_groups.sort(
        key=lambda x: (len(x[1]), len(x[0])),
        reverse=True,
    )

    group_rows = []
    entity_rows = []
    molecule_rows = []

    # Molecule metadata lookup.
    molecule_table = molecules.copy()
    molecule_table["pubchem_id"] = pd.to_numeric(
        molecule_table["pubchem_id"],
        errors="coerce",
    )

    molecule_lookup = {
        int(row["pubchem_id"]): row
        for _, row in molecule_table.dropna(subset=["pubchem_id"]).iterrows()
    }

    for group_id, (profile, entity_ids) in enumerate(duplicate_groups, start=1):
        signature = profile_signature(set(profile))
        names = [
            first_alias(entity_aliases.get(entity_id, set()))
            for entity_id in entity_ids
        ]

        categories = {
            entity_meta.get(entity_id, {}).get("category", "")
            for entity_id in entity_ids
        }
        category_readable = {
            entity_meta.get(entity_id, {}).get("category_readable", "")
            for entity_id in entity_ids
        }
        natural_sources = {
            entity_meta.get(entity_id, {}).get("natural_source_name", "")
            for entity_id in entity_ids
        }

        categories.discard("")
        category_readable.discard("")
        natural_sources.discard("")

        group_rows.append({
            "profile_group_id": group_id,
            "profile_signature": signature,
            "entity_count": len(entity_ids),
            "molecule_count": len(profile),
            "entity_ids": safe_join(sorted(entity_ids)),
            "entities": safe_join(sorted(names)),
            "category_count": len(categories),
            "categories": safe_join(sorted(categories)),
            "category_readable_count": len(category_readable),
            "category_readable": safe_join(sorted(category_readable)),
            "natural_source_count": len(natural_sources),
            "natural_sources": safe_join(sorted(natural_sources)),
            "molecule_ids": safe_join(profile),
        })

        for entity_id in sorted(entity_ids):
            meta = entity_meta.get(entity_id, {})
            entity_rows.append({
                "profile_group_id": group_id,
                "profile_signature": signature,
                "entity_id": entity_id,
                "entity_name": first_alias(entity_aliases.get(entity_id, set())),
                "molecule_count": len(profile),
                "category": meta.get("category", ""),
                "category_readable": meta.get("category_readable", ""),
                "natural_source_name": meta.get("natural_source_name", ""),
                "entity_alias_readable": meta.get("entity_alias_readable", ""),
                "entity_alias": meta.get("entity_alias", ""),
                "entity_alias_basket": meta.get("entity_alias_basket", ""),
                "entity_alias_synonyms": meta.get("entity_alias_synonyms", ""),
            })

        for molecule_id in profile:
            row = molecule_lookup.get(molecule_id, {})

            def molecule_value(column):
                value = row.get(column, "")
                return clean_text(value)

            molecule_rows.append({
                "profile_group_id": group_id,
                "profile_signature": signature,
                "molecule_count_in_profile": len(profile),
                "pubchem_id": molecule_id,
                "common_name": molecule_value("common_name"),
                "iupac_name": molecule_value("iupac_name"),
                "odor": molecule_value("odor"),
                "taste": molecule_value("taste"),
                "functional_groups": molecule_value("functional_groups"),
                "flavor_profile": molecule_value("flavor_profile"),
                "fema_flavor_profile": molecule_value("fema_flavor_profile"),
                "fooddb_flavor_profile": molecule_value("fooddb_flavor_profile"),
                "molecular_weight": molecule_value("molecular_weight"),
                "xlogp": molecule_value("xlogp"),
                "heavy_atom_count": molecule_value("heavy_atom_count"),
            })

    groups_df = pd.DataFrame(group_rows)
    entity_df = pd.DataFrame(entity_rows)
    molecule_df = pd.DataFrame(molecule_rows)

    total_duplicate_entities = sum(len(ids) for _, ids in duplicate_groups)

    print(f"Unique molecular profiles       : {len(profile_groups):,}")
    print(f"Identical-profile groups        : {len(duplicate_groups):,}")
    print(f"Entities in duplicate groups   : {total_duplicate_entities:,}")

    if duplicate_groups:
        print("\nLargest identical-profile groups:")
        for group_id, (profile, entity_ids) in enumerate(
            duplicate_groups[:10], start=1
        ):
            names = sorted(
                first_alias(entity_aliases.get(i, set()))
                for i in entity_ids
            )
            print(
                f"Group {group_id}: {len(entity_ids)} entities, "
                f"{len(profile)} molecules"
            )
            print("  " + ", ".join(names[:20]))
            if len(names) > 20:
                print(f"  ... and {len(names) - 20} more")

    return groups_df, entity_df, molecule_df


# ================================================================
# 14. UNRESOLVED NAME CANDIDATES
# ================================================================

def build_candidate_index(entity_aliases, entity_meta, molecule_sets):
    rows = []

    for entity_id, aliases in entity_aliases.items():
        for alias in aliases:
            rows.append({
                "entity_id": entity_id,
                "alias": alias,
                "category": entity_meta.get(entity_id, {}).get("category", ""),
                "category_readable": entity_meta.get(entity_id, {}).get(
                    "category_readable", ""
                ),
                "natural_source_name": entity_meta.get(entity_id, {}).get(
                    "natural_source_name", ""
                ),
                "molecule_count": len(molecule_sets.get(entity_id, set())),
            })

    return rows


def investigate_name_matching(
    molecule_sets,
    entity_aliases,
    entity_meta,
):
    section("8. NAME MATCHING + CANDIDATE GENERATION")

    name_index = build_name_index(entity_aliases)
    rows = []
    candidate_rows = []

    alias_rows = build_candidate_index(
        entity_aliases,
        entity_meta,
        molecule_sets,
    )

    for requested_a, requested_b in EXAMPLE_PAIRS:
        for side, requested_name in [
            ("a", requested_a),
            ("b", requested_b),
        ]:
            normalized = normalize_name(requested_name)
            exact_matches = sorted(name_index.get(normalized, set()))

            if len(exact_matches) == 1:
                entity_id = exact_matches[0]
                aliases = sorted(entity_aliases.get(entity_id, set()))

                rows.append({
                    "pair": f"{requested_a} <-> {requested_b}",
                    "side": side,
                    "requested_name": requested_name,
                    "normalized_name": normalized,
                    "match_status": "EXACT_ALIAS",
                    "matched_entity_id": entity_id,
                    "matched_entity_name": first_alias(aliases),
                    "molecule_count": len(molecule_sets.get(entity_id, set())),
                    "available_aliases": safe_join(aliases),
                })
                continue

            if len(exact_matches) > 1:
                rows.append({
                    "pair": f"{requested_a} <-> {requested_b}",
                    "side": side,
                    "requested_name": requested_name,
                    "normalized_name": normalized,
                    "match_status": "AMBIGUOUS_ALIAS",
                    "matched_entity_id": "",
                    "matched_entity_name": "",
                    "molecule_count": "",
                    "available_aliases": "",
                })
                continue

            # Unresolved: generate candidates, but DO NOT accept one.
            scored = []

            for item in alias_rows:
                alias = item["alias"]
                score = similarity_score(requested_name, alias)

                if score >= MIN_CANDIDATE_SCORE:
                    scored.append((score, item))

            scored.sort(
                key=lambda x: (x[0], x[1]["molecule_count"]),
                reverse=True,
            )

            seen = set()
            top = []

            for score, item in scored:
                key = item["entity_id"]
                if key in seen:
                    continue
                seen.add(key)
                top.append((score, item))
                if len(top) >= TOP_NAME_CANDIDATES:
                    break

            rows.append({
                "pair": f"{requested_a} <-> {requested_b}",
                "side": side,
                "requested_name": requested_name,
                "normalized_name": normalized,
                "match_status": "NOT_FOUND",
                "matched_entity_id": "",
                "matched_entity_name": "",
                "molecule_count": "",
                "available_aliases": "",
            })

            for rank, (score, item) in enumerate(top, start=1):
                candidate_rows.append({
                    "pair": f"{requested_a} <-> {requested_b}",
                    "side": side,
                    "requested_name": requested_name,
                    "candidate_rank": rank,
                    "candidate_entity_id": item["entity_id"],
                    "candidate_alias": item["alias"],
                    "candidate_score": round(score, 6),
                    "candidate_category": item["category"],
                    "candidate_category_readable": item["category_readable"],
                    "candidate_natural_source_name": item["natural_source_name"],
                    "candidate_molecule_count": item["molecule_count"],
                    "mapping_decision": "REVIEW_ONLY",
                })

    matching_df = pd.DataFrame(rows)
    candidates_df = pd.DataFrame(candidate_rows)

    print("\nName matching:")
    for _, row in matching_df.iterrows():
        print(
            f"  {row['requested_name']:<25} "
            f"{row['match_status']}"
            + (
                f" -> {row['matched_entity_name']}"
                if row["match_status"] == "EXACT_ALIAS"
                else ""
            )
        )

    if not candidates_df.empty:
        print("\nTop unresolved-name candidates:")
        print(
            candidates_df[
                [
                    "requested_name",
                    "candidate_alias",
                    "candidate_score",
                    "candidate_category_readable",
                    "candidate_molecule_count",
                ]
            ].head(20).to_string(index=False)
        )

    return matching_df, candidates_df


# ================================================================
# 15. SUMMARY
# ================================================================

def write_summary(
    summary,
    entities,
    molecule_sets,
    frequency_df,
    pairwise_df,
    identical_profiles_df,
    name_matching_df,
    candidates_df,
    missing_ids,
):
    small_profile_count = sum(
        len(profile) < MIN_PROFILE_SIZE
        for profile in molecule_sets.values()
    )

    summary.extend([
        "FINAL MOLECULAR-SIMILARITY DECISION",
        "-" * 72,
        "Primary molecular similarity = IDF-weighted Jaccard.",
        "Formula = sum(IDF of shared molecules) / "
        "sum(IDF of union molecules).",
        "Common molecules contribute less than rare molecules.",
        "Low-evidence profiles are retained but flagged; they should not "
        "be treated as equally reliable downstream.",
        "Identical molecular profiles are explicitly flagged.",
        "Name candidates are review-only and are never silently mapped.",
        "",
        "FINAL AUDIT COUNTS",
        "-" * 72,
        f"FlavorDB entities: {len(entities):,}",
        f"Entities with molecule profiles: {len(molecule_sets):,}",
        f"Small profiles (< {MIN_PROFILE_SIZE}): {small_profile_count:,}",
        f"Unique linked molecules: {len(frequency_df):,}",
        f"Pairs with >= 1 shared molecule: {len(pairwise_df):,}",
        f"Identical-profile groups: {len(identical_profiles_df):,}",
        f"Name checks: {len(name_matching_df):,}",
        f"Generated review candidates: {len(candidates_df):,}",
        f"Missing referenced molecule IDs: {len(missing_ids):,}",
        "",
        "IMPORTANT INTERPRETATION",
        "-" * 72,
        "A high molecular similarity means the two FlavorDB entities "
        "share a large weighted molecular profile.",
        "It does NOT by itself mean that one ingredient is a valid "
        "substitute for the other.",
        "The final substitution framework should combine this feature "
        "with graph/contextual compatibility, carbon footprint, "
        "nutrition preservation, and hard constraints.",
        "",
        "OUTPUTS",
        "-" * 72,
        str(ENTITY_STATS_FILE),
        str(MOLECULE_FREQ_FILE),
        str(PAIR_EXAMPLES_FILE),
        str(PAIRWISE_FILE),
        str(IDENTICAL_PROFILES_FILE),
        str(IDENTICAL_ENTITIES_FILE),
        str(IDENTICAL_MOLECULES_FILE),
        str(NAME_MATCH_FILE),
        str(NAME_CANDIDATES_FILE),
        str(SUMMARY_FILE),
    ])

    SUMMARY_FILE.write_text("\n".join(summary), encoding="utf-8")


# ================================================================
# 16. MAIN
# ================================================================

def main():
    summary = []

    section("FINAL FLAVORDB MOLECULE AUDIT")

    print(
        "Primary molecular similarity: IDF-weighted Jaccard."
    )
    print(
        "This script prepares the molecular feature for the "
        "substitution framework; it does not create the final "
        "multi-objective substitution score."
    )

    entities, entity_molecules, molecules = load_data()

    missing_ids = audit_basic_structure(
        entities,
        entity_molecules,
        molecules,
        summary,
    )

    molecule_sets, entity_aliases, entity_meta = build_profiles(
        entities,
        entity_molecules,
    )

    entity_stats = build_entity_statistics(
        entities,
        molecule_sets,
        entity_aliases,
    )

    frequency_df, weights = build_frequency_table(
        molecule_sets,
        len(entities),
    )

    pair_examples_df = audit_example_pairs(
        molecule_sets,
        entity_aliases,
        weights,
    )

    pairwise_df = build_pairwise_similarity(
        molecule_sets,
        entity_aliases,
        weights,
    )

    identical_profiles_df, identical_entities_df, identical_molecules_df = (
        investigate_identical_profiles(
            entities,
            molecules,
            molecule_sets,
            entity_aliases,
            entity_meta,
        )
    )

    name_matching_df, name_candidates_df = investigate_name_matching(
        molecule_sets,
        entity_aliases,
        entity_meta,
    )

    # ------------------------------------------------------------
    # Save all outputs
    # ------------------------------------------------------------

    entity_stats.to_csv(ENTITY_STATS_FILE, index=False)
    frequency_df.to_csv(MOLECULE_FREQ_FILE, index=False)
    pair_examples_df.to_csv(PAIR_EXAMPLES_FILE, index=False)
    pairwise_df.to_csv(PAIRWISE_FILE, index=False)

    identical_profiles_df.to_csv(
        IDENTICAL_PROFILES_FILE,
        index=False,
    )
    identical_entities_df.to_csv(
        IDENTICAL_ENTITIES_FILE,
        index=False,
    )
    identical_molecules_df.to_csv(
        IDENTICAL_MOLECULES_FILE,
        index=False,
    )

    name_matching_df.to_csv(
        NAME_MATCH_FILE,
        index=False,
    )
    name_candidates_df.to_csv(
        NAME_CANDIDATES_FILE,
        index=False,
    )

    write_summary(
        summary,
        entities,
        molecule_sets,
        frequency_df,
        pairwise_df,
        identical_profiles_df,
        name_matching_df,
        name_candidates_df,
        missing_ids,
    )

    section("AUDIT COMPLETE")

    print(f"Entity statistics      : {ENTITY_STATS_FILE}")
    print(f"Molecule frequency     : {MOLECULE_FREQ_FILE}")
    print(f"Pair examples          : {PAIR_EXAMPLES_FILE}")
    print(f"Pairwise similarity    : {PAIRWISE_FILE}")
    print(f"Identical profiles     : {IDENTICAL_PROFILES_FILE}")
    print(f"Identical entities     : {IDENTICAL_ENTITIES_FILE}")
    print(f"Identical molecules    : {IDENTICAL_MOLECULES_FILE}")
    print(f"Name matching          : {NAME_MATCH_FILE}")
    print(f"Name candidates        : {NAME_CANDIDATES_FILE}")
    print(f"Summary                : {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
