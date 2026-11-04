"""Build one ingredient master row per GastroGraph ingredient.

Nutrient trust policy:
- Only source='usda', tier='trusted', non-empty USDA description, and physically
  valid finite nutrient values are eligible for ranking.
- RecipeDB per-100g comparison tiers are diagnostic only. They are NOT used to
  downgrade or promote explicit USDA mappings because RecipeDB ingredient
  quantity/value semantics are not a reliable ground truth for per-100g values.
"""
import numpy as np
import pandas as pd
import config as C

NUTRIENT_COLS = ["protein_100g", "carbs_100g", "fat_100g", "kcal_100g"]


def load_nutrients():
    if not C.NUTRIENTS_FILE.is_file():
        raise FileNotFoundError(f"Missing nutrient master: {C.NUTRIENTS_FILE}")
    nut = pd.read_csv(C.NUTRIENTS_FILE)
    required = ["ingredient_name", *NUTRIENT_COLS, "source", "tier", "usda_description"]
    missing = [c for c in required if c not in nut.columns]
    if missing:
        raise ValueError(f"Nutrient master missing columns {missing}; found {nut.columns.tolist()}")
    nut["ingredient_name"] = nut["ingredient_name"].astype("string").str.strip().str.lower()
    if nut["ingredient_name"].duplicated().any():
        dupes = nut.loc[nut["ingredient_name"].duplicated(False), "ingredient_name"].tolist()
        raise ValueError(f"Duplicate ingredient names in nutrient master: {dupes[:20]}")
    for col in NUTRIENT_COLS:
        nut[col] = pd.to_numeric(nut[col], errors="coerce")

    valid_numeric = np.isfinite(nut[NUTRIENT_COLS]).all(axis=1)
    nonnegative = nut[NUTRIENT_COLS].ge(0).all(axis=1)
    macro_mass_valid = nut[["protein_100g", "carbs_100g", "fat_100g"]].sum(axis=1).le(100.5)
    kcal_valid = nut["kcal_100g"].between(0, 950, inclusive="both")
    has_description = nut["usda_description"].notna() & nut["usda_description"].astype("string").str.strip().ne("")
    explicit_usda = nut["source"].astype("string").str.casefold().eq("usda")
    declared_trusted = nut["tier"].astype("string").str.casefold().eq("trusted")

    checks = pd.DataFrame({
        "finite_all_nutrients": valid_numeric,
        "nonnegative_nutrients": nonnegative,
        "macros_sum_le_100_5": macro_mass_valid,
        "kcal_between_0_and_950": kcal_valid,
        "has_usda_description": has_description,
        "source_is_usda": explicit_usda,
        "input_tier_trusted": declared_trusted,
    })
    nut["nutrient_validation_pass"] = checks.all(axis=1)
    nut["nutrient_validation_issues"] = checks.apply(
        lambda row: ";".join(row.index[~row].tolist()), axis=1
    )
    nut["tier_input"] = nut["tier"]
    nut["tier"] = np.where(nut["nutrient_validation_pass"], "trusted", "uncertain")
    # Keep missing/non-USDA values uncertain even if their numbers look plausible.
    nut.loc[~(explicit_usda & declared_trusted), "tier"] = "uncertain"
    return nut, checks


nodes = pd.read_csv(C.GRAPH_DIR / "nodes_our_recipeflavor.csv")
edges = pd.read_csv(C.GRAPH_DIR / "edges_our_recipeflavor.csv")
ing = nodes.loc[nodes.node_type.eq("ingredient"), ["node_id", "name"]].copy()
ing["name"] = ing["name"].astype("string").str.strip().str.lower()

mol = edges.loc[edges.edge_type.eq("ingr-dcomp")].groupby("id_1").size().rename("n_molecules")
cc = edges.loc[edges.edge_type.eq("ingr-ingr")]
deg = pd.concat([cc.id_1, cc.id_2]).value_counts().rename("cooc_degree")

rec = pd.read_csv(C.RECIPE_FILE, usecols=["ingredient_name"], low_memory=False)
freq = rec.ingredient_name.astype("string").str.strip().str.lower().value_counts().rename("recipedb_recipe_rows")# USDA food taxonomy is also used to validate that every mapped description
# actually exists in the installed USDA files.
cat_file = C.USDA_SR / "food_category.csv"
sr_food_file = C.USDA_SR / "food.csv"
ff_food_file = C.USDA_FF / "food.csv"
for p in [cat_file, sr_food_file, ff_food_file]:
    if not p.is_file():
        raise FileNotFoundError(f"Required USDA food taxonomy file missing: {p}")
cat = pd.read_csv(cat_file).set_index("id").description
food = pd.concat([
    pd.read_csv(sr_food_file, usecols=["description", "food_category_id"]),
    pd.read_csv(ff_food_file, usecols=["description", "food_category_id"])
]).dropna()
food["food_group"] = food.food_category_id.astype(int).map(cat)
grp = food.drop_duplicates("description").set_index("description").food_group
known_usda_descriptions = set(food.description.astype(str))

nut, nutrient_checks = load_nutrients()
desc_exists = nut["usda_description"].astype("string").isin(known_usda_descriptions)
nut["usda_description_exists"] = desc_exists
nut.loc[~desc_exists & nut["source"].astype("string").str.casefold().eq("usda"), "tier"] = "uncertain"
nut.loc[~desc_exists & nut["source"].astype("string").str.casefold().eq("usda"),
        "nutrient_validation_issues"] += ";usda_description_not_found"
nut = nut.set_index("ingredient_name")
m = (ing.set_index("name").join(nut, how="left")
     .join(freq, how="left").reset_index())
m = m.merge(mol, left_on="node_id", right_index=True, how="left")
m = m.merge(deg, left_on="node_id", right_index=True, how="left")
m["tier"] = m["tier"].fillna("unavailable")
m["source"] = m["source"].fillna("unavailable")

# Optional RecipeDB audit comparison: report discrepancies but do not use it as
# a truth label for manually mapped USDA values.
review_path = C.OUT / "nutrient_source_review.csv"
if getattr(C, "NUTRIENT_AUDIT_FILE", None) is not None and C.NUTRIENT_AUDIT_FILE.is_file():
    audit = pd.read_csv(C.NUTRIENT_AUDIT_FILE)
    if "ingredient_name" in audit.columns:
        audit["ingredient_name"] = audit["ingredient_name"].astype("string").str.strip().str.lower()
        keep = [c for c in ["ingredient_name", "tier", "internally_consistent",
                            "protein_100g", "carbs_100g", "fat_100g", "kcal_100g"] if c in audit.columns]
        audit = audit[keep].drop_duplicates("ingredient_name")
        audit = audit.rename(columns={"tier": "recipedb_audit_tier"})
        m = m.merge(audit, left_on="name", right_on="ingredient_name", how="left", suffixes=("", "_audit"))
        m.drop(columns=["ingredient_name"], errors="ignore", inplace=True)
        m["recipedb_disagreement_review"] = (
            m["tier"].eq("trusted") & m["recipedb_audit_tier"].notna()
            & m["recipedb_audit_tier"].ne("trusted")
        )
        m["nutrient_source_note"] = np.where(
            m["recipedb_disagreement_review"],
            "USDA mapping retained; RecipeDB per-100g audit differs and is diagnostic only",
            np.where(m["tier"].eq("trusted"), "Validated explicit USDA mapping", "Not eligible for nutrient ranking")
        )
        m.loc[m["tier"].eq("trusted") | m["recipedb_audit_tier"].notna()].to_csv(review_path, index=False)
    else:
        m["nutrient_source_note"] = np.where(m["tier"].eq("trusted"), "Validated explicit USDA mapping", "Not eligible")
else:
    m["nutrient_source_note"] = np.where(m["tier"].eq("trusted"), "Validated explicit USDA mapping", "Not eligible")

# USDA food group -> recipe-job family
m["food_group"] = m.usda_description.map(grp)
m["family"] = m.food_group.map(C.FAMILY_BY_GROUP)
for prefix, family in C.FAMILY_BY_DESC_PREFIX.items():
    m.loc[m.usda_description.fillna("").str.startswith(prefix), "family"] = family
m["family"] = [C.FAMILY_OVERRIDE.get(n, f) for n, f in zip(m.name, m.family)]
for col in ["n_molecules", "cooc_degree", "recipe_rows"]:
    if col not in m:
        m[col] = 0
    m[col] = m[col].fillna(0).astype(int)
m["has_molecules"] = m.n_molecules.gt(0)
m = m.sort_values("recipe_rows", ascending=False).reset_index(drop=True)
C.OUT.mkdir(parents=True, exist_ok=True)
m.to_csv(C.OUT / "ingredient_master.csv", index=False)

print("Ingredients in graph:", len(m))
print("Nutrient tiers:")
print(m.tier.value_counts(dropna=False).to_string())
print("Trusted USDA ingredients:", int(m.tier.eq("trusted").sum()))
print("Trusted USDA ingredients with molecules:", int((m.tier.eq("trusted") & m.has_molecules).sum()))
print("Trusted ingredients without a family:", m.loc[m.tier.eq("trusted") & m.family.isna(), "name"].tolist())
print("RecipeDB audit disagreements (diagnostic only):", int(m.get("recipedb_disagreement_review", pd.Series(False, index=m.index)).sum()))
print("Saved:", C.OUT / "ingredient_master.csv")
if review_path.exists():
    print("Saved:", review_path)
