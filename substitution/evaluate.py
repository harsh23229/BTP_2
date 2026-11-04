"""Evaluate all configured nutritional objectives and export ranked result CSVs."""
import numpy as np
import pandas as pd
import config as C
import ranking as RK
from retrieval import Retriever
from recipes import load_recipes

TARGETS = [
    "beef", "chicken breast", "pork", "lamb", "turkey", "shrimp", "bacon", "egg",
    "butter", "milk", "heavy cream", "cream cheese", "cheddar cheese", "sugar",
    "flour", "rice", "pasta", "potato", "bread", "mayonnaise"
]
N_RECIPES, TOP, CANDIDATE_POOL = 10, 5, 40

R = Retriever()
M = R.master
rec = load_recipes(M.name)
available_targets = [name for name in TARGETS if name in R.name_idx]
missing_targets = [name for name in TARGETS if name not in R.name_idx]

ingredient_rows, recipe_rows, notes = [], [], []
violations = 0
objective_counts = {name: {"cases": 0, "ranked_cases": 0, "ranked_rows": 0} for name in RK.PRESETS}

for name in available_targets:
    orig = M.iloc[R.name_idx[name]]
    if orig.tier != "trusted":
        continue
    candidates, report = R.candidates(name, top_k=CANDIDATE_POOL)
    for objective_name, objective in RK.PRESETS.items():
        if objective.applies_to_family and orig.family not in objective.applies_to_family:
            continue
        ok, rejected = RK.rank(orig, candidates, objective, has_recipe=False)
        objective_counts[objective_name]["cases"] += 1
        objective_counts[objective_name]["ranked_cases"] += int(not ok.empty)
        objective_counts[objective_name]["ranked_rows"] += min(TOP, len(ok))

        for _, row in ok.head(TOP).iterrows():
            ingredient_rows.append({
                "ingredient": name, "objective": objective_name,
                "original_source": orig.source, "original_usda_description": orig.usda_description,
                "original_protein_100g": orig.protein_100g,
                "original_carbs_100g": orig.carbs_100g,
                "original_fat_100g": orig.fat_100g,
                "original_kcal_100g": orig.kcal_100g,
                "candidate": row["name"], "candidate_usda_description": row.usda_description,
                "rank": row["rank"], "score": row.score, "pareto": row.pareto,
                "swap_g_per_100g": row.swap_g_per_100g,
                "d_protein_g_per100g": row.d_protein,
                "d_carbs_g_per100g": row.d_carbs,
                "d_fat_g_per100g": row.d_fat,
                "d_kcal_per100g": row.d_kcal,
                "ctx_sim": row.ctx_sim, "flavour_sim": row.flavour_sim,
                "flavour_identical_profile": row.get("flavour_identical_profile", False),
                "flavour_low_evidence": row.get("flavour_low_evidence", False),
                "compat": row.get("compat", np.nan), "explain": row.explain
            })
        for _, row in ok.iterrows():
            violations += int(RK.violates(row, objective))
        notes.append({
            "ingredient": name, "objective": objective_name, "candidate_pool": len(candidates),
            "passed": len(ok), "rejected": len(rejected),
            "top50_context_neighbours_without_trusted_nutrition": report["top50_without_trusted_nutrition"],
            "most_common_rejection": (
                rejected.reject_reason.str.split(";").explode().str.strip()
                .replace("", np.nan).dropna().value_counts().index[0]
                if not rejected.empty and rejected.reject_reason.str.split(";").explode().str.strip().replace("", np.nan).dropna().size
                else ""
            )
        })

ingredient_df = pd.DataFrame(ingredient_rows)
ingredient_df.to_csv(C.OUT / "review_ingredient_level.csv", index=False)
notes_df = pd.DataFrame(notes)

# Recipe-level sample is selected deterministically from recipes with >=6 graph ingredients
# and at least two target ingredients that have trusted nutrition.
if not rec.empty:
    good = rec.groupby("recipe_id").agg(
        n=("ingredient_name", "nunique"),
        hits=("ingredient_name", lambda s: len(set(s) & set(available_targets)))
    )
    candidate_ids = good[(good.n >= 6) & (good.hits >= 2)].index.to_numpy()
else:
    candidate_ids = np.array([])

rng = np.random.default_rng(7)
chosen = rng.choice(candidate_ids, size=min(N_RECIPES, len(candidate_ids)), replace=False) if len(candidate_ids) else []

for recipe_id in chosen:
    recipe = rec[rec.recipe_id.eq(recipe_id)]
    title = recipe.recipe_title.iloc[0]
    names = recipe.ingredient_name.unique().tolist()
    for name in [n for n in names if n in available_targets and n in R.name_idx]:
        orig = M.iloc[R.name_idx[name]]
        if orig.tier != "trusted":
            continue
        mass_values = recipe.loc[recipe.ingredient_name.eq(name), "mass_g"]
        mass_g = mass_values.iloc[0] if not mass_values.empty else np.nan
        candidates, _ = R.candidates(name, top_k=CANDIDATE_POOL, recipe_names=names)
        candidates = candidates[~candidates.in_recipe]
        for objective_name, objective in RK.PRESETS.items():
            if objective.applies_to_family and orig.family not in objective.applies_to_family:
                continue
            ok, rejected = RK.rank(orig, candidates, objective, mass_g=mass_g, has_recipe=True)
            for _, row in ok.head(TOP).iterrows():
                recipe_rows.append({
                    "recipe_id": recipe_id, "recipe_title": title,
                    "ingredient": name, "objective": objective_name,
                    "original_mass_g": mass_g, "rank": row["rank"],
                    "candidate": row["name"], "pareto": row.pareto, "score": row.score,
                    "candidate_mass_g": row.get("recipe_candidate_mass_g", np.nan),
                    "swap_g_per_100g": row.swap_g_per_100g,
                    "d_protein_g_per100g": row.d_protein,
                    "d_carbs_g_per100g": row.d_carbs,
                    "d_fat_g_per100g": row.d_fat,
                    "d_kcal_per100g": row.d_kcal,
                    "recipe_d_protein_g": row.get("recipe_d_protein", np.nan),
                    "recipe_d_carbs_g": row.get("recipe_d_carbs", np.nan),
                    "recipe_d_fat_g": row.get("recipe_d_fat", np.nan),
                    "recipe_d_kcal": row.get("recipe_d_kcal", np.nan),
                    "compat": row.get("compat", np.nan), "explain": row.explain
                })

recipe_df = pd.DataFrame(recipe_rows)
recipe_df.to_csv(C.OUT / "review_recipe_level.csv", index=False)

lines = [
    "SUBSTITUTION EVALUATION SUMMARY", "=" * 72,
    f"Trusted ingredients in master: {int(M.tier.eq('trusted').sum())}",
    f"Target ingredients available: {len(available_targets)} / {len(TARGETS)}",
    f"Target ingredients absent from graph: {', '.join(missing_targets) if missing_targets else 'none'}",
    f"Ingredient-level objective cases: {len(notes_df)}",
    f"Ingredient-level ranked rows: {len(ingredient_df)}",
    f"Recipe-level recipes sampled: {len(chosen)}",
    f"Recipe-level ranked rows: {len(recipe_df)}",
    f"Hard-constraint violations in all returned ranked candidates: {violations} (must be 0)",
    "",
    "OBJECTIVE COVERAGE"
]
for name, counts in objective_counts.items():
    lines.append(
        f"{name}: {counts['ranked_cases']} / {counts['cases']} cases returned at least one candidate; "
        f"{counts['ranked_rows']} top-ranked rows exported"
    )
lines += [
    "",
    "DATA LIMITS",
    f"Mean top-50 context neighbours without trusted nutrition: "
    f"{notes_df.top50_context_neighbours_without_trusted_nutrition.mean():.1f}"
    if not notes_df.empty else "No ingredient-level cases were evaluated.",
    "RecipeDB-derived conditional per-100g profiles are not used for ranking.",
    "Only validated USDA-sourced trusted profiles are eligible for ranking.",
    "",
    "CASES WITH FEWER THAN THREE PASSING CANDIDATES"
]
if not notes_df.empty:
    few = notes_df[notes_df.passed < 3][["ingredient", "objective", "candidate_pool", "passed", "rejected"]]
    lines.append(few.to_string(index=False) if not few.empty else "None.")
else:
    lines.append("None.")

summary = "\n".join(lines)
(C.OUT / "evaluation_summary.txt").write_text(summary, encoding="utf-8")
print(summary)
if violations != 0:
    raise AssertionError(f"Hard-constraint audit failed: {violations} violations")
