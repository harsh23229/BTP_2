"""STEP 5 - run the pipeline on 10 recipes x 20 ingredients and write the ranked result sheets.

Run:  python evaluate.py
Writes (outputs/):
  review_ingredient_level.csv  - top-5 per ingredient per objective, no recipe context. Candidates that fail a hard rule are never written.
  review_recipe_level.csv      - same, inside 10 real recipes (adds compatibility + gram change). 
  evaluation_summary.txt       - constraint check and data-limitation report
"""
import numpy as np
import pandas as pd
import config as C
import ranking as RK
from retrieval import Retriever
from recipes import load_recipes

TARGETS = ["beef", "chicken breast", "pork", "lamb", "turkey", "shrimp", "bacon", "egg", "butter", "milk",
           "heavy cream", "cream cheese", "cheddar cheese", "sugar", "flour", "rice", "pasta", "potato",
           "bread", "mayonnaise"]
N_RECIPES, TOP = 10, 5

R = Retriever()
M = R.master
rec = load_recipes(M.name)

keep_cols = ["rank", "candidate", "score", "pareto", "explain", "d_protein", "d_carbs", "sim_score", "ctx_sim",
             "flavor_sim", "compat", "n_judges"]
rows_ing, rows_rec, notes = [], [], []
violations = 0

# ------------------------------------------------------------------ ingredient level
for name in TARGETS:
    q = R.name_idx[name]; orig = M.iloc[q]
    cands, rep = R.candidates(name, top_k=40)
    for oname, obj in RK.PRESETS.items():
        if obj.applies_to_family and orig.family not in obj.applies_to_family:
            continue
        ok, rej = RK.rank(orig, cands, obj, has_recipe=False)
        for _, r in ok.head(TOP).iterrows():
            rows_ing.append({"ingredient": name, "objective": oname, "orig_protein": orig.protein_100g,
                             "orig_carbs": orig.carbs_100g, "candidate": r["name"], "rank": r["rank"], "pareto": r.pareto,
                             "swap_g_per_100g": round(r.swap_g_per_100g, 0),
                             "d_protein": round(r.d_protein, 1), "d_carbs": round(r.d_carbs, 1),
                             "d_fat": round(r.d_fat, 1), "d_kcal": round(r.d_kcal, 0),
                             "ctx_sim": round(r.ctx_sim, 2), "flavor_sim": r.flavor_sim, "explain": r.explain})
        # hard-filter audit: nothing that passed may violate a guard
        for _, r in ok.iterrows():
            violations += int(RK.violates(r, obj))
        n_improving = len(ok) + int((~rej.reject_reason.str.contains("gain")).sum())
        notes.append({"ingredient": name, "objective": oname, "pool": len(cands), "passed": len(ok), "rejected": len(rej),
                      "improving": n_improving,
                      "top50_ctx_lost_to_unknown_nutrition": rep["top50_without_trusted_nutrition"]})

pd.DataFrame(rows_ing).to_csv(C.OUT / "review_ingredient_level.csv", index=False)
nt = pd.DataFrame(notes)

# ------------------------------------------------------------------ recipe level
good = rec.groupby("recipe_id").agg(n=("ingredient_name", "nunique"),
                                    hits=("ingredient_name", lambda s: len(set(s) & set(TARGETS))))
cand_ids = good[(good.n >= 6) & (good.hits >= 2)].index
rng = np.random.default_rng(7)
chosen = rng.choice(cand_ids, size=min(N_RECIPES, len(cand_ids)), replace=False)

for rid in chosen:
    rr = rec[rec.recipe_id == rid]
    title = rr.recipe_title.iloc[0]
    names = rr.ingredient_name.unique().tolist()
    for name in [n for n in names if n in TARGETS]:
        q = R.name_idx[name]; orig = M.iloc[q]
        mass = rr[rr.ingredient_name == name].mass_g.iloc[0]
        cands, _ = R.candidates(name, top_k=40, recipe_names=names)
        cands = cands[~cands.in_recipe]                        # do not swap in something already in the recipe
        for oname, obj in RK.PRESETS.items():
            if obj.applies_to_family and orig.family not in obj.applies_to_family:
                continue
            ok, _ = RK.rank(orig, cands, obj, mass_g=mass, has_recipe=True)
            for _, r in ok.head(TOP).iterrows():
                rows_rec.append({"recipe_id": rid, "recipe_title": title, "ingredient": name, "objective": oname,
                                 "line_mass_g": mass, "rank": r["rank"], "candidate": r["name"], "pareto": r.pareto,
                                 "swap_g_per_100g": round(r.swap_g_per_100g, 0), "recipe_candidate_g": round(r.get("recipe_candidate_mass_g", np.nan), 0),
                                 "d_protein_per100g": round(r.d_protein, 1), "d_carbs_per100g": round(r.d_carbs, 1),
                                 "d_fat_per100g": round(r.d_fat, 1), "d_kcal_per100g": round(r.d_kcal, 0),
                                 "recipe_d_protein_g": round(r.get("recipe_d_protein_g", np.nan), 1), "recipe_d_carbs_g": round(r.get("recipe_d_carbs_g", np.nan), 1),
                                 "compat": round(r.compat, 2), "explain": r.explain})
pd.DataFrame(rows_rec).to_csv(C.OUT / "review_recipe_level.csv", index=False)

# ------------------------------------------------------------------ summary
lines = [
    "EVALUATION SUMMARY", "=" * 60,
    f"trusted candidate pool size           : {len(R.trusted)} ingredients",
    f"ingredient-level cases                : {len(nt)}",
    f"  cases with a ranked answer (>=1)     : {(nt.passed >= 1).sum()}",
    f"  NO candidate improves the target     : {((nt.improving == 0)).sum()}   (original already near the best in its family: not a data problem)",
    f"  improvers exist but guard blocks all : {((nt.improving > 0) & (nt.passed == 0)).sum()}   (trade-off cases, worth reporting)",
    f"  cases with only 1-2 answers          : {((nt.passed >= 1) & (nt.passed < 3)).sum()}",
    f"hard-constraint violations in output  : {violations}   (must be 0)",
    f"recipes reviewed                      : {len(chosen)}  ({len(rows_rec)} ranked rows)",
    f"recipe lines with a gram mass         : {int(pd.DataFrame(rows_rec).recipe_d_protein_g.notna().sum()) if rows_rec else 0} rows",
    "",
    "DATA LIMITS THAT PREVENT RECOMMENDATIONS",
    f"mean of the top-50 context neighbours with NO trusted nutrition: {nt.top50_ctx_lost_to_unknown_nutrition.mean():.1f} / 50",
    "(those candidates are reported as 'nutrition unknown', never ranked)",
    f"candidates with flavour evidence      : {int(pd.DataFrame(rows_ing).flavor_sim.notna().sum())} / {len(rows_ing)} top-5 rows",
    f"embeddings used                       : {'yes' if R.E is not None else 'no (EMBEDDINGS_FILE is None)'}",
    "", "Cases with few/no candidates:", nt[nt.passed < 3][["ingredient", "objective", "pool", "improving", "passed"]].to_string(index=False),
]
(C.OUT / "evaluation_summary.txt").write_text("\n".join(lines))
print("\n".join(lines))