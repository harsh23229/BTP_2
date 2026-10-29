"""The single entry point: given a recipe, an ingredient and an objective, return ranked substitutions
with evidence, and say where the data prevents a reliable answer.

Examples (run inside BTP_2/substitution):
  python substitute.py --recipe 100000 --ingredient "beef" --objective more_protein
  python substitute.py --recipe 100000 --ingredient "beef" --objective plant_swap --top 5
  python substitute.py --ingredient "sugar" --objective less_carbs          (no recipe: similarity + nutrition only)
  python substitute.py --list-recipes                                       (some recipe ids to try)
"""
import argparse
import numpy as np
import pandas as pd
import config as C
import ranking as RK
from retrieval import Retriever
from recipes import load_recipes


def substitute(R, rec, ingredient, objective, recipe_id=None, top=5):
    M = R.master
    if ingredient not in R.name_idx:
        return f"'{ingredient}' is not an ingredient node in the graph."
    orig = M.iloc[R.name_idx[ingredient]]
    obj = RK.PRESETS[objective]
    out = []

    names, mass = None, None
    if recipe_id is not None:
        rr = rec[rec.recipe_id == recipe_id]
        if rr.empty:
            return f"recipe {recipe_id} not found (or none of its ingredients are graph nodes)."
        names = rr.ingredient_name.unique().tolist()
        out.append(f"Recipe {recipe_id}: {rr.recipe_title.iloc[0]}")
        if ingredient not in names:
            out.append(f"  note: '{ingredient}' is not in this recipe; ranking without recipe compatibility.")
            names = None
        else:
            mass = rr[rr.ingredient_name == ingredient].mass_g.iloc[0]

    if orig.tier == "trusted":
        out.append(f"Replace: {ingredient}   (protein {orig.protein_100g:.1f} g, carbs {orig.carbs_100g:.1f} g per 100 g, source: {orig.source})")
    else:
        out.append(f"Replace: {ingredient}")
    out.append(f"Objective: {objective}")

    if orig.tier != "trusted":
        out.append(f"\nNO RECOMMENDATION: '{ingredient}' has no trusted nutrition ({orig.tier}). "
                   f"Map it to a USDA entry in USDA_MAP to enable this.")
        return "\n".join(out)
    if obj.applies_to_family and orig.family not in obj.applies_to_family:
        out.append(f"\nNO RECOMMENDATION: objective '{objective}' applies to {obj.applies_to_family}, "
                   f"but '{ingredient}' is in family '{orig.family}'.")
        return "\n".join(out)

    cands, rep = R.candidates(ingredient, top_k=40, recipe_names=names)
    if names:
        cands = cands[~cands.in_recipe]
    ok, rej = RK.rank(orig, cands, obj, mass_g=mass, has_recipe=bool(names))

    if ok.empty:
        out.append("\nNO CANDIDATE PASSES the hard constraints.")
        why = rej.reject_reason.str.split(";").str[0].str.replace(r"[-+]?\d+\.?\d*", "#", regex=True).value_counts().head(3)
        out.append("  most common reasons: " + "; ".join(f"{k.strip()} ({v})" for k, v in why.items()))
    else:
        out.append("")
        for _, r in ok.head(top).iterrows():
            line = f"{int(r['rank'])}. {r['name']}   [{'best trade-off' if r.pareto else 'ok'}]"
            out.append(line)
            out.append("     " + r.explain)
            if "recipe_candidate_mass_g" in r and pd.notna(r.get("recipe_candidate_mass_g")):
                out.append(f"     in this recipe: {mass:.0f} g {ingredient} -> {r.recipe_candidate_mass_g:.0f} g {r['name']} "
                           f"(carbs {r.recipe_d_carbs_g:+.0f} g, protein {r.recipe_d_protein_g:+.0f} g, fat {r.recipe_d_fat_g:+.0f} g)")
            elif mass is None or (isinstance(mass, float) and np.isnan(mass)):
                if names:
                    out.append("     (recipe amount is not in grams, so only per-100 g change is shown)")
    out.append("")
    out.append("DATA LIMITS")
    out.append(f"  {rep['top50_without_trusted_nutrition']} of the 50 closest ingredients by recipe context have no trusted nutrition "
               f"and were NOT ranked, e.g. {', '.join(rep['unknown_examples'][:6])}.")
    out.append("  Comparison basis: " + ("equal protein, raw weights." if obj.equal_protein else "equal weight (100 g for 100 g)."))
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=int)
    ap.add_argument("--ingredient")
    ap.add_argument("--objective", default="more_protein", choices=list(RK.PRESETS))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--list-recipes", action="store_true")
    a = ap.parse_args()

    R = Retriever()
    rec = load_recipes(R.master.name)
    if a.list_recipes:
        trusted = set(R.master.name[R.master.tier == "trusted"])
        g = rec[rec.ingredient_name.isin(trusted)].groupby("recipe_id").agg(title=("recipe_title", "first"), n=("ingredient_name", "nunique"))
        print(g[g.n >= 5].sample(10, random_state=1).to_string())
    else:
        if not a.ingredient:
            ap.error("--ingredient is required")
        print(substitute(R, rec, a.ingredient, a.objective, a.recipe, a.top))