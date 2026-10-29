"""Interactive substitution: pick a recipe, an ingredient, an objective, get ranked swaps with the evidence.

Run (inside BTP_2/substitution):   python cook.py
"""
import sys
import numpy as np
import pandas as pd
import ranking as RK
from retrieval import Retriever
from recipes import load_recipes
from substitute import substitute


def ask(prompt, default=None):
    try:
        v = input(prompt).strip()
    except EOFError:
        print(); sys.exit(0)
    return v if v else default


def pick(prompt, n):
    """ask for a number 1..n, or q to quit, or b to go back (returns None)."""
    while True:
        v = ask(prompt)
        if v is None or v.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        if v.lower() in ("b", "back"):
            return None
        if v.isdigit() and 1 <= int(v) <= n:
            return int(v) - 1
        print(f"  enter a number from 1 to {n} (b = back, q = quit)")


def search_recipes(rec, trusted, text, limit=15):
    words = text.lower().split()
    titles = rec.drop_duplicates("recipe_id")[["recipe_id", "recipe_title"]]
    mask = np.ones(len(titles), bool)
    for w in words:
        mask &= titles.recipe_title.str.lower().str.contains(w, regex=False).to_numpy()
    hit = titles[mask]
    if hit.empty:
        return hit
    n_ing = rec.groupby("recipe_id").ingredient_name.nunique()
    n_tr = rec[rec.ingredient_name.isin(trusted)].groupby("recipe_id").ingredient_name.nunique()
    hit = hit.assign(ingredients=hit.recipe_id.map(n_ing), with_nutrition=hit.recipe_id.map(n_tr).fillna(0).astype(int))
    return hit.sort_values(["with_nutrition", "ingredients"], ascending=False).head(limit).reset_index(drop=True)


def main():
    print("Loading data (first run takes a minute)...")
    R = Retriever()
    rec = load_recipes(R.master.name)
    M = R.master
    trusted = set(M.name[M.tier == "trusted"])
    tier = M.set_index("name").tier.to_dict()
    family = M.set_index("name").family.to_dict()
    print("Ready.  (b = go back, q = quit at any prompt)\n")

    while True:
        # ---- 1. recipe ----
        text = ask("Which recipe are you cooking? (type part of the name): ")
        if text is None:
            continue
        if text.lower() in ("q", "quit", "exit"):
            return
        hits = search_recipes(rec, trusted, text)
        if hits.empty:
            print("  no recipe matches that. Try fewer or different words.\n")
            continue
        print()
        for i, r in hits.iterrows():
            print(f"  {i+1:>2}. {r.recipe_title}   [{r.with_nutrition} of {r.ingredients} ingredients have trusted nutrition]")
        k = pick("\nPick a recipe number: ", len(hits))
        if k is None:
            continue
        rid = int(hits.recipe_id[k])

        while True:
            rr = rec[rec.recipe_id == rid].drop_duplicates("ingredient_name").reset_index(drop=True)
            print(f"\nRecipe: {rr.recipe_title.iloc[0]}")
            for i, r in rr.iterrows():
                ok = r.ingredient_name in trusted
                amt = f"{r.mass_g:.0f} g" if pd.notna(r.mass_g) else "amount not in grams"
                print(f"  {i+1:>2}. {r.ingredient_name:<28} {amt:<20} {'' if ok else '(no trusted nutrition, cannot optimise)'}")
            k = pick("\nWhich ingredient do you want to optimise? (number): ", len(rr))
            if k is None:
                break
            ing = rr.ingredient_name[k]
            if ing not in trusted:
                print(f"  '{ing}' has no trusted nutrition yet, so a change cannot be computed. Pick another.")
                continue

            # ---- 3. objective ----
            options = [(n, o) for n, o in RK.PRESETS.items()
                       if not (o.applies_to_family and family.get(ing) not in o.applies_to_family)]
            desc = {"more_protein": "more protein, carbs not higher",
                    "less_carbs": "fewer carbs, protein not lower",
                    "plant_swap": "replace with a plant protein (equal protein, shows the carb cost)"}
            print(f"\nObjective for '{ing}':")
            for i, (n, o) in enumerate(options):
                print(f"  {i+1}. {n:<13} {desc.get(n, '')}")
            j = pick("Pick an objective: ", len(options))
            if j is None:
                continue
            obj_name = options[j][0]

            print("\n" + "=" * 78)
            print(substitute(R, rec, ing, obj_name, rid, top=5))
            print("=" * 78)

            nxt = ask("\n[Enter] another ingredient in this recipe | n = new recipe | q = quit: ", "")
            if nxt.lower() in ("q", "quit", "exit"):
                return
            if nxt.lower() in ("n", "new"):
                break


if __name__ == "__main__":
    main()