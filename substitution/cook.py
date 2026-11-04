"""Interactive recipe substitution interface."""
import sys
import numpy as np
import pandas as pd
import ranking as RK
from retrieval import Retriever
from recipes import load_recipes
from substitute import substitute


def ask(prompt, default=None):
    try:
        value = input(prompt).strip()
    except EOFError:
        print()
        sys.exit(0)
    return value if value else default


def pick(prompt, n):
    while True:
        value = ask(prompt)
        if value is None or value.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        if value.lower() in ("b", "back"):
            return None
        if value.isdigit() and 1 <= int(value) <= n:
            return int(value) - 1
        print(f"Enter a number from 1 to {n} (b = back, q = quit)")


def search_recipes(recipes, trusted, text, limit=15):
    words = text.lower().split()
    titles = recipes.drop_duplicates("recipe_id")[["recipe_id", "recipe_title"]]
    mask = np.ones(len(titles), dtype=bool)
    for word in words:
        mask &= titles.recipe_title.str.lower().str.contains(word, regex=False).to_numpy()
    hits = titles[mask]
    if hits.empty:
        return hits
    total = recipes.groupby("recipe_id").ingredient_name.nunique()
    with_nutrition = recipes[recipes.ingredient_name.isin(trusted)].groupby("recipe_id").ingredient_name.nunique()
    hits = hits.assign(
        ingredients=hits.recipe_id.map(total),
        with_nutrition=hits.recipe_id.map(with_nutrition).fillna(0).astype(int)
    )
    return hits.sort_values(["with_nutrition", "ingredients"], ascending=False).head(limit).reset_index(drop=True)


def main():
    print("Loading data (first run takes a minute)...")
    retriever = Retriever()
    recipes = load_recipes(retriever.master.name)
    master = retriever.master
    trusted = set(master.name[master.tier.eq("trusted")])
    family = master.set_index("name").family.to_dict()
    print("Ready. (b = go back, q = quit at any prompt)")

    descriptions = {
        "more_protein": "more protein; carbohydrate increase limited",
        "less_carbs": "fewer carbohydrates; protein loss limited",
        "less_calories": "fewer calories; protein loss limited",
        "less_fat": "less fat; protein loss limited",
        "plant_swap": "plant-protein replacement at equal protein",
    }

    while True:
        query = ask("\nWhich recipe are you cooking? (type part of the name): ")
        if query is None:
            continue
        if query.lower() in ("q", "quit", "exit"):
            return
        hits = search_recipes(recipes, trusted, query)
        if hits.empty:
            print("No recipe matches. Try fewer or different words.")
            continue
        for i, row in hits.iterrows():
            print(f"{i+1:>2}. {row.recipe_title} [{row.with_nutrition}/{row.ingredients} ingredients with trusted nutrition]")
        choice = pick("Pick a recipe number: ", len(hits))
        if choice is None:
            continue
        recipe_id = int(hits.recipe_id.iloc[choice])

        while True:
            rows = recipes[recipes.recipe_id.eq(recipe_id)].drop_duplicates("ingredient_name").reset_index(drop=True)
            print(f"\nRecipe: {rows.recipe_title.iloc[0]}")
            for i, row in rows.iterrows():
                known = row.ingredient_name in trusted
                amount = f"{row.mass_g:.1f} g" if pd.notna(row.mass_g) else "amount not in grams"
                print(f"{i+1:>2}. {row.ingredient_name:<28} {amount:<20} "
                      f"{'' if known else '(no trusted nutrition)'}")
            choice = pick("Which ingredient do you want to optimise? ", len(rows))
            if choice is None:
                break
            ingredient = rows.ingredient_name.iloc[choice]
            if ingredient not in trusted:
                print(f"'{ingredient}' has no trusted nutrient profile. Choose another ingredient.")
                continue

            options = [
                (name, objective) for name, objective in RK.PRESETS.items()
                if not (objective.applies_to_family and family.get(ingredient) not in objective.applies_to_family)
            ]
            print(f"\nObjectives for '{ingredient}':")
            for i, (name, _) in enumerate(options, 1):
                print(f"{i}. {name:<15} {descriptions.get(name, '')}")
            choice = pick("Pick an objective: ", len(options))
            if choice is None:
                continue
            objective_name = options[choice][0]
            print("\n" + "=" * 78)
            print(substitute(retriever, recipes, ingredient, objective_name, recipe_id, top=5))
            print("=" * 78)

            next_action = ask("[Enter] another ingredient | n = new recipe | q = quit: ", "")
            if next_action.lower() in ("q", "quit", "exit"):
                return
            if next_action.lower() in ("n", "new"):
                break


if __name__ == "__main__":
    main()
