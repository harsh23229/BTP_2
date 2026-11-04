"""Single entry point for evidence-backed ingredient substitutions."""
import argparse
import numpy as np
import pandas as pd
import config as C
import ranking as RK
from retrieval import Retriever
from recipes import load_recipes


def substitute(R, rec, ingredient, objective, recipe_id=None, top=5):
    if objective not in RK.PRESETS:
        return f"Unknown objective '{objective}'. Available: {', '.join(RK.PRESETS)}"
    if ingredient not in R.name_idx:
        return f"'{ingredient}' is not an ingredient node in the graph."

    master = R.master
    orig = master.iloc[R.name_idx[ingredient]]
    obj = RK.PRESETS[objective]
    output = []
    recipe_names, mass_g = None, None

    if recipe_id is not None:
        recipe_rows = rec[rec.recipe_id == recipe_id]
        if recipe_rows.empty:
            return f"Recipe {recipe_id} not found (or none of its ingredients are graph nodes)."
        recipe_names = recipe_rows.ingredient_name.unique().tolist()
        output.append(f"Recipe {recipe_id}: {recipe_rows.recipe_title.iloc[0]}")
        if ingredient not in recipe_names:
            output.append(f"Note: '{ingredient}' is not in this recipe; ranking without recipe compatibility.")
            recipe_names = None
        else:
            mass_g = recipe_rows.loc[recipe_rows.ingredient_name.eq(ingredient), "mass_g"].iloc[0]

    if orig.tier == "trusted":
        output.append(
            f"Replace: {ingredient} | "
            f"Protein {orig.protein_100g:.2f} g, carbs {orig.carbs_100g:.2f} g, "
            f"fat {orig.fat_100g:.2f} g, kcal {orig.kcal_100g:.1f} per 100 g | "
            f"source: {orig.source} ({orig.usda_description})"
        )
    else:
        output.append(f"Replace: {ingredient} (nutrition tier: {orig.tier})")
    output.append(f"Objective: {objective}")

    if orig.tier != "trusted":
        output.append(
            f"NO RECOMMENDATION: '{ingredient}' does not have a validated trusted nutrient profile."
        )
        return "\n".join(output)
    if obj.applies_to_family and orig.family not in obj.applies_to_family:
        output.append(
            f"NO RECOMMENDATION: '{objective}' applies to {obj.applies_to_family}, "
            f"but '{ingredient}' belongs to family '{orig.family}'."
        )
        return "\n".join(output)

    candidates, report = R.candidates(ingredient, top_k=40, recipe_names=recipe_names)
    if recipe_names:
        candidates = candidates[~candidates.in_recipe]
    ranked, rejected = RK.rank(orig, candidates, obj, mass_g=mass_g, has_recipe=bool(recipe_names))

    if ranked.empty:
        output.append("NO CANDIDATE PASSES the hard nutritional constraints.")
        if not rejected.empty and "reject_reason" in rejected:
            reasons = (rejected.reject_reason.str.split(";").explode().str.strip()
                       .replace("", np.nan).dropna().value_counts().head(5))
            if not reasons.empty:
                output.append("Most common rejection reasons: " + "; ".join(
                    f"{reason} ({count})" for reason, count in reasons.items()
                ))
    else:
        output.append("")
        for _, row in ranked.head(top).iterrows():
            output.append(
                f"{int(row['rank'])}. {row['name']} "
                f"[{'best trade-off' if row.pareto else 'eligible'}]"
            )
            output.append("   " + row.explain)
            output.append(
                f"   Changes per 100 g original: protein {row.d_protein:+.2f} g, "
                f"carbs {row.d_carbs:+.2f} g, fat {row.d_fat:+.2f} g, "
                f"calories {row.d_kcal:+.1f} kcal"
            )
            if pd.notna(mass_g) and "recipe_candidate_mass_g" in row:
                output.append(
                    f"   Recipe amount: {mass_g:.1f} g {ingredient} -> "
                    f"{row.recipe_candidate_mass_g:.1f} g {row['name']}"
                )
                output.append(
                    f"   Recipe-line changes: protein {row.recipe_d_protein:+.2f} g, "
                    f"carbs {row.recipe_d_carbs:+.2f} g, fat {row.recipe_d_fat:+.2f} g, "
                    f"calories {row.recipe_d_kcal:+.1f} kcal"
                )
            elif recipe_names:
                output.append("   Recipe quantity is not available in a supported mass unit.")

    output.append("")
    output.append("DATA LIMITS")
    output.append(
        f"{report['top50_without_trusted_nutrition']} of the 50 closest context neighbours "
        "lack trusted nutrition and were not ranked."
    )
    output.append(
        "Comparison basis: equal weight (100 g for 100 g)."
        if not obj.equal_protein else
        "Comparison basis: equal protein, raw ingredient weights."
    )
    output.append(
        "FlavorDB molecular similarity uses IDF-weighted Jaccard. It is supporting evidence, "
        "not proof that two ingredients are interchangeable."
    )
    output.append(
        "Identical-profile and low-evidence flags are retained in candidate results; "
        "identical profiles are not automatically discarded."
    )
    output.append(
        "Nutrient values are USDA per-100-g values; RecipeDB-derived conditional profiles "
        "are not used for ranking."
    )
    return "\n".join(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=int)
    parser.add_argument("--ingredient")
    parser.add_argument("--objective", default="more_protein", choices=list(RK.PRESETS))
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--list-recipes", action="store_true")
    args = parser.parse_args()

    retriever = Retriever()
    recipes = load_recipes(retriever.master.name)
    if args.list_recipes:
        trusted = set(retriever.master.name[retriever.master.tier.eq("trusted")])
        grouped = recipes[recipes.ingredient_name.isin(trusted)].groupby("recipe_id").agg(
            title=("recipe_title", "first"), n=("ingredient_name", "nunique")
        )
        eligible = grouped[grouped.n >= 5]
        print(eligible.sample(min(10, len(eligible)), random_state=1).to_string())
    else:
        if not args.ingredient:
            parser.error("--ingredient is required unless --list-recipes is used")
        print(substitute(retriever, recipes, args.ingredient, args.objective, args.recipe, args.top))
