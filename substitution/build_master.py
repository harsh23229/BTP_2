"""STEP 1 - build the ingredient master table (one row per graph ingredient).

Run:  python build_master.py
Writes: outputs/ingredient_master.csv
"""
import numpy as np
import pandas as pd
import config as C

nodes = pd.read_csv(C.GRAPH_DIR / "nodes_our_recipeflavor.csv")
edges = pd.read_csv(C.GRAPH_DIR / "edges_our_recipeflavor.csv")
ing = nodes[nodes.node_type == "ingredient"][["node_id", "name"]].copy()

# molecules per ingredient (CONTAINS edges)
mol = edges[edges.edge_type == "ingr-dcomp"].groupby("id_1").size().rename("n_molecules")
# co-occurrence degree (CO_OCCURS edges, both directions)
cc = edges[edges.edge_type == "ingr-ingr"]
deg = pd.concat([cc.id_1, cc.id_2]).value_counts().rename("cooc_degree")

# recipe frequency
rec = pd.read_csv(C.RECIPE_FILE, usecols=["ingredient_name"], low_memory=False)
freq = rec.ingredient_name.astype("string").str.strip().str.lower().value_counts().rename("recipe_rows")

nut = pd.read_csv(C.NUTRIENTS_FILE, index_col=0)[
    ["protein_100g", "carbs_100g", "fat_100g", "kcal_100g", "source", "tier", "usda_description"]]

m = (ing.set_index("name")
        .join(nut, how="left").join(freq, how="left").reset_index())
m = m.merge(mol, left_on="node_id", right_index=True, how="left").merge(deg, left_on="node_id", right_index=True, how="left")
m["tier"] = m.tier.fillna("unavailable")

# USDA food group -> recipe-job family
cat = pd.read_csv(C.USDA_SR / "food_category.csv").set_index("id").description
food = pd.concat([pd.read_csv(C.USDA_SR / "food.csv", usecols=["description", "food_category_id"]),
                  pd.read_csv(C.USDA_FF / "food.csv", usecols=["description", "food_category_id"])]).dropna()
food["food_group"] = food.food_category_id.astype(int).map(cat)
grp = food.drop_duplicates("description").set_index("description").food_group
m["food_group"] = m.usda_description.map(grp)
m["family"] = m.food_group.map(C.FAMILY_BY_GROUP)
for pre, fam in C.FAMILY_BY_DESC_PREFIX.items():
    m.loc[m.usda_description.fillna("").str.startswith(pre), "family"] = fam
m["family"] = [C.FAMILY_OVERRIDE.get(n, f) for n, f in zip(m.name, m.family)]
m[["n_molecules", "cooc_degree", "recipe_rows"]] = m[["n_molecules", "cooc_degree", "recipe_rows"]].fillna(0).astype(int)
m["has_molecules"] = m.n_molecules > 0
m = m.sort_values("recipe_rows", ascending=False).reset_index(drop=True)
m.to_csv(C.OUT / "ingredient_master.csv", index=False)

print("ingredients in graph:", len(m))
print(m.tier.value_counts().to_string())
print("trusted with molecules:", int(((m.tier == "trusted") & m.has_molecules).sum()), "of", int((m.tier == "trusted").sum()))
print("trusted without a family (fix FAMILY_OVERRIDE):", m[(m.tier == "trusted") & m.family.isna()].name.tolist())
print(m[m.tier == "trusted"].family.value_counts().to_string())
print("saved", C.OUT / "ingredient_master.csv")