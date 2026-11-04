"""Paths for the substitution pipeline. Edit only if your folders differ."""
from pathlib import Path

HERE = Path(__file__).resolve().parent            # BTP_2/substitution
BASE_DIR = HERE.parent                            # BTP_2

# folder that contains nodes_our_recipeflavor.csv and edges_our_recipeflavor.csv
GRAPH_DIR = BASE_DIR / "GastroGraph-custom" / "data"

# recipe table (same file the nutrient audit used)
RECIPE_FILE = BASE_DIR / "dataset" / "raw" / "RecipeDB_full" / "recipe_ingredients.csv"

# output of the nutrient audit notebook (Step 5)
NUTRIENTS_FILE = BASE_DIR / "EDA" / "results" / "nutrient_audit" / "ingredient_nutrients_final.csv"
# RecipeDB per-100g audit is diagnostic only; it must NOT override USDA source tiers.
NUTRIENT_AUDIT_FILE = BASE_DIR / "EDA" / "results" / "nutrient_audit" / "ingredient_per100g_with_tiers.csv"

# GastroSAGE embeddings (set both to None to run without embeddings).
# node2idx.pkl maps node id -> row of the .npy; it is used to align rows, so no ordering is assumed.
EMBEDDINGS_FILE = BASE_DIR / "GastroGraph-custom" / "models" / "node_embeddings.npy"
NODE2IDX_FILE = BASE_DIR / "GastroGraph-custom" / "models" / "node2idx.pkl"


# USDA folders (for food groups). Same folders the nutrient audit used.
USDA_SR = BASE_DIR / "dataset" / "external" / "usda" / "FoodData_Central_sr_legacy_food_csv_2018-04"
USDA_FF = BASE_DIR / "dataset" / "external" / "usda" / "FoodData_Central_foundation_food_csv_2026-04-30"

# A substitute must do the same JOB in a recipe. USDA food group -> job family.
FAMILY_BY_GROUP = {
    "Beef Products": "main_protein", "Pork Products": "main_protein", "Poultry Products": "main_protein",
    "Lamb, Veal, and Game Products": "main_protein", "Finfish and Shellfish Products": "main_protein",
    "Sausages and Luncheon Meats": "main_protein", "Legumes and Legume Products": "main_protein",
    "Dairy and Egg Products": "milk_cream", "Fats and Oils": "fat",
    "Cereal Grains and Pasta": "staple_grain", "Baked Products": "bread",
    "Vegetables and Vegetable Products": "vegetable", "Fruits and Fruit Juices": "fruit",
    "Spices and Herbs": "herb_spice", "Sweets": "sweetener", "Nut and Seed Products": "nut_seed",
    "Soups, Sauces, and Gravies": "sauce_soup", "Beverages": "beverage",
}
# per-ingredient fixes where the USDA group is not the recipe job
FAMILY_OVERRIDE = {
    "egg": "egg", "egg yolk": "egg", "egg white": "egg",
    "butter": "fat", "margarine": "fat", "mayonnaise": "fat",
    "peanut butter": "nut_seed", "peanut": "nut_seed", "coconut milk": "milk_cream",
    "soy sauce": "sauce_soup",
    "baking powder": "additive", "baking soda": "additive", "vanilla extract": "additive", "vanilla": "additive",
    "salt": "herb_spice", "cocoa powder": "additive",
    "flour": "flour_meal", "purpose flour": "flour_meal", "plain flour": "flour_meal", "wheat flour": "flour_meal",
    "cornmeal": "flour_meal", "cornstarch": "flour_meal", "oat": "flour_meal",
    "potato": "staple_grain", "red potato": "staple_grain", "water": "beverage",
    "tortilla chip": "snack", "yeast": "additive", "active yeast": "additive", "ghee": "fat",
    "cornflour": "flour_meal", "self raising flour": "flour_meal",
}
# USDA descriptions that start with these are a different family than their food group
FAMILY_BY_DESC_PREFIX = {"Cheese": "cheese"}

# Precomputed FlavorDB molecular similarity outputs from EDA/eda_03_flavour.ipynb.
FLAVOUR_RESULTS_DIR = BASE_DIR / "EDA" / "results" / "flavour"
FLAVOUR_SIMILARITY_FILE = FLAVOUR_RESULTS_DIR / "molecular_similarity.csv"
FLAVOUR_ENTITY_STATS_FILE = FLAVOUR_RESULTS_DIR / "molecule_audit_entity_stats.csv"

OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)