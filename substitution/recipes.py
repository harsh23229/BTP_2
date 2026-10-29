"""STEP 4 (library) - recipe loader. Parses quantity/unit to grams only for mass units (g, kg, oz, lb)."""
from fractions import Fraction
import numpy as np
import pandas as pd
import config as C

TO_G = {"g": 1, "gram": 1, "grams": 1, "kg": 1000, "kilogram": 1000, "kilograms": 1000,
        "oz": 28.3495, "ounce": 28.3495, "ounces": 28.3495,
        "lb": 453.592, "lbs": 453.592, "pound": 453.592, "pounds": 453.592}


def _qty(s):
    if pd.isna(s) or "-" in str(s):
        return np.nan
    try:
        return sum(float(Fraction(p)) for p in str(s).split())
    except (ValueError, ZeroDivisionError):
        return np.nan


def load_recipes(graph_names):
    cache = C.OUT / "recipes_cache.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = pd.read_csv(C.RECIPE_FILE, usecols=["recipe_id", "recipe_title", "ingredient_name", "quantity", "unit"], low_memory=False)
    df["ingredient_name"] = df.ingredient_name.astype("string").str.strip().str.lower()
    df = df[df.ingredient_name.isin(set(graph_names))].copy()
    df["qty"] = df.quantity.map(_qty)
    df["mass_g"] = df.unit.astype("string").str.strip().str.lower().map(TO_G) * df.qty
    df = df[["recipe_id", "recipe_title", "ingredient_name", "mass_g"]]
    df.to_pickle(cache)
    return df