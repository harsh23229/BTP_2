"""STEP 3 (library) - Stage B: nutrition objective, hard guards, ranking.

Order of operations (this is the point of the design):
  1. HARD FILTERS first  : minimum gain on the target nutrient + guard nutrients (+ optional food-group filter).
                           A candidate that fails is rejected, with the reason recorded, and can never be
                           rescued by similarity.
  2. Then rank survivors : weighted rank-percentiles of (nutrient gain, similarity, compatibility) + Pareto flag.

Two comparison bases:
  * equal weight  (default)      : 100 g of the original is replaced by 100 g of the candidate.
  * equal protein (equal_protein): the candidate amount is scaled so protein is the SAME as the original,
                                   mass_ratio = protein_orig / protein_cand. All other nutrients are scaled by that ratio.
                                   All values stay RAW (as in USDA), so "100 g beef -> 70 g dry lentil" is raw weight.
If the recipe line has a mass unit (g/oz/lb) the changes in grams for that line are also reported.
Harsh's fat/kcal objectives plug in through the same Objective class.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

COL = {"protein": "protein_100g", "carbs": "carbs_100g", "fat": "fat_100g", "kcal": "kcal_100g"}


@dataclass
class Guard:
    nutrient: str
    max_increase: float | None = None      # candidate may raise it by at most this (per 100 g of the original)
    max_decrease: float | None = None      # candidate may lower it by at most this


@dataclass
class Objective:
    name: str
    target: str                             # protein | carbs | fat | kcal
    direction: str                          # increase | decrease
    guards: list = field(default_factory=list)
    min_gain: float = 3.0                   # target must improve by at least this (negative = may get worse by that much)
    weights: tuple = (0.4, 0.3, 0.3)        # gain, similarity, compatibility
    require_groups: list | None = None      # candidate's USDA food group must be one of these (e.g. plant-only)
    applies_to_family: tuple | None = None  # only run for originals in these job families
    equal_protein: bool = False             # compare at equal protein instead of equal weight
    max_mass_ratio: float = 3.0             # equal_protein only: reject if candidate amount > this x the original amount


PRESETS = {
    "more_protein": Objective("more_protein", "protein", "increase", [Guard("carbs", max_increase=2.0)]),
    "less_carbs": Objective("less_carbs", "carbs", "decrease", [Guard("protein", max_decrease=2.0)]),
    # Separate objective: replace an animal protein with a legume/tofu at EQUAL PROTEIN, ranked by the smallest
    # carb increase. Carbs may rise by at most 80 g per 100 g of the original (a sanity bound, tune freely).
    "plant_swap": Objective("plant_swap", "carbs", "decrease", [], min_gain=-80.0,
                            require_groups=["Legumes and Legume Products"], applies_to_family=("main_protein",),
                            equal_protein=True, max_mass_ratio=3.0),
}


def violates(row, obj):
    """independent re-check of the hard rules on a returned row (used by the evaluation audit)."""
    sign = 1.0 if obj.direction == "increase" else -1.0
    if sign * row[f"d_{obj.target}"] < obj.min_gain - 1e-9:
        return True
    for g in obj.guards:
        d = row[f"d_{g.nutrient}"]
        if g.max_increase is not None and d > g.max_increase + 1e-9:
            return True
        if g.max_decrease is not None and -d > g.max_decrease + 1e-9:
            return True
    if obj.equal_protein and (row["mass_ratio"] > obj.max_mass_ratio + 1e-9 or abs(row["d_protein"]) > 1e-6):
        return True
    if obj.require_groups and row["food_group"] not in obj.require_groups:
        return True
    return False


def _pct(x):
    return pd.Series(x, dtype=float).rank(pct=True, method="average").to_numpy()


def _pareto(M):
    """True for rows not dominated on any column (all columns: higher is better)."""
    n = len(M); keep = np.ones(n, bool)
    for i in range(n):
        dom = np.all(M >= M[i], axis=1) & np.any(M > M[i], axis=1)
        keep[i] = not dom.any()
    return keep


def rank(orig, cands, obj, mass_g=None, has_recipe=True):
    """orig: master row (Series) of the ingredient being replaced. cands: output of Retriever.candidates().
    Returns (ranked, rejected)."""
    if orig.tier != "trusted":
        raise ValueError(f"'{orig['name']}' has no trusted nutrition ({orig.tier}); change cannot be computed")
    c = cands.copy()

    # ---- comparison basis ----
    if obj.equal_protein:
        p0 = orig[COL["protein"]]
        pc = c[COL["protein"]].to_numpy(dtype=float)
        c["mass_ratio"] = np.where(pc > 0, p0 / np.where(pc > 0, pc, 1.0), np.nan)
    else:
        c["mass_ratio"] = 1.0
    for k, col in COL.items():
        c[f"d_{k}"] = c[col] * c.mass_ratio - orig[col]           # per 100 g of the ORIGINAL
    sign = 1.0 if obj.direction == "increase" else -1.0
    c["gain"] = sign * c[f"d_{obj.target}"]

    # ---- hard filters, with a recorded reason ----
    reasons = []
    for _, row in c.iterrows():
        why = []
        if obj.equal_protein:
            if np.isnan(row.mass_ratio):
                why.append("no protein, cannot equalise")
            elif row.mass_ratio > obj.max_mass_ratio:
                why.append(f"needs {row.mass_ratio:.1f}x the amount (> {obj.max_mass_ratio})")
        if obj.require_groups and row.food_group not in obj.require_groups:
            why.append(f"not in allowed group ({row.food_group})")
        if not np.isnan(row.gain) and row.gain < obj.min_gain:
            why.append(f"{obj.target} gain {row.gain:+.1f} < {obj.min_gain}")
        for g in obj.guards:
            d = row[f"d_{g.nutrient}"]
            if g.max_increase is not None and d > g.max_increase:
                why.append(f"{g.nutrient} +{d:.1f} > allowed +{g.max_increase}")
            if g.max_decrease is not None and -d > g.max_decrease:
                why.append(f"{g.nutrient} {d:.1f} < allowed -{g.max_decrease}")
        reasons.append("; ".join(why))
    c["reject_reason"] = reasons
    rejected = c[c.reject_reason != ""].copy()
    ok = c[c.reject_reason == ""].copy().reset_index(drop=True)
    if ok.empty:
        return ok, rejected

    # ---- score survivors ----
    w_g, w_s, w_c = obj.weights
    comp = ok.compat.to_numpy(dtype=float)
    use_comp = has_recipe and not np.all(np.isnan(comp))
    if not use_comp:
        w_s, w_c = w_s + w_c, 0.0
    ok["gain_pct"] = _pct(ok.gain)
    ok["sim_pct"] = _pct(ok.sim_score)
    ok["compat_pct"] = _pct(np.nan_to_num(comp, nan=0.0)) if use_comp else 0.0
    tot = w_g + w_s + w_c
    ok["score"] = (w_g * ok.gain_pct + w_s * ok.sim_pct + w_c * ok.compat_pct) / tot
    M = np.column_stack([ok.gain, ok.sim_score, np.nan_to_num(comp, nan=0.0) if use_comp else np.zeros(len(ok))])
    ok["pareto"] = _pareto(M)

    ok["swap_g_per_100g"] = 100.0 * ok.mass_ratio                  # candidate grams (raw) per 100 g of the original
    if mass_g is not None and not np.isnan(mass_g):
        ok["recipe_line_mass_g"] = mass_g
        ok["recipe_candidate_mass_g"] = mass_g * ok.mass_ratio
        for k in COL:
            ok[f"recipe_d_{k}_g"] = ok[f"d_{k}"] * mass_g / 100.0

    def _explain(r):
        if obj.equal_protein:
            head = (f"{r.swap_g_per_100g:.0f} g {r['name']} per 100 g {orig['name']} (raw wt), protein held: "
                    f"carbs {r.d_carbs:+.1f}, fat {r.d_fat:+.1f}, kcal {r.d_kcal:+.0f}")
        else:
            head = (f"{obj.target} {r[f'd_{obj.target}']:+.1f} g/100g"
                    + "".join(f", {g.nutrient} {r[f'd_{g.nutrient}']:+.1f}" for g in obj.guards))
        return (head + f" | context {r.ctx_sim:.2f}"
                + (f", flavour {r.flavor_sim:.2f}" if pd.notna(r.flavor_sim) else ", flavour n/a")
                + (f", embedding rank {r.emb_pct:.0%}" if "emb_pct" in r and pd.notna(r.emb_pct) else "")
                + (f", co-occurs with {r.compat:.0%} of recipe ingredients" if use_comp and pd.notna(r.compat) else ""))
    ok["explain"] = [_explain(r) for _, r in ok.iterrows()]
    ok = ok.sort_values("score", ascending=False).reset_index(drop=True)
    ok.insert(0, "rank", np.arange(1, len(ok) + 1))
    return ok, rejected