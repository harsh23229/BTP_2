"""Nutrient-aware substitution ranking with hard constraints before similarity ranking."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

COL = {
    "protein": "protein_100g",
    "carbs": "carbs_100g",
    "fat": "fat_100g",
    "kcal": "kcal_100g",
}
UNITS = {"protein": "g/100 g", "carbs": "g/100 g", "fat": "g/100 g", "kcal": "kcal/100 g"}


@dataclass
class Guard:
    nutrient: str
    max_increase: float | None = None
    max_decrease: float | None = None


@dataclass
class Objective:
    name: str
    target: str
    direction: str
    guards: list = field(default_factory=list)
    min_gain: float = 3.0
    weights: tuple = (0.4, 0.3, 0.3)
    require_groups: list | None = None
    applies_to_family: tuple | None = None
    equal_protein: bool = False
    max_mass_ratio: float = 3.0


PRESETS = {
    "more_protein": Objective(
        "more_protein", "protein", "increase",
        [Guard("carbs", max_increase=2.0)], min_gain=3.0
    ),
    "less_carbs": Objective(
        "less_carbs", "carbs", "decrease",
        [Guard("protein", max_decrease=2.0)], min_gain=3.0
    ),
    "less_calories": Objective(
        "less_calories", "kcal", "decrease",
        [Guard("protein", max_decrease=2.0)], min_gain=10.0
    ),
    "less_fat": Objective(
        "less_fat", "fat", "decrease",
        [Guard("protein", max_decrease=2.0)], min_gain=1.0
    ),
    "plant_swap": Objective(
        "plant_swap", "carbs", "decrease", [], min_gain=-80.0,
        require_groups=["Legumes and Legume Products"],
        applies_to_family=("main_protein",), equal_protein=True, max_mass_ratio=3.0
    ),
}


def _finite(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def violates(row, obj):
    """Independent hard-rule check; missing/non-finite data always fails."""
    target_delta = row.get(f"d_{obj.target}", np.nan)
    if not _finite(target_delta):
        return True
    sign = 1.0 if obj.direction == "increase" else -1.0
    if sign * float(target_delta) < obj.min_gain - 1e-9:
        return True
    for guard in obj.guards:
        delta = row.get(f"d_{guard.nutrient}", np.nan)
        if not _finite(delta):
            return True
        delta = float(delta)
        if guard.max_increase is not None and delta > guard.max_increase + 1e-9:
            return True
        if guard.max_decrease is not None and -delta > guard.max_decrease + 1e-9:
            return True
    if obj.equal_protein:
        ratio = row.get("mass_ratio", np.nan)
        protein_delta = row.get("d_protein", np.nan)
        if not _finite(ratio) or not _finite(protein_delta):
            return True
        if float(ratio) > obj.max_mass_ratio + 1e-9 or abs(float(protein_delta)) > 1e-6:
            return True
    if obj.require_groups and row.get("food_group") not in obj.require_groups:
        return True
    return False


def _pct(values):
    s = pd.Series(values, dtype=float)
    return s.rank(pct=True, method="average").to_numpy()


def _pareto(matrix):
    n = len(matrix)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(matrix >= matrix[i], axis=1) & np.any(matrix > matrix[i], axis=1)
        keep[i] = not dominated.any()
    return keep


def rank(orig, cands, obj, mass_g=None, has_recipe=True):
    """Return (ranked, rejected). Missing nutrient values are rejected, never passed."""
    if obj.target not in COL:
        raise ValueError(f"Unknown target nutrient: {obj.target}")
    if orig.get("tier") != "trusted":
        raise ValueError(f"{orig.get('name', 'ingredient')} has no trusted nutrition ({orig.get('tier')})")

    c = cands.copy()
    if c.empty:
        c["reject_reason"] = pd.Series(dtype="object")
        return c, c.copy()

    required_nutrients = list(COL)
    missing_columns = [COL[k] for k in required_nutrients if COL[k] not in c.columns]
    if missing_columns:
        raise ValueError(f"Candidate table missing nutrient columns: {missing_columns}")
    missing_orig = [k for k, col in COL.items() if not _finite(orig.get(col, np.nan))]
    if missing_orig:
        raise ValueError(f"Original ingredient has missing/non-finite nutrients: {missing_orig}")

    # All four nutrient deltas are reported; candidates without any of them are ineligible.
    for key, col in COL.items():
        c[col] = pd.to_numeric(c[col], errors="coerce")
    valid_candidate = np.isfinite(c[list(COL.values())]).all(axis=1)
    if obj.equal_protein:
        p0 = float(orig[COL["protein"]])
        pc = c[COL["protein"]].to_numpy(dtype=float)
        c["mass_ratio"] = np.where(pc > 0, p0 / np.where(pc > 0, pc, 1.0), np.nan)
    else:
        c["mass_ratio"] = 1.0

    for key, col in COL.items():
        c[f"d_{key}"] = c[col] * c.mass_ratio - float(orig[col])
    sign = 1.0 if obj.direction == "increase" else -1.0
    c["gain"] = sign * c[f"d_{obj.target}"]

    reasons = []
    for idx, row in c.iterrows():
        why = []
        if not valid_candidate.loc[idx]:
            why.append("missing/non-finite trusted nutrient value")
        if obj.equal_protein:
            if not _finite(row.mass_ratio):
                why.append("no protein, cannot equalise")
            elif row.mass_ratio > obj.max_mass_ratio:
                why.append(f"needs {row.mass_ratio:.1f}x the amount (> {obj.max_mass_ratio})")
        if obj.require_groups and row.get("food_group") not in obj.require_groups:
            why.append(f"not in allowed group ({row.get('food_group')})")
        if not _finite(row.gain):
            why.append(f"missing target nutrient: {obj.target}")
        elif row.gain < obj.min_gain - 1e-9:
            why.append(f"{obj.target} gain {row.gain:+.1f} < {obj.min_gain}")
        for guard in obj.guards:
            delta = row.get(f"d_{guard.nutrient}", np.nan)
            if not _finite(delta):
                why.append(f"missing guard nutrient: {guard.nutrient}")
                continue
            if guard.max_increase is not None and delta > guard.max_increase + 1e-9:
                why.append(f"{guard.nutrient} +{delta:.1f} > allowed +{guard.max_increase}")
            if guard.max_decrease is not None and -delta > guard.max_decrease + 1e-9:
                why.append(f"{guard.nutrient} {delta:.1f} < allowed -{guard.max_decrease}")
        reasons.append("; ".join(dict.fromkeys(why)))

    c["reject_reason"] = reasons
    rejected = c[c.reject_reason.ne("")].copy()
    ok = c[c.reject_reason.eq("")].copy().reset_index(drop=True)
    if ok.empty:
        return ok, rejected

    # Retrieval similarity is required for ranking; reject missing values rather than silently score as zero.
    for col in ["sim_score", "ctx_sim"]:
        if col not in ok:
            raise ValueError(f"Candidate table missing retrieval score column: {col}")
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    missing_similarity = ~np.isfinite(ok["sim_score"]) | ~np.isfinite(ok["ctx_sim"])
    if missing_similarity.any():
        bad = ok.loc[missing_similarity].copy()
        bad["reject_reason"] = "missing/non-finite retrieval similarity"
        rejected = pd.concat([rejected, bad], ignore_index=True)
        ok = ok.loc[~missing_similarity].copy().reset_index(drop=True)
    if ok.empty:
        return ok, rejected

    w_gain, w_sim, w_compat = obj.weights
    compat = pd.to_numeric(ok.get("compat", pd.Series(np.nan, index=ok.index)), errors="coerce").to_numpy(float)
    use_compat = has_recipe and np.isfinite(compat).any()
    if not use_compat:
        w_sim += w_compat
        w_compat = 0.0
    ok["gain_pct"] = _pct(ok.gain)
    ok["sim_pct"] = _pct(ok.sim_score)
    ok["compat_pct"] = _pct(np.nan_to_num(compat, nan=0.0)) if use_compat else 0.0
    total = w_gain + w_sim + w_compat
    ok["score"] = (
        w_gain * ok.gain_pct + w_sim * ok.sim_pct + w_compat * ok.compat_pct
    ) / total
    pareto_compat = np.nan_to_num(compat, nan=0.0) if use_compat else np.zeros(len(ok))
    ok["pareto"] = _pareto(np.column_stack([ok.gain, ok.sim_score, pareto_compat]))

    ok["swap_g_per_100g"] = 100.0 * ok.mass_ratio
    if mass_g is not None and _finite(mass_g):
        ok["recipe_line_mass_g"] = float(mass_g)
        ok["recipe_candidate_mass_g"] = float(mass_g) * ok.mass_ratio
        for key in COL:
            ok[f"recipe_d_{key}"] = ok[f"d_{key}"] * float(mass_g) / 100.0

    def explain(row):
        if obj.equal_protein:
            head = (
                f"{row.swap_g_per_100g:.0f} g {row['name']} per 100 g {orig['name']} (raw weight; "
                f"protein held), carbs {row.d_carbs:+.1f} g, fat {row.d_fat:+.1f} g, "
                f"kcal {row.d_kcal:+.0f}"
            )
        else:
            unit = UNITS[obj.target]
            head = f"{obj.target} {row[f'd_{obj.target}']:+.1f} {unit}"
            for guard in obj.guards:
                head += f", {guard.nutrient} {row[f'd_{guard.nutrient}']:+.1f} g/100 g"
        return (
            head + f" | context {row.ctx_sim:.2f}"
            + (f", FlavorDB molecular similarity {row.flavour_sim:.3f}" if "flavour_sim" in row and _finite(row.flavour_sim) else ", FlavorDB molecular similarity n/a")
            + (" [identical FlavorDB profiles; interpret cautiously]" if bool(row.get("flavour_identical_profile", False)) else "")
            + (" [low-evidence FlavorDB profile]" if bool(row.get("flavour_low_evidence", False)) else "")
            + (f", embedding rank {row.emb_pct:.0%}" if "emb_pct" in row and _finite(row.emb_pct) else "")
            + (f", co-occurs with {row.compat:.0%} of recipe ingredients"
               if use_compat and _finite(row.compat) else "")
        )

    ok["explain"] = [explain(row) for _, row in ok.iterrows()]
    ok = ok.sort_values("score", ascending=False).reset_index(drop=True)
    ok.insert(0, "rank", np.arange(1, len(ok) + 1))
    return ok, rejected
