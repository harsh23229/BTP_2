#!/usr/bin/env python3
"""Create poster-ready BTP evaluation graphs (300 DPI).

Expected project layout:
BTP_2/
├── EDA/eda_08_graphs.py
├── EDA/results/flavour/molecule_audit_entity_stats.csv
└── substitution/outputs/
    ├── evaluation_summary.txt
    ├── gold_summary.txt
    └── ingredient_master.csv

Run from anywhere with: python3 EDA/eda_08_graphs.py
All figures and the exact plotted data are saved under EDA/results/graph/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results" / "graph"
SUB_OUTPUTS = PROJECT_ROOT / "substitution" / "outputs"
EVAL_SUMMARY = SUB_OUTPUTS / "evaluation_summary.txt"
GOLD_SUMMARY = SUB_OUTPUTS / "gold_summary.txt"
MASTER_CSV = SUB_OUTPUTS / "ingredient_master.csv"
FLAVOR_ENTITY_CSV = SCRIPT_DIR / "results" / "flavour" / "molecule_audit_entity_stats.csv"
DPI = 300

# Consistent, print-friendly typography. Matplotlib's default color cycle is retained.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 16,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            "Check that the project folders and evaluation outputs are in their expected locations."
        )


def save_figure(fig: plt.Figure, filename: str) -> None:
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def parse_objective_coverage(summary_text: str) -> pd.DataFrame:
    rows = []
    pattern = re.compile(
        r"^\s*(more_protein|less_carbs|less_calories|less_fat|plant_swap):\s*"
        r"(\d+)\s*/\s*(\d+)\s+cases returned at least one candidate;\s*(\d+)\s+"
        r"top-ranked rows exported\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(summary_text):
        objective, returned, total, ranked_rows = match.groups()
        rows.append({
            "objective": objective,
            "cases_with_candidates": int(returned),
            "total_cases": int(total),
            "cases_without_candidates": int(total) - int(returned),
            "ranked_rows_exported": int(ranked_rows),
            "coverage_percent": 100 * int(returned) / int(total) if int(total) else np.nan,
        })
    if not rows:
        raise ValueError("Could not parse OBJECTIVE COVERAGE from evaluation_summary.txt")
    return pd.DataFrame(rows)


def parse_gold_section(gold_text: str, section_label: str) -> pd.DataFrame:
    """Parse one all-pairs table from gold_summary.txt without hardcoding metric values."""
    lines = gold_text.splitlines()
    start = next((i for i, line in enumerate(lines) if section_label in line), None)
    if start is None:
        raise ValueError(f"Could not find gold evaluation section: {section_label}")

    rows = []
    row_pattern = re.compile(
        r"^\s*(context only|flavour only|context \+ flavour|embedding only|"
        r"context \+ embedding|all three \(our ranking\)|random baseline)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
    )
    for line in lines[start + 1:]:
        if line.strip().startswith(("ALL pairs | family gate", "STRONG pairs", "Queries where", "queries whose")):
            break
        match = row_pattern.match(line)
        if match:
            method, h1, h3, h5, h10, mrr, n = match.groups()
            rows.append({
                "method": method,
                "Hit@1": float(h1),
                "Hit@3": float(h3),
                "Hit@5": float(h5),
                "Hit@10": float(h10),
                "MRR": float(mrr),
                "n_queries": int(float(n)),
            })
    if not rows:
        raise ValueError(f"Could not parse rows for section: {section_label}")
    return pd.DataFrame(rows)


def plot_objective_coverage(df: pd.DataFrame) -> None:
    labels = {
        "more_protein": "More protein",
        "less_carbs": "Fewer carbohydrates",
        "less_calories": "Fewer calories",
        "less_fat": "Less fat",
        "plant_swap": "Equal-protein plant swap",
    }
    plot_df = df.copy()
    plot_df["label"] = plot_df["objective"].map(labels).fillna(plot_df["objective"])
    plot_df = plot_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(plot_df))
    returned = plot_df["cases_with_candidates"].to_numpy()
    missing = plot_df["cases_without_candidates"].to_numpy()
    ax.barh(y, returned, label="At least one candidate")
    ax.barh(y, missing, left=returned, label="No candidate")
    ax.set_yticks(y, plot_df["label"])
    ax.set_xlabel("Evaluation cases")
    ax.set_title("Candidate availability across nutritional objectives", pad=14, weight="bold")
    ax.set_xlim(0, max(plot_df["total_cases"]) * 1.22)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for yi, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(row["total_cases"] + 0.15, yi,
                f'{row["cases_with_candidates"]}/{row["total_cases"]}  ({row["coverage_percent"]:.0f}%)',
                va="center", fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "01_objective_candidate_coverage.png")


def plot_retrieval_ablation(gate_on: pd.DataFrame, gate_off: pd.DataFrame) -> None:
    merged = gate_on[["method", "Hit@5"]].merge(
        gate_off[["method", "Hit@5"]], on="method", suffixes=("_family_gate_on", "_family_gate_off")
    )
    merged.to_csv(RESULTS_DIR / "retrieval_hit5_plot_data.csv", index=False)
    methods = merged["method"].tolist()
    labels = [m.replace(" (our ranking)", "").title() for m in methods]
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bars1 = ax.bar(x - width / 2, merged["Hit@5_family_gate_on"], width, label="Food-family gate ON")
    bars2 = ax.bar(x + width / 2, merged["Hit@5_family_gate_off"], width, label="Food-family gate OFF")
    ax.set_ylabel("Hit@5")
    ax.set_title("Effect of the food-family gate on gold-set retrieval", pad=14, weight="bold")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "02_gold_set_hit5_family_gate_ablation.png")


def plot_retrieval_mrr(gate_on: pd.DataFrame) -> None:
    df = gate_on.copy()
    df.to_csv(RESULTS_DIR / "retrieval_mrr_plot_data.csv", index=False)
    labels = [m.replace(" (our ranking)", "").title() for m in df["method"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bars = ax.bar(labels, df["MRR"])
    ax.set_ylabel("Mean Reciprocal Rank (MRR)")
    ax.set_title("Gold-set ranking quality (food-family gate ON)", pad=14, weight="bold")
    ax.set_ylim(0, 0.78)
    ax.tick_params(axis="x", rotation=18)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "03_gold_set_mrr_family_gate_on.png")


def plot_nutrient_and_flavour_coverage(master: pd.DataFrame) -> None:
    required = {"tier", "has_molecules"}
    if not required.issubset(master.columns):
        raise ValueError(f"ingredient_master.csv must contain columns: {sorted(required)}")
    tier_order = ["trusted", "uncertain", "unavailable"]
    tier_counts = master["tier"].fillna("unavailable").value_counts().reindex(tier_order, fill_value=0)
    trusted = master[master["tier"].eq("trusted")]
    molecule_count = int(trusted["has_molecules"].fillna(False).astype(bool).sum())
    trusted_count = int(len(trusted))
    no_molecules = trusted_count - molecule_count

    coverage_data = pd.DataFrame([
        {"measure": "Trusted USDA nutrition", "category": "Trusted", "ingredients": int(tier_counts["trusted"])},
        {"measure": "Nutrient trust tier", "category": "Uncertain", "ingredients": int(tier_counts["uncertain"])},
        {"measure": "Nutrient trust tier", "category": "Unavailable", "ingredients": int(tier_counts["unavailable"])},
        {"measure": "FlavorDB coverage within trusted pool", "category": "Has usable molecules", "ingredients": molecule_count},
        {"measure": "FlavorDB coverage within trusted pool", "category": "No usable molecules", "ingredients": no_molecules},
    ])
    coverage_data.to_csv(RESULTS_DIR / "ingredient_data_coverage_plot_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    left_labels = ["Trusted", "Uncertain", "Unavailable"]
    left_values = [int(tier_counts[x]) for x in tier_order]
    bars = axes[0].bar(left_labels, left_values)
    axes[0].set_title(f"Nutritional data tiers\n({len(master):,} graph ingredients)", weight="bold")
    axes[0].set_ylabel("Number of ingredients")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)
    axes[0].bar_label(bars, labels=[f"{v:,}" for v in left_values], padding=3, fontsize=9)

    right_values = [molecule_count, no_molecules]
    right_labels = ["Molecules available", "No usable profile"]
    bars = axes[1].bar(right_labels, right_values)
    pct = 100 * molecule_count / trusted_count if trusted_count else 0
    axes[1].set_title(f"FlavorDB coverage in trusted pool\n({molecule_count}/{trusted_count}; {pct:.1f}%)", weight="bold")
    axes[1].set_ylabel("Number of trusted ingredients")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)
    axes[1].bar_label(bars, labels=[f"{v:,}" for v in right_values], padding=3, fontsize=9)
    for ax in axes:
        ax.tick_params(axis="x", rotation=12)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    fig.suptitle("Coverage limits of the substitution evidence", weight="bold")
    fig.tight_layout()
    save_figure(fig, "04_nutrient_and_flavour_data_coverage.png")


def write_readme(objective_df: pd.DataFrame, gate_on: pd.DataFrame, gate_off: pd.DataFrame,
                 master: pd.DataFrame) -> None:
    trusted = int(master["tier"].eq("trusted").sum())
    total = len(master)
    has_molecules = int(master.loc[master["tier"].eq("trusted"), "has_molecules"].fillna(False).astype(bool).sum())
    lines = [
        "# BTP poster graphs", "",
        "Generated by `EDA/eda_08_graphs.py`.",
        "All PNG figures are exported at 300 DPI. CSV files preserve the plotted values.", "",
        "## Input sources", "",
        "- `substitution/outputs/evaluation_summary.txt` — objective-wise candidate coverage.",
        "- `substitution/outputs/gold_summary.txt` — gold-set retrieval ablation and MRR.",
        "- `substitution/outputs/ingredient_master.csv` — nutrient trust tiers and FlavorDB availability.", "",
        "## Key values represented", "",
        f"- Ingredient master rows: {total:,}.",
        f"- Trusted nutrition: {trusted:,} ingredients.",
        f"- FlavorDB molecules available among trusted ingredients: {has_molecules}/{trusted} "
        f"({100 * has_molecules / trusted:.1f}% if trusted else 0).",
        f"- Gold-set queries: {int(gate_on['n_queries'].iloc[0])}.",
        f"- Context-only Hit@5, family gate ON: {float(gate_on.loc[gate_on.method.eq('context only'), 'Hit@5'].iloc[0]):.2f}.",
        f"- Context-only Hit@5, family gate OFF: {float(gate_off.loc[gate_off.method.eq('context only'), 'Hit@5'].iloc[0]):.2f}.", "",
        "## Objective coverage", "",
        objective_df.to_string(index=False), "",
        "## Interpretation notes", "",
        "- Hit@5 is the fraction of gold queries with an acceptable answer in the top five.",
        "- The family gate restricts candidates to trusted ingredients in the target's food family.",
        "- FlavorDB molecular overlap is supporting evidence, not proof of sensory equivalence.",
        "- These plots describe the saved evaluation outputs; they do not imply human sensory validation.",
    ]
    (RESULTS_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [EVAL_SUMMARY, GOLD_SUMMARY, MASTER_CSV]:
        require_file(path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    eval_text = EVAL_SUMMARY.read_text(encoding="utf-8")
    gold_text = GOLD_SUMMARY.read_text(encoding="utf-8")
    objective_df = parse_objective_coverage(eval_text)
    gate_on = parse_gold_section(gold_text, "ALL pairs | family gate ON")
    gate_off = parse_gold_section(gold_text, "ALL pairs | family gate OFF")
    master = pd.read_csv(MASTER_CSV)

    # Save every chart's source values alongside the image.
    objective_df.to_csv(RESULTS_DIR / "objective_candidate_coverage_plot_data.csv", index=False)
    gate_on.to_csv(RESULTS_DIR / "gold_set_family_gate_on_plot_data.csv", index=False)
    gate_off.to_csv(RESULTS_DIR / "gold_set_family_gate_off_plot_data.csv", index=False)

    plot_objective_coverage(objective_df)
    plot_retrieval_ablation(gate_on, gate_off)
    plot_retrieval_mrr(gate_on)
    plot_nutrient_and_flavour_coverage(master)
    write_readme(objective_df, gate_on, gate_off, master)

    print("\nCompleted. Poster graphs and their plotted data are in:")
    print(RESULTS_DIR)
    print(f"\nFlavorDB coverage input: {FLAVOR_ENTITY_CSV} (not needed for these trusted-pool counts)")
    print("Note: this script intentionally does not recreate the report's beef-to-plant Figure 16,")
    print("because that example needs a fresh, reproducible run from the current substitution engine.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
