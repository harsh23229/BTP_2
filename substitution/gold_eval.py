"""STEP 6 - does the combined retrieval beat its own parts and a random baseline?

For each known substitution in gold_pairs.csv (query -> acceptable substitutes) every judge ranks the candidate pool
and we ask: how high is the first acceptable answer? Judges: context, flavour, embedding, and their combinations.
Candidates are deduplicated by USDA food, because 'garlic' and 'garlic clove' are one candidate.

Run:  python gold_eval.py
Writes: outputs/gold_results_per_query.csv, outputs/gold_summary.txt
"""
import warnings
import numpy as np
import pandas as pd
import config as C

warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean of an all-NaN row (a judge with no evidence)
from retrieval import Retriever, _pct

KS = [1, 3, 5, 10]
R = Retriever()
M = R.master
has_emb = R.E is not None
gold = pd.read_csv(C.HERE / "gold_pairs.csv")
usda_of = M.set_index("name").usda_description.to_dict()
tier_of = M.set_index("name").tier.to_dict()

JUDGES = {"context only": ["ctx"], "flavour only": ["fl"], "context + flavour": ["ctx", "fl"]}
if has_emb:
    JUDGES.update({"embedding only": ["emb"], "context + embedding": ["ctx", "emb"], "all three (our ranking)": ["ctx", "fl", "emb"]})
else:
    JUDGES["all available (our ranking)"] = ["ctx", "fl"]


def first_rank(order_usda, acc_usda):
    for i, u in enumerate(order_usda, 1):
        if u in acc_usda:
            return i
    return None


rows, missing = [], []
for _, g in gold.iterrows():
    q = g["query"]
    acc = [a.strip() for a in g.acceptable.split(";")]
    bad = [n for n in [q] + acc if tier_of.get(n) != "trusted"]
    if bad:
        missing.append((q, bad)); continue
    acc_usda = {usda_of[a] for a in acc} - {usda_of[q]}
    if not acc_usda:
        continue
    for gate in (False, True):
        df, _ = R.candidates(q, top_k=10**6, same_family=gate, dedupe=False)
        df = df.assign(ctx=_pct(df.ctx_sim), fl=_pct(df.flavor_sim), emb=_pct(df.emb_sim))
        in_pool = bool(set(df.usda_description) & acc_usda)
        for jname, parts in JUDGES.items():
            d = df.assign(score=np.nanmean(df[parts].to_numpy(dtype=float), axis=1)).dropna(subset=["score"])
            d = d.sort_values("score", ascending=False).drop_duplicates("usda_description")
            rows.append({"query": q, "confidence": g.confidence, "gate": gate, "judge": jname, "in_pool": in_pool,
                         "n_candidates": len(d), "rank": first_rank(d.usda_description.tolist(), acc_usda)})
        # random baseline: expected first-hit rank distribution by simulation over distinct foods
        foods = df.drop_duplicates("usda_description").usda_description.to_numpy()
        rng = np.random.default_rng(0)
        rr = []
        for _ in range(300):
            rr.append(first_rank(rng.permutation(foods).tolist(), acc_usda))
        for k in range(1):
            rows.append({"query": q, "confidence": g.confidence, "gate": gate, "judge": "random baseline", "in_pool": in_pool,
                         "n_candidates": len(foods), "rank": np.nan, "rand_ranks": [x for x in rr if x is not None]})

res = pd.DataFrame(rows)
res["rank"] = pd.to_numeric(res["rank"], errors="coerce")
res.drop(columns=["rand_ranks"], errors="ignore").to_csv(C.OUT / "gold_results_per_query.csv", index=False)


def metrics(jname, sub):
    n = len(sub)
    out = {}
    if jname == "random baseline":
        for k in KS:
            out[f"Hit@{k}"] = np.mean([np.mean([x <= k for x in rr]) if len(rr) else 0.0 for rr in sub.rand_ranks])
        out["MRR"] = np.mean([np.mean([1 / x for x in rr]) if len(rr) else 0.0 for rr in sub.rand_ranks])
    else:
        r = sub["rank"]
        for k in KS:
            out[f"Hit@{k}"] = float((r <= k).sum()) / n
        out["MRR"] = float(np.nansum(1.0 / r.to_numpy(dtype=float))) / n
    out["n"] = n
    return out


lines = ["GOLD-SET EVALUATION", "=" * 72,
         f"queries scored: {res['query'].nunique()}  (skipped: {len(missing)} with an untrusted name)",
         f"embeddings: {'used' if has_emb else 'NOT used'}", ""]
for subset, title in [(res, "ALL pairs"), (res[res.confidence == "strong"], "STRONG pairs only")]:
    for gate in (False, True):
        s = subset[subset.gate == gate]
        t = pd.DataFrame({j: metrics(j, s[s.judge == j]) for j in s.judge.drop_duplicates()}).T.round(2)
        lines += [f"{title} | family gate {'ON ' if gate else 'OFF'} | pool = trusted ingredients" + (" of the same job family" if gate else ""),
                  t.to_string(), ""]
# fairness check for the flavour judge: only queries where it had evidence, against random on the SAME queries
cov_q = set(res[(res.judge == "flavour only") & res.gate & res["rank"].notna()]["query"])
sc = res[res.gate & res["query"].isin(cov_q)]
tc = pd.DataFrame({j: metrics(j, sc[sc.judge == j]) for j in sc.judge.drop_duplicates()}).T.round(2)
lines += [f"Queries where the flavour judge HAS evidence (n={len(cov_q)}) | family gate ON", tc.to_string(), ""]
excluded = res[(res.gate) & (res.judge == "random baseline") & (~res.in_pool)]["query"].tolist()
lines.append(f"queries whose acceptable answer is excluded by the family gate: {len(excluded)} {excluded}")
fl = res[(res.judge == "flavour only") & (res.gate)]
lines.append(f"flavour judge produced an answer rank for {int(fl['rank'].notna().sum())} of {len(fl)} queries "
             f"(others lack molecule data for the query or for every acceptable answer)")
if missing:
    lines.append("skipped (name not trusted): " + "; ".join(f"{q}: {b}" for q, b in missing))
(C.OUT / "gold_summary.txt").write_text("\n".join(lines))
print("\n".join(lines))