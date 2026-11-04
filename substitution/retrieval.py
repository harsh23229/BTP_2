"""Stage A: candidate retrieval and compatibility.

Similarity evidence:
  * ctx_sim: PPMI-weighted context cosine from recipe co-occurrence.
  * flavour_sim: precomputed FlavorDB IDF-weighted Jaccard score.
  * emb_sim: optional GastroSAGE cosine (ranked within candidate pool).
  * compat: direct co-occurrence with the other ingredients in the recipe.

FlavorDB scores are joined by normalized entity names. Pairs absent from the
pairwise file receive 0 only when BOTH entities have a non-small FlavorDB
profile. Missing/unmapped or small profiles remain NaN. Identical profiles
retain their computed score (often 1.0) and are flagged, not discarded.
"""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import config as C


def _pct(values):
    return pd.Series(values, dtype=float).rank(pct=True, method="average").to_numpy()


def _name_key(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().casefold())


class Retriever:
    def __init__(self, min_count=2):
        self.master = pd.read_csv(C.OUT / "ingredient_master.csv")
        required_nutrients = ["protein_100g", "carbs_100g", "fat_100g", "kcal_100g"]
        missing = [c for c in required_nutrients if c not in self.master.columns]
        if missing:
            raise ValueError(f"ingredient_master.csv is missing nutrient columns: {missing}")
        for col in required_nutrients:
            self.master[col] = pd.to_numeric(self.master[col], errors="coerce")
        self.master["nutrients_finite"] = np.isfinite(self.master[required_nutrients]).all(axis=1)

        nodes = pd.read_csv(C.GRAPH_DIR / "nodes_our_recipeflavor.csv")
        edges = pd.read_csv(C.GRAPH_DIR / "edges_our_recipeflavor.csv")
        self.ing_ids = self.master.node_id.tolist()
        self.idx = {n: i for i, n in enumerate(self.ing_ids)}
        self.name_idx = {n: i for i, n in enumerate(self.master.name)}
        N = len(self.ing_ids)

        # Symmetric ingredient co-occurrence matrix.
        cc = edges[edges.edge_type == "ingr-ingr"]
        r = cc.id_1.map(self.idx).to_numpy()
        c = cc.id_2.map(self.idx).to_numpy()
        w = cc.score.to_numpy(dtype=float)
        A = sp.coo_matrix((w, (r, c)), shape=(N, N)).tocsr()
        self.cooc = (A + A.T).tocsr()

        # PPMI context matrix.
        K = self.cooc.copy()
        K.data[K.data < min_count] = 0
        K.eliminate_zeros()
        tot = K.sum()
        if tot <= 0:
            raise ValueError("No ingredient co-occurrence edges remain after min_count filtering.")
        rs = np.asarray(K.sum(1)).ravel()
        cs = rs ** 0.75
        cs = cs / cs.sum() * tot
        K = K.tocoo()
        pmi = np.log((K.data * tot) / (rs[K.row] * cs[K.col] + 1e-12) + 1e-12)
        keep = pmi > 0
        P = sp.coo_matrix((pmi[keep], (K.row[keep], K.col[keep])), shape=(N, N)).tocsr()
        self.ppmi = P
        norms = np.sqrt(np.asarray(P.multiply(P).sum(1)).ravel()) + 1e-12
        self.Pn = sp.diags(1 / norms) @ P

        # Load FlavorDB molecular similarity and profile-availability evidence.
        self._load_flavour_similarity()

        # Optional embeddings, aligned by node ID/name mapping.
        self.E = None
        if C.EMBEDDINGS_FILE is not None and C.EMBEDDINGS_FILE.is_file():
            import pickle
            E = np.load(C.EMBEDDINGS_FILE)
            n2i = None
            if getattr(C, "NODE2IDX_FILE", None) is not None and C.NODE2IDX_FILE.is_file():
                with open(C.NODE2IDX_FILE, "rb") as fh:
                    n2i = pickle.load(fh)
            if n2i is not None:
                wrapper = n2i if isinstance(n2i, dict) and "node2idx" in n2i else None
                if wrapper is not None:
                    n2i = wrapper["node2idx"]
                keys = list(n2i.keys())[:3] if hasattr(n2i, "keys") else None
                if all(n in n2i for n in self.ing_ids):
                    rows = [n2i[n] for n in self.ing_ids]
                elif all(n in n2i for n in self.master.name):
                    rows = [n2i[n] for n in self.master.name]
                elif wrapper is not None and "nodes" in wrapper and all(n in set(wrapper["nodes"]) for n in self.ing_ids):
                    pos = {n: i for i, n in enumerate(wrapper["nodes"])}
                    rows = [pos[n] for n in self.ing_ids]
                else:
                    raise ValueError(f"node2idx keys do not match node ids or names; first keys: {keys}")
            else:
                if E.shape[0] != len(nodes):
                    raise ValueError("no node2idx given and embedding rows != number of nodes")
                order = {n: i for i, n in enumerate(nodes.node_id)}
                rows = [order[n] for n in self.ing_ids]
            E = E[np.asarray(rows)]
            self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

        self.trusted = self.master.index[
            self.master.tier.eq("trusted") & self.master.nutrients_finite
        ].to_numpy()
        if len(self.trusted) == 0:
            raise ValueError("No trusted ingredients with complete finite nutrient values.")

    def _load_flavour_similarity(self):
        sim_path = C.FLAVOUR_SIMILARITY_FILE
        stats_path = C.FLAVOUR_ENTITY_STATS_FILE
        if not sim_path.is_file():
            raise FileNotFoundError(
                f"Missing FlavorDB similarity file: {sim_path}\n"
                "Run EDA/eda_03_flavour.ipynb first and confirm it writes molecular_similarity.csv."
            )
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Missing FlavorDB entity statistics: {stats_path}\n"
                "The entity statistics file is needed to distinguish a true zero-overlap pair "
                "from an ingredient with no usable FlavorDB profile."
            )

        sim = pd.read_csv(sim_path, low_memory=False)
        stats = pd.read_csv(stats_path, low_memory=False)
        required_sim = {"entity_a", "entity_b", "molecular_similarity", "identical_profile_flag", "low_evidence_flag"}
        required_stats = {"entity_name", "molecule_count", "small_profile_flag"}
        if missing := required_sim - set(sim.columns):
            raise ValueError(f"FlavorDB similarity file missing columns: {sorted(missing)}")
        if missing := required_stats - set(stats.columns):
            raise ValueError(f"FlavorDB entity statistics missing columns: {sorted(missing)}")

        sim["key_a"] = sim.entity_a.map(_name_key)
        sim["key_b"] = sim.entity_b.map(_name_key)
        sim["molecular_similarity"] = pd.to_numeric(sim.molecular_similarity, errors="coerce")
        sim = sim[
            sim.key_a.ne("") & sim.key_b.ne("")
            & sim.molecular_similarity.between(0, 1, inclusive="both")
        ].copy()

        # Aggregate duplicate/alias name combinations conservatively: retain the
        # maximum observed score and preserve whether any source pair was flagged.
        forward = sim[["key_a", "key_b", "molecular_similarity", "identical_profile_flag", "low_evidence_flag"]].copy()
        reverse = forward.rename(columns={"key_a": "key_b", "key_b": "key_a"})
        pairs = pd.concat([forward, reverse], ignore_index=True)
        self.flavour_scores = (
            pairs.groupby(["key_a", "key_b"], as_index=False)
            .agg(
                flavour_sim=("molecular_similarity", "max"),
                flavour_identical_profile=("identical_profile_flag", "max"),
                flavour_low_evidence=("low_evidence_flag", "max"),
            )
        )
        stats["key"] = stats.entity_name.map(_name_key)
        stats["molecule_count"] = pd.to_numeric(stats.molecule_count, errors="coerce")
        stats["small_profile_flag"] = stats.small_profile_flag.astype("string").str.casefold().isin(["true", "1", "yes"])
        # A name is usable if at least one corresponding entity has a non-small profile.
        usable = stats[stats.key.ne("") & stats.molecule_count.gt(0) & ~stats.small_profile_flag]
        self.flavour_profile_names = set(usable.key.unique())

        self.flavour_name_coverage = int(
            self.master.name.map(_name_key).isin(self.flavour_profile_names).sum()
        )

    def ctx_sim_all(self, q):
        return np.asarray((self.Pn @ self.Pn[q].T).todense()).ravel()

    def flavour_sim(self, q, js):
        qname = _name_key(self.master.name.iloc[q])
        candidate_names = [_name_key(self.master.name.iloc[j]) for j in js]
        out = pd.DataFrame({"key_a": qname, "key_b": candidate_names})
        out = out.merge(self.flavour_scores, on=["key_a", "key_b"], how="left")
        scores = out["flavour_sim"].to_numpy(dtype=float)

        # No pair row means zero only if both names have usable FlavorDB profiles.
        # Otherwise keep NaN: no evidence is not the same as zero similarity.
        has_profiles = np.array([
            qname in self.flavour_profile_names and name in self.flavour_profile_names
            for name in candidate_names
        ])
        missing_pair = ~np.isfinite(scores) & has_profiles
        scores = scores.copy()
        scores[missing_pair] = 0.0
        out["flavour_sim"] = scores
        out["flavour_identical_profile"] = out["flavour_identical_profile"].eq(True)
        out["flavour_low_evidence"] = out["flavour_low_evidence"].eq(True)
        return out[["flavour_sim", "flavour_identical_profile", "flavour_low_evidence"]]

    def emb_sim(self, q, js):
        if self.E is None:
            return np.full(len(js), np.nan)
        return self.E[js] @ self.E[q]

    def compat(self, js, recipe_idx):
        recipe_idx = [i for i in recipe_idx if i is not None]
        if not recipe_idx:
            return np.full(len(js), np.nan), np.full(len(js), np.nan)
        sub = self.cooc[js][:, recipe_idx]
        cov = np.asarray((sub > 0).mean(1)).ravel()
        pp = np.asarray(self.ppmi[js][:, recipe_idx].mean(1)).ravel()
        return cov, pp

    def candidates(self, name, top_k=30, recipe_names=None, same_family=True, dedupe=True):
        q = self.name_idx[name]
        ctx = self.ctx_sim_all(q)
        ctx[q] = -1

        top_all = np.argsort(-ctx)[:50]
        unk = top_all[self.master.tier.to_numpy()[top_all] != "trusted"]
        n_unknown = int(len(unk))
        unknown_names = self.master.name.iloc[unk].head(10).tolist()

        pool = self.trusted[self.trusted != q]
        if same_family:
            fam = self.master.family.iloc[q]
            pool = pool[self.master.family.iloc[pool].to_numpy() == fam]
        qd = self.master.usda_description.iloc[q]
        if isinstance(qd, str):
            pool = pool[self.master.usda_description.iloc[pool].to_numpy() != qd]

        cols = ["name", "protein_100g", "carbs_100g", "fat_100g", "kcal_100g",
                "usda_description", "food_group"]
        df = self.master.iloc[pool][cols].copy()
        df["ctx_sim"] = ctx[pool]
        flav = self.flavour_sim(q, pool)
        df = pd.concat([df.reset_index(drop=True), flav.reset_index(drop=True)], axis=1)
        df["emb_sim"] = self.emb_sim(q, pool)

        # Similarity = mean rank-percentile of available judges.
        P = np.vstack([_pct(df.ctx_sim), _pct(df.flavour_sim), _pct(df.emb_sim)])
        df["emb_pct"] = P[2]
        df["sim_score"] = np.nanmean(P, axis=0)
        df["n_judges"] = (~np.isnan(P)).sum(0)

        rn = [self.name_idx[n] for n in (recipe_names or []) if n in self.name_idx and n != name]
        df["compat"], df["compat_ppmi"] = self.compat(pool, rn)
        df["in_recipe"] = df.name.isin(recipe_names or [])
        df = df.sort_values("sim_score", ascending=False)
        if dedupe:
            df = df.drop_duplicates("usda_description")
        df = df.head(top_k).reset_index(drop=True)
        return df, {
            "top50_all": 50,
            "top50_without_trusted_nutrition": n_unknown,
            "unknown_examples": unknown_names,
            "flavour_profile_names_in_master": self.flavour_name_coverage,
        }
