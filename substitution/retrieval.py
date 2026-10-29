"""STEP 2 (library) - Stage A: candidate retrieval and compatibility.

Evidence used, each kept separate so it can be judged on its own:
  * ctx_sim   : CONTEXT similarity from the co-occurrence graph. Two ingredients are similar if they
                appear with the same partners (second-order, PPMI-weighted cosine). Direct co-occurrence
                finds partners (beef-bay leaf), not substitutes, so it is NOT used for similarity.
  * flavor_sim: Jaccard over FlavorDB molecules (only ~4.7% of ingredients have molecules; NaN otherwise,
                and NaN when the two profiles are identical, which the molecule audit showed is an artifact).
  * emb_sim   : GastroSAGE cosine (optional). Embeddings were judged unreliable alone, so only its RANK
                inside the candidate pool is used, never the raw cosine.
  * compat    : how many of the other recipe ingredients the candidate directly co-occurs with.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import config as C


def _pct(x):
    """rank percentile in [0,1]; NaN stays NaN."""
    s = pd.Series(x, dtype=float)
    return s.rank(pct=True, method="average").to_numpy()


class Retriever:
    def __init__(self, min_count=2):
        self.master = pd.read_csv(C.OUT / "ingredient_master.csv")
        nodes = pd.read_csv(C.GRAPH_DIR / "nodes_our_recipeflavor.csv")
        edges = pd.read_csv(C.GRAPH_DIR / "edges_our_recipeflavor.csv")

        self.ing_ids = self.master.node_id.tolist()
        self.idx = {n: i for i, n in enumerate(self.ing_ids)}
        self.name_idx = {n: i for i, n in enumerate(self.master.name)}
        N = len(self.ing_ids)

        # ---- co-occurrence matrix (symmetric counts) ----
        cc = edges[edges.edge_type == "ingr-ingr"]
        r = cc.id_1.map(self.idx).to_numpy(); c = cc.id_2.map(self.idx).to_numpy()
        w = cc.score.to_numpy(dtype=float)
        A = sp.coo_matrix((w, (r, c)), shape=(N, N)).tocsr()
        self.cooc = (A + A.T).tocsr()                       # raw counts, for direct compat

        # ---- PPMI with a minimum-count filter (count-1 pairs are noise on rare names) ----
        K = self.cooc.copy(); K.data[K.data < min_count] = 0; K.eliminate_zeros()
        tot = K.sum(); rs = np.asarray(K.sum(1)).ravel(); cs = rs ** 0.75; cs = cs / cs.sum() * tot   # context smoothing
        K = K.tocoo()
        pmi = np.log((K.data * tot) / (rs[K.row] * cs[K.col] + 1e-12) + 1e-12)
        keep = pmi > 0
        P = sp.coo_matrix((pmi[keep], (K.row[keep], K.col[keep])), shape=(N, N)).tocsr()
        self.ppmi = P
        norms = np.sqrt(np.asarray(P.multiply(P).sum(1)).ravel()) + 1e-12
        self.Pn = sp.diags(1 / norms) @ P                    # row-normalised for cosine

        # ---- molecules ----
        mol = edges[edges.edge_type == "ingr-dcomp"]
        self.mols = {}
        for a, b in zip(mol.id_1, mol.id_2):
            if a in self.idx:
                self.mols.setdefault(self.idx[a], set()).add(b)

        # ---- optional embeddings (aligned through node2idx.pkl, never by assumed order) ----
        self.E = None
        if C.EMBEDDINGS_FILE is not None:
            import pickle
            E = np.load(C.EMBEDDINGS_FILE)
            n2i = None
            if getattr(C, "NODE2IDX_FILE", None) is not None:
                with open(C.NODE2IDX_FILE, "rb") as fh:
                    n2i = pickle.load(fh)
            if n2i is not None:
                # the pickle may be a wrapper such as {"node2idx": {...}, "nodes": [...]}
                wrapper = n2i if isinstance(n2i, dict) and "node2idx" in n2i else None
                if wrapper is not None:
                    n2i = wrapper["node2idx"]
                keys = list(n2i.keys())[:3] if hasattr(n2i, "keys") else None
                if all(n in n2i for n in self.ing_ids):                  # keyed by node id
                    rows = [n2i[n] for n in self.ing_ids]
                elif all(n in n2i for n in self.master.name):            # keyed by ingredient name
                    rows = [n2i[n] for n in self.master.name]
                elif wrapper is not None and "nodes" in wrapper and all(n in set(wrapper["nodes"]) for n in self.ing_ids):
                    pos = {n: i for i, n in enumerate(wrapper["nodes"])}  # row order = order of the saved nodes list
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

        self.trusted = self.master.index[self.master.tier == "trusted"].to_numpy()

    # ------------------------------------------------------------------
    def ctx_sim_all(self, q):
        return np.asarray((self.Pn @ self.Pn[q].T).todense()).ravel()

    def flavor_sim(self, q, js):
        a = self.mols.get(q)
        out = np.full(len(js), np.nan)
        if not a:
            return out
        for k, j in enumerate(js):
            b = self.mols.get(j)
            if b:
                if a == b:
                    continue                                # identical profile -> artifact, leave NaN
                out[k] = len(a & b) / len(a | b)
        return out

    def emb_sim(self, q, js):
        if self.E is None:
            return np.full(len(js), np.nan)
        return self.E[js] @ self.E[q]

    def compat(self, js, recipe_idx):
        """fraction of the other recipe ingredients the candidate directly co-occurs with, and mean PPMI."""
        recipe_idx = [i for i in recipe_idx if i is not None]
        if not recipe_idx:
            return np.full(len(js), np.nan), np.full(len(js), np.nan)
        sub = self.cooc[js][:, recipe_idx]
        cov = np.asarray((sub > 0).mean(1)).ravel()
        pp = np.asarray(self.ppmi[js][:, recipe_idx].mean(1)).ravel()
        return cov, pp

    # ------------------------------------------------------------------
    def candidates(self, name, top_k=30, recipe_names=None, same_family=True, dedupe=True):
        """Top-k TRUSTED candidates for `name`, plus a report of how much was lost to missing nutrition."""
        q = self.name_idx[name]
        ctx = self.ctx_sim_all(q); ctx[q] = -1

        # what plain retrieval over ALL ingredients would have returned (for the "unknown nutrition" report)
        top_all = np.argsort(-ctx)[:50]
        unk = top_all[self.master.tier.to_numpy()[top_all] != "trusted"]
        n_unknown = int(len(unk))
        unknown_names = self.master.name.iloc[unk].head(10).tolist()

        pool = self.trusted[self.trusted != q]
        if same_family:
            fam = self.master.family.iloc[q]
            pool = pool[self.master.family.iloc[pool].to_numpy() == fam]
        # drop things that are the same USDA food as the query (garlic vs garlic clove)
        qd = self.master.usda_description.iloc[q]
        if isinstance(qd, str):
            pool = pool[self.master.usda_description.iloc[pool].to_numpy() != qd]

        df = self.master.iloc[pool][["name", "protein_100g", "carbs_100g", "fat_100g", "kcal_100g", "usda_description", "food_group"]].copy()
        df["ctx_sim"] = ctx[pool]
        df["flavor_sim"] = self.flavor_sim(q, pool)
        df["emb_sim"] = self.emb_sim(q, pool)

        # similarity = mean rank-percentile of whichever judges are available (ctx always is)
        P = np.vstack([_pct(df.ctx_sim), _pct(df.flavor_sim), _pct(df.emb_sim)])
        df["emb_pct"] = P[2]                                  # embedding rank-percentile inside this pool (NaN if no embeddings)
        df["sim_score"] = np.nanmean(P, axis=0)
        df["n_judges"] = (~np.isnan(P)).sum(0)

        rn = [self.name_idx[n] for n in (recipe_names or []) if n in self.name_idx and n != name]
        df["compat"], df["compat_ppmi"] = self.compat(pool, rn)
        df["in_recipe"] = df.name.isin(recipe_names or [])
        # synonyms that resolve to the same USDA food are one candidate (keep the best-scoring name)
        df = df.sort_values("sim_score", ascending=False)
        if dedupe:
            df = df.drop_duplicates("usda_description")
        df = df.head(top_k).reset_index(drop=True)
        return df, {"top50_all": 50, "top50_without_trusted_nutrition": n_unknown, "unknown_examples": unknown_names}