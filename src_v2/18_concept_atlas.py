"""Stage 18 v2 — APOLLO-style figures:
   (A) concept atlas  : UMAP of the per-variable embeddings, coloured by category (cf APOLLO Fig 2)
   (B) patient atlas by age : UMAP of patient embeddings, coloured by age gradient (cf APOLLO Fig 3a)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
sys.path.insert(0, str(Path(__file__).parent))
from small_ehr_model import SmallEHRTransformer


def concept_atlas():
    import umap
    ck = torch.load("outputs/v2_small_ce/best.pt", map_location="cpu")
    sd = ck["model"]
    cat_emb = sd["cat_feat_emb.weight"].numpy()      # (n_cat, D)
    flt_emb = sd["float_feat_emb.weight"].numpy()    # (n_float, D)
    feat = json.load(open(WORK / "feat_info.json"))
    cat_names = list(feat.get("category_cols", []))
    flt_names = sorted(feat["float_cols"].keys())
    names = cat_names + flt_names
    E = np.concatenate([cat_emb, flt_emb], 0)
    cats = [n.split("::")[0] if "::" in n else n.split("_")[0] for n in names]
    print(f"{len(names)} variables, {len(set(cats))} categories: {sorted(set(cats))}")

    U = umap.UMAP(n_neighbors=12, min_dist=0.25, metric="cosine", random_state=0).fit_transform(E)
    uniq = sorted(set(cats))
    cmap = plt.get_cmap("tab20")
    cidx = {c: i for i, c in enumerate(uniq)}
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in uniq:
        m = np.array([x == c for x in cats])
        ax.scatter(U[m, 0], U[m, 1], s=45, color=cmap(cidx[c] % 20), label=f"{c} ({m.sum()})", alpha=0.85, edgecolors="white", linewidths=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Concept atlas: UMAP of the model's per-variable embeddings, coloured by category")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(WORK / "final_results" / "concept_atlas.png", dpi=130, bbox_inches="tight")
    print("concept atlas done")


def patient_atlas_age():
    import umap
    from sklearn.preprocessing import StandardScaler
    e = np.load(WORK / "downstream_v2" / "small_ce_embeddings.npz")
    X, sids = e["embeddings"].astype(np.float32), e["stay_ids"].astype(np.int64)
    meta = pd.DataFrame({"stay_id_v1": sids, "row": np.arange(len(sids))})
    v1c = pd.read_parquet(WORK/"cohort"/"cohort.parquet")[["stay_id","hadm_id"]].rename(columns={"stay_id":"stay_id_v1","hadm_id":"hospitalization_id"})
    v2c = pd.read_parquet(WORK/"cohort_v2"/"cohort.parquet")[["hospitalization_id","stay_id","age_at_admission"]]
    meta = meta.merge(v1c,on="stay_id_v1").merge(v2c,on="hospitalization_id")
    X = X[meta["row"].values]
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(0); idx = rng.choice(len(meta), min(15000, len(meta)), replace=False)
    U = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(Xs[idx])
    age = meta["age_at_admission"].values[idx]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(U[:,0], U[:,1], s=4, c=age, cmap="viridis", alpha=0.7, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Patient atlas: UMAP of patient embeddings, coloured by age")
    fig.colorbar(sc, ax=ax, label="age", shrink=0.7)
    fig.tight_layout(); fig.savefig(WORK / "final_results" / "patient_atlas_age.png", dpi=130, bbox_inches="tight")
    print("patient age atlas done")


if __name__ == "__main__":
    concept_atlas()
    patient_atlas_age()
