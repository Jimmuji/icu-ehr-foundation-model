"""Stage 17 v2 — two summary figures:
   (A) train/val loss curves for the v2-CE model
   (B) APOLLO-style patient atlas: UMAP coloured by discovered phenotype
"""
from __future__ import annotations
import re, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
DIS = ["ami", "stroke", "masld"]


def fig_curves():
    log = sorted(glob.glob(str(WORK / "v2_small_ce" / "*.log")))[-1]
    txt = open(log).read()
    tr = [float(x) for x in re.findall(r"train: loss=([\d.]+)", txt)]
    va = [float(x) for x in re.findall(r"val  : loss=([\d.]+)", txt)]
    ep = list(range(1, len(tr) + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, tr, "-o", ms=4, label="train loss")
    ax.plot(ep, va, "-o", ms=4, label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked-reconstruction loss")
    ax.set_title("v2-CE pre-training: train vs val loss"); ax.legend()
    fig.tight_layout(); fig.savefig(WORK / "final_results" / "train_val_loss.png", dpi=130, bbox_inches="tight")
    print("curves done:", len(tr), "epochs")


def fig_atlas():
    import umap
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    e = np.load(WORK / "downstream_v2" / "small_ce_embeddings.npz")
    X, sids = e["embeddings"].astype(np.float32), e["stay_ids"].astype(np.int64)
    meta = pd.DataFrame({"stay_id_v1": sids, "row": np.arange(len(sids))})
    v1c = pd.read_parquet(WORK/"cohort"/"cohort.parquet")[["stay_id","hadm_id"]].rename(columns={"stay_id":"stay_id_v1","hadm_id":"hospitalization_id"})
    v2c = pd.read_parquet(WORK/"cohort_v2"/"cohort.parquet")[["hospitalization_id","stay_id"]]
    surv = pd.read_parquet(WORK/"downstream_v2"/"survival_labels.parquet")[["stay_id","surv_event"]]
    lab = pd.read_parquet(WORK/"downstream_v2"/"labels.parquet")[["stay_id"]+[f"y_{d}" for d in DIS]]
    meta = meta.merge(v1c,on="stay_id_v1").merge(v2c,on="hospitalization_id").merge(surv,on="stay_id").merge(lab,on="stay_id")
    X = X[meta["row"].values]
    Xs = StandardScaler().fit_transform(X)
    meta["pheno"] = KMeans(n_clusters=6, random_state=0, n_init=10).fit_predict(Xs)
    # descriptor per phenotype
    overall = {d: meta[f"y_{d}"].mean() for d in DIS}
    desc = {}
    for c in range(6):
        s = meta[meta["pheno"] == c]
        enr = {d: s[f"y_{d}"].mean()/overall[d] for d in DIS}
        top = max(enr, key=enr.get)
        tag = f", {top} {enr[top]:.1f}x" if enr[top] > 1.3 else ""
        desc[c] = f"P{c}: mort {s['surv_event'].mean()*100:.0f}%{tag}"
    # subsample + UMAP
    rng = np.random.default_rng(0)
    idx = rng.choice(len(meta), min(15000, len(meta)), replace=False)
    U = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(Xs[idx])
    ph = meta["pheno"].values[idx]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = plt.get_cmap("tab10")
    for c in range(6):
        m = ph == c
        ax.scatter(U[m,0], U[m,1], s=4, color=cmap(c), label=desc[c], alpha=0.6, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Patient atlas: UMAP of v2 embeddings, coloured by discovered phenotype")
    ax.legend(fontsize=8, markerscale=2.5, loc="best")
    fig.tight_layout(); fig.savefig(WORK / "final_results" / "patient_atlas.png", dpi=130, bbox_inches="tight")
    print("atlas done")


if __name__ == "__main__":
    fig_curves()
    fig_atlas()
