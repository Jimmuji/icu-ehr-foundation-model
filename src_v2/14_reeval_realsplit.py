"""Stage 14 v2 — re-evaluate EXISTING embeddings under the REAL temporal split.

CPU-only (no re-extraction, no GPU). Loads a saved embedding set, aligns to the
v2 cohort whose `split` column is now the real anchor_year_group split, and
re-runs: PCA effective dim, linear probe (binary), k-NN AUROC.

NOTE (honest caveat): the small-model embeddings here come from a model
pretrained under the OLD shifted split, so the encoder saw some real-test-era
patients during self-supervised pretraining. These numbers are a PREVIEW;
the clean version requires retraining on the real-train split.

Run:
    python src_v2/14_reeval_realsplit.py --emb outputs/downstream_v2/small_ce_embeddings.npz --tag ce
    python src_v2/14_reeval_realsplit.py --emb v1 --tag v1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
BINARY = ["mortality", "los_gt_7d", "readmit_30d", "celiac", "masld", "ami", "stroke"]
KNN_TASKS = ["mortality", "los_gt_7d", "ami", "stroke", "masld"]


def load_emb(spec):
    if spec == "v1":
        Xs, sids = [], []
        for s in ["train", "val", "test"]:
            d = np.load(WORK / "embeddings" / f"{s}_embeddings.npz")
            Xs.append(d["embeddings"].astype(np.float32)); sids.append(d["stay_ids"].astype(np.int64))
        return np.concatenate(Xs), np.concatenate(sids)
    d = np.load(spec)
    return d["embeddings"].astype(np.float32), d["stay_ids"].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    X, sids = load_emb(args.emb)
    meta = pd.DataFrame({"stay_id_v1": sids, "row_idx": np.arange(len(sids))})
    v1c = pd.read_parquet(WORK / "cohort" / "cohort.parquet")[["stay_id", "hadm_id"]].rename(
        columns={"stay_id": "stay_id_v1", "hadm_id": "hospitalization_id"})
    v2c = pd.read_parquet(WORK / "cohort_v2" / "cohort.parquet")[["hospitalization_id", "stay_id", "split"]]
    labels = pd.read_parquet(WORK / "downstream_v2" / "labels.parquet").drop(columns=["split"], errors="ignore")
    meta = (meta.merge(v1c, on="stay_id_v1", how="inner")
                .merge(v2c, on="hospitalization_id", how="inner")
                .merge(labels, on="stay_id", how="inner"))
    X = X[meta["row_idx"].values]
    sp = meta["split"].values
    print(f"[{args.tag}] aligned {len(meta):,} | split {pd.Series(sp).value_counts().to_dict()}")

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, normalize
    from sklearn.metrics import roc_auc_score, average_precision_score

    pca = PCA(n_components=min(100, X.shape[1])).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    effdim = {int(t*100): int(np.searchsorted(cum, t))+1 for t in [0.5, 0.9, 0.99]}
    print(f"[{args.tag}] PCA effdim 50/90/99% = {effdim[50]}/{effdim[90]}/{effdim[99]}")

    tr, te = sp == "train", sp == "test"
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    rows = []
    print(f"[{args.tag}] linear probe (REAL temporal test):")
    for t in BINARY:
        y = meta[f"y_{t}"].values; m = meta[f"m_{t}"].values == 1
        trm, tem = tr & m, te & m
        if trm.sum() == 0 or y[trm].sum() in (0, trm.sum()) or len(np.unique(y[tem])) < 2:
            print(f"   {t:12s} skip"); continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1).fit(Xs[trm], y[trm])
        p = clf.predict_proba(Xs[tem])[:, 1]
        au, ap_ = roc_auc_score(y[tem], p), average_precision_score(y[tem], p)
        print(f"   {t:12s} AUROC={au:.3f} AUPRC={ap_:.3f}")
        rows.append((args.tag, t, "linear_probe", round(au, 4), round(ap_, 4)))

    Xn = normalize(X, axis=1)
    db, q = Xn[tr], Xn[te]
    dbm, qm = meta[tr].reset_index(drop=True), meta[te].reset_index(drop=True)
    rng = np.random.default_rng(0); nq = min(2000, q.shape[0])
    sel = rng.choice(q.shape[0], nq, replace=False)
    topk = np.argpartition(-(q[sel] @ db.T), args.k, axis=1)[:, :args.k]
    print(f"[{args.tag}] k-NN AUROC:")
    for t in KNN_TASKS:
        dy = dbm[f"y_{t}"].values; qy = qm[f"y_{t}"].values[sel]
        if len(np.unique(qy)) < 2: continue
        sc_ = dy[topk].mean(axis=1); au = roc_auc_score(qy, sc_)
        print(f"   {t:12s} knn-AUROC={au:.3f}")
        rows.append((args.tag, t, "knn_auroc", round(au, 4), np.nan))

    out = WORK / "downstream_v2" / "realsplit_scorecard.csv"
    df = pd.DataFrame(rows, columns=["tag", "task", "metric_type", "metric", "metric2"])
    if out.exists():
        prev = pd.read_csv(out); prev = prev[prev["tag"] != args.tag]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"[{args.tag}] wrote {out}")


if __name__ == "__main__":
    main()
