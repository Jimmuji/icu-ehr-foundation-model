"""Stage 15 v2 — retrieval precision@k vs baseline (APOLLO-style sanity check).

APOLLO's main embedding sanity check is retrieval: query a patient, pull the
nearest neighbours, and check they share the query's clinical cohort more often
than chance. Here, for each condition, I take test-set positives as queries,
retrieve their k nearest TRAIN neighbours (cosine), and measure the fraction of
neighbours that share the condition (precision@k), against the train baseline
prevalence. lift = precision / baseline; >1 means the embedding genuinely
retrieves clinically similar patients.

Run:
    python src_v2/15_retrieval_precision.py --emb outputs/downstream_v2/small_ce_embeddings.npz --tag ce
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
TASKS = ["mortality", "los_gt_7d", "ami", "stroke", "masld"]


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

    from sklearn.preprocessing import normalize
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
    Xn = normalize(X, axis=1)
    tr, te = meta["split"].values == "train", meta["split"].values == "test"
    db, dbm = Xn[tr], meta[tr].reset_index(drop=True)
    q, qm = Xn[te], meta[te].reset_index(drop=True)

    rows = []
    for t in TASKS:
        dby = dbm[f"y_{t}"].values
        baseline = dby.mean()
        qpos = np.where(qm[f"y_{t}"].values == 1)[0]
        if len(qpos) < 20:
            continue
        # retrieve k nearest train neighbours for each positive query
        sims = q[qpos] @ db.T
        topk = np.argpartition(-sims, args.k, axis=1)[:, :args.k]
        prec = dby[topk].mean()  # fraction of retrieved neighbours that are positive
        rows.append({"task": t, "n_queries": len(qpos), "precision@10": round(float(prec), 3),
                     "baseline": round(float(baseline), 3), "lift": round(float(prec / baseline), 2)})
        print(f"  {t:12s} precision@{args.k}={prec:.3f}  baseline={baseline:.3f}  lift={prec/baseline:.2f}x  (n_q={len(qpos)})")

    df = pd.DataFrame(rows)
    df.to_csv(WORK / "downstream_v2" / f"retrieval_precision_{args.tag}.csv", index=False)
    print(f"[{args.tag}] wrote retrieval_precision_{args.tag}.csv")


if __name__ == "__main__":
    main()
