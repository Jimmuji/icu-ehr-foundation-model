"""Stage 04 v2 — Apollo-style embedding sanity check.

In addition to v1's UMAP + KMeans clustering, this runs:

(1) k-NN retrieval label concordance:
    For random query patients, look up top-k nearest neighbors (cosine)
    and check what fraction share the query's label for each binary task.
    → A good embedding should have neighbors that share clinically meaningful
       traits at higher-than-baseline rate.

(2) Per-dimension variance + effective dimensionality (PCA explained variance):
    → Detect representation collapse (low effective dim = bad).

(3) Cluster purity by label:
    Use the existing UMAP+KMeans labels; for each task, measure the
    label distribution within each cluster (entropy / max-class fraction).

(4) Top-k retrieval AUC for binary tasks:
    For each query stay, score positive class by mean label of top-k
    neighbors. AUROC of this score against true label.

Outputs CSV summary + a few plots.

Run on server:
    python src_v2/04_apollo_sanity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_config, setup_logging, stage_dir


BINARY_TASKS = ["mortality", "los_gt_7d", "ami", "stroke", "masld"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-queries", type=int, default=2000, help="Sample of patients to compute k-NN concordance for")
    ap.add_argument("--k", type=int, default=10, help="k for k-NN retrieval")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("04_apollo_sanity", cfg["paths"]["work_dir"])
    out_dir = stage_dir(cfg, "downstream_v2")
    work = Path(cfg["paths"]["work_dir"])

    # ---- Load embeddings + labels (re-using the alignment from 03) ----
    emb_dir = work / "embeddings"
    parts_X, parts_meta = [], []
    for split in ["train", "val", "test"]:
        d = np.load(emb_dir / f"{split}_embeddings.npz")
        parts_X.append(d["embeddings"].astype(np.float32))
        parts_meta.append(pd.DataFrame({"stay_id_v1": d["stay_ids"].astype(np.int64), "v1_split": split}))
    X = np.concatenate(parts_X, axis=0)
    meta = pd.concat(parts_meta, ignore_index=True)
    meta["row_idx"] = np.arange(len(meta), dtype=np.int64)
    log.info(f"Loaded {X.shape[0]:,} embeddings × {X.shape[1]}-d")

    v1_cohort = pd.read_parquet(work / "cohort" / "cohort.parquet")[["stay_id", "hadm_id"]].rename(
        columns={"stay_id": "stay_id_v1", "hadm_id": "hospitalization_id"})
    v2_cohort = pd.read_parquet(work / "cohort_v2" / "cohort.parquet")[["hospitalization_id", "stay_id", "split"]]
    labels = pd.read_parquet(out_dir / "labels.parquet").drop(columns=["split"], errors="ignore")
    v2 = v2_cohort.merge(labels, on="stay_id", how="left")
    meta = meta.merge(v1_cohort, on="stay_id_v1", how="inner").merge(v2, on="hospitalization_id", how="inner")
    X = X[meta["row_idx"].values]
    log.info(f"Aligned: {X.shape[0]:,} rows have v2 labels")

    # ============== (1) Effective dimensionality via PCA ==============
    log.info("\n=== (1) Effective dimensionality (PCA explained variance) ===")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(100, X.shape[1])).fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
        k = int(np.searchsorted(cumvar, thresh)) + 1
        log.info(f"  components for {thresh*100:.0f}% variance: {k}")
    # Per-dim std (already noted in v1 report)
    per_dim_std = X.std(axis=0)
    log.info(f"  per-dim std: min={per_dim_std.min():.4f}, "
             f"median={np.median(per_dim_std):.4f}, max={per_dim_std.max():.4f}")

    # ============== (2) k-NN retrieval (cosine) — label concordance ==============
    log.info(f"\n=== (2) k-NN retrieval (k={args.k}, cosine) — label concordance ===")
    # Use train embeddings as the "database", queries from val+test (out-of-time)
    from sklearn.preprocessing import normalize
    Xn = normalize(X, axis=1)
    train_mask = meta["split"].values == "train"
    test_mask = (meta["split"].values == "test")
    log.info(f"  database (train): {train_mask.sum():,}, queries (test): {test_mask.sum():,}")

    db = Xn[train_mask]
    qmask = test_mask  # use all test as queries (manageable size)
    queries = Xn[qmask]
    db_meta = meta[train_mask].reset_index(drop=True)
    q_meta = meta[qmask].reset_index(drop=True)

    # For each query, find top-k nearest neighbors via cosine similarity
    # Use chunked computation to avoid huge matrix
    log.info("  computing nearest neighbors…")
    n_queries = min(args.n_queries, len(queries))
    sel = np.random.default_rng(0).choice(len(queries), n_queries, replace=False)
    q_sub = queries[sel]
    sims = q_sub @ db.T  # (n_queries, n_db)
    topk_idx = np.argpartition(-sims, args.k, axis=1)[:, :args.k]
    # Sort each row's top-k
    rows = np.arange(n_queries)[:, None]
    sorted_idx = np.take_along_axis(topk_idx, np.argsort(-sims[rows, topk_idx], axis=1), axis=1)
    log.info(f"  retrieved top-{args.k} for {n_queries:,} queries")

    rows_out = []
    for task in BINARY_TASKS:
        y_col, m_col = f"y_{task}", f"m_{task}"
        q_y = q_meta[y_col].values[sel]
        q_m = q_meta[m_col].values[sel] == 1
        db_y = db_meta[y_col].values
        db_m = db_meta[m_col].values == 1
        # For each query (with mask=1), get mean label of top-k neighbors (with neighbor mask=1)
        scores = []
        valid_queries = []
        for i in range(n_queries):
            if not q_m[i]:
                continue
            nbr = sorted_idx[i]
            nbr_y = db_y[nbr]
            nbr_m = db_m[nbr]
            if nbr_m.sum() == 0:
                continue
            scores.append(nbr_y[nbr_m].mean())
            valid_queries.append(i)
        if len(valid_queries) < 50:
            log.info(f"  {task:14s} | too few valid queries"); continue
        scores = np.array(scores)
        y_q = q_y[valid_queries]
        # AUROC of knn-score against true label
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_q)) == 2:
            auroc = roc_auc_score(y_q, scores)
        else:
            auroc = np.nan
        # Label concordance: prob that neighbors share the query label
        # For positives: avg(scores | y=1), for negatives: avg(1-scores | y=0)
        pos_concord = scores[y_q == 1].mean() if (y_q == 1).any() else np.nan
        neg_concord = (1 - scores[y_q == 0]).mean() if (y_q == 0).any() else np.nan
        baseline = db_y[db_m].mean()
        log.info(f"  {task:14s} | knn-AUROC={auroc:.3f} | "
                 f"pos-concord={pos_concord:.3f} (baseline {baseline:.3f}) | "
                 f"neg-concord={neg_concord:.3f}")
        rows_out.append((task, auroc, pos_concord, neg_concord, baseline))

    knn_df = pd.DataFrame(rows_out, columns=["task", "knn_auroc", "pos_concordance", "neg_concordance", "baseline_rate"])
    knn_df.to_csv(out_dir / "knn_retrieval.csv", index=False)
    log.info(f"  wrote {out_dir / 'knn_retrieval.csv'}")

    # ============== (3) Cluster purity ==============
    log.info(f"\n=== (3) Cluster purity (re-use existing v1 KMeans k=5 labels) ===")
    cl_path = work / "clusters" / "cluster_labels.npz"
    if cl_path.exists():
        cl = np.load(cl_path, allow_pickle=True)
        # cluster_labels.npz has 'stay_ids' + per-k arrays; figure out keys
        keys = list(cl.files)
        log.info(f"  cluster file keys: {keys}")
        if "kmeans_k5" in keys and "stay_ids" in keys:
            cl_sid = cl["stay_ids"].astype(np.int64)
            k5 = cl["kmeans_k5"]
            cluster_df = pd.DataFrame({"stay_id_v1": cl_sid, "cluster": k5})
            merged = meta.merge(cluster_df, on="stay_id_v1", how="left")
            purity_rows = []
            for task in BINARY_TASKS:
                y = merged[f"y_{task}"].values
                m = merged[f"m_{task}"].values == 1
                c = merged["cluster"].values
                if not (m & (~np.isnan(c))).any():
                    continue
                sub = merged[m & merged["cluster"].notna()]
                stats = sub.groupby("cluster").agg(
                    n=(f"y_{task}", "size"),
                    pos_rate=(f"y_{task}", "mean"),
                )
                purity_rows.append((task, stats["pos_rate"].max(), stats["pos_rate"].min(),
                                    stats["pos_rate"].std(), len(stats)))
                log.info(f"  {task:14s} | cluster pos_rate range: "
                         f"[{stats['pos_rate'].min():.3f}, {stats['pos_rate'].max():.3f}], "
                         f"std={stats['pos_rate'].std():.3f}")
            purity_df = pd.DataFrame(purity_rows, columns=["task", "max_cluster_pos_rate", "min_cluster_pos_rate",
                                                            "std_across_clusters", "n_clusters"])
            purity_df.to_csv(out_dir / "cluster_purity.csv", index=False)
            log.info(f"  wrote {out_dir / 'cluster_purity.csv'}")
    else:
        log.info("  (v1 cluster labels not found, skipping)")

    log.info("\n=== Apollo sanity checks done ===")


if __name__ == "__main__":
    main()
