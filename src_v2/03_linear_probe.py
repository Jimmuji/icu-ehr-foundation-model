"""Stage 03 v2 — Linear probe on (existing v1) embeddings against (new v2) labels.

For each downstream task, train a logistic regression (binary) or linear
regression (continuous) on top of FROZEN encoder embeddings. This is the
"linear probe" protocol — the standard way to measure how much downstream
signal lives in a foundation model's representation.

Re-uses v1's already-extracted 768-d embeddings (encoder mean-pool over
valid timesteps), and the v2 cohort + labels + temporal split.

Memory-safe: embeddings stay in a pure numpy matrix (X), metadata in a
small DataFrame keyed by row index. Never put embeddings in a pandas
object column — that bloats memory >100×.

Run on server:
    python src_v2/03_linear_probe.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_config, setup_logging, stage_dir


BINARY_TASKS = ["mortality", "los_gt_7d", "readmit_30d", "celiac", "masld", "ami", "stroke"]
REGRESSION_TASKS = ["reg_platelets", "reg_creatinine", "reg_spo2"]


def load_v1_embeddings_compact(emb_dir: Path):
    """Combine v1 embeddings into ONE numpy matrix X + a small metadata DataFrame.
    Returns (X, meta) where X.shape = (N, D) and meta has stay_id_v1 + v1_split.
    """
    X_parts = []
    meta_parts = []
    for split in ["train", "val", "test"]:
        d = np.load(emb_dir / f"{split}_embeddings.npz")
        emb = d["embeddings"].astype(np.float32)  # (n, 768)
        sid = d["stay_ids"].astype(np.int64)
        X_parts.append(emb)
        meta_parts.append(pd.DataFrame({"stay_id_v1": sid, "v1_split": split}))
    X = np.concatenate(X_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True)
    return X, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("03_linear_probe", cfg["paths"]["work_dir"])
    out_dir = stage_dir(cfg, "downstream_v2")
    work = Path(cfg["paths"]["work_dir"])

    # ---- Load v1 embeddings into numpy matrix ----
    emb_dir = work / "embeddings"
    log.info("Loading v1 embeddings (compact)…")
    X_full, meta = load_v1_embeddings_compact(emb_dir)
    log.info(f"  X shape: {X_full.shape}, dtype: {X_full.dtype}, "
             f"memory: {X_full.nbytes/1e9:.2f} GB; metadata: {len(meta):,}")

    # ---- Map v1 stay_id → hospitalization_id via v1 cohort ----
    v1_cohort = pd.read_parquet(work / "cohort" / "cohort.parquet")[["stay_id", "hadm_id"]]
    v1_cohort = v1_cohort.rename(columns={"stay_id": "stay_id_v1", "hadm_id": "hospitalization_id"})
    meta = meta.merge(v1_cohort, on="stay_id_v1", how="inner")
    log.info(f"  joined v1 cohort: {len(meta):,} rows have hospitalization_id")

    # ---- Map → v2 stay_id + labels + split ----
    v2_cohort = pd.read_parquet(work / "cohort_v2" / "cohort.parquet")[["hospitalization_id", "stay_id", "split"]]
    labels = pd.read_parquet(out_dir / "labels.parquet").drop(columns=["split"], errors="ignore")
    v2 = v2_cohort.merge(labels, on="stay_id", how="left")
    meta = meta.merge(v2, on="hospitalization_id", how="inner")
    log.info(f"  joined v2 labels: {len(meta):,} rows")
    log.info(f"  per-split: {meta['split'].value_counts().to_dict()}")

    # ---- Align X_full to the (filtered, possibly reordered) meta ----
    # We do this by indexing X_full with the kept row indices in original order.
    # First, rebuild meta with original-order index preserved.
    X_full, meta_full = load_v1_embeddings_compact(emb_dir)
    meta_full["row_idx"] = np.arange(len(meta_full), dtype=np.int64)
    meta_full = meta_full.merge(v1_cohort, on="stay_id_v1", how="inner")
    meta_full = meta_full.merge(v2, on="hospitalization_id", how="inner")
    X = X_full[meta_full["row_idx"].values]
    log.info(f"  Final aligned X: {X.shape}, meta: {len(meta_full):,}")

    splits = meta_full["split"].values

    # ---- Sklearn imports ----
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    log.info(f"  train/val/test sizes: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    scaler = StandardScaler().fit(X[train_mask])
    Xs = scaler.transform(X)

    results = []
    log.info(f"\n=== Binary tasks (linear probe + class-weight balanced LR) ===")
    for task in BINARY_TASKS:
        y_col, m_col = f"y_{task}", f"m_{task}"
        y = meta_full[y_col].values
        m = meta_full[m_col].values == 1
        tr_m = train_mask & m
        va_m = val_mask & m
        te_m = test_mask & m
        if tr_m.sum() == 0 or y[tr_m].sum() == 0 or y[tr_m].sum() == tr_m.sum():
            log.info(f"  [SKIP {task}] insufficient positives in train (pos={int(y[tr_m].sum())}/{tr_m.sum()})")
            results.append((task, "binary", tr_m.sum(), te_m.sum(), int(y[te_m].sum()), np.nan, np.nan, np.nan, np.nan))
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", n_jobs=-1)
        clf.fit(Xs[tr_m], y[tr_m])
        p_val = clf.predict_proba(Xs[va_m])[:, 1] if va_m.sum() > 0 else None
        p_test = clf.predict_proba(Xs[te_m])[:, 1] if te_m.sum() > 0 else None
        auroc_v = roc_auc_score(y[va_m], p_val) if (p_val is not None and len(np.unique(y[va_m])) == 2) else np.nan
        auprc_v = average_precision_score(y[va_m], p_val) if (p_val is not None and len(np.unique(y[va_m])) == 2) else np.nan
        auroc_t = roc_auc_score(y[te_m], p_test) if (p_test is not None and len(np.unique(y[te_m])) == 2) else np.nan
        auprc_t = average_precision_score(y[te_m], p_test) if (p_test is not None and len(np.unique(y[te_m])) == 2) else np.nan
        log.info(f"  {task:14s} | train n={tr_m.sum():>6,} pos={int(y[tr_m].sum()):>5,} | "
                 f"val AUROC={auroc_v:.3f} AUPRC={auprc_v:.3f} | "
                 f"test AUROC={auroc_t:.3f} AUPRC={auprc_t:.3f}")
        results.append((task, "binary", tr_m.sum(), te_m.sum(), int(y[te_m].sum()), auroc_v, auprc_v, auroc_t, auprc_t))

    log.info(f"\n=== Regression tasks (Ridge α=1.0) ===")
    for task in REGRESSION_TASKS:
        y_col, m_col = f"y_{task}", f"m_{task}"
        y = meta_full[y_col].values.astype(np.float32)
        m = meta_full[m_col].values == 1
        tr_m = train_mask & m
        va_m = val_mask & m
        te_m = test_mask & m
        if tr_m.sum() < 100:
            log.info(f"  [SKIP {task}] too few samples")
            results.append((task, "regression", tr_m.sum(), te_m.sum(), 0, np.nan, np.nan, np.nan, np.nan))
            continue
        lo, hi = np.percentile(y[tr_m], [0.5, 99.5])
        y_c = np.clip(y, lo, hi)
        reg = Ridge(alpha=1.0).fit(Xs[tr_m], y_c[tr_m])
        p_val = reg.predict(Xs[va_m])
        p_test = reg.predict(Xs[te_m])
        r2_v = r2_score(y_c[va_m], p_val)
        mae_v = mean_absolute_error(y_c[va_m], p_val)
        r2_t = r2_score(y_c[te_m], p_test)
        mae_t = mean_absolute_error(y_c[te_m], p_test)
        log.info(f"  {task:14s} | train n={tr_m.sum():>6,} | "
                 f"val R²={r2_v:.3f} MAE={mae_v:.2f} | "
                 f"test R²={r2_t:.3f} MAE={mae_t:.2f}")
        results.append((task, "regression", tr_m.sum(), te_m.sum(), 0, r2_v, mae_v, r2_t, mae_t))

    res = pd.DataFrame(results, columns=[
        "task", "type", "n_train", "n_test", "n_test_pos",
        "val_metric", "val_metric2", "test_metric", "test_metric2",
    ])
    res.to_csv(out_dir / "linear_probe_results.csv", index=False)
    log.info(f"\nWrote {out_dir / 'linear_probe_results.csv'}")
    log.info(f"\n=== Final results ===\n{res.to_string(index=False)}")


if __name__ == "__main__":
    main()
