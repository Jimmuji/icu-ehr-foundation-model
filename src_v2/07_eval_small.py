"""Stage 07 v2 — extract small-model embeddings + run the verification suite.

The lesson from v1 was: do NOT trust pretraining loss. So as soon as a
SmallEHRTransformer checkpoint is trained, this script measures whether the
representation is actually useful, using the same diagnostics that exposed
v1's collapse:

  1. Extract a hidden-dim patient embedding per stay
     (encoder z, mean-pooled over valid timesteps, NO masking).
  2. Sanity / collapse check:
       - PCA effective dimensionality (components for 50/80/90/95/99% variance)
       - per-dimension std
  3. k-NN retrieval AUROC on the v2 temporal TEST set (cosine; train as DB).
  4. Linear probe (LogisticRegression, class-balanced) on the binary tasks.

Everything is keyed so results line up directly against v1's numbers.

Run (after training finishes):
    python src_v2/07_eval_small.py --ckpt outputs/v2_small_ce/best.pt --tag ce
    python src_v2/07_eval_small.py --ckpt outputs/v2_small_mse/best.pt --tag mse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from small_ehr_model import SmallEHRTransformer
from train_small import EHRCacheDataset

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
CACHE = WORK / "ehr_cache"
FEAT_INFO = WORK / "feat_info.json"
BINARY_TASKS = ["mortality", "los_gt_7d", "readmit_30d", "celiac", "masld", "ami", "stroke"]


@torch.no_grad()
def extract_embeddings(model, cache_dir, pids, device, batch_size=64):
    """Mean-pool encoder output over valid timesteps. No masking (clean input)."""
    ds = EHRCacheDataset(cache_dir, pids, float_mean_std=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    embs, out_pids = [], []
    for batch in loader:
        cat = batch["cat"].to(device)
        float_bin = batch["float_bin"].to(device)
        valid = batch["valid"].to(device)
        tindex = batch["time"].to(device)
        amp_device = "cuda" if device == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16):
            _, _, z = model(cat, float_bin, tindex, valid)   # z: (B, T, D)
        z = z.float()
        m = valid.unsqueeze(-1).float()                       # (B, T, 1)
        pooled = (z * m).sum(1) / m.sum(1).clamp(min=1.0)     # (B, D)
        embs.append(pooled.cpu().numpy())
        out_pids.extend([int(p) for p in batch["pid"]])
    return np.concatenate(embs, axis=0).astype(np.float32), np.array(out_pids, dtype=np.int64)


def load_labels_aligned(meta):
    """meta has columns: stay_id_v1 (= cache pid), row_idx. Join to v2 labels."""
    v1_cohort = pd.read_parquet(WORK / "cohort" / "cohort.parquet")[["stay_id", "hadm_id"]].rename(
        columns={"stay_id": "stay_id_v1", "hadm_id": "hospitalization_id"})
    v2_cohort = pd.read_parquet(WORK / "cohort_v2" / "cohort.parquet")[["hospitalization_id", "stay_id", "split"]]
    labels = pd.read_parquet(WORK / "downstream_v2" / "labels.parquet").drop(columns=["split"], errors="ignore")
    v2 = v2_cohort.merge(labels, on="stay_id", how="left")
    m = meta.merge(v1_cohort, on="stay_id_v1", how="inner").merge(v2, on="hospitalization_id", how="inner")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="label for outputs, e.g. 'ce' or 'mse'")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-queries", type=int, default=2000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = WORK / "downstream_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Rebuild model from checkpoint ----
    ckpt = torch.load(args.ckpt, map_location=device)
    cargs = ckpt.get("args", {})
    with open(FEAT_INFO) as f:
        feat_info = json.load(f)
    n_cat = len(feat_info.get("category_cols", []))
    n_float = len(feat_info.get("float_cols", {}))
    model = SmallEHRTransformer(
        n_cat_feats=n_cat, n_float_feats=n_float, n_cat_values=50, n_float_bins=256,
        hidden_dim=cargs.get("hidden_dim", 192), n_layers=cargs.get("n_layers", 6),
        n_heads=cargs.get("n_heads", 4), ff_dim=cargs.get("hidden_dim", 192) * 2, max_seq_len=512,
        float_mode=cargs.get("float_mode", "mse"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[{args.tag}] loaded {args.ckpt} | epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f} "
          f"| float_mode={model.float_mode} | device={device}")

    # ---- Extract embeddings for all splits ----
    metadata = pd.read_parquet(CACHE / "metadata.parquet")
    parts_X, parts_meta = [], []
    for split in ["train", "val", "test"]:
        pids = metadata.loc[metadata["split"] == split, "pid"].tolist()
        X, pid_arr = extract_embeddings(model, CACHE, pids, device)
        parts_X.append(X)
        parts_meta.append(pd.DataFrame({"stay_id_v1": pid_arr}))
        print(f"  {split}: {X.shape}")
    X = np.concatenate(parts_X, axis=0)
    meta = pd.concat(parts_meta, ignore_index=True)
    meta["row_idx"] = np.arange(len(meta), dtype=np.int64)
    np.savez_compressed(out_dir / f"small_{args.tag}_embeddings.npz",
                        embeddings=X, stay_ids=meta["stay_id_v1"].values)

    # ---- Align to v2 labels ----
    meta = load_labels_aligned(meta)
    X = X[meta["row_idx"].values]
    splits = meta["split"].values
    print(f"  aligned to v2 labels: {X.shape}")

    # ============== (1) Effective dimensionality ==============
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(100, X.shape[1])).fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    eff = {}
    print(f"\n[{args.tag}] === effective dimensionality (PCA) ===")
    for thr in [0.5, 0.8, 0.9, 0.95, 0.99]:
        k = int(np.searchsorted(cumvar, thr)) + 1
        eff[thr] = k
        print(f"  {int(thr*100)}% variance: {k} components")
    per_dim_std = X.std(axis=0)
    print(f"  per-dim std: min={per_dim_std.min():.4f} median={np.median(per_dim_std):.4f} max={per_dim_std.max():.4f}")

    # ============== (2) Linear probe ==============
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
    tr, va, te = splits == "train", splits == "val", splits == "test"
    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)
    rows = []
    print(f"\n[{args.tag}] === linear probe (test AUROC) ===")
    for task in BINARY_TASKS:
        y = meta[f"y_{task}"].values
        msk = meta[f"m_{task}"].values == 1
        trm, tem = tr & msk, te & msk
        if trm.sum() == 0 or y[trm].sum() in (0, trm.sum()):
            print(f"  {task:12s} skipped (insufficient positives)"); continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1).fit(Xs[trm], y[trm])
        p = clf.predict_proba(Xs[tem])[:, 1]
        auroc = roc_auc_score(y[tem], p) if len(np.unique(y[tem])) == 2 else np.nan
        auprc = average_precision_score(y[tem], p) if len(np.unique(y[tem])) == 2 else np.nan
        print(f"  {task:12s} AUROC={auroc:.3f} AUPRC={auprc:.3f} (test_pos={int(y[tem].sum())})")
        rows.append((args.tag, task, "linear_probe", auroc, auprc))

    # ============== (3) k-NN retrieval AUROC ==============
    from sklearn.preprocessing import normalize
    Xn = normalize(X, axis=1)
    db, q = Xn[tr], Xn[te]
    db_meta, q_meta = meta[tr].reset_index(drop=True), meta[te].reset_index(drop=True)
    rng = np.random.default_rng(0)
    nq = min(args.n_queries, len(q))
    sel = rng.choice(len(q), nq, replace=False)
    sims = q[sel] @ db.T
    topk = np.argpartition(-sims, args.k, axis=1)[:, :args.k]
    print(f"\n[{args.tag}] === k-NN retrieval AUROC (k={args.k}) ===")
    for task in ["mortality", "los_gt_7d", "ami", "stroke", "masld"]:
        db_y = db_meta[f"y_{task}"].values
        q_y = q_meta[f"y_{task}"].values[sel]
        scores = db_y[topk].mean(axis=1)
        if len(np.unique(q_y)) < 2:
            continue
        auroc = roc_auc_score(q_y, scores)
        print(f"  {task:12s} knn-AUROC={auroc:.3f}")
        rows.append((args.tag, task, "knn_auroc", auroc, np.nan))

    res = pd.DataFrame(rows, columns=["tag", "task", "metric_type", "metric", "metric2"])
    res.to_csv(out_dir / f"small_{args.tag}_scorecard.csv", index=False)
    eff_df = pd.DataFrame([{"tag": args.tag, **{f"pca_{int(t*100)}": k for t, k in eff.items()}}])
    eff_df.to_csv(out_dir / f"small_{args.tag}_effdim.csv", index=False)
    print(f"\n[{args.tag}] wrote scorecard + effdim to {out_dir}")


if __name__ == "__main__":
    main()
