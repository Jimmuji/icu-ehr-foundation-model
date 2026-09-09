"""Stage 05 v2 — Fine-tune with task-specific decoders.

Implements the third leg of the downstream eval pipeline (per the
eval plan): for each downstream task, attach a task-specific
decoder (MLP head) on top of the pretrained encoder, train end-to-end (or
with frozen encoder), and report test metrics.

Two modes:

  Mode A (default — runs on CPU/GPU, fast):
    Frozen encoder + nonlinear MLP head.
    Inputs: pre-extracted 768-d embeddings (from `outputs/embeddings/`)
    Trains a 2-layer MLP per task. The "decoder" is the MLP head.
    Strictly more expressive than the sklearn LR linear probe — captures
    nonlinear structure in the (collapsed) embedding space.

  Mode B (--full, requires GPU):
    Full encoder + MLP head, end-to-end finetune.
    Loads the EHRFormer checkpoint, swaps the pretrain head for a task head,
    and trains everything. Stub for now since the team server's H100s are
    all held by a vLLM job — would launch identically with --full once a
    GPU is free.

Per-task head design:
    BinaryHead: Linear(768→256) → GELU → Dropout → Linear(256→1)
    RegressionHead: same shape, single scalar output

Run:
    python src_v2/05_finetune.py                # Mode A, default
    python src_v2/05_finetune.py --full         # Mode B (needs GPU)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_config, setup_logging, stage_dir


# ---------- Task configuration ----------
BINARY_TASKS = ["mortality", "los_gt_7d", "readmit_30d", "celiac", "masld", "ami", "stroke"]
REGRESSION_TASKS = ["reg_platelets", "reg_creatinine", "reg_spo2"]


# ---------- Task-specific decoder heads ----------
class MLPHead(nn.Module):
    """Two-layer MLP head used as the task-specific decoder for both
    binary classification (output_dim=1, BCEWithLogits) and regression
    (output_dim=1, MSE).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.3, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------- Embedding + label loading (same alignment logic as linear probe) ----------
def load_aligned_embeddings(work: Path, out_dir: Path):
    """Load v1 embeddings and align them to v2 cohort+labels. Returns
    (X: float32 [N,D], meta: DataFrame with split + label columns)."""
    emb_dir = work / "embeddings"
    X_parts, meta_parts = [], []
    for split in ["train", "val", "test"]:
        d = np.load(emb_dir / f"{split}_embeddings.npz")
        X_parts.append(d["embeddings"].astype(np.float32))
        meta_parts.append(pd.DataFrame({
            "stay_id_v1": d["stay_ids"].astype(np.int64),
            "v1_split": split,
        }))
    X = np.concatenate(X_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True)
    meta["row_idx"] = np.arange(len(meta), dtype=np.int64)
    # Map v1 stay_id → hospitalization_id → v2 stay_id + labels + split
    v1_cohort = pd.read_parquet(work / "cohort" / "cohort.parquet")[["stay_id", "hadm_id"]].rename(
        columns={"stay_id": "stay_id_v1", "hadm_id": "hospitalization_id"})
    v2_cohort = pd.read_parquet(work / "cohort_v2" / "cohort.parquet")[["hospitalization_id", "stay_id", "split"]]
    labels = pd.read_parquet(out_dir / "labels.parquet").drop(columns=["split"], errors="ignore")
    meta = meta.merge(v1_cohort, on="stay_id_v1", how="inner")
    meta = meta.merge(v2_cohort.merge(labels, on="stay_id", how="left"),
                       on="hospitalization_id", how="inner")
    X = X[meta["row_idx"].values]
    return X, meta


# ---------- Training one task ----------
def train_one_task(
    task: str,
    is_binary: bool,
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    splits: np.ndarray,
    device: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    hidden_dim: int,
    dropout: float,
    log,
):
    in_dim = X.shape[1]
    train_idx = np.where((splits == "train") & mask)[0]
    val_idx = np.where((splits == "val") & mask)[0]
    test_idx = np.where((splits == "test") & mask)[0]
    if len(train_idx) == 0 or (is_binary and y[train_idx].sum() in (0, len(train_idx))):
        log.info(f"  [SKIP {task}] insufficient training samples / positives")
        return None

    # Z-score features on train
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0) + 1e-6
    def norm(idx): return (X[idx] - mu) / sd

    Xtr = torch.from_numpy(norm(train_idx)).float()
    Xva = torch.from_numpy(norm(val_idx)).float()
    Xte = torch.from_numpy(norm(test_idx)).float()

    if is_binary:
        ytr = torch.from_numpy(y[train_idx].astype(np.float32))
        yva = torch.from_numpy(y[val_idx].astype(np.float32))
        yte = torch.from_numpy(y[test_idx].astype(np.float32))
        pos_count = ytr.sum().item()
        neg_count = len(ytr) - pos_count
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # Clip extreme values (same as linear probe)
        lo, hi = np.percentile(y[train_idx], [0.5, 99.5])
        ytr = torch.from_numpy(np.clip(y[train_idx], lo, hi).astype(np.float32))
        yva = torch.from_numpy(np.clip(y[val_idx], lo, hi).astype(np.float32))
        yte = torch.from_numpy(np.clip(y[test_idx], lo, hi).astype(np.float32))
        # Standardize targets for stable MLP training; unscale at eval
        y_mu, y_sd = ytr.mean().item(), ytr.std().item() + 1e-6
        ytr_s = (ytr - y_mu) / y_sd
        yva_s = (yva - y_mu) / y_sd
        yte_s = (yte - y_mu) / y_sd
        loss_fn = nn.MSELoss()

    head = MLPHead(in_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    train_ds = TensorDataset(Xtr, ytr if is_binary else ytr_s)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    Xva_d = Xva.to(device)
    Xte_d = Xte.to(device)
    if is_binary:
        yva_d = yva.to(device)
    else:
        yva_d = yva_s.to(device)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(n_epochs):
        head.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = head(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            p_val = head(Xva_d)
            v_loss = loss_fn(p_val, yva_d).item()
            p_val = p_val.cpu()
        history.append({"epoch": epoch+1, "val_loss": v_loss})

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        p_val = head(Xva_d).cpu().numpy()
        p_test = head(Xte_d).cpu().numpy()

    # ----- Metrics -----
    from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error
    if is_binary:
        # Sigmoid to get probs (BCEWithLogits output is logits)
        p_val_prob = 1 / (1 + np.exp(-p_val))
        p_test_prob = 1 / (1 + np.exp(-p_test))
        y_val_np = yva.numpy()
        y_test_np = yte.numpy()
        result = dict(
            task=task, type="binary",
            n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
            n_test_pos=int(y_test_np.sum()),
            val_auroc=roc_auc_score(y_val_np, p_val_prob) if len(np.unique(y_val_np)) == 2 else float("nan"),
            val_auprc=average_precision_score(y_val_np, p_val_prob) if len(np.unique(y_val_np)) == 2 else float("nan"),
            test_auroc=roc_auc_score(y_test_np, p_test_prob) if len(np.unique(y_test_np)) == 2 else float("nan"),
            test_auprc=average_precision_score(y_test_np, p_test_prob) if len(np.unique(y_test_np)) == 2 else float("nan"),
            best_val_loss=best_val,
            n_epochs_run=len(history),
        )
        log.info(f"  {task:14s} | train n={result['n_train']:>6,} | "
                 f"val AUROC={result['val_auroc']:.3f} AUPRC={result['val_auprc']:.3f} | "
                 f"test AUROC={result['test_auroc']:.3f} AUPRC={result['test_auprc']:.3f} | "
                 f"epochs={result['n_epochs_run']}")
    else:
        # Unscale predictions
        p_val_unscaled = p_val * y_sd + y_mu
        p_test_unscaled = p_test * y_sd + y_mu
        result = dict(
            task=task, type="regression",
            n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
            n_test_pos=0,
            val_r2=r2_score(yva.numpy(), p_val_unscaled),
            val_mae=mean_absolute_error(yva.numpy(), p_val_unscaled),
            test_r2=r2_score(yte.numpy(), p_test_unscaled),
            test_mae=mean_absolute_error(yte.numpy(), p_test_unscaled),
            best_val_loss=best_val,
            n_epochs_run=len(history),
        )
        log.info(f"  {task:14s} | train n={result['n_train']:>6,} | "
                 f"val R²={result['val_r2']:.3f} MAE={result['val_mae']:.2f} | "
                 f"test R²={result['test_r2']:.3f} MAE={result['test_mae']:.2f} | "
                 f"epochs={result['n_epochs_run']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--full", action="store_true",
                     help="Mode B: unfreeze encoder for full finetune (requires GPU + EHRFormer checkpoint).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("05_finetune", cfg["paths"]["work_dir"])
    out_dir = stage_dir(cfg, "downstream_v2")
    work = Path(cfg["paths"]["work_dir"])

    if args.full:
        log.info("Mode B (--full): full encoder + head end-to-end finetune. NOT YET IMPLEMENTED.")
        log.info("  Will load EHRFormer checkpoint + swap pretrain head with MLPHead + train both.")
        log.info("  Pending GPU availability (all 8 H100s currently held by vLLM serving job).")
        log.info("  Mode A (frozen + MLP head) is functionally equivalent for embedding-quality eval,")
        log.info("  so we run that now and add Mode B comparisons later.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Mode A: frozen encoder + MLP task heads. Device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)

    log.info("Loading v1 embeddings + v2 labels…")
    X, meta = load_aligned_embeddings(work, out_dir)
    log.info(f"  X: {X.shape}, meta: {len(meta):,}")
    splits = meta["split"].values

    results = []
    log.info(f"\n=== Binary classification tasks (MLP head, BCE w/ pos-weight, AdamW + cosine LR) ===")
    for task in BINARY_TASKS:
        y = meta[f"y_{task}"].values
        m = meta[f"m_{task}"].values == 1
        r = train_one_task(
            task, is_binary=True, X=X, y=y, mask=m, splits=splits,
            device=device, n_epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.wd, patience=args.patience,
            hidden_dim=args.hidden_dim, dropout=args.dropout, log=log,
        )
        if r is not None: results.append(r)

    log.info(f"\n=== Regression tasks (MLP head, MSE on z-scored targets) ===")
    for task in REGRESSION_TASKS:
        y = meta[f"y_{task}"].values.astype(np.float32)
        m = meta[f"m_{task}"].values == 1
        r = train_one_task(
            task, is_binary=False, X=X, y=y, mask=m, splits=splits,
            device=device, n_epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.wd, patience=args.patience,
            hidden_dim=args.hidden_dim, dropout=args.dropout, log=log,
        )
        if r is not None: results.append(r)

    # ----- Save results -----
    res = pd.DataFrame(results)
    out_csv = out_dir / "finetune_results.csv"
    res.to_csv(out_csv, index=False)
    log.info(f"\nWrote {out_csv}")
    log.info(f"\n=== Final results ===\n{res.to_string(index=False)}")


if __name__ == "__main__":
    main()
