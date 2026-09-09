"""Train SmallEHRTransformer (~2M params) on the existing v1 diskcache.

Reuses v1's tokenized data (50k ICU stays) — much faster to iterate on than
re-tokenizing the full 523k v2 cohort. Once this small-model recipe is
validated, we can scale up.

Differences from v1 training:
- No VAE (fixes posterior collapse)
- Smaller model (2.16M vs 228M params)
- Wandb logging (responding to the onboarding requirement)
- Cleaner masked-feature reconstruction loss
- Early stopping on val_loss
- bf16 + 1 GPU (no DDP needed at this size)

Run:
    python src_v2/train_small.py --epochs 15 --batch-size 32 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from small_ehr_model import SmallEHRTransformer, loss_pretrain


class EHRCacheDataset(Dataset):
    """Wraps the v1 diskcache. Each sample comes pre-padded to seq_max_len=512."""
    def __init__(self, cache_dir: Path, pids: list[int], float_mean_std: dict[str, tuple[float, float]] | None = None):
        import diskcache
        self.cache = diskcache.Cache(str(cache_dir), eviction_policy="none")
        self.pids = pids
        # Optional: standardize raw float values for regression loss
        # float_mean_std: maps feature_idx → (mean, std)
        self.float_mean = None
        self.float_std = None
        if float_mean_std is not None:
            n_feat = max(float_mean_std.keys()) + 1
            self.float_mean = np.zeros(n_feat, dtype=np.float32)
            self.float_std = np.ones(n_feat, dtype=np.float32)
            for i, (m, s) in float_mean_std.items():
                self.float_mean[i] = m
                self.float_std[i] = max(s, 1e-3)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = int(self.pids[idx])
        d = self.cache[pid]
        # All arrays in cache use feature-first layout (F, T)
        out = {
            "pid": pid,
            "cat": torch.from_numpy(d["tokenized_category_feats"]).long(),
            "float_bin": torch.from_numpy(d["tokenized_float_feats"]).long(),
            "float_raw": torch.from_numpy(d["float_feats"]).float(),
            "valid": torch.from_numpy(d["valid_mask"]).bool(),
            "time": torch.from_numpy(d["time_index"]).long(),
        }
        if self.float_mean is not None:
            # Standardize raw values (avoid normalizing -1 = missing)
            raw = out["float_raw"]
            missing = (raw == -1) | torch.isnan(raw)
            m = torch.from_numpy(self.float_mean).view(-1, 1)
            s = torch.from_numpy(self.float_std).view(-1, 1)
            normalized = (raw - m) / s
            out["float_raw"] = torch.where(missing, torch.full_like(raw, float("nan")), normalized)
        return out


def random_mask(valid_mask, n_cat, n_float, mask_ratio=0.50, device="cpu"):
    """For each (sample, feature, timestep) where the position is VALID,
    randomly select mask_ratio fraction to mask. Returns:
      cat_mask_input  : (B, F_cat, T) bool — True where to mask in INPUT
      float_mask_input: (B, F_float, T) bool
      cat_mask_loss   : same — True where we compute loss
      float_mask_loss : same
    For simplicity we apply the same mask to input and loss (model must predict
    the masked positions).
    """
    B, T = valid_mask.shape
    # Cat mask: per (B, F_cat, T)
    cat_rand = torch.rand(B, n_cat, T, device=device)
    float_rand = torch.rand(B, n_float, T, device=device)
    # Only mask within valid timesteps
    valid_3d_cat = valid_mask.unsqueeze(1).expand(B, n_cat, T)
    valid_3d_float = valid_mask.unsqueeze(1).expand(B, n_float, T)
    cat_m = (cat_rand < mask_ratio) & valid_3d_cat
    float_m = (float_rand < mask_ratio) & valid_3d_float
    return cat_m, float_m


def apply_mask(tensor, mask, missing_val):
    """Replace tensor values where mask=True with missing_val. Used to construct masked INPUT."""
    return torch.where(mask, torch.full_like(tensor, missing_val), tensor)


def train_one_epoch(model, loader, opt, scheduler, scaler, device, mask_ratio, log_every=50):
    model.train()
    n_cat = model.n_cat
    n_float = model.n_float
    metrics = {"loss": 0.0, "cls": 0.0, "reg": 0.0, "n": 0}
    t0 = time.time()
    for step, batch in enumerate(loader):
        cat = batch["cat"].to(device)        # (B, F_cat, T) int
        float_bin = batch["float_bin"].to(device)  # (B, F_float, T) int
        float_raw = batch["float_raw"].to(device)  # (B, F_float, T) float
        valid = batch["valid"].to(device)    # (B, T) bool
        tindex = batch["time"].to(device)    # (B, T) int

        # Random masking
        cat_mask, float_mask = random_mask(valid, n_cat, n_float, mask_ratio, device=device)
        cat_input = apply_mask(cat, cat_mask, missing_val=-1)
        float_input = apply_mask(float_bin, float_mask, missing_val=-1)

        amp_device = "cuda" if device == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16):
            cls_logits, float_out, _ = model(cat_input, float_input, tindex, valid)
            loss, parts = loss_pretrain(
                cls_logits, float_out,
                cat_target=cat, cat_mask_loss=cat_mask, float_mask_loss=float_mask,
                float_target_raw=float_raw, float_target_bin=float_bin,
                float_mode=model.float_mode,
            )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        bs = cat.shape[0]
        metrics["loss"] += parts["total"] * bs
        metrics["cls"] += parts["cls"] * bs
        metrics["reg"] += parts["reg"] * bs
        metrics["n"] += bs

        if step % log_every == 0:
            print(f"  step {step:>4} | loss={parts['total']:.4f} cls={parts['cls']:.4f} reg={parts['reg']:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")

    for k in ["loss", "cls", "reg"]:
        metrics[k] /= max(metrics["n"], 1)
    return metrics


@torch.no_grad()
def eval_one_epoch(model, loader, device, mask_ratio):
    model.eval()
    n_cat = model.n_cat
    n_float = model.n_float
    metrics = {"loss": 0.0, "cls": 0.0, "reg": 0.0, "n": 0}
    for batch in loader:
        cat = batch["cat"].to(device)
        float_bin = batch["float_bin"].to(device)
        float_raw = batch["float_raw"].to(device)
        valid = batch["valid"].to(device)
        tindex = batch["time"].to(device)
        cat_mask, float_mask = random_mask(valid, n_cat, n_float, mask_ratio, device=device)
        cat_input = apply_mask(cat, cat_mask, missing_val=-1)
        float_input = apply_mask(float_bin, float_mask, missing_val=-1)
        amp_device = "cuda" if device == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16):
            cls_logits, float_out, _ = model(cat_input, float_input, tindex, valid)
            _, parts = loss_pretrain(
                cls_logits, float_out,
                cat_target=cat, cat_mask_loss=cat_mask, float_mask_loss=float_mask,
                float_target_raw=float_raw, float_target_bin=float_bin,
                float_mode=model.float_mode,
            )
        bs = cat.shape[0]
        metrics["loss"] += parts["total"] * bs
        metrics["cls"] += parts["cls"] * bs
        metrics["reg"] += parts["reg"] * bs
        metrics["n"] += bs
    for k in ["loss", "cls", "reg"]:
        metrics[k] /= max(metrics["n"], 1)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/data/mimic_data/physionet.org/script/ehrformer/outputs/ehr_cache")
    ap.add_argument("--feat-info", default="/data/mimic_data/physionet.org/script/ehrformer/outputs/feat_info.json")
    ap.add_argument("--out-dir", default="/data/mimic_data/physionet.org/script/ehrformer/outputs/v2_small")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--hidden-dim", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--mask-ratio", type=float, default=0.50)
    ap.add_argument("--float-mode", choices=["mse", "ce"], default="mse",
                    help="continuous-value head: 'mse' regresses raw value; 'ce' predicts quantile bin (APOLLO/MOTOR style)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load feat_info ----
    with open(args.feat_info) as f:
        feat_info = json.load(f)
    n_cat_feats = len(feat_info.get("category_cols", []))
    n_float_feats = len(feat_info.get("float_cols", {}))
    print(f"n_cat={n_cat_feats}, n_float={n_float_feats}")

    # Build float mean/std dict by feature_idx (alphabetical order matches stage 06)
    float_features = sorted(feat_info["float_cols"].keys())
    float_mean_std = {}
    for i, fname in enumerate(float_features):
        info = feat_info["float_cols"][fname]
        float_mean_std[i] = (float(info.get("mean", 0.0)), float(info.get("std", 1.0)))

    # ---- Load split assignment ----
    import pandas as pd
    metadata = pd.read_parquet(Path(args.cache) / "metadata.parquet")
    train_pids = metadata.loc[metadata["split"] == "train", "pid"].tolist()
    val_pids = metadata.loc[metadata["split"] == "val", "pid"].tolist()
    test_pids = metadata.loc[metadata["split"] == "test", "pid"].tolist()
    print(f"train/val/test sizes: {len(train_pids)}/{len(val_pids)}/{len(test_pids)}")

    train_ds = EHRCacheDataset(Path(args.cache), train_pids, float_mean_std)
    val_ds = EHRCacheDataset(Path(args.cache), val_pids, float_mean_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # ---- Build model ----
    model = SmallEHRTransformer(
        n_cat_feats=n_cat_feats, n_float_feats=n_float_feats,
        n_cat_values=50, n_float_bins=256,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        n_heads=args.n_heads, ff_dim=args.hidden_dim * 2, max_seq_len=512,
        float_mode=args.float_mode,
    ).to(device)
    print(f"Params: {model.num_params()/1e6:.2f}M | float_mode={args.float_mode}")

    # ---- Optim + scheduler ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * (len(train_loader))
    warmup_steps = int(total_steps * 0.1)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 doesn't need GradScaler

    # ---- Wandb ----
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="ehrformer-mimic-v2", config=vars(args),
                       name=f"small_d{args.hidden_dim}_L{args.n_layers}_{args.float_mode}")
        except Exception as e:
            print(f"  (wandb disabled: {e})")
            use_wandb = False

    # ---- Train loop ----
    best_val = float("inf")
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        tr = train_one_epoch(model, train_loader, opt, scheduler, scaler, device, args.mask_ratio)
        va = eval_one_epoch(model, val_loader, device, args.mask_ratio)
        print(f"  train: loss={tr['loss']:.4f} cls={tr['cls']:.4f} reg={tr['reg']:.4f}")
        print(f"  val  : loss={va['loss']:.4f} cls={va['cls']:.4f} reg={va['reg']:.4f}")
        if use_wandb:
            wandb.log({"epoch": epoch+1,
                       "train_loss": tr["loss"], "train_cls": tr["cls"], "train_reg": tr["reg"],
                       "val_loss": va["loss"], "val_cls": va["cls"], "val_reg": va["reg"],
                       "lr": opt.param_groups[0]["lr"]})
        # Save last + best
        torch.save({"model": model.state_dict(), "epoch": epoch+1, "val_loss": va["loss"], "args": vars(args)},
                   out_dir / "last.pt")
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({"model": model.state_dict(), "epoch": epoch+1, "val_loss": va["loss"], "args": vars(args)},
                       out_dir / "best.pt")
            print(f"  ★ new best val_loss={best_val:.4f}")

    print(f"\nDone. Best val_loss={best_val:.4f}. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
