"""Step 4a — extract patient embeddings from a trained EHRFormer checkpoint.

Loads the LightningModule, runs forward on each chunk, mean-pools `mu_z` (the VAE
latent mean — more stable than sampled z) across valid timesteps to get a
per-patient embedding, and writes {N, D} float32 matrix + parallel metadata.

Run on the H100 server (1 GPU is plenty; embedding extraction is fast).

Usage:
    python -m src.clustering.extract_embeddings \
        --ckpt outputs/pretrain/log/version_0/checkpoint/last.ckpt \
        --chunks outputs/chunks \
        --out outputs/embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_lightning_checkpoint(ckpt_path: str, ehrformer_repo: str | None):
    """Load the EHRModule + return its inner model. Falls back to raw state_dict if needed."""
    if ehrformer_repo:
        sys.path.insert(0, ehrformer_repo)
    from ehr_model_module_pretrain import EHRModule  # noqa: E402
    from transformers import GPT2Config  # noqa: E402

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("hyper_parameters", {}).get("config") or ckpt.get("config")
    if cfg is None:
        raise RuntimeError(
            "Checkpoint missing config in hyper_parameters. "
            "Pass --config explicitly to override."
        )
    cfg["transformer"] = GPT2Config.from_pretrained("gpt2")
    cfg["transformer"].n_positions = 8192
    cfg["mode"] = "pretrain"

    module = EHRModule(cfg)
    module.load_state_dict(ckpt["state_dict"], strict=False)
    module.eval()
    return module, cfg


class ChunkDataset(Dataset):
    """Iterate over per-stay records inside our chunked .pt files."""

    def __init__(self, chunks_dir: Path, split: str):
        self.files = sorted((chunks_dir / split).glob("chunk_*.pt"))
        # Build a flat index (file_idx, record_idx) → linear i
        self.index: list[tuple[int, int]] = []
        for fi, f in enumerate(self.files):
            recs = torch.load(f, map_location="cpu", weights_only=False)
            self.index.extend((fi, ri) for ri in range(len(recs)))
        self._cache: dict[int, list] = {}

    def _load(self, fi: int) -> list:
        if fi not in self._cache:
            # Single-file LRU: keep last 2 to bound RAM
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            self._cache[fi] = torch.load(self.files[fi], map_location="cpu", weights_only=False)
        return self._cache[fi]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, ri = self.index[i]
        return self._load(fi)[ri]


def collate(records: list[dict]):
    """Pad cat / cont / valid_mask to the max length in the batch."""
    max_T = max(r["cat"].shape[0] for r in records)
    F_cat = records[0]["cat"].shape[1]
    F_cont = records[0]["cont"].shape[1]

    cats, conts, masks, times, sids = [], [], [], [], []
    for r in records:
        T = r["cat"].shape[0]
        pad = max_T - T
        cats.append(torch.nn.functional.pad(r["cat"].T, (0, pad), value=-1))    # (F_cat, max_T)
        conts.append(torch.nn.functional.pad(r["cont"].T, (0, pad), value=-1)) # (F_cont, max_T)
        masks.append(torch.nn.functional.pad(r["valid_mask"].T.bool(), (0, pad), value=False))
        times.append(torch.nn.functional.pad(r["time_index"], (0, pad), value=0))
        sids.append(r["stay_id"])
    return {
        "cat_feats": torch.stack(cats).long(),     # (B, F_cat, T)
        "float_feats": torch.stack(conts).long(),  # (B, F_cont, T)
        "valid_mask": torch.stack(masks),          # (B, F_total, T)
        "time_index": torch.stack(times).long(),   # (B, T)
        "stay_id": sids,
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chunks", required=True, help="dir containing train/, val/, test/")
    ap.add_argument("--ehrformer-repo", default=None,
                    help="path to EHRFormer repo (for ehr_model_module_pretrain import)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    module, cfg = load_lightning_checkpoint(args.ckpt, args.ehrformer_repo)
    model = module.model.to(args.device).eval()

    chunks_dir = Path(args.chunks)
    for split in args.splits:
        ds = ChunkDataset(chunks_dir, split)
        if len(ds) == 0:
            print(f"  {split}: no chunks, skipping")
            continue
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)

        all_emb, all_sids = [], []
        for batch in tqdm(dl, desc=f"embed {split}"):
            cat = batch["cat_feats"].to(args.device)
            flo = batch["float_feats"].to(args.device)
            tim = batch["time_index"].to(args.device)
            valid = batch["valid_mask"].to(args.device)
            # Use the model's encoding path; bypass VAE sampling by reading mu_z directly.
            ft_emb = model.ehr_embed(cat, flo)
            y = model.transformer(
                inputs_embeds=ft_emb,
                position_ids=tim,
                attention_mask=valid.any(dim=1),
            ).last_hidden_state
            mu = model.ehr_mu(y)
            # mu shape: (B, T, D). Pool over valid timesteps for one vector per stay.
            time_valid = valid.any(dim=1).unsqueeze(-1).float()  # (B, T, 1)
            denom = time_valid.sum(dim=1).clamp(min=1.0)
            pooled = (mu * time_valid).sum(dim=1) / denom        # (B, D)
            all_emb.append(pooled.cpu().numpy().astype(np.float32))
            all_sids.extend(batch["stay_id"])

        embs = np.concatenate(all_emb, axis=0)
        np.savez(out_dir / f"{split}_embeddings.npz",
                 embeddings=embs, stay_ids=np.asarray(all_sids, dtype=np.int64))
        print(f"  {split}: wrote {embs.shape} → {out_dir / f'{split}_embeddings.npz'}")

    # Stash the config so downstream scripts know D etc
    with open(out_dir / "extract_info.json", "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "embedding_dim": int(embs.shape[1]) if 'embs' in locals() else None,
        }, f, indent=2)


if __name__ == "__main__":
    main()
