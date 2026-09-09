"""Stage 05 — fuse per-stay .npz files into chunked .pt shards for EHRFormer.

EHRFormer's `ehr_dataset_chunk.py` expects pre-shuffled chunks with multiple
patients per file (faster than reading thousands of tiny files at training time).

Output: {work_dir}/chunks/{train,val,test}/chunk_NNNN.pt
Each .pt is a list[dict] with keys: cat, cont, time_index, valid_mask, stay_id, length.

Run:
    python src/05_chunk.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import load_config, setup_logging, stage_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("05_chunk", cfg["paths"]["work_dir"])
    work = Path(cfg["paths"]["work_dir"])

    splits = pd.read_parquet(work / "vocab" / "splits.parquet")
    tokens_dir = work / "tokens"
    out_root = stage_dir(cfg, "chunks")

    chunk_size = cfg["chunking"]["patients_per_chunk"]
    rng = np.random.default_rng(cfg["split"]["seed"])

    for split_name in cfg["chunking"]["shard_subdirs"]:
        out_dir = Path(out_root) / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        stays = splits.loc[splits["split"] == split_name, "stay_id"].astype(int).tolist()
        rng.shuffle(stays)
        log.info(f"{split_name}: {len(stays):,} stays")

        records: list[dict] = []
        chunk_idx = 0
        for sid in tqdm(stays, desc=split_name):
            f = tokens_dir / f"{sid}.npz"
            if not f.exists():
                continue
            arr = np.load(f)
            rec = {
                "stay_id": int(sid),
                "cat": torch.from_numpy(arr["cat"].astype(np.int32)),
                "cont": torch.from_numpy(arr["cont"].astype(np.int32)),
                "time_index": torch.from_numpy(arr["time_index"].astype(np.int32)),
                "valid_mask": torch.from_numpy(arr["valid_mask"]),
                "length": int(arr["time_index"].shape[0]),
            }
            records.append(rec)
            if len(records) >= chunk_size:
                torch.save(records, out_dir / f"chunk_{chunk_idx:05d}.pt")
                chunk_idx += 1
                records = []
        if records:
            torch.save(records, out_dir / f"chunk_{chunk_idx:05d}.pt")
            chunk_idx += 1
        log.info(f"  wrote {chunk_idx} chunks to {out_dir}")


if __name__ == "__main__":
    main()
