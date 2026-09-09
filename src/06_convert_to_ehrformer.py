"""Stage 06 — convert our preprocessing outputs to EHRFormer's expected format.

EHRFormer's data loader (ehr_dataset_chunk.py) expects:
  - {work_dir}/ehr_cache/metadata.parquet  — rows: pid, split
  - diskcache stored in {work_dir}/ehr_cache/ — key=pid, value=dict with:
      tokenized_category_feats : (F_cat,   T) int  — quantized cat token IDs
      tokenized_float_feats    : (F_float, T) int  — quantized float bin IDs (-1=missing)
      category_feats           : (F_cat,   T) int  — same as tokenized for cat
      float_feats              : (F_float, T) float — RAW (un-quantized) float values
      valid_mask               : (T,)        bool  — any feature valid at this timestep
      time_index               : (T,)        int

Also updates outputs/feat_info.json to include per-float mean/std (needed for
the loader to normalize regression targets).

Run on server (CPU-only, ~10-20 min):
    python src/06_convert_to_ehrformer.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# diskcache imported lazily so the script is importable without it on Mac
def _open_cache(path: Path):
    import diskcache
    return diskcache.Cache(str(path), eviction_policy="none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="/data/mimic_data/physionet.org/script/ehrformer/outputs")
    ap.add_argument("--seq-max-len", type=int, default=512)
    ap.add_argument("--bin-hours", type=int, default=1)
    args = ap.parse_args()

    PROJ = Path(args.work_dir)
    SEQ_MAX = args.seq_max_len
    BIN_HOURS = args.bin_hours

    # --- Load all artifacts ---
    print(f"[load] artifacts from {PROJ}")
    feat_idx = pd.read_parquet(PROJ / "vocab" / "feature_index.parquet")
    splits = pd.read_parquet(PROJ / "vocab" / "splits.parquet")
    events = pd.read_parquet(PROJ / "events" / "events.parquet")
    print(f"  events: {len(events):,} rows")

    cat_features = sorted(feat_idx[feat_idx.feature_kind == "cat"].feature_name.tolist())
    float_features = sorted(feat_idx[feat_idx.feature_kind == "cont"].feature_name.tolist())
    F_cat, F_float = len(cat_features), len(float_features)
    print(f"  F_cat={F_cat}, F_float={F_float}, SEQ_MAX={SEQ_MAX}")

    float_to_col = {f: i for i, f in enumerate(float_features)}

    # --- Compute mean/std per float feature (using train events only) ---
    train_stays = set(splits.loc[splits["split"] == "train", "stay_id"].astype(int))
    print(f"  train stays: {len(train_stays):,}")

    events_cont = events[events.feature_kind == "cont"].copy()
    events_cont["value"] = pd.to_numeric(events_cont["value"], errors="coerce")
    events_cont = events_cont.dropna(subset=["value"])
    train_cont = events_cont[events_cont["stay_id"].isin(train_stays)]
    print(f"  train cont events: {len(train_cont):,}")

    ms = train_cont.groupby("feature_name")["value"].agg(["mean", "std"]).reset_index()
    mean_lookup = dict(zip(ms.feature_name, ms["mean"]))
    std_lookup = dict(zip(ms.feature_name, ms["std"]))
    print(f"  computed mean/std for {len(ms)} features")

    # --- Update feat_info.json with mean/std ---
    feat_info_path = PROJ / "feat_info.json"
    with open(feat_info_path) as f:
        feat_info = json.load(f)
    for fname in float_features:
        if fname not in feat_info["float_cols"]:
            feat_info["float_cols"][fname] = {}
        feat_info["float_cols"][fname]["mean"] = float(mean_lookup.get(fname, 0.0))
        feat_info["float_cols"][fname]["std"] = float(std_lookup.get(fname, 1.0)) if not np.isnan(std_lookup.get(fname, 1.0)) else 1.0
    with open(feat_info_path, "w") as f:
        json.dump(feat_info, f, indent=2)
    print(f"  wrote feat_info.json with mean/std")

    # --- Aggregate raw float per (stay, feature, bin) ---
    print("[agg] aggregating raw float values per (stay, feature, bin)…")
    events_cont["bin"] = (events_cont["t_hours"] // BIN_HOURS).astype(int)
    events_cont = events_cont[(events_cont["bin"] >= 0)]
    agg = (events_cont
           .groupby(["stay_id", "feature_name", "bin"])["value"]
           .mean()
           .reset_index())
    print(f"  agg rows: {len(agg):,}")

    # Index for fast per-stay lookup
    agg_by_stay = agg.groupby("stay_id")
    del events, events_cont, train_cont
    gc.collect()

    # --- Build per-stay structures + write to diskcache + metadata ---
    cache_dir = PROJ / "ehr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _open_cache(cache_dir)

    metadata_rows = []
    all_stays_with_split = splits[["stay_id", "split"]].itertuples(index=False)
    n_total = len(splits)
    print(f"[convert] {n_total:,} stays → {cache_dir}")

    tokens_dir = PROJ / "tokens"
    n_done = 0
    n_skip = 0
    for stay_id, split_name in all_stays_with_split:
        stay_id = int(stay_id)
        npz_path = tokens_dir / f"{stay_id}.npz"
        if not npz_path.exists():
            n_skip += 1
            continue
        npz = np.load(npz_path)
        tok_cat = npz["cat"]              # (T, F_cat)
        tok_cont = npz["cont"]            # (T, F_cont)
        valid_mask_2d = npz["valid_mask"] # (T, F_total)
        time_index = npz["time_index"]    # (T,)
        T = tok_cat.shape[0]

        # Pad to SEQ_MAX
        if T < SEQ_MAX:
            pad = SEQ_MAX - T
            tok_cat = np.pad(tok_cat, ((0, pad), (0, 0)), constant_values=0)        # 0=PAD
            tok_cont = np.pad(tok_cont, ((0, pad), (0, 0)), constant_values=-1)     # -1=missing
            valid_mask_2d = np.pad(valid_mask_2d, ((0, pad), (0, 0)), constant_values=False)
            time_index = np.pad(time_index, (0, pad), constant_values=0)
        elif T > SEQ_MAX:
            tok_cat = tok_cat[:SEQ_MAX]
            tok_cont = tok_cont[:SEQ_MAX]
            valid_mask_2d = valid_mask_2d[:SEQ_MAX]
            time_index = time_index[:SEQ_MAX]

        # Build raw float (T, F_float) with -1 = missing
        raw_float = np.full((SEQ_MAX, F_float), -1.0, dtype=np.float32)
        if stay_id in agg_by_stay.groups:
            stay_agg = agg.iloc[agg_by_stay.indices[stay_id]]
            # Vectorized fill
            cols = stay_agg["feature_name"].map(float_to_col).values
            bins = stay_agg["bin"].values.astype(int)
            vals = stay_agg["value"].values.astype(np.float32)
            in_range = (bins < SEQ_MAX) & (cols != -1) & (~pd.isna(cols))
            cols = cols[in_range]
            bins = bins[in_range]
            vals = vals[in_range]
            raw_float[bins, cols.astype(int)] = vals

        # Transpose to features-first and build 1D valid mask
        data_dict = {
            "pid": stay_id,
            "tokenized_category_feats": tok_cat.T.astype(np.int32),
            "tokenized_float_feats": tok_cont.T.astype(np.int32),
            "category_feats": tok_cat.T.astype(np.int32),  # same as tokenized for cat
            "float_feats": raw_float.T.astype(np.float32),
            "valid_mask": valid_mask_2d.any(axis=1).astype(bool),
            "time_index": time_index.astype(np.int64),
        }
        cache[stay_id] = data_dict
        metadata_rows.append({"pid": stay_id, "split": split_name})

        n_done += 1
        if n_done % 5000 == 0:
            print(f"  [{n_done:,}/{n_total:,}]")

    cache.close()
    print(f"[convert] done: {n_done:,} cached, {n_skip:,} skipped (missing .npz)")

    # --- Write metadata.parquet ---
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_parquet(cache_dir / "metadata.parquet", index=False)
    print(f"  metadata.parquet: {len(metadata):,} rows → {cache_dir / 'metadata.parquet'}")
    print(f"  split counts: {metadata['split'].value_counts().to_dict()}")

    print("\n=== Updates needed to pretrain_mimic.json ===")
    print(f'  df_paths: "{cache_dir}"')
    print(f'  use_cache: true')


if __name__ == "__main__":
    main()
