"""Stage 04 — tokenize each ICU stay into the EHRFormer tensor format.

For each stay we produce a dict of arrays:
  cat        : int16 [T, F_cat]     — categorical token id (PAD=0 if missing)
  cont       : int16 [T, F_cont]    — quantized bin id (MISSING= -1)
  time_index : int32 [T]            — time bin index (0..T-1 with admit-aligned)
  valid_mask : bool  [T, F_total]   — True where the value was observed at this bin

Aggregation rule per (stay, bin, feature):
  - categorical: last observed value within the bin
  - continuous : mean of observed values within the bin (then quantize)

If a stay has more than max_seq_len bins, we keep the LAST max_seq_len bins
(the most-recent window) — common ICU pretraining choice; configurable.

Output: {work_dir}/tokens/{stay_id}.npz  (one file per stay) — chunker fuses these.

Run:
    python src/04_tokenize.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_config, setup_logging, stage_dir


def quantize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map values to bin index in [0, n_bins-1] via np.searchsorted on bin edges."""
    # edges has length n_bins+1, monotonic
    idx = np.searchsorted(edges, values, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("04_tokenize", cfg["paths"]["work_dir"])
    out_dir = stage_dir(cfg, "tokens")
    work = Path(cfg["paths"]["work_dir"])

    # --- Load artifacts from prior stages ---
    events = pd.read_parquet(work / "events" / "events.parquet")
    feat_idx = pd.read_parquet(work / "vocab" / "feature_index.parquet")
    edge_df = pd.read_parquet(work / "vocab" / "bin_edges.parquet")
    with open(work / "vocab" / "vocab_cat.json") as f:
        cat_vocab = json.load(f)

    bin_h = cfg["time_binning"]["bin_hours"]
    max_T = cfg["time_binning"]["max_seq_len"]
    miss_token = cfg["quantization"]["missing_value_token"]
    pad_id = cfg["vocab"]["pad_token_id"]
    unk_id = cfg["vocab"]["unk_token_id"]

    # --- Build feature index lookups ---
    cat_features = feat_idx[feat_idx["feature_kind"] == "cat"]["feature_name"].tolist()
    cont_features = feat_idx[feat_idx["feature_kind"] == "cont"]["feature_name"].tolist()
    cat_to_col = {f: i for i, f in enumerate(cat_features)}
    cont_to_col = {f: i for i, f in enumerate(cont_features)}
    F_cat = len(cat_features)
    F_cont = len(cont_features)
    F_total = F_cat + F_cont
    log.info(f"F_cat={F_cat}, F_cont={F_cont}, F_total={F_total}, max_T={max_T}")

    # Edge lookup: feature_name -> np.ndarray
    edge_lookup = {row["feature_name"]: np.asarray(row["edges"]) for _, row in edge_df.iterrows()}

    # Pre-compute time bin per event
    events = events.copy()
    events["bin"] = (events["t_hours"] // bin_h).astype(int)
    events = events[events["bin"] >= 0]

    # Sort once for last-value semantics
    events = events.sort_values(["stay_id", "bin", "t_hours"])

    n_stays = events["stay_id"].nunique()
    log.info(f"Tokenizing {n_stays:,} stays")

    n_written = 0
    for stay_id, grp in tqdm(events.groupby("stay_id", sort=False), total=n_stays):
        bins = grp["bin"].to_numpy()
        T_full = int(bins.max()) + 1
        # Truncate to last max_T bins
        if T_full > max_T:
            offset = T_full - max_T
            grp = grp[grp["bin"] >= offset].copy()
            grp["bin"] = grp["bin"] - offset
            T = max_T
        else:
            T = T_full

        cat = np.full((T, F_cat), pad_id, dtype=np.int16)
        cont = np.full((T, F_cont), miss_token, dtype=np.int16)
        valid = np.zeros((T, F_total), dtype=bool)

        # Categorical: last value per (bin, feature)
        cat_grp = grp[grp["feature_kind"] == "cat"]
        if len(cat_grp):
            for (b, fname), sub in cat_grp.groupby(["bin", "feature_name"], sort=False):
                col = cat_to_col.get(fname)
                if col is None:
                    continue
                vocab = cat_vocab.get(fname, {})
                last_val = str(sub["value"].iloc[-1])
                tok = vocab.get(last_val, unk_id)
                cat[int(b), col] = tok
                valid[int(b), col] = True

        # Continuous: mean per (bin, feature) → quantize
        cont_grp = grp[grp["feature_kind"] == "cont"].copy()
        if len(cont_grp):
            cont_grp["value"] = pd.to_numeric(cont_grp["value"], errors="coerce")
            cont_grp = cont_grp.dropna(subset=["value"])
            agg = cont_grp.groupby(["bin", "feature_name"], sort=False)["value"].mean().reset_index()
            for _, r in agg.iterrows():
                col = cont_to_col.get(r["feature_name"])
                if col is None:
                    continue
                edges = edge_lookup.get(r["feature_name"])
                if edges is None:
                    continue
                bin_id = int(quantize(np.asarray([r["value"]]), edges)[0])
                cont[int(r["bin"]), col] = bin_id
                valid[int(r["bin"]), F_cat + col] = True

        time_index = np.arange(T, dtype=np.int32)
        np.savez_compressed(
            Path(out_dir) / f"{int(stay_id)}.npz",
            cat=cat,
            cont=cont,
            time_index=time_index,
            valid_mask=valid,
        )
        n_written += 1

    log.info(f"Wrote {n_written:,} tokenized stays → {out_dir}")


if __name__ == "__main__":
    main()
