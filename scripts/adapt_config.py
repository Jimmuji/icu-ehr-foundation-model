"""Fill in MIMIC-specific fields in pretrain_mimic.json from preprocessing outputs.

Reads:
  outputs/vocab/feature_index.parquet  →  n_category_feats, n_float_feats
  outputs/vocab/vocab_cat.json         →  n_category_values (max per-feature vocab size)
  outputs/vocab/bin_edges.parquet      →  (sanity check on continuous count)
  configs/preprocess.yaml              →  seq_max_len

Writes:
  configs/pretrain_mimic.json          →  (in place: fills n_category_feats etc)
  outputs/feat_info.json               →  feat_info schema upstream EHRModule expects:
                                          { "category_cols": [...], "float_cols": {feat: meta} }

Run AFTER stage 03 (build vocab) has produced its outputs.

Usage:
    python scripts/adapt_config.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "preprocess.yaml"))
    ap.add_argument("--pretrain-config", default=str(REPO / "configs" / "pretrain_mimic.json"))
    ap.add_argument("--feat-info-out", default=str(REPO / "outputs" / "feat_info.json"))
    args = ap.parse_args()

    # Load preprocess config to find work_dir
    with open(args.config) as f:
        pre_cfg = yaml.safe_load(f)
    work_dir = Path(pre_cfg["paths"]["work_dir"])
    if not work_dir.is_absolute():
        work_dir = REPO / work_dir

    vocab_dir = work_dir / "vocab"
    feat_idx = pd.read_parquet(vocab_dir / "feature_index.parquet")
    with open(vocab_dir / "vocab_cat.json") as f:
        cat_vocab = json.load(f)
    bin_edges = pd.read_parquet(vocab_dir / "bin_edges.parquet")

    cat_feats = feat_idx[feat_idx["feature_kind"] == "cat"]["feature_name"].tolist()
    cont_feats = feat_idx[feat_idx["feature_kind"] == "cont"]["feature_name"].tolist()

    # n_category_values = max vocab size across features (upstream uses a single shared int range)
    max_cat_vocab = max((len(v) for v in cat_vocab.values()), default=0)

    # --- Build feat_info.json (schema expected by upstream EHRModule) ---
    feat_info = {
        "category_cols": sorted(cat_feats),
        "float_cols": {
            row["feature_name"]: {
                "n_obs": int(row["n_obs"]),
                "n_bins": len(row["edges"]) - 1,
            }
            for _, row in bin_edges.iterrows()
        },
    }
    Path(args.feat_info_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.feat_info_out, "w") as f:
        json.dump(feat_info, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.feat_info_out} (cat={len(cat_feats)}, cont={len(cont_feats)})")

    # --- Patch pretrain_mimic.json with measured sizes ---
    with open(args.pretrain_config) as f:
        pcfg = json.load(f)
    pcfg["n_category_feats"] = len(cat_feats)
    pcfg["n_float_feats"] = len(cont_feats)
    pcfg["n_category_values"] = max_cat_vocab
    pcfg["seq_max_len"] = int(pre_cfg["time_binning"]["max_seq_len"])
    pcfg["feat_info_path"] = args.feat_info_out
    pcfg["float_feats"] = str(vocab_dir / "bin_edges.parquet")
    pcfg["df_paths"] = str(work_dir / "chunks")

    with open(args.pretrain_config, "w") as f:
        json.dump(pcfg, f, indent=2)
    print(f"Patched {args.pretrain_config}:")
    for k in ["n_category_feats", "n_float_feats", "n_category_values",
              "n_float_values", "seq_max_len", "batch_size", "n_epoch", "n_gpus"]:
        print(f"  {k}: {pcfg[k]}")


if __name__ == "__main__":
    main()
