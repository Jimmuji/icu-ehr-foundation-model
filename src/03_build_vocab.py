"""Stage 03 — build feature/value vocabularies and quantile bin edges.

Two artifacts:
1. `feature_index.parquet`  — every feature_name we see → integer id (separately for
   categorical and continuous), plus its kind.
2. `vocab_cat.json`         — mapping `{feature_name: {category_value: int_id}}`,
                              with reserved IDs (PAD/UNK/MASK) and frequency cutoff.
3. `bin_edges.parquet`      — for each continuous feature, 257 bin edges (n_bins+1).

Why split feature_index from vocab_cat: EHRFormer treats *features* as separate
embedding tables — the model needs to know how many features and their kinds before
seeing values.

Run:
    python src/03_build_vocab.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_config, setup_logging, stage_dir


def build_split_assignment(cfg: dict, stay_ids: pd.Series, cohort: pd.DataFrame) -> pd.DataFrame:
    """Produce a stay_id → split DataFrame, splitting *by patient* to avoid leakage."""
    rng = np.random.default_rng(cfg["split"]["seed"])
    sub_to_stays = cohort.groupby("subject_id")["stay_id"].apply(list)
    subjects = sub_to_stays.index.to_numpy()
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = int(n * cfg["split"]["test_frac"])
    n_val = int(n * cfg["split"]["val_frac"])
    test_subj = set(subjects[:n_test])
    val_subj = set(subjects[n_test : n_test + n_val])

    rows = []
    for subj, stays in sub_to_stays.items():
        if subj in test_subj:
            split = "test"
        elif subj in val_subj:
            split = "val"
        else:
            split = "train"
        for s in stays:
            rows.append((s, subj, split))
    return pd.DataFrame(rows, columns=["stay_id", "subject_id", "split"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("03_vocab", cfg["paths"]["work_dir"])
    out_dir = stage_dir(cfg, "vocab")
    work = Path(cfg["paths"]["work_dir"])

    cohort = pd.read_parquet(work / "cohort" / "cohort.parquet")
    events = pd.read_parquet(work / "events" / "events.parquet")
    log.info(f"Loaded cohort {len(cohort):,}, events {len(events):,}")

    # --- Patient-level split first; vocab is computed on TRAIN only ---
    splits = build_split_assignment(cfg, events["stay_id"], cohort)
    splits.to_parquet(out_dir / "splits.parquet", index=False)
    log.info(f"Splits: {splits['split'].value_counts().to_dict()}")

    train_stays = set(splits.loc[splits["split"] == "train", "stay_id"].astype(int))
    train_events = events[events["stay_id"].isin(train_stays)]
    log.info(f"Train events: {len(train_events):,}")

    # --- Feature index: every distinct feature_name → int id, plus its kind ---
    feat = (
        train_events.groupby(["feature_name", "feature_kind"])
        .size().rename("freq").reset_index()
        .sort_values(["feature_kind", "feature_name"])
        .reset_index(drop=True)
    )
    feat["feature_id"] = np.arange(len(feat))
    feat.to_parquet(out_dir / "feature_index.parquet", index=False)
    log.info(f"Features: {len(feat):,} ({(feat.feature_kind=='cat').sum()} cat / "
             f"{(feat.feature_kind=='cont').sum()} cont)")

    # --- Categorical vocab (per-feature value→id) ---
    reserved = cfg["vocab"]["reserved"]
    pad_id = cfg["vocab"]["pad_token_id"]
    unk_id = cfg["vocab"]["unk_token_id"]
    mask_id = cfg["vocab"]["mask_token_id"]
    min_freq = cfg["vocab"]["min_frequency"]

    cat_events = train_events[train_events["feature_kind"] == "cat"]
    cat_vocab: dict[str, dict[str, int]] = {}
    for fname, grp in cat_events.groupby("feature_name"):
        counts = grp["value"].astype(str).value_counts()
        vocab = {"<PAD>": pad_id, "<UNK>": unk_id, "<MASK>": mask_id}
        next_id = reserved
        for v, c in counts.items():
            if c < min_freq:
                continue
            vocab[v] = next_id
            next_id += 1
        cat_vocab[fname] = vocab
    with open(out_dir / "vocab_cat.json", "w") as f:
        json.dump(cat_vocab, f, ensure_ascii=False, indent=2)
    log.info(f"Categorical vocabularies: {len(cat_vocab)} features, "
             f"avg vocab size {np.mean([len(v) for v in cat_vocab.values()]):.1f}")

    # --- Continuous bin edges (per feature, computed on train) ---
    nb = cfg["quantization"]["n_bins"]
    lo = cfg["quantization"]["clip_quantile_lo"]
    hi = cfg["quantization"]["clip_quantile_hi"]
    cont_events = train_events[train_events["feature_kind"] == "cont"].copy()
    cont_events["value"] = pd.to_numeric(cont_events["value"], errors="coerce")
    cont_events = cont_events.dropna(subset=["value"])

    edge_rows = []
    if cfg["quantization"]["strategy"] == "quantile":
        qs = np.linspace(lo, hi, nb + 1)
    else:
        qs = None  # uniform handled below

    for fname, grp in cont_events.groupby("feature_name"):
        v = grp["value"].to_numpy(dtype=np.float64)
        if len(v) < 10:
            continue
        v_lo, v_hi = np.quantile(v, [lo, hi])
        v = np.clip(v, v_lo, v_hi)
        if qs is not None:
            edges = np.quantile(v, qs)
        else:
            edges = np.linspace(v.min(), v.max(), nb + 1)
        # Make edges strictly increasing
        edges = np.maximum.accumulate(edges + np.arange(len(edges)) * 1e-12)
        edge_rows.append({
            "feature_name": fname,
            "n_obs": int(len(v)),
            "edges": edges.tolist(),
        })
    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_parquet(out_dir / "bin_edges.parquet", index=False)
    log.info(f"Bin edges built for {len(edge_df)} continuous features (n_bins={nb})")


if __name__ == "__main__":
    main()
