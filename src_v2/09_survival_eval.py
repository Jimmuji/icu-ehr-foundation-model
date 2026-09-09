"""Stage 09 v2 — survival (time-to-event) eval on FROZEN embeddings.

Protocol (the survival analog of the linear probe):
  1. Load patient embeddings (frozen encoder output).
  2. PCA -> 32 dims (fit on train; Cox is unstable with hundreds of covariates).
  3. Fit a Cox proportional-hazards model on the TRAIN split.
  4. Report Harrell's C-index on the TEMPORAL TEST split.

C-index is the survival analog of AUROC: the probability the model ranks
"who has the event sooner" correctly. 0.5 = chance.

Compares any number of embedding sets so v1 vs the small models line up.

Run (after 07 has produced the small-model embeddings):
    python src_v2/09_survival_eval.py --emb v1 --tag v1
    python src_v2/09_survival_eval.py --emb outputs/downstream_v2/small_ce_embeddings.npz --tag ce
    python src_v2/09_survival_eval.py --emb outputs/downstream_v2/small_mse_embeddings.npz --tag mse
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")


def load_embeddings(spec):
    """spec == 'v1'  -> concat work/embeddings/{train,val,test}_embeddings.npz
       else          -> a single npz with keys 'embeddings','stay_ids'."""
    if spec == "v1":
        Xs, sids = [], []
        for split in ["train", "val", "test"]:
            d = np.load(WORK / "embeddings" / f"{split}_embeddings.npz")
            Xs.append(d["embeddings"].astype(np.float32))
            sids.append(d["stay_ids"].astype(np.int64))
        return np.concatenate(Xs), np.concatenate(sids)
    d = np.load(spec)
    return d["embeddings"].astype(np.float32), d["stay_ids"].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="'v1' or path to a *_embeddings.npz")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pca", type=int, default=32)
    args = ap.parse_args()

    X, sids = load_embeddings(args.emb)
    meta = pd.DataFrame({"stay_id_v1": sids, "row_idx": np.arange(len(sids))})

    # Map v1 stay_id -> hospitalization_id -> v2 stay_id -> survival labels + split
    v1c = pd.read_parquet(WORK / "cohort" / "cohort.parquet")[["stay_id", "hadm_id"]].rename(
        columns={"stay_id": "stay_id_v1", "hadm_id": "hospitalization_id"})
    v2c = pd.read_parquet(WORK / "cohort_v2" / "cohort.parquet")[["hospitalization_id", "stay_id"]]
    surv = pd.read_parquet(WORK / "downstream_v2" / "survival_labels.parquet")
    meta = (meta.merge(v1c, on="stay_id_v1", how="inner")
                .merge(v2c, on="hospitalization_id", how="inner")
                .merge(surv, on="stay_id", how="inner"))
    X = X[meta["row_idx"].values]
    splits = meta["split"].values
    print(f"[{args.tag}] aligned {X.shape[0]:,} stays to survival labels | X dim {X.shape[1]}")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    tr, te = splits == "train", splits == "test"
    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)
    pca = PCA(n_components=min(args.pca, X.shape[1])).fit(Xs[tr])
    Z = pca.transform(Xs)

    cols = [f"z{i}" for i in range(Z.shape[1])]
    tr_df = pd.DataFrame(Z[tr], columns=cols)
    tr_df["time"] = meta.loc[tr, "surv_time_hours"].values
    tr_df["event"] = meta.loc[tr, "surv_event"].values
    te_df = pd.DataFrame(Z[te], columns=cols)
    te_df["time"] = meta.loc[te, "surv_time_hours"].values
    te_df["event"] = meta.loc[te, "surv_event"].values

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(tr_df, duration_col="time", event_col="event")

    # C-index: higher partial hazard => earlier event, so negate for concordance
    risk_tr = cph.predict_partial_hazard(tr_df)
    risk_te = cph.predict_partial_hazard(te_df)
    c_tr = concordance_index(tr_df["time"], -risk_tr, tr_df["event"])
    c_te = concordance_index(te_df["time"], -risk_te, te_df["event"])
    print(f"[{args.tag}] in-hospital survival C-index: train={c_tr:.3f} | TEST={c_te:.3f} "
          f"(test events={int(te_df['event'].sum())}/{len(te_df)})")

    # Append to a shared results CSV
    res_path = WORK / "downstream_v2" / "survival_results.csv"
    row = pd.DataFrame([{"tag": args.tag, "task": "in_hosp_survival",
                         "c_index_train": round(c_tr, 4), "c_index_test": round(c_te, 4),
                         "test_events": int(te_df["event"].sum()), "test_n": len(te_df)}])
    if res_path.exists():
        prev = pd.read_csv(res_path)
        prev = prev[prev["tag"] != args.tag]
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(res_path, index=False)
    print(f"[{args.tag}] wrote {res_path}")


if __name__ == "__main__":
    main()
