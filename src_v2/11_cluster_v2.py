"""Stage 11 v2 — unsupervised clustering analysis of patient embeddings.

The clustering leg of the downstream eval: group patients by embedding (no
labels), then characterise each cluster clinically, to see whether coherent
phenotypes emerge. Runs identically on v1 or v2 so they can be compared.

For each KMeans cluster it reports:
  - size, mean age, in-hospital mortality, mean LOS
  - disease ENRICHMENT (cluster rate / cohort rate) for AMI / stroke / MASLD

It also reports cluster PURITY: how widely each label's positive-rate varies
across clusters (a collapsed embedding -> clusters differ on acuity only, so
mortality/LOS vary a lot but specific diseases barely vary; a good embedding
-> disease rates also vary across clusters).

Run:
    python src_v2/11_cluster_v2.py --emb outputs/downstream_v2/small_ce_embeddings.npz --tag ce --k 6
    python src_v2/11_cluster_v2.py --emb v1 --tag v1 --k 6
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
DISEASES = ["ami", "stroke", "masld", "celiac"]


def load_emb(spec):
    if spec == "v1":
        Xs, sids = [], []
        for split in ["train", "val", "test"]:
            d = np.load(WORK / "embeddings" / f"{split}_embeddings.npz")
            Xs.append(d["embeddings"].astype(np.float32)); sids.append(d["stay_ids"].astype(np.int64))
        return np.concatenate(Xs), np.concatenate(sids)
    d = np.load(spec)
    return d["embeddings"].astype(np.float32), d["stay_ids"].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    X, sids = load_emb(args.emb)
    meta = pd.DataFrame({"stay_id_v1": sids, "row_idx": np.arange(len(sids))})

    v1c = pd.read_parquet(WORK / "cohort" / "cohort.parquet")[["stay_id", "hadm_id"]].rename(
        columns={"stay_id": "stay_id_v1", "hadm_id": "hospitalization_id"})
    v2c = pd.read_parquet(WORK / "cohort_v2" / "cohort.parquet")[
        ["hospitalization_id", "stay_id", "age_at_admission", "died_in_hosp", "hospital_los_hours"]]
    labels = pd.read_parquet(WORK / "downstream_v2" / "labels.parquet")[
        ["stay_id"] + [f"y_{d}" for d in DISEASES]]
    meta = (meta.merge(v1c, on="stay_id_v1", how="inner")
                .merge(v2c, on="hospitalization_id", how="inner")
                .merge(labels, on="stay_id", how="inner"))
    X = X[meta["row_idx"].values]
    print(f"[{args.tag}] aligned {len(meta):,} stays")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=args.k, random_state=0, n_init=10).fit(Xs)
    meta["cluster"] = km.labels_
    sil = silhouette_score(Xs[:20000], km.labels_[:20000])  # subsample for speed

    overall = {d: meta[f"y_{d}"].mean() for d in DISEASES}
    rows = []
    for c in range(args.k):
        sub = meta[meta["cluster"] == c]
        row = {"cluster": c, "n": len(sub),
               "mean_age": round(sub["age_at_admission"].mean(), 1),
               "mortality_%": round(sub["died_in_hosp"].mean() * 100, 1),
               "mean_los_h": round(sub["hospital_los_hours"].mean(), 1)}
        for d in DISEASES:
            rate = sub[f"y_{d}"].mean()
            row[f"{d}_enrich"] = round(rate / overall[d], 2) if overall[d] > 0 else np.nan
        rows.append(row)
    prof = pd.DataFrame(rows).sort_values("mortality_%", ascending=False)
    prof.to_csv(WORK / "downstream_v2" / f"cluster_profile_{args.tag}.csv", index=False)

    # Purity: spread of per-cluster positive rate across clusters (std), for each label
    pur_rows = []
    for lab in ["died_in_hosp"] + [f"y_{d}" for d in DISEASES]:
        rates = meta.groupby("cluster")[lab].mean()
        pur_rows.append({"label": lab, "min": round(rates.min(), 4), "max": round(rates.max(), 4),
                         "spread_std": round(rates.std(), 4)})
    pur = pd.DataFrame(pur_rows)

    print(f"[{args.tag}] k={args.k} silhouette={sil:.3f}")
    print(f"\n[{args.tag}] cluster profiles (enrich = cluster_rate / cohort_rate):")
    print(prof.to_string(index=False))
    print(f"\n[{args.tag}] cluster purity (how much each label varies across clusters):")
    print(pur.to_string(index=False))
    print(f"\n[{args.tag}] wrote cluster_profile_{args.tag}.csv")


if __name__ == "__main__":
    main()
