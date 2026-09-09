"""Stage 12 v2 — Kaplan-Meier survival by discovered phenotype + log-rank test.

Clinical-statistics payoff of the v2 representation: the phenotypes found by
unsupervised clustering have statistically distinct survival trajectories.

  1. KMeans(k) on the v2-CE embedding -> phenotype per stay.
  2. Kaplan-Meier survival curve per phenotype (in-hospital survival,
     survivors right-censored at discharge).
  3. Multivariate log-rank test: are the curves significantly different?
  4. Each phenotype gets a short data-driven descriptor for the legend.

Run:
    python src_v2/12_km_survival.py --k 6
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
DISEASES = ["ami", "stroke", "masld"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--cap-hours", type=float, default=720.0, help="x-axis cap (30d)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    e = np.load(WORK / "downstream_v2" / "small_ce_embeddings.npz")
    X, sids = e["embeddings"].astype(np.float32), e["stay_ids"].astype(np.int64)
    meta = pd.DataFrame({"stay_id_v1": sids, "row_idx": np.arange(len(sids))})

    v1c = pd.read_parquet(WORK / "cohort" / "cohort.parquet")[["stay_id", "hadm_id"]].rename(
        columns={"stay_id": "stay_id_v1", "hadm_id": "hospitalization_id"})
    v2c = pd.read_parquet(WORK / "cohort_v2" / "cohort.parquet")[
        ["hospitalization_id", "stay_id", "age_at_admission"]]
    surv = pd.read_parquet(WORK / "downstream_v2" / "survival_labels.parquet")
    labels = pd.read_parquet(WORK / "downstream_v2" / "labels.parquet")[["stay_id"] + [f"y_{d}" for d in DISEASES]]
    meta = (meta.merge(v1c, on="stay_id_v1", how="inner")
                .merge(v2c, on="hospitalization_id", how="inner")
                .merge(surv, on="stay_id", how="inner")
                .merge(labels, on="stay_id", how="inner"))
    X = X[meta["row_idx"].values]
    print(f"aligned {len(meta):,} stays")

    Xs = StandardScaler().fit_transform(X)
    meta["cluster"] = KMeans(n_clusters=args.k, random_state=0, n_init=10).fit_predict(Xs)

    # Data-driven descriptor per cluster
    overall = {d: meta[f"y_{d}"].mean() for d in DISEASES}
    desc = {}
    for c in range(args.k):
        sub = meta[meta["cluster"] == c]
        mort = sub["surv_event"].mean() * 100
        enr = {d: (sub[f"y_{d}"].mean() / overall[d]) for d in DISEASES}
        top = max(enr, key=enr.get)
        tag = f"{top}↑{enr[top]:.1f}x" if enr[top] > 1.3 else "mixed"
        desc[c] = f"C{c} (n={len(sub)}, mort {mort:.0f}%, {tag})"

    # Log-rank across clusters
    lr = multivariate_logrank_test(meta["surv_time_hours"], meta["cluster"], meta["surv_event"])
    print(f"\nMultivariate log-rank: chi2={lr.test_statistic:.1f}, p={lr.p_value:.2e}")

    # KM plot
    fig, ax = plt.subplots(figsize=(9, 6))
    kmf = KaplanMeierFitter()
    summary = []
    for c in sorted(meta["cluster"].unique()):
        m = meta["cluster"] == c
        kmf.fit(meta.loc[m, "surv_time_hours"].clip(upper=args.cap_hours),
                meta.loc[m, "surv_event"], label=desc[c])
        kmf.plot_survival_function(ax=ax, ci_show=False)
        summary.append({"cluster": int(c), "n": int(m.sum()),
                        "mortality_%": round(meta.loc[m, "surv_event"].mean() * 100, 1),
                        "mean_age": round(meta.loc[m, "age_at_admission"].mean(), 1),
                        "surv_at_30d": round(float(kmf.predict(args.cap_hours)), 3)})
    ax.set_title(f"In-hospital survival by phenotype (v2-CE, k={args.k})\n"
                 f"log-rank p = {lr.p_value:.1e}", fontsize=12)
    ax.set_xlabel("hours since admission (capped at 30 days)")
    ax.set_ylabel("survival probability")
    ax.set_ylim(0.6, 1.005)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = WORK / "final_results" / "km_survival_by_phenotype.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")

    sdf = pd.DataFrame(summary).sort_values("mortality_%", ascending=False)
    sdf["logrank_p"] = lr.p_value
    sdf.to_csv(WORK / "downstream_v2" / "km_phenotype_summary.csv", index=False)
    print(f"\n{sdf.to_string(index=False)}")
    print(f"\nwrote {out} and km_phenotype_summary.csv")


if __name__ == "__main__":
    main()
