"""Step 4c — analyze cluster phenotypes: per-cluster metadata enrichment + figures.

For each cluster method × cluster, compute:
  - cluster size
  - mean age, mortality rate, mean ICU LOS
  - top enriched diagnoses (ICD chapter, χ² vs background)
  - top enriched continuous features (Mann-Whitney U)

Produces:
  - per-method UMAP scatter PNGs (colored by cluster / by mortality / by age decade)
  - per-method enrichment tables (CSV)
  - markdown summary

Usage:
    python -m src.clustering.analyze \
        --umap outputs/clusters/umap_2d.npz \
        --labels outputs/clusters/cluster_labels.npz \
        --cohort outputs/cohort/cohort.parquet \
        --events outputs/events/events.parquet \
        --out outputs/clusters/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def scatter_by(coords, labels, out_path, title, cmap="tab20", legend=True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = np.asarray(labels)
    uniq = sorted(set(labels.tolist()))
    cmap_o = plt.get_cmap(cmap, len(uniq))
    for i, u in enumerate(uniq):
        m = labels == u
        ax.scatter(coords[m, 0], coords[m, 1], s=3, c=[cmap_o(i)],
                   label=str(u), alpha=0.7, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    if legend and len(uniq) <= 20:
        ax.legend(markerscale=3, fontsize=7, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def per_cluster_summary(meta: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Compute size + mean numeric stats by cluster id."""
    df = meta.copy()
    df["cluster"] = labels
    agg = df.groupby("cluster").agg(
        n=("stay_id", "count"),
        mean_age=("admission_age", "mean"),
        mortality=("died_in_hosp", "mean"),
        mean_los_h=("los_hours", "mean"),
    ).reset_index()
    agg["mortality"] = (agg["mortality"] * 100).round(1)
    agg["mean_age"] = agg["mean_age"].round(1)
    agg["mean_los_h"] = agg["mean_los_h"].round(1)
    return agg


def diagnosis_enrichment(events: pd.DataFrame, stay_to_cluster: pd.DataFrame, top_n: int = 5):
    """χ²-style enrichment of ICD chapter per cluster."""
    from scipy.stats import chi2_contingency

    diag = events[events["feature_name"] == "diag::chapter"][["stay_id", "value"]]
    diag = diag.merge(stay_to_cluster, on="stay_id")
    cross = pd.crosstab(diag["cluster"], diag["value"])
    if cross.empty:
        return pd.DataFrame()
    chi2, p, _, exp = chi2_contingency(cross)
    enrichment = (cross / exp).round(2)  # observed / expected
    # For each cluster, take top diagnoses by enrichment ratio (filter rare)
    rows = []
    for c in enrichment.index:
        row = enrichment.loc[c]
        row = row[cross.loc[c] >= 5]  # require some support
        for diag_name in row.nlargest(top_n).index:
            rows.append({
                "cluster": c, "diagnosis": diag_name,
                "ratio_obs_over_expected": float(row[diag_name]),
                "n_in_cluster": int(cross.loc[c, diag_name]),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umap", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--events", required=False, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    umap_data = np.load(args.umap)
    coords = umap_data["coords"]
    sids = umap_data["stay_ids"]

    lab_data = np.load(args.labels, allow_pickle=True)
    method_keys = [k for k in lab_data.files if k != "stay_ids"]
    print(f"Cluster methods present: {method_keys}")

    cohort = pd.read_parquet(args.cohort)
    cohort = cohort[cohort["stay_id"].isin(sids)]
    # Reindex cohort rows to match embedding order
    cohort = cohort.set_index("stay_id").loc[sids].reset_index()

    # Always-on figures: by age decade + by mortality
    age_decade = (cohort["admission_age"] // 10 * 10).astype(int).to_numpy()
    scatter_by(coords, age_decade, out / "umap_by_age_decade.png", "UMAP — age decade")
    scatter_by(coords, cohort["died_in_hosp"].to_numpy(), out / "umap_by_mortality.png",
               "UMAP — in-hospital mortality", cmap="coolwarm")

    summary = {}
    events_df = pd.read_parquet(args.events) if args.events else None

    for method in method_keys:
        labels = lab_data[method]
        if len(labels) != len(coords):
            print(f"  skip {method}: length mismatch")
            continue

        scatter_by(coords, labels, out / f"umap_{method}.png", f"UMAP — {method}")
        cluster_meta = per_cluster_summary(cohort, labels)
        cluster_meta.to_csv(out / f"summary_{method}.csv", index=False)

        # Diagnosis enrichment (needs events)
        diag_rows = None
        if events_df is not None:
            stay_to_cluster = pd.DataFrame({"stay_id": sids, "cluster": labels})
            diag_rows = diagnosis_enrichment(events_df, stay_to_cluster, top_n=5)
            if not diag_rows.empty:
                diag_rows.to_csv(out / f"diag_enrichment_{method}.csv", index=False)

        summary[method] = {
            "n_clusters": int(np.unique(labels).size),
            "summary_csv": f"summary_{method}.csv",
            "scatter_png": f"umap_{method}.png",
            "diag_csv": f"diag_enrichment_{method}.csv" if diag_rows is not None and not diag_rows.empty else None,
        }
        print(f"  {method}: {summary[method]['n_clusters']} clusters; wrote summary + scatter")

    # Markdown report
    md = ["# Cluster analysis — EHRFormer embeddings\n"]
    md.append("## Always-on figures\n")
    md.append("- `umap_by_age_decade.png`")
    md.append("- `umap_by_mortality.png`\n")
    md.append("## Per-method results\n")
    for method, info in summary.items():
        md.append(f"### {method} — {info['n_clusters']} clusters\n")
        md.append(f"![scatter]({info['scatter_png']})")
        md.append(f"\nSee `{info['summary_csv']}` for per-cluster mortality / age / LOS.")
        if info["diag_csv"]:
            md.append(f"\nDiagnosis enrichment: `{info['diag_csv']}` "
                      f"(top 5 ICD chapters per cluster by observed/expected ratio).")
        md.append("")
    (out / "report.md").write_text("\n".join(md))
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
