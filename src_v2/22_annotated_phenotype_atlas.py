"""Stage 22 v2 — annotated phenotype atlas (APOLLO Fig 3a style with callouts).

UMAP of patient embeddings coloured by the 6 discovered phenotypes, with a
callout box drawn out from each cluster describing it (mortality, age, any
enriched disease). Like APOLLO's annotated patient atlas, minus trajectories.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
DIS = ["ami", "stroke", "masld"]
K = 6


def fmt_mortality(pct: float) -> str:
    """Avoid over-interpreting tiny rounded mortality rates as exactly zero."""
    if pct < 0.5:
        return "<1%"
    return f"{pct:.0f}%"


def main():
    import umap
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    e = np.load(WORK/"downstream_v2"/"small_ce_embeddings.npz")
    X, sids = e["embeddings"].astype(np.float32), e["stay_ids"].astype(np.int64)
    m = pd.DataFrame({"stay_id_v1": sids, "row": np.arange(len(sids))})
    v1c = pd.read_parquet(WORK/"cohort"/"cohort.parquet")[["stay_id","hadm_id"]].rename(columns={"stay_id":"stay_id_v1","hadm_id":"hospitalization_id"})
    v2c = pd.read_parquet(WORK/"cohort_v2"/"cohort.parquet")[["hospitalization_id","stay_id","age_at_admission"]]
    surv = pd.read_parquet(WORK/"downstream_v2"/"survival_labels.parquet")[["stay_id","surv_event"]]
    lab = pd.read_parquet(WORK/"downstream_v2"/"labels.parquet")[["stay_id"]+[f"y_{d}" for d in DIS]]
    m = m.merge(v1c,on="stay_id_v1").merge(v2c,on="hospitalization_id").merge(surv,on="stay_id").merge(lab,on="stay_id")
    X = X[m["row"].values]
    Xs = StandardScaler().fit_transform(X)
    m["pheno"] = KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(Xs)

    overall = {d: m[f"y_{d}"].mean() for d in DIS}
    desc = {}
    for c in range(K):
        s = m[m["pheno"]==c]
        mort = s["surv_event"].mean()*100; age = s["age_at_admission"].mean()
        enr = {d: s[f"y_{d}"].mean()/overall[d] for d in DIS}
        top = max(enr, key=enr.get)
        dtag = f", {top} {enr[top]:.1f}x" if enr[top] > 1.3 else ""
        if mort >= 20: head = "Critically ill"
        elif mort >= 10: head = "High risk"
        elif mort >= 4: head = "Moderate risk"
        else: head = "Low risk"
        desc[c] = f"P{c}: {head}, n={len(s):,}\n{fmt_mortality(mort)} mortality, age {age:.0f}{dtag}"

    rng = np.random.default_rng(0); idx = rng.choice(len(m), min(15000,len(m)), replace=False)
    U = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(Xs[idx])
    ph = m["pheno"].values[idx]
    cmap = plt.get_cmap("tab10")
    ctr = U.mean(0)

    # short legend labels (like the original): "P{c}: mort X%[, disease Nx]"
    leg = {}
    for c in range(K):
        s = m[m["pheno"]==c]; mort = s["surv_event"].mean()*100
        enr = {d: s[f"y_{d}"].mean()/overall[d] for d in DIS}; top = max(enr, key=enr.get)
        dtag = f", {top} {enr[top]:.1f}x" if enr[top] > 1.3 else ""
        leg[c] = f"P{c}: n={len(s):,}, mort {fmt_mortality(mort)}{dtag}"

    fig, ax = plt.subplots(figsize=(11, 8.5))
    for c in range(K):
        mm = ph==c
        ax.scatter(U[mm,0], U[mm,1], s=4, color=cmap(c), alpha=0.55, linewidths=0, label=leg[c])
    ax.legend(loc="upper left", fontsize=9, markerscale=2.5, framealpha=0.9)
    # Callouts are placed in fixed axes-relative positions so the labels stay
    # legible and do not collide when the figure is inserted into a short report.
    callout_xytext = {
        0: (0.53, -0.055),
        1: (0.92, 0.72),
        2: (0.72, 0.92),
        3: (0.70, 0.035),
        4: (0.80, -0.055),
        5: (0.30, 0.22),
    }
    span = (U.max(0)-U.min(0))
    for c in range(K):
        cen = np.median(U[ph==c], axis=0)
        ax.annotate(desc[c], xy=cen, xytext=callout_xytext[c],
                    textcoords="axes fraction",
                    fontsize=9, color="black", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=cmap(c), lw=1.5),
                    arrowprops=dict(arrowstyle="-", color=cmap(c), lw=1.2),
                    annotation_clip=False)
    ax.set_xticks([]); ax.set_yticks([])
    pad = span*0.36
    ax.set_xlim(U[:,0].min()-pad[0], U[:,0].max()+pad[0])
    ax.set_ylim(U[:,1].min()-pad[1], U[:,1].max()+pad[1])
    ax.set_title("Patient atlas: unsupervised phenotypes with clinical annotation")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12)
    fig.savefig(WORK/"final_results"/"annotated_phenotype_atlas.png", dpi=130, bbox_inches="tight", pad_inches=0.25)
    print("done"); [print(desc[c].replace(chr(10)," | ")) for c in range(K)]


if __name__ == "__main__":
    main()
