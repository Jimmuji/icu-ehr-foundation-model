"""Stage 20 v2 — annotated patient atlas (APOLLO Fig 3a style).

UMAP of patient embeddings with disease-enriched regions highlighted and
labelled, like APOLLO's annotated patient atlas. Background = all patients in
grey; for a few conditions, the patients carrying that diagnosis are coloured,
and a label is placed at the centre of their region.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
CLIF = Path("/data/mimic_data/physionet.org/files/mimic-iv-ext-clif-data/1.1.0")
N_SUB = 15000

# label -> ICD-10 prefixes
DISEASES = {
    "Sepsis": ("A41",), "Heart failure": ("I50",), "Stroke": ("I63",),
    "Heart attack": ("I21",), "Acute kidney injury": ("N17",),
    "Respiratory failure": ("J96",), "Liver disease": ("K70", "K72", "K74", "K76"),
}
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]


def main():
    import umap
    from sklearn.preprocessing import StandardScaler
    e = np.load(WORK / "downstream_v2" / "small_ce_embeddings.npz")
    X, sids = e["embeddings"].astype(np.float32), e["stay_ids"].astype(np.int64)
    m = pd.DataFrame({"stay_id_v1": sids, "row": np.arange(len(sids))})
    v1c = pd.read_parquet(WORK/"cohort"/"cohort.parquet")[["stay_id","hadm_id"]].rename(
        columns={"stay_id":"stay_id_v1","hadm_id":"hospitalization_id"})
    m = m.merge(v1c, on="stay_id_v1", how="inner")
    m["hospitalization_id"] = m["hospitalization_id"].astype(str)

    dx = pd.read_parquet(CLIF/"clif_hospital_diagnosis.parquet", columns=["hospitalization_id","diagnosis_code"])
    dx["hospitalization_id"] = dx["hospitalization_id"].astype(str)
    dx["code"] = dx["diagnosis_code"].astype(str).str.upper().str.replace(".","",regex=False)
    has = {}
    for name, prefs in DISEASES.items():
        hadms = dx.loc[dx["code"].str.startswith(prefs), "hospitalization_id"].unique()
        has[name] = set(hadms)

    Xs = StandardScaler().fit_transform(X[m["row"].values])
    rng = np.random.default_rng(0); idx = rng.choice(len(m), min(N_SUB, len(m)), replace=False)
    U = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(Xs[idx])
    sub = m.iloc[idx].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(U[:,0], U[:,1], s=4, color="lightgray", alpha=0.4, linewidths=0)
    for (name, prefs), col in zip(DISEASES.items(), COLORS):
        mask = sub["hospitalization_id"].isin(has[name]).values
        if mask.sum() < 20:
            continue
        ax.scatter(U[mask,0], U[mask,1], s=7, color=col, alpha=0.55, linewidths=0)
        # label at the densest area: use median of positives
        cx, cy = np.median(U[mask,0]), np.median(U[mask,1])
        ax.annotate(f"{name}\n(n={mask.sum()})", (cx, cy), fontsize=10, fontweight="bold",
                    color=col, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Patient atlas with disease-enriched regions highlighted")
    fig.tight_layout(); fig.savefig(WORK/"final_results"/"annotated_atlas.png", dpi=130, bbox_inches="tight")
    print("annotated atlas done")
    for name in DISEASES:
        mask = sub["hospitalization_id"].isin(has[name]).values
        print(f"  {name}: {mask.sum()} patients in subsample")


if __name__ == "__main__":
    main()
