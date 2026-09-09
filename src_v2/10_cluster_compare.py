"""Stage 10 v2 — v1 vs v2 clustering comparison figure.

Visualises the collapse and its fix on the SAME patients:
  rows = colour by {in-hospital mortality, AMI diagnosis}
  cols = {v1 (228M VAE), v2-CE (2M, VAE-free)}

Reading:
  - v1 colour-by-mortality is a smooth gradient and colour-by-AMI is
    scattered  -> the embedding is organised by acuity only (collapsed).
  - v2 shows AMI patients grouping together and richer structure beyond a
    single axis -> disease identity is now encoded.

Run:
    python src_v2/10_cluster_compare.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
N_SUB = 15000
SEED = 0


def load_v1():
    Xs, sids = [], []
    for split in ["train", "val", "test"]:
        d = np.load(WORK / "embeddings" / f"{split}_embeddings.npz")
        Xs.append(d["embeddings"].astype(np.float32))
        sids.append(d["stay_ids"].astype(np.int64))
    return np.concatenate(Xs), np.concatenate(sids)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    from sklearn.preprocessing import StandardScaler

    X1, s1 = load_v1()
    e = np.load(WORK / "downstream_v2" / "small_ce_embeddings.npz")
    X2, s2 = e["embeddings"].astype(np.float32), e["stay_ids"].astype(np.int64)

    m1 = pd.DataFrame({"stay_id_v1": s1, "i1": np.arange(len(s1))})
    m2 = pd.DataFrame({"stay_id_v1": s2, "i2": np.arange(len(s2))})

    # labels
    v1c = pd.read_parquet(WORK / "cohort" / "cohort.parquet")[["stay_id", "hadm_id"]].rename(
        columns={"stay_id": "stay_id_v1", "hadm_id": "hospitalization_id"})
    v2c = pd.read_parquet(WORK / "cohort_v2" / "cohort.parquet")[["hospitalization_id", "stay_id"]]
    labels = pd.read_parquet(WORK / "downstream_v2" / "labels.parquet")[["stay_id", "y_mortality", "y_ami"]]
    lab = v2c.merge(labels, on="stay_id", how="inner")

    meta = (m1.merge(m2, on="stay_id_v1", how="inner")
              .merge(v1c, on="stay_id_v1", how="inner")
              .merge(lab, on="hospitalization_id", how="inner"))
    print(f"common stays with labels: {len(meta):,}")

    rng = np.random.default_rng(SEED)
    if len(meta) > N_SUB:
        meta = meta.iloc[rng.choice(len(meta), N_SUB, replace=False)].reset_index(drop=True)
    A = X1[meta["i1"].values]
    B = X2[meta["i2"].values]
    y_mort = meta["y_mortality"].values
    y_ami = meta["y_ami"].values
    print(f"subsampled to {len(meta):,} | mortality+={y_mort.sum()} ami+={y_ami.sum()}")

    def embed(X):
        Xs = StandardScaler().fit_transform(X)
        return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(Xs)

    print("UMAP v1 ..."); U1 = embed(A)
    print("UMAP v2 ..."); U2 = embed(B)

    fig, ax = plt.subplots(2, 2, figsize=(12, 11))
    panels = [
        (U1, y_mort, "v1 (228M, VAE) — by mortality"),
        (U2, y_mort, "v2-CE (2M, no VAE) — by mortality"),
        (U1, y_ami,  "v1 (228M, VAE) — by AMI"),
        (U2, y_ami,  "v2-CE (2M, no VAE) — by AMI"),
    ]
    for a, (U, y, title) in zip(ax.flat, panels):
        neg, pos = y == 0, y == 1
        a.scatter(U[neg, 0], U[neg, 1], s=3, c="lightgray", alpha=0.4, linewidths=0)
        a.scatter(U[pos, 0], U[pos, 1], s=6, c="crimson", alpha=0.8, linewidths=0)
        a.set_title(title, fontsize=12)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Embedding structure: v1 collapse vs v2 (same patients)", fontsize=14, y=0.99)
    fig.tight_layout()
    out = WORK / "final_results" / "v1_vs_v2_umap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
