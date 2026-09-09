"""Stage 19 v2 — disease atlas in the style of APOLLO Fig 2b.

We do not feed ICD codes as input tokens, so we build a disease embedding the
indirect way: represent each ICD code by the MEAN patient embedding over the
patients who carry that code, then UMAP and colour by ICD-10 chapter. If our
patient representation is clinically organised, related diseases should land
near each other (circulatory together, respiratory together, etc.).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
CLIF = Path("/data/mimic_data/physionet.org/files/mimic-iv-ext-clif-data/1.1.0")
MINP = 40   # min patients per code prefix

CHAP = {  # ICD-10 first letter -> chapter
    "A": "Infectious", "B": "Infectious", "C": "Neoplasms", "D": "Neoplasms/Blood",
    "E": "Endocrine/Metabolic", "F": "Mental", "G": "Nervous", "H": "Eye/Ear",
    "I": "Circulatory", "J": "Respiratory", "K": "Digestive", "L": "Skin",
    "M": "Musculoskeletal", "N": "Genitourinary", "O": "Pregnancy", "P": "Perinatal",
    "Q": "Congenital", "R": "Symptoms", "S": "Injury", "T": "Injury",
    "Z": "Health status",
}


def main():
    import umap
    e = np.load(WORK / "downstream_v2" / "small_ce_embeddings.npz")
    X, sids = e["embeddings"].astype(np.float32), e["stay_ids"].astype(np.int64)
    m = pd.DataFrame({"stay_id_v1": sids, "row": np.arange(len(sids))})
    v1c = pd.read_parquet(WORK/"cohort"/"cohort.parquet")[["stay_id","hadm_id"]].rename(
        columns={"stay_id":"stay_id_v1","hadm_id":"hospitalization_id"})
    m = m.merge(v1c, on="stay_id_v1", how="inner")
    hadm2row = dict(zip(m["hospitalization_id"].astype(str), m["row"]))

    dx = pd.read_parquet(CLIF/"clif_hospital_diagnosis.parquet", columns=["hospitalization_id","diagnosis_code"])
    dx["hospitalization_id"] = dx["hospitalization_id"].astype(str)
    dx = dx[dx["hospitalization_id"].isin(hadm2row)]
    dx["code3"] = dx["diagnosis_code"].astype(str).str.upper().str.replace(".","",regex=False).str[:3]
    dx = dx[dx["code3"].str[0].isin(CHAP.keys())]            # ICD-10 only
    dx["row"] = dx["hospitalization_id"].map(hadm2row)

    rows = []
    for code, g in dx.groupby("code3"):
        idx = g["row"].unique()
        if len(idx) < MINP:
            continue
        rows.append((code, CHAP[code[0]], len(idx), X[idx].mean(0)))
    print(f"{len(rows)} ICD-10 code prefixes with >= {MINP} patients")
    codes = [r[0] for r in rows]; chaps = [r[1] for r in rows]
    C = np.vstack([r[3] for r in rows])

    U = umap.UMAP(n_neighbors=15, min_dist=0.2, metric="cosine", random_state=0).fit_transform(C)
    uniq = sorted(set(chaps)); cmap = plt.get_cmap("tab20")
    ci = {c:i for i,c in enumerate(uniq)}
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in uniq:
        mm = np.array([x==c for x in chaps])
        ax.scatter(U[mm,0], U[mm,1], s=28, color=cmap(ci[c]%20), label=f"{c}", alpha=0.8, edgecolors="white", linewidths=0.3)
    # annotate a few well-known codes
    for code, (x, y) in zip(codes, U):
        if code in {"I21","I50","J18","J96","N17","E11","C34","I63","K70","A41"}:
            ax.annotate(code, (x, y), fontsize=7, alpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Disease atlas: each ICD code = mean patient embedding of its patients, coloured by ICD-10 chapter")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(WORK/"final_results"/"disease_atlas.png", dpi=130, bbox_inches="tight")
    print("disease atlas done")


if __name__ == "__main__":
    main()
