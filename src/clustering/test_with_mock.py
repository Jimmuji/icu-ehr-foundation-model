"""Smoke test — generate mock embeddings + cohort and run cluster.py + analyze.py.

Verifies the Step 4 pipeline works end-to-end without needing a real checkpoint.
Useful while we wait for pretraining to finish on the server.

Run:
    python -m src.clustering.test_with_mock
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
TMP = REPO / "outputs" / "_smoke"


def make_mock(n_stays: int = 1500, dim: int = 64, k_true: int = 6, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    TMP.mkdir(parents=True, exist_ok=True)

    # Synthetic embedding clusters
    centers = rng.normal(size=(k_true, dim)) * 3.0
    sizes = rng.multinomial(n_stays, np.ones(k_true) / k_true)
    embs, ground = [], []
    for ci, n in enumerate(sizes):
        embs.append(centers[ci] + rng.normal(scale=0.6, size=(n, dim)))
        ground.extend([ci] * n)
    embs = np.concatenate(embs).astype(np.float32)
    stay_ids = np.arange(len(embs), dtype=np.int64)
    ground = np.asarray(ground)

    np.savez(TMP / "train_embeddings.npz", embeddings=embs, stay_ids=stay_ids)

    # Synthetic cohort metadata correlated with ground-truth cluster (so analyze sees signal)
    age_means = rng.uniform(30, 80, size=k_true)
    mort_rates = rng.uniform(0.05, 0.35, size=k_true)
    los_means = rng.uniform(48, 240, size=k_true)
    cohort = pd.DataFrame({
        "stay_id": stay_ids,
        "subject_id": stay_ids,
        "hadm_id": stay_ids,
        "intime": pd.Timestamp("2020-01-01"),
        "outtime": pd.Timestamp("2020-01-05"),
        "los_hours": [rng.normal(los_means[g], 24) for g in ground],
        "admission_age": [int(np.clip(rng.normal(age_means[g], 6), 18, 89)) for g in ground],
        "gender": rng.choice(["M", "F"], len(stay_ids)),
        "dod": pd.NaT,
        "died_in_hosp": [int(rng.random() < mort_rates[g]) for g in ground],
        "hospital_los_hours": rng.uniform(72, 480, len(stay_ids)),
    })
    cohort_dir = TMP / "cohort"; cohort_dir.mkdir(exist_ok=True)
    cohort.to_parquet(cohort_dir / "cohort.parquet", index=False)

    # Synthetic events: one diagnosis chapter event per stay, biased by cluster
    chapters = ["icd10_I", "icd10_N", "icd10_J", "icd10_R", "icd10_K", "icd10_E"]
    dx = pd.DataFrame({
        "stay_id": stay_ids,
        "t_hours": 0.0,
        "feature_name": "diag::chapter",
        "feature_kind": "cat",
        "value": [chapters[g % len(chapters)] for g in ground],
    })
    events_dir = TMP / "events"; events_dir.mkdir(exist_ok=True)
    dx.to_parquet(events_dir / "events.parquet", index=False)

    print(f"  wrote mock data: {len(embs):,} stays, dim={dim}, k_true={k_true}")


def run(cmd):
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"failed: {r.returncode}")


def main():
    make_mock()

    out_clusters = TMP / "clusters"
    out_analysis = TMP / "analysis"

    run([sys.executable, "-m", "src.clustering.cluster",
         "--emb", str(TMP / "train_embeddings.npz"),
         "--out", str(out_clusters),
         "--skip", "leiden", "hdbscan",   # skip optional deps for fast smoke test
         "--kmeans-k", "4", "6", "8"])

    run([sys.executable, "-m", "src.clustering.analyze",
         "--umap", str(out_clusters / "umap_2d.npz"),
         "--labels", str(out_clusters / "cluster_labels.npz"),
         "--cohort", str(TMP / "cohort" / "cohort.parquet"),
         "--events", str(TMP / "events" / "events.parquet"),
         "--out", str(out_analysis)])

    print("\n=== SMOKE TEST PASSED ===")
    print("Artifacts:")
    for p in sorted(out_analysis.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
