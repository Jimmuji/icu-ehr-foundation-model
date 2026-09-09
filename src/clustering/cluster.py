"""Step 4b — UMAP + multiple clustering methods on patient embeddings.

Loads {split}_embeddings.npz, runs UMAP to 2D, then KMeans / HDBSCAN / Leiden.
Saves cluster assignments + 2D coordinates + silhouette scores.

Designed to be run on Mac (CPU) once embeddings are downloaded from server.

Usage:
    python -m src.clustering.cluster \
        --emb outputs/embeddings/train_embeddings.npz \
        --out outputs/clusters

Methods:
    - UMAP            : 2D projection for visualization
    - KMeans (k=8)    : default fast baseline; sweep k via --kmeans-k
    - HDBSCAN         : density-based, auto-detect cluster count + noise
    - Leiden (optional, requires `igraph` + `leidenalg`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_umap(X: np.ndarray, n_neighbors: int, min_dist: float, seed: int):
    import umap
    print(f"  UMAP fit: n_neighbors={n_neighbors}, min_dist={min_dist}")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="euclidean", random_state=seed,
    )
    return reducer.fit_transform(X)


def fit_kmeans(X: np.ndarray, k_values: list[int], seed: int) -> dict:
    """Fit KMeans for each k; return best by silhouette."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    results = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
        print(f"  KMeans k={k}: silhouette={sil:.3f}")
        results.append({"k": k, "labels": labels.tolist(), "silhouette": float(sil)})
    best = max(results, key=lambda r: r["silhouette"] if r["silhouette"] == r["silhouette"] else -1)
    return {"sweep": results, "best": best}


def fit_hdbscan(X: np.ndarray, min_cluster_size: int) -> dict:
    try:
        import hdbscan
    except ImportError:
        print("  HDBSCAN not installed; pip install hdbscan to enable")
        return {}
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = cl.fit_predict(X)
    n_clusters = int((labels >= 0).max() + 1) if (labels >= 0).any() else 0
    n_noise = int((labels == -1).sum())
    print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
    return {"labels": labels.tolist(), "n_clusters": n_clusters, "n_noise": n_noise}


def fit_leiden(X: np.ndarray, n_neighbors: int, seed: int) -> dict:
    try:
        from sklearn.neighbors import NearestNeighbors
        import igraph as ig
        import leidenalg
    except ImportError:
        print("  Leiden deps missing; pip install python-igraph leidenalg to enable")
        return {}

    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(X)
    dist, idx = knn.kneighbors(X)
    # Build undirected weighted graph; drop self-loops (first col)
    edges = []
    weights = []
    for i in range(idx.shape[0]):
        for j_pos in range(1, idx.shape[1]):
            j = int(idx[i, j_pos])
            w = float(1.0 / (1.0 + dist[i, j_pos]))
            if i < j:
                edges.append((i, j))
                weights.append(w)
    g = ig.Graph(n=X.shape[0], edges=edges, edge_attrs={"weight": weights}, directed=False)
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights="weight", seed=seed,
    )
    labels = np.asarray(part.membership)
    print(f"  Leiden: {labels.max() + 1} clusters")
    return {"labels": labels.tolist(), "n_clusters": int(labels.max() + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="path to {split}_embeddings.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--kmeans-k", nargs="+", type=int, default=[5, 8, 10, 12, 15, 20])
    ap.add_argument("--hdbscan-min-size", type=int, default=50)
    ap.add_argument("--skip", nargs="+", default=[], choices=["umap", "kmeans", "hdbscan", "leiden"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.emb)
    X = data["embeddings"].astype(np.float32)
    sids = data["stay_ids"]
    print(f"Loaded embeddings: shape={X.shape}, n_stays={len(sids)}")

    results: dict = {"n_stays": int(len(sids)), "embedding_dim": int(X.shape[1])}

    if "umap" not in args.skip:
        coords = fit_umap(X, args.umap_neighbors, args.umap_min_dist, args.seed)
        np.savez(out / "umap_2d.npz", coords=coords, stay_ids=sids)
        results["umap_path"] = "umap_2d.npz"

    if "kmeans" not in args.skip:
        results["kmeans"] = fit_kmeans(X, args.kmeans_k, args.seed)

    if "hdbscan" not in args.skip:
        results["hdbscan"] = fit_hdbscan(X, args.hdbscan_min_size)

    if "leiden" not in args.skip:
        results["leiden"] = fit_leiden(X, args.umap_neighbors, args.seed)

    # Persist (drop large 'labels' lists to a separate npz for sanity)
    cluster_labels = {}
    for key in ["kmeans", "hdbscan", "leiden"]:
        if key in results and results[key]:
            if "labels" in results[key]:
                cluster_labels[key] = np.asarray(results[key].pop("labels"))
            if key == "kmeans" and "sweep" in results[key]:
                for r in results[key]["sweep"]:
                    cluster_labels[f"kmeans_k{r['k']}"] = np.asarray(r.pop("labels"))
                if "labels" in results[key]["best"]:
                    cluster_labels["kmeans_best"] = np.asarray(results[key]["best"].pop("labels"))
    np.savez(out / "cluster_labels.npz", stay_ids=sids, **cluster_labels)

    with open(out / "cluster_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out / 'cluster_results.json'}")
    print(f"Wrote {out / 'cluster_labels.npz'}")


if __name__ == "__main__":
    main()
