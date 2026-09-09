"""Stage 21 v2 — first-48h re-evaluation for a FAIR comparison to the literature.

The whole-stay mortality AUROC (0.92) is not comparable to standard MIMIC
benchmarks, which predict from the first 24-48h. Here I truncate each patient
to time_index <= 48 (bin_hours=1, admit-aligned), re-encode with the frozen
model, and re-run the linear probe. Now ours is on the same setup as the
benchmark (~0.86 mortality), so the comparison is head-to-head.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
sys.path.insert(0, str(Path(__file__).parent))
from small_ehr_model import SmallEHRTransformer

CUT = 48   # hours
TMAX = CUT + 1


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    ck = torch.load("outputs/v2_small_ce/best.pt", map_location="cpu")
    ca = ck.get("args", {})
    feat = json.load(open(WORK / "feat_info.json"))
    n_cat = len(feat.get("category_cols", [])); n_float = len(feat["float_cols"])
    model = SmallEHRTransformer(n_cat_feats=n_cat, n_float_feats=n_float, n_cat_values=50, n_float_bins=256,
                                hidden_dim=ca.get("hidden_dim",192), n_layers=ca.get("n_layers",6),
                                n_heads=ca.get("n_heads",4), ff_dim=ca.get("hidden_dim",192)*2,
                                max_seq_len=512, float_mode=ca.get("float_mode","ce"))
    model.load_state_dict(ck["model"]); model.eval()

    import diskcache
    cache = diskcache.Cache(str(WORK/"ehr_cache"), eviction_policy="none")
    meta = pd.read_parquet(WORK/"ehr_cache"/"metadata.parquet")
    pids = meta["pid"].astype(int).tolist()

    embs, out_pids = [], []
    B = 128; buf_c, buf_f, buf_t, buf_v, buf_p = [], [], [], [], []
    def flush():
        if not buf_p: return
        cat = torch.stack(buf_c); fb = torch.stack(buf_f); ti = torch.stack(buf_t); vm = torch.stack(buf_v)
        _, _, z = model(cat, fb, ti, vm)
        m = vm.unsqueeze(-1).float()
        pooled = (z*m).sum(1)/m.sum(1).clamp(min=1.0)
        embs.append(pooled.numpy()); out_pids.extend(buf_p)
        buf_c.clear(); buf_f.clear(); buf_t.clear(); buf_v.clear(); buf_p.clear()

    for pid in pids:
        d = cache[int(pid)]
        vm = d["valid_mask"].astype(bool); ti = d["time_index"]
        keep = np.where(vm & (ti <= CUT))[0]
        k = min(len(keep), TMAX)
        keep = keep[:k]
        cat = np.full((n_cat, TMAX), -1, np.int64); fb = np.full((n_float, TMAX), -1, np.int64)
        tix = np.zeros(TMAX, np.int64); val = np.zeros(TMAX, bool)
        if k > 0:
            cat[:, :k] = d["tokenized_category_feats"][:, keep]
            fb[:, :k] = d["tokenized_float_feats"][:, keep]
            tix[:k] = ti[keep]; val[:k] = True
        buf_c.append(torch.from_numpy(cat)); buf_f.append(torch.from_numpy(fb))
        buf_t.append(torch.from_numpy(tix)); buf_v.append(torch.from_numpy(val)); buf_p.append(int(pid))
        if len(buf_p) >= B: flush()
    flush()
    X = np.concatenate(embs); sids = np.array(out_pids, np.int64)
    np.savez_compressed(WORK/"downstream_v2"/"small_ce_first48h.npz", embeddings=X, stay_ids=sids)
    print(f"extracted first-{CUT}h embeddings: {X.shape}")

    # linear probe vs whole-stay
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    m = pd.DataFrame({"stay_id_v1": sids, "row": np.arange(len(sids))})
    v1c = pd.read_parquet(WORK/"cohort"/"cohort.parquet")[["stay_id","hadm_id"]].rename(columns={"stay_id":"stay_id_v1","hadm_id":"hospitalization_id"})
    v2c = pd.read_parquet(WORK/"cohort_v2"/"cohort.parquet")[["hospitalization_id","stay_id","split"]]
    lab = pd.read_parquet(WORK/"downstream_v2"/"labels.parquet").drop(columns=["split"],errors="ignore")
    m = m.merge(v1c,on="stay_id_v1").merge(v2c,on="hospitalization_id").merge(lab,on="stay_id")
    X = X[m["row"].values]; sp = m["split"].values
    tr, te = sp=="train", sp=="test"
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    print("\nfirst-48h linear probe (real test AUROC):")
    for t in ["mortality","los_gt_7d","readmit_30d","ami","stroke"]:
        y = m[f"y_{t}"].values; msk = m[f"m_{t}"].values==1
        trm, tem = tr&msk, te&msk
        if y[trm].sum() in (0,trm.sum()) or len(np.unique(y[tem]))<2: continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs[trm], y[trm])
        au = roc_auc_score(y[tem], clf.predict_proba(Xs[tem])[:,1])
        print(f"  {t:12s} {au:.3f}")


if __name__ == "__main__":
    main()
