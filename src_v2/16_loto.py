"""Stage 16 v2 — Leave-One-Feature-Out interpretability (APOLLO LOTO-style).

APOLLO uses Leave-One-Token-Out to explain which medical events drive a
patient's risk. Feature-level analog here: for sampled patients, ablate one
clinical variable at a time (set all its timesteps to missing), re-encode with
the frozen model, and measure the change in predicted in-hospital mortality.
Averaged over patients, this ranks which variables the representation relies on
for mortality risk. Positive importance = removing it lowers predicted risk
(the variable was pushing risk up).

CPU-only; per-patient tensors are truncated to valid timesteps for speed.

Run:
    python src_v2/16_loto.py --ckpt outputs/v2_small_ce/best.pt --n 150
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
sys.path.insert(0, str(Path(__file__).parent))
from small_ehr_model import SmallEHRTransformer


@torch.no_grad()
def encode_batch(model, cat, fb, tix, val):
    _, _, z = model(cat, fb, tix, val)            # (B,T,D)
    m = val.unsqueeze(-1).float()
    return ((z * m).sum(1) / m.sum(1).clamp(min=1.0)).numpy()  # (B,D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/v2_small_ce/best.pt")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--chunk", type=int, default=48)
    args = ap.parse_args()
    torch.set_num_threads(8)

    # ---- model ----
    ck = torch.load(args.ckpt, map_location="cpu")
    ca = ck.get("args", {})
    feat = json.load(open(WORK / "feat_info.json"))
    cat_names = list(feat.get("category_cols", []))
    float_names = sorted(feat["float_cols"].keys())
    n_cat, n_float = len(cat_names), len(float_names)
    model = SmallEHRTransformer(n_cat_feats=n_cat, n_float_feats=n_float, n_cat_values=50, n_float_bins=256,
                                hidden_dim=ca.get("hidden_dim", 192), n_layers=ca.get("n_layers", 6),
                                n_heads=ca.get("n_heads", 4), ff_dim=ca.get("hidden_dim", 192)*2,
                                max_seq_len=512, float_mode=ca.get("float_mode", "ce"))
    model.load_state_dict(ck["model"]); model.eval()
    names = [f"cat:{c}" for c in cat_names] + [f"flt:{f}" for f in float_names]
    F = n_cat + n_float

    # ---- mortality probe (fit on existing npz train embeddings) ----
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    e = np.load(WORK / "downstream_v2" / "small_ce_embeddings.npz")
    emb, sid = e["embeddings"].astype(np.float32), e["stay_ids"].astype(np.int64)
    m0 = pd.DataFrame({"stay_id_v1": sid, "row": np.arange(len(sid))})
    v1c = pd.read_parquet(WORK/"cohort"/"cohort.parquet")[["stay_id","hadm_id"]].rename(columns={"stay_id":"stay_id_v1","hadm_id":"hospitalization_id"})
    v2c = pd.read_parquet(WORK/"cohort_v2"/"cohort.parquet")[["hospitalization_id","stay_id","split"]]
    lab = pd.read_parquet(WORK/"downstream_v2"/"labels.parquet")[["stay_id","y_mortality"]]
    m0 = m0.merge(v1c,on="stay_id_v1").merge(v2c,on="hospitalization_id").merge(lab,on="stay_id")
    Xe = emb[m0["row"].values]
    tr = m0["split"].values == "train"
    sc = StandardScaler().fit(Xe[tr]); lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xe[tr]), m0["y_mortality"].values[tr])
    def prob(E): return lr.predict_proba(sc.transform(E))[:, 1]

    # ---- sample patients (stratified to include deaths) ----
    import diskcache
    cache = diskcache.Cache(str(WORK/"ehr_cache"), eviction_policy="none")
    rng = np.random.default_rng(0)
    pos = m0[m0["y_mortality"] == 1]; neg = m0[m0["y_mortality"] == 0]
    take = pd.concat([pos.sample(min(len(pos), args.n//2), random_state=0),
                      neg.sample(args.n - min(len(pos), args.n//2), random_state=0)])
    pids = take["stay_id_v1"].tolist()

    imp = np.zeros(F); cnt = 0
    for pid in pids:
        d = cache[int(pid)]
        v = torch.from_numpy(d["valid_mask"]).bool()
        T = int(v.sum())
        if T < 2: continue
        cat = torch.from_numpy(d["tokenized_category_feats"]).long()[:, :T]   # (n_cat,T)
        fb = torch.from_numpy(d["tokenized_float_feats"]).long()[:, :T]       # (n_float,T)
        tix = torch.from_numpy(d["time_index"]).long()[:T]
        val = torch.ones(T, dtype=torch.bool)
        base_emb = encode_batch(model, cat[None], fb[None], tix[None], val[None])
        base_p = prob(base_emb)[0]
        # build all F ablations, run in chunks
        ablated_p = np.zeros(F)
        idx = 0
        while idx < F:
            js = list(range(idx, min(idx+args.chunk, F)))
            B = len(js)
            C = cat[None].repeat(B,1,1).clone(); Fb = fb[None].repeat(B,1,1).clone()
            for bi, j in enumerate(js):
                if j < n_cat: C[bi, j, :] = -1
                else: Fb[bi, j-n_cat, :] = -1
            embs = encode_batch(model, C, Fb, tix[None].repeat(B,1), val[None].repeat(B,1))
            ablated_p[js] = prob(embs)
            idx += args.chunk
        imp += (base_p - ablated_p)   # positive = feature pushed risk up
        cnt += 1

    imp /= max(cnt, 1)
    res = pd.DataFrame({"feature": names, "mortality_importance": imp}).sort_values(
        "mortality_importance", key=lambda s: s.abs(), ascending=False)
    res.to_csv(WORK/"downstream_v2"/"loto_mortality.csv", index=False)
    print(f"LOTO over {cnt} patients. Top 15 drivers of predicted mortality:")
    print(res.head(15).to_string(index=False))
    print(f"\nwrote loto_mortality.csv")


if __name__ == "__main__":
    main()
