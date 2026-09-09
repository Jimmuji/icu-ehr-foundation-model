"""Stage 13 v2 — REAL temporal split using MIMIC-IV anchor_year_group.

Problem: MIMIC shifts each patient's dates by a random per-patient offset
(into 2100-2200), so the shifted admit_year does NOT reflect true chronology
across patients. Our earlier split on shifted years was therefore not a true
temporal split.

Fix: MIMIC-IV patients.csv provides anchor_year_group, the REAL 3-year window
(2008-2010 ... 2020-2022) each patient's record falls in. We split on that.

  train = 2008-2010, 2011-2013, 2014-2016
  val   = 2017-2019
  test  = 2020-2022

patient_id in CLIF == MIMIC subject_id, so the join is direct.

Writes:
  cohort_v2/cohort.parquet  gains  anchor_year_group, split_real (keeps old `split`)
Run:
    python src_v2/13_real_temporal_split.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")
TRAIN_G = {"2008 - 2010", "2011 - 2013", "2014 - 2016"}
VAL_G = {"2017 - 2019"}
TEST_G = {"2020 - 2022"}


def assign(g):
    if g in TRAIN_G: return "train"
    if g in VAL_G: return "val"
    if g in TEST_G: return "test"
    return None


def main():
    pats = pd.read_csv(WORK / "patients.csv.gz", usecols=["subject_id", "anchor_year_group"])
    pats["patient_id"] = pats["subject_id"].astype(str)
    g = pats.set_index("patient_id")["anchor_year_group"]

    coh = pd.read_parquet(WORK / "cohort_v2" / "cohort.parquet")
    coh["patient_id"] = coh["patient_id"].astype(str)
    coh["anchor_year_group"] = coh["patient_id"].map(g)
    miss = coh["anchor_year_group"].isna().sum()
    coh["split_real"] = coh["anchor_year_group"].map(assign)
    print(f"cohort {len(coh):,} | unmapped patients: {miss:,}")

    # Report: real-time distribution + mortality drift
    print("\n=== per real anchor_year_group ===")
    rep = coh.groupby("anchor_year_group").agg(
        n_hosp=("hospitalization_id", "size"),
        n_patients=("patient_id", "nunique"),
        mortality=("died_in_hosp", "mean")).reset_index()
    rep["mortality"] = (rep["mortality"] * 100).round(2)
    print(rep.to_string(index=False))

    print("\n=== REAL temporal split ===")
    sp = coh.groupby("split_real").agg(
        n_hosp=("hospitalization_id", "size"),
        n_patients=("patient_id", "nunique"),
        mortality_pct=("died_in_hosp", lambda x: round(x.mean() * 100, 2))).reindex(["train", "val", "test"])
    print(sp.to_string())

    # Compare with how much the OLD (shifted) split overlaps the new one
    if "split" in coh.columns:
        xtab = pd.crosstab(coh["split"], coh["split_real"])
        print("\n=== old (shifted) split  vs  new (real) split — crosstab ===")
        print(xtab.to_string())

    coh.to_parquet(WORK / "cohort_v2" / "cohort.parquet", index=False)
    print(f"\nwrote split_real + anchor_year_group into cohort_v2/cohort.parquet")


if __name__ == "__main__":
    main()
