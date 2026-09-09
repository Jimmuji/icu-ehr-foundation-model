"""Stage 01 v2 — build full adult hospitalization cohort + temporal split.

Key changes from v1:
- Drops "ICU only" filter (was: location_category == 'icu')
- Drops "first_icu_stay_only" filter
- Keeps "adult, age 18-89" filter
- Allows multiple admissions per patient (needed for readmission task + general FM scale)
- Implements joint patient + temporal split:
    train: patients whose FIRST admit_year < TRAIN_END_YEAR
    val:   patients whose FIRST admit_year ∈ [TRAIN_END_YEAR, VAL_END_YEAR)
    test:  patients whose FIRST admit_year >= VAL_END_YEAR
  This ensures (a) no patient leakage across splits, (b) test set is temporally held out
  (more realistic than random split, addresses review concern #1).

Output: {work_dir}/cohort_v2/cohort.parquet with columns
    patient_id, hospitalization_id, stay_id (synthetic, 1-indexed),
    admission_dttm, discharge_dttm, age_at_admission, sex_category,
    discharge_category, died_in_hosp, hospital_los_hours, admit_year,
    split.

Run:
    python src_v2/01_build_cohort_v2.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_config, setup_logging, stage_dir


# --------- Temporal split anchor years (MIMIC-IV CLIF uses ~2100-offset dates) ---------
# Empirically year distribution centers around 2110-2214; pick 80/10/10-ish cutoffs.
# We'll determine the actual cutoffs dynamically based on the data distribution.

def pick_temporal_cutoffs(admit_years: pd.Series, train_frac=0.80, val_frac=0.10):
    """Pick year cutoffs so that ~train_frac of admissions land in train, val_frac in val.
    Returns (train_end_year, val_end_year).
    """
    yrs = admit_years.value_counts().sort_index()
    cum = yrs.cumsum() / yrs.sum()
    train_end = cum[cum <= train_frac].index[-1] if (cum <= train_frac).any() else cum.index[0]
    val_end_cum = train_frac + val_frac
    val_end = cum[cum <= val_end_cum].index[-1] if (cum <= val_end_cum).any() else cum.index[0]
    return int(train_end) + 1, int(val_end) + 1  # +1 because we use < strict less-than


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("01_cohort_v2", cfg["paths"]["work_dir"])
    out_dir = stage_dir(cfg, "cohort_v2")

    clif_root = Path(cfg["paths"]["clif_root"])
    log.info(f"CLIF root: {clif_root}")

    # ---- Load source tables ----
    hosp = pd.read_parquet(clif_root / "clif_hospitalization.parquet")
    log.info(f"clif_hospitalization rows: {len(hosp):,}")

    patient = pd.read_parquet(clif_root / "clif_patient.parquet")
    log.info(f"clif_patient rows: {len(patient):,}")

    # ---- Strip timezones for arithmetic ----
    for c in ["admission_dttm", "discharge_dttm"]:
        if hasattr(hosp[c].dt, "tz") and hosp[c].dt.tz is not None:
            hosp[c] = hosp[c].dt.tz_localize(None)
    if hasattr(patient["death_dttm"].dt, "tz") and patient["death_dttm"].dt.tz is not None:
        patient["death_dttm"] = patient["death_dttm"].dt.tz_localize(None)

    # ---- Adult filter ----
    cf = cfg["cohort"]
    n0 = len(hosp)
    hosp = hosp[(hosp["age_at_admission"] >= cf["min_age"]) & (hosp["age_at_admission"] <= cf["max_age"])]
    log.info(f"  age filter [{cf['min_age']}, {cf['max_age']}]: {n0:,} → {len(hosp):,}")

    # ---- Compute hospital LOS + mortality flag ----
    hosp["hospital_los_hours"] = (hosp["discharge_dttm"] - hosp["admission_dttm"]).dt.total_seconds() / 3600.0
    hosp["died_in_hosp"] = (hosp["discharge_category"].astype(str).str.lower() == "expired").astype(int)
    hosp["admit_year"] = hosp["admission_dttm"].dt.year

    # ---- Drop hospitalizations with bad LOS (negative or > 365 days) ----
    n0 = len(hosp)
    hosp = hosp[(hosp["hospital_los_hours"] > 0) & (hosp["hospital_los_hours"] <= 365 * 24)]
    log.info(f"  LOS sanity [0, 365 days]: {n0:,} → {len(hosp):,}")

    # ---- Join patient demographics ----
    pat = patient[["patient_id", "sex_category", "birth_date", "death_dttm"]].copy()
    df = hosp.merge(pat, on="patient_id", how="left")

    # ---- Assign synthetic stay_id (sorted by patient, admit time so it's stable) ----
    df = df.sort_values(["patient_id", "admission_dttm"]).reset_index(drop=True)
    df["stay_id"] = df.index.astype("int64") + 1

    # ---- Temporal split (anchored on patient's FIRST admit year, not per-hospitalization) ----
    # Determine cutoffs from admission year distribution
    train_end, val_end = pick_temporal_cutoffs(df["admit_year"])
    log.info(f"Temporal split cutoffs: train < {train_end} <= val < {val_end} <= test")

    # Patient → first admit year (anchor)
    pat_first = df.groupby("patient_id")["admit_year"].min().rename("first_admit_year")
    df = df.join(pat_first, on="patient_id")

    def assign_split(yr):
        if yr < train_end: return "train"
        elif yr < val_end: return "val"
        else: return "test"
    df["split"] = df["first_admit_year"].map(assign_split)

    # ---- Persist ----
    keep_cols = [
        "patient_id", "hospitalization_id", "stay_id",
        "admission_dttm", "discharge_dttm", "admit_year",
        "age_at_admission", "sex_category", "death_dttm",
        "discharge_category", "died_in_hosp", "hospital_los_hours",
        "first_admit_year", "split",
    ]
    out = df[keep_cols].reset_index(drop=True)
    out_path = out_dir / "cohort.parquet"
    out.to_parquet(out_path, index=False)
    log.info(f"Wrote cohort: {len(out):,} hospitalizations → {out_path}")

    # ---- Sanity stats ----
    log.info(f"\n=== Cohort summary ===")
    log.info(f"  total hospitalizations: {len(out):,}")
    log.info(f"  unique patients:        {out['patient_id'].nunique():,}")
    log.info(f"  mean age:               {out['age_at_admission'].mean():.1f}")
    log.info(f"  in-hospital mortality:  {out['died_in_hosp'].mean()*100:.2f}%")
    log.info(f"  mean hospital LOS (h):  {out['hospital_los_hours'].mean():.1f}")
    log.info(f"\n=== Split distribution ===")
    split_summary = out.groupby("split").agg(
        n_hosp=("stay_id", "count"),
        n_patient=("patient_id", "nunique"),
        mortality=("died_in_hosp", "mean"),
        admit_yr_min=("admit_year", "min"),
        admit_yr_max=("admit_year", "max"),
    )
    log.info(f"\n{split_summary}")


if __name__ == "__main__":
    main()
