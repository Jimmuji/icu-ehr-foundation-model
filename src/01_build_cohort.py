"""Stage 01 — build the ICU cohort from CLIF tables.

We use MIMIC-IV-Ext-CLIF instead of raw MIMIC-IV because:
- CLIF gives us patient + hospitalization + adt + diagnoses, all in one harmonized format
- We don't need to wait for the large raw MIMIC-IV files (chartevents etc.) to download
- CLIF is the format the lab uses across multiple hospitals, so the pipeline transfers

Logic:
- ICU stays = clif_adt rows where location_category == "icu"
- Each ADT row is one ICU stay (one in_dttm/out_dttm pair)
- Filter: age in [min_age, max_age], LOS in [min_los, max_los] hours
- Filter: at least min_time_in_icu_before_death_h before any death
- If first_icu_stay_only: keep earliest ICU stay per patient_id

Output: {work_dir}/cohort/cohort.parquet with MIMIC-style column names
(so stages 02-05 stay unchanged).

Run:
    python src/01_build_cohort.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import load_config, setup_logging, stage_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("01_cohort", cfg["paths"]["work_dir"])
    out_dir = stage_dir(cfg, "cohort")

    clif_root = Path(cfg["paths"]["clif_root"])
    log.info(f"CLIF root: {clif_root}")

    # --- Load CLIF source tables ---
    adt = pd.read_parquet(clif_root / "clif_adt.parquet")
    log.info(f"clif_adt rows: {len(adt):,}")

    hosp = pd.read_parquet(clif_root / "clif_hospitalization.parquet")
    log.info(f"clif_hospitalization rows: {len(hosp):,}")

    patient = pd.read_parquet(clif_root / "clif_patient.parquet")
    log.info(f"clif_patient rows: {len(patient):,}")

    # --- ICU stays from ADT ---
    icu = adt[adt["location_category"] == "icu"].copy()
    log.info(f"ICU ADT events: {len(icu):,}")

    # Strip timezone for consistent arithmetic (CLIF stores UTC)
    for c in ["in_dttm", "out_dttm"]:
        if hasattr(icu[c].dt, "tz") and icu[c].dt.tz is not None:
            icu[c] = icu[c].dt.tz_localize(None)
    icu["los_hours"] = (icu["out_dttm"] - icu["in_dttm"]).dt.total_seconds() / 3600.0

    # Each ICU ADT row is one stay; synthetic stay_id (stable when sorted by patient,intime)
    icu = icu.sort_values(["patient_id", "in_dttm"]).reset_index(drop=True)
    icu["stay_id"] = icu.index.astype("int64") + 1

    # --- Join hospitalization (admission timing + discharge + age) ---
    hosp_cols = hosp[[
        "patient_id", "hospitalization_id", "admission_dttm", "discharge_dttm",
        "age_at_admission", "discharge_category",
    ]].copy()
    for c in ["admission_dttm", "discharge_dttm"]:
        if hasattr(hosp_cols[c].dt, "tz") and hosp_cols[c].dt.tz is not None:
            hosp_cols[c] = hosp_cols[c].dt.tz_localize(None)

    df = icu.merge(hosp_cols, on=["patient_id", "hospitalization_id"], how="left")
    df["hospital_los_hours"] = (
        df["discharge_dttm"] - df["admission_dttm"]
    ).dt.total_seconds() / 3600.0

    # --- Join patient demographics ---
    pat = patient[["patient_id", "sex_category", "birth_date", "death_dttm"]].copy()
    if hasattr(pat["death_dttm"].dt, "tz") and pat["death_dttm"].dt.tz is not None:
        pat["death_dttm"] = pat["death_dttm"].dt.tz_localize(None)
    df = df.merge(pat, on="patient_id", how="left")

    # --- Mortality flag (discharge_category == "Expired") ---
    df["died_in_hosp"] = (df["discharge_category"].astype(str).str.lower() == "expired").astype(int)
    df["admission_age"] = df["age_at_admission"]

    # --- Filters ---
    cf = cfg["cohort"]
    n0 = len(df)
    df = df[(df["admission_age"] >= cf["min_age"]) & (df["admission_age"] <= cf["max_age"])]
    log.info(f"  age filter [{cf['min_age']}, {cf['max_age']}]: {n0:,} → {len(df):,}")

    n0 = len(df)
    df = df[(df["los_hours"] >= cf["min_los_hours"]) & (df["los_hours"] <= cf["max_los_hours"])]
    log.info(f"  LOS filter [{cf['min_los_hours']}, {cf['max_los_hours']}]h: {n0:,} → {len(df):,}")

    # Early-death filter: deaths within X hours of ICU intime are excluded
    min_h = cf["min_time_in_icu_before_death_h"]
    time_to_death_h = (df["death_dttm"] - df["in_dttm"]).dt.total_seconds() / 3600.0
    bad = time_to_death_h.notna() & (time_to_death_h < min_h)
    n0 = len(df)
    df = df[~bad]
    log.info(f"  early-death filter (>{min_h}h after ICU intime): {n0:,} → {len(df):,}")

    if cf["first_icu_stay_only"]:
        n0 = len(df)
        df = df.sort_values(["patient_id", "in_dttm"]).drop_duplicates("patient_id", keep="first")
        log.info(f"  first stay only: {n0:,} → {len(df):,}")

    # --- Persist with MIMIC-style column names so downstream stages are unchanged ---
    out = df.rename(columns={
        "patient_id": "subject_id",
        "hospitalization_id": "hadm_id",
        "in_dttm": "intime",
        "out_dttm": "outtime",
        "sex_category": "gender",
        "death_dttm": "dod",
    })[[
        "subject_id", "hadm_id", "stay_id", "intime", "outtime", "los_hours",
        "admission_age", "gender", "dod", "died_in_hosp", "hospital_los_hours",
    ]].reset_index(drop=True)

    out_path = Path(out_dir) / "cohort.parquet"
    out.to_parquet(out_path, index=False)
    log.info(f"Wrote cohort: {len(out):,} stays → {out_path}")
    log.info(
        f"Mean age {out['admission_age'].mean():.1f}, "
        f"mean LOS {out['los_hours'].mean():.1f} h, "
        f"in-hospital mortality {out['died_in_hosp'].mean()*100:.1f}%, "
        f"unique patients {out['subject_id'].nunique():,}"
    )


if __name__ == "__main__":
    main()
