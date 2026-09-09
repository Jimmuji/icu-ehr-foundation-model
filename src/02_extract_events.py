"""Stage 02 — extract clinical events for the cohort from CLIF.

Schemas hardcoded from CLIF v1.1.0 (verified on actual data).
Each table contributes one or more (cat|cont) features at timestamp -> long format.

Output: {work_dir}/events/events.parquet
    stay_id, t_hours, feature_name, feature_kind ("cat"|"cont"), value (str|float)

`t_hours` = float hours since ICU intime; rows outside the ICU window dropped.

Run:
    python src/02_extract_events.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_config, setup_logging, stage_dir


def _strip_tz(s: pd.Series) -> pd.Series:
    """Strip timezone if present (CLIF stores UTC; arithmetic needs naive)."""
    if hasattr(s.dt, "tz") and s.dt.tz is not None:
        return s.dt.tz_localize(None)
    return s


def attach_stay(df: pd.DataFrame, cohort_idx: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Inner-join with cohort to (a) keep only cohort hospitalization_id rows and
    (b) compute t_hours since ICU intime + drop rows outside [intime, outtime]."""
    df = df.dropna(subset=[time_col])
    df = df.join(cohort_idx, on="hospitalization_id", how="inner")
    df = df.dropna(subset=["intime", "outtime"])
    df["t_hours"] = (df[time_col] - df["intime"]).dt.total_seconds() / 3600.0
    df = df[(df[time_col] >= df["intime"]) & (df[time_col] <= df["outtime"])]
    df = df[df["t_hours"] >= 0]
    return df


def extract_vitals(clif_root: Path, cohort_idx: pd.DataFrame, log) -> pd.DataFrame:
    p = clif_root / "clif_vitals.parquet"
    if not p.exists():
        log.warning("  clif_vitals missing"); return pd.DataFrame()
    df = pd.read_parquet(p, columns=["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"])
    df["recorded_dttm"] = _strip_tz(df["recorded_dttm"])
    df = df.dropna(subset=["vital_category", "vital_value"])
    df = attach_stay(df, cohort_idx, "recorded_dttm")
    df["feature_name"] = "vital::" + df["vital_category"].astype(str)
    df["feature_kind"] = "cont"
    df["value"] = df["vital_value"]
    out = df[["stay_id", "t_hours", "feature_name", "feature_kind", "value"]]
    log.info(f"  vitals: {len(out):,} events")
    return out


def extract_labs(clif_root: Path, cohort_idx: pd.DataFrame, log) -> pd.DataFrame:
    p = clif_root / "clif_labs.parquet"
    if not p.exists():
        log.warning("  clif_labs missing"); return pd.DataFrame()
    df = pd.read_parquet(p, columns=["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"])
    df["lab_result_dttm"] = _strip_tz(df["lab_result_dttm"])
    df = df.dropna(subset=["lab_category", "lab_value_numeric", "lab_result_dttm"])
    df = attach_stay(df, cohort_idx, "lab_result_dttm")
    df["feature_name"] = "lab::" + df["lab_category"].astype(str)
    df["feature_kind"] = "cont"
    df["value"] = df["lab_value_numeric"]
    out = df[["stay_id", "t_hours", "feature_name", "feature_kind", "value"]]
    log.info(f"  labs: {len(out):,} events")
    return out


def extract_meds(clif_root: Path, cohort_idx: pd.DataFrame, log) -> pd.DataFrame:
    pieces = []
    for table, kind_prefix in [
        ("clif_medication_admin_continuous", "med_cont"),
        ("clif_medication_admin_intermittent", "med_int"),
    ]:
        p = clif_root / f"{table}.parquet"
        if not p.exists():
            log.warning(f"  {table} missing"); continue
        df = pd.read_parquet(p, columns=["hospitalization_id", "admin_dttm", "med_category", "med_dose"])
        df["admin_dttm"] = _strip_tz(df["admin_dttm"])
        df = df.dropna(subset=["med_category", "admin_dttm"])
        df = attach_stay(df, cohort_idx, "admin_dttm")
        df["feature_name"] = f"{kind_prefix}::" + df["med_category"].astype(str)
        # If dose is NaN, treat as a categorical "given" event (presence-only)
        df_cont = df.dropna(subset=["med_dose"]).copy()
        df_cont["feature_kind"] = "cont"
        df_cont["value"] = df_cont["med_dose"].astype(float)
        df_cat = df[df["med_dose"].isna()].copy()
        df_cat["feature_kind"] = "cat"
        df_cat["value"] = "given"
        for d in (df_cont, df_cat):
            pieces.append(d[["stay_id", "t_hours", "feature_name", "feature_kind", "value"]])
        log.info(f"  {table}: cont={len(df_cont):,}, cat={len(df_cat):,}")
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def extract_resp_support(clif_root: Path, cohort_idx: pd.DataFrame, log) -> pd.DataFrame:
    p = clif_root / "clif_respiratory_support.parquet"
    if not p.exists():
        log.warning("  clif_respiratory_support missing"); return pd.DataFrame()
    # Wide table: pull device + several ventilator settings as separate features
    use_cols = [
        "hospitalization_id", "recorded_dttm",
        "device_category", "mode_category",
        "fio2_set", "peep_set", "tidal_volume_set", "resp_rate_set",
        "tidal_volume_obs", "resp_rate_obs", "peak_inspiratory_pressure_obs",
        "minute_vent_obs", "mean_airway_pressure_obs",
    ]
    df = pd.read_parquet(p, columns=use_cols)
    df["recorded_dttm"] = _strip_tz(df["recorded_dttm"])
    df = df.dropna(subset=["recorded_dttm"])
    df = attach_stay(df, cohort_idx, "recorded_dttm")

    pieces = []
    # Categorical features
    for col, name in [("device_category", "resp::device"), ("mode_category", "resp::mode")]:
        if col in df.columns:
            sub = df.dropna(subset=[col])[["stay_id", "t_hours"]].copy()
            sub["feature_name"] = name
            sub["feature_kind"] = "cat"
            sub["value"] = df.dropna(subset=[col])[col].astype(str)
            pieces.append(sub)
    # Continuous features
    for col in [
        "fio2_set", "peep_set", "tidal_volume_set", "resp_rate_set",
        "tidal_volume_obs", "resp_rate_obs", "peak_inspiratory_pressure_obs",
        "minute_vent_obs", "mean_airway_pressure_obs",
    ]:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col])[["stay_id", "t_hours"]].copy()
        sub["feature_name"] = f"resp::{col}"
        sub["feature_kind"] = "cont"
        sub["value"] = df.dropna(subset=[col])[col].astype(float)
        pieces.append(sub)
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    log.info(f"  respiratory_support: {len(out):,} events")
    return out


def extract_assessments(clif_root: Path, cohort_idx: pd.DataFrame, log) -> pd.DataFrame:
    p = clif_root / "clif_patient_assessments.parquet"
    if not p.exists():
        log.warning("  clif_patient_assessments missing"); return pd.DataFrame()
    df = pd.read_parquet(p, columns=[
        "hospitalization_id", "recorded_dttm", "assessment_category",
        "numerical_value", "categorical_value",
    ])
    df["recorded_dttm"] = _strip_tz(df["recorded_dttm"])
    df = df.dropna(subset=["assessment_category", "recorded_dttm"])
    df = attach_stay(df, cohort_idx, "recorded_dttm")
    df["feature_name"] = "assess::" + df["assessment_category"].astype(str)

    cont = df.dropna(subset=["numerical_value"]).copy()
    cont["feature_kind"] = "cont"; cont["value"] = cont["numerical_value"].astype(float)

    cat = df[df["numerical_value"].isna() & df["categorical_value"].notna()].copy()
    cat["feature_kind"] = "cat"; cat["value"] = cat["categorical_value"].astype(str)

    out = pd.concat([
        cont[["stay_id", "t_hours", "feature_name", "feature_kind", "value"]],
        cat[["stay_id", "t_hours", "feature_name", "feature_kind", "value"]],
    ], ignore_index=True)
    log.info(f"  assessments: cont={len(cont):,}, cat={len(cat):,}")
    return out


def extract_demographics(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in cohort.iterrows():
        rows.append((r.stay_id, 0.0, "demo::gender", "cat", str(r.gender) if pd.notna(r.gender) else "UNK"))
        age_bin = f"age_{int(r.admission_age // 10) * 10}s"
        rows.append((r.stay_id, 0.0, "demo::age_decade", "cat", age_bin))
    return pd.DataFrame(rows, columns=["stay_id", "t_hours", "feature_name", "feature_kind", "value"])


def extract_diagnoses(clif_root: Path, cohort: pd.DataFrame, log) -> pd.DataFrame:
    """ICD chapter as static categorical (t=0 per hospitalization)."""
    p = clif_root / "clif_hospital_diagnosis.parquet"
    if not p.exists():
        log.warning("  clif_hospital_diagnosis missing"); return pd.DataFrame()
    dx = pd.read_parquet(p, columns=["hospitalization_id", "diagnosis_code", "diagnosis_code_format"])
    cohort_keys = cohort[["hadm_id", "stay_id"]].rename(columns={"hadm_id": "hospitalization_id"})
    dx = dx.merge(cohort_keys, on="hospitalization_id", how="inner")

    def chapter(row):
        code = str(row["diagnosis_code"]).strip()
        fmt = str(row["diagnosis_code_format"]).lower()
        if "10" in fmt:
            return f"icd10_{code[:1]}" if code else "icd10_UNK"
        # ICD-9 chapter buckets
        try:
            n = int(code[:3])
        except Exception:
            return "icd9_UNK"
        for lo, hi, name in [
            (1, 139, "infectious"), (140, 239, "neoplasms"), (240, 279, "endocrine"),
            (280, 289, "blood"), (290, 319, "mental"), (320, 389, "nervous"),
            (390, 459, "circulatory"), (460, 519, "respiratory"), (520, 579, "digestive"),
            (580, 629, "genitourinary"), (630, 679, "pregnancy"), (680, 709, "skin"),
            (710, 739, "musculoskeletal"), (740, 759, "congenital"), (760, 779, "perinatal"),
            (780, 799, "symptoms"), (800, 999, "injury"),
        ]:
            if lo <= n <= hi:
                return f"icd9_{name}"
        return "icd9_UNK"

    dx["chap"] = dx.apply(chapter, axis=1)
    out = dx[["stay_id", "chap"]].drop_duplicates()
    out["t_hours"] = 0.0
    out["feature_name"] = "diag::chapter"
    out["feature_kind"] = "cat"
    out = out.rename(columns={"chap": "value"})[["stay_id", "t_hours", "feature_name", "feature_kind", "value"]]
    log.info(f"  diagnoses: {len(out):,} chapter events")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("02_events", cfg["paths"]["work_dir"])
    out_dir = stage_dir(cfg, "events")

    cohort_path = Path(cfg["paths"]["work_dir"]) / "cohort" / "cohort.parquet"
    if not cohort_path.exists():
        raise SystemExit(f"Run stage 01 first; missing {cohort_path}")
    cohort = pd.read_parquet(cohort_path)
    log.info(f"Cohort size: {len(cohort):,}")

    # Build cohort_idx for fast lookup: hospitalization_id → (stay_id, intime, outtime)
    cohort_idx = cohort[["hadm_id", "stay_id", "intime", "outtime"]].rename(
        columns={"hadm_id": "hospitalization_id"}
    ).set_index("hospitalization_id")
    for c in ["intime", "outtime"]:
        if hasattr(cohort_idx[c].dt, "tz") and cohort_idx[c].dt.tz is not None:
            cohort_idx[c] = cohort_idx[c].dt.tz_localize(None)

    clif_root = Path(cfg["paths"]["clif_root"])
    log.info(f"CLIF root: {clif_root}")

    pieces = []
    log.info("Extracting vitals…")
    pieces.append(extract_vitals(clif_root, cohort_idx, log))
    log.info("Extracting labs…")
    pieces.append(extract_labs(clif_root, cohort_idx, log))
    log.info("Extracting meds…")
    pieces.append(extract_meds(clif_root, cohort_idx, log))
    log.info("Extracting respiratory support…")
    pieces.append(extract_resp_support(clif_root, cohort_idx, log))
    log.info("Extracting assessments…")
    pieces.append(extract_assessments(clif_root, cohort_idx, log))

    if cfg["features"].get("use_demographics", True):
        log.info("Extracting demographics…")
        pieces.append(extract_demographics(cohort))
    if cfg["features"].get("use_diagnoses", True):
        log.info("Extracting diagnoses…")
        pieces.append(extract_diagnoses(clif_root, cohort, log))

    events = pd.concat([p for p in pieces if len(p)], ignore_index=True)

    # Drop NaN values, then cast everything to string for parquet (continuous values
    # are parsed back to float in stage 03). Mixed-type object columns crash pyarrow.
    events = events.dropna(subset=["value"])
    is_cont = events["feature_kind"] == "cont"
    # Validate continuous values are numeric before casting
    cont_numeric = pd.to_numeric(events.loc[is_cont, "value"], errors="coerce")
    keep_mask = ~is_cont | cont_numeric.notna()
    events = events.loc[keep_mask].copy()
    events["value"] = events["value"].astype(str)

    log.info(f"Total events: {len(events):,}")
    log.info(f"Distinct features: {events['feature_name'].nunique():,}")
    log.info(f"Distinct stays touched: {events['stay_id'].nunique():,}")

    out_path = Path(out_dir) / "events.parquet"
    events.to_parquet(out_path, index=False)
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
