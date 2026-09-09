"""Stage 02 v2 — extract downstream-task labels for each hospitalization.

Produces a single `labels.parquet` keyed by stay_id with columns:

Binary classification (10):
  y_mortality       — died_in_hosp (already in cohort)
  y_los_gt_7d       — LOS > 7 days
  y_readmit_30d     — next admission of same patient within 30 days (or NaN if no future)
  y_celiac          — ICD K90.0 / 579.0 present in this hospitalization
  y_masld           — ICD K76.0 / K75.81 / 571.5 / 571.8
  y_ami             — ICD I21.x / 410.x
  y_stroke          — ICD I63.x / 433.x / 434.x

Regression (3, at 4h post-admission):
  y_reg_platelets   — platelet_count value (float, NaN if missing)
  y_reg_creatinine  — creatinine value
  y_reg_spo2        — SpO2 from clif_vitals (proxy for "Oxygen"; po2_arterial coverage too low)

Each task additionally has a `m_<task>` mask column (1 if label available, 0 otherwise).

Note on chronic vs acute (per Pang et al. + the project brief):
- The Pang/ORA setup uses "future first diagnosis" as label. MIMIC-IV CLIF median
  follow-up = 0 years (half of patients have 1 admit total), so future-event labeling
  yields very few positives. We instead use "present diagnosis at this admission",
  which is a standard simplification for cross-sectional EHR FM evals when followup
  is limited. Time-to-event variants can be added later via lifelines on a
  filtered cohort of patients with ≥1 follow-up admission.

Run:
    python src_v2/02_extract_labels.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_config, setup_logging, stage_dir


# --- ICD code prefix sets (uppercased, dots removed) ---
ICD_CODES = {
    "celiac": {"icd10": ["K900"], "icd9": ["5790"]},
    "masld":  {"icd10": ["K760", "K7581"], "icd9": ["5715", "5718"]},
    "ami":    {"icd10": [f"I21{x}" for x in "0123456789A"] + [f"I22{x}" for x in "0123456789"],
               "icd9":  [f"410{x}" for x in "0123456789"]},
    "stroke": {"icd10": [f"I63{x}" for x in "0123456789A"],
               "icd9":  [f"43{a}{b}" for a in ["3","4"] for b in "0123456789"]},
}


def prefix_match(code_series: pd.Series, prefixes: list[str]) -> pd.Series:
    s = code_series.astype(str)
    mask = pd.Series(False, index=s.index)
    for p in prefixes:
        mask = mask | s.str.startswith(p)
    return mask


def label_diseases(cohort: pd.DataFrame, dx: pd.DataFrame, log) -> pd.DataFrame:
    dx["code_norm"] = dx["diagnosis_code"].astype(str).str.upper().str.replace(".", "", regex=False)
    out = cohort[["stay_id", "hospitalization_id"]].copy()
    for name, codes in ICD_CODES.items():
        all_codes = codes["icd10"] + codes["icd9"]
        mask = prefix_match(dx["code_norm"], all_codes)
        positive_hosp = dx.loc[mask, "hospitalization_id"].unique()
        out[f"y_{name}"] = out["hospitalization_id"].isin(positive_hosp).astype(int)
        log.info(f"  y_{name}: {out[f'y_{name}'].sum():,} positives ({out[f'y_{name}'].mean()*100:.2f}%)")
    return out.drop(columns=["hospitalization_id"])


def label_readmission_30d(cohort: pd.DataFrame, log) -> pd.DataFrame:
    """For each hospitalization H of patient P, find next admission of P after discharge of H.
    Label 1 if next admission is within 30 days of discharge, 0 if not, NaN if no next admission
    (right-censored, mask=0).
    """
    c = cohort.sort_values(["patient_id", "admission_dttm"]).reset_index(drop=True).copy()
    c["next_admit"] = c.groupby("patient_id")["admission_dttm"].shift(-1)
    c["days_to_next"] = (c["next_admit"] - c["discharge_dttm"]).dt.total_seconds() / (24 * 3600)
    c["y_readmit_30d"] = (c["days_to_next"] <= 30).astype("Int64")  # nullable int
    c.loc[c["next_admit"].isna(), "y_readmit_30d"] = pd.NA  # censored
    c["m_readmit_30d"] = c["y_readmit_30d"].notna().astype(int)
    c.loc[c["m_readmit_30d"] == 0, "y_readmit_30d"] = 0  # placeholder 0 for masked
    c["y_readmit_30d"] = c["y_readmit_30d"].astype(int)
    log.info(f"  y_readmit_30d: {c['y_readmit_30d'].sum():,} positives, "
             f"{c['m_readmit_30d'].sum():,} unmasked")
    return c[["stay_id", "y_readmit_30d", "m_readmit_30d"]]


def label_regression_labs(cohort: pd.DataFrame, clif_root: Path, log) -> pd.DataFrame:
    """Pull the first lab value within [0, 4]h after admission for each target lab.
    Uses clif_labs for platelets + creatinine, and clif_vitals (vital_category='spo2') for SpO2.
    """
    labs = pd.read_parquet(
        clif_root / "clif_labs.parquet",
        columns=["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"],
    )
    if hasattr(labs["lab_result_dttm"].dt, "tz") and labs["lab_result_dttm"].dt.tz is not None:
        labs["lab_result_dttm"] = labs["lab_result_dttm"].dt.tz_localize(None)
    labs = labs.dropna(subset=["lab_result_dttm", "lab_value_numeric"])

    # Time relative to admission
    admit_lookup = cohort.set_index("hospitalization_id")[["admission_dttm", "stay_id"]]
    labs = labs.join(admit_lookup, on="hospitalization_id", how="inner")
    labs["hrs"] = (labs["lab_result_dttm"] - labs["admission_dttm"]).dt.total_seconds() / 3600.0
    labs_4h = labs[(labs["hrs"] >= 0) & (labs["hrs"] <= 4)]

    out = cohort[["stay_id"]].copy()
    for cat, alias in [("platelet_count", "platelets"), ("creatinine", "creatinine")]:
        sub = labs_4h[labs_4h["lab_category"] == cat]
        # Use the earliest measurement per hospitalization
        first = sub.sort_values("hrs").groupby("stay_id")["lab_value_numeric"].first()
        out[f"y_reg_{alias}"] = out["stay_id"].map(first).astype(float)
        out[f"m_reg_{alias}"] = out[f"y_reg_{alias}"].notna().astype(int)
        log.info(f"  y_reg_{alias}: {out[f'm_reg_{alias}'].sum():,} measurements available")

    # SpO2 from vitals
    vitals = pd.read_parquet(
        clif_root / "clif_vitals.parquet",
        columns=["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"],
    )
    if hasattr(vitals["recorded_dttm"].dt, "tz") and vitals["recorded_dttm"].dt.tz is not None:
        vitals["recorded_dttm"] = vitals["recorded_dttm"].dt.tz_localize(None)
    spo2 = vitals[(vitals["vital_category"].astype(str).str.lower() == "spo2") &
                  vitals["vital_value"].notna()].copy()
    spo2 = spo2.join(admit_lookup, on="hospitalization_id", how="inner")
    spo2["hrs"] = (spo2["recorded_dttm"] - spo2["admission_dttm"]).dt.total_seconds() / 3600.0
    spo2_4h = spo2[(spo2["hrs"] >= 0) & (spo2["hrs"] <= 4)]
    first_spo2 = spo2_4h.sort_values("hrs").groupby("stay_id")["vital_value"].first()
    out["y_reg_spo2"] = out["stay_id"].map(first_spo2).astype(float)
    out["m_reg_spo2"] = out["y_reg_spo2"].notna().astype(int)
    log.info(f"  y_reg_spo2: {out['m_reg_spo2'].sum():,} measurements available")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("02_extract_labels", cfg["paths"]["work_dir"])
    out_dir = stage_dir(cfg, "downstream_v2")

    clif_root = Path(cfg["paths"]["clif_root"])
    cohort_path = Path(cfg["paths"]["work_dir"]) / "cohort_v2" / "cohort.parquet"
    if not cohort_path.exists():
        raise SystemExit(f"Run 01_build_cohort_v2 first. Missing {cohort_path}")
    cohort = pd.read_parquet(cohort_path)
    log.info(f"Loaded cohort_v2: {len(cohort):,} hospitalizations")

    # ---- Mortality + LOS (from cohort) ----
    out = cohort[["stay_id", "patient_id", "split"]].copy()
    out["y_mortality"] = cohort["died_in_hosp"]
    out["m_mortality"] = 1
    out["y_los_gt_7d"] = (cohort["hospital_los_hours"] > 7 * 24).astype(int)
    out["m_los_gt_7d"] = 1
    log.info(f"  y_mortality: {out['y_mortality'].sum():,} positives ({out['y_mortality'].mean()*100:.2f}%)")
    log.info(f"  y_los_gt_7d: {out['y_los_gt_7d'].sum():,} positives ({out['y_los_gt_7d'].mean()*100:.2f}%)")

    # ---- Readmission ----
    log.info("Computing 30-day readmission labels…")
    readmit = label_readmission_30d(cohort, log)
    out = out.merge(readmit, on="stay_id", how="left")

    # ---- Disease present at this admission ----
    log.info("Loading hospital diagnoses + matching ICD codes…")
    dx = pd.read_parquet(
        clif_root / "clif_hospital_diagnosis.parquet",
        columns=["hospitalization_id", "diagnosis_code"],
    )
    dis_labels = label_diseases(cohort, dx, log)
    out = out.merge(dis_labels, on="stay_id", how="left")
    for name in ICD_CODES.keys():
        out[f"m_{name}"] = 1  # always observed if we have hospital_diagnosis rows

    # ---- Regression labs ----
    log.info("Pulling regression labs (Platelets, Creatinine, SpO2 at 4h)…")
    reg = label_regression_labs(cohort, clif_root, log)
    out = out.merge(reg, on="stay_id", how="left")

    # ---- Save ----
    out_path = out_dir / "labels.parquet"
    out.to_parquet(out_path, index=False)
    log.info(f"\nWrote labels: {len(out):,} rows × {len(out.columns)} cols → {out_path}")

    # ---- Summary table per task ----
    log.info(f"\n=== Per-task summary ===")
    tasks = ["mortality", "los_gt_7d", "readmit_30d", "celiac", "masld", "ami", "stroke",
             "reg_platelets", "reg_creatinine", "reg_spo2"]
    rows = []
    for t in tasks:
        y_col = f"y_{t}"
        m_col = f"m_{t}"
        m = out[m_col] == 1
        if t.startswith("reg_"):
            vals = out.loc[m, y_col]
            rows.append((t, "regression", m.sum(), "—", f"mean={vals.mean():.1f}, std={vals.std():.1f}"))
        else:
            pos = (out.loc[m, y_col] == 1).sum()
            rows.append((t, "binary", m.sum(), pos, f"{pos / m.sum() * 100:.2f}%"))
    summary = pd.DataFrame(rows, columns=["task", "type", "n_unmasked", "n_positive", "rate/stats"])
    log.info(f"\n{summary.to_string(index=False)}")

    # Per-split breakdown for top tasks
    log.info(f"\n=== Per-split positive rate ===")
    for t in ["mortality", "los_gt_7d", "readmit_30d", "ami", "stroke"]:
        log.info(f"  {t}: " + ", ".join(
            f"{s}: {out.loc[out['split'] == s, f'y_{t}'].mean()*100:.2f}%"
            for s in ["train", "val", "test"]
        ))


if __name__ == "__main__":
    main()
