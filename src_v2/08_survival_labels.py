"""Stage 08 v2 — time-to-event (survival) labels.

Adds a time-to-event task. The cleanest survival task
on MIMIC (given the short per-patient follow-up) is IN-HOSPITAL SURVIVAL:

  duration = hours from admission to in-hospital death OR discharge
  event    = 1 if the patient died in hospital (event observed)
             0 if discharged alive (right-CENSORED at discharge)

This is a proper survival setup: survivors are not labelled "negative", they
are censored at the time we stop observing them. That is exactly the
censoring mechanism MOTOR / SurvivEHR handle, applied to the one horizon
MIMIC observes cleanly.

(Post-discharge time-to-readmission is documented as a future extension: its
censoring time per patient is not well defined in MIMIC, so we keep the clean
in-hospital task as the headline.)

Output: survival_labels.parquet  [stay_id, surv_time_hours, surv_event, split]

Run:
    python src_v2/08_survival_labels.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

WORK = Path("/data/mimic_data/physionet.org/script/ehrformer/outputs")


def main():
    cohort = pd.read_parquet(WORK / "cohort_v2" / "cohort.parquet")
    out = cohort[["stay_id", "split"]].copy()
    # Duration: clamp to a small positive minimum so Cox has no zero/neg times
    out["surv_time_hours"] = cohort["hospital_los_hours"].clip(lower=0.1)
    out["surv_event"] = cohort["died_in_hosp"].astype(int)

    out_path = WORK / "downstream_v2" / "survival_labels.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Wrote {len(out):,} rows -> {out_path}")
    print("\n=== In-hospital survival summary ===")
    print(f"  events (deaths): {out['surv_event'].sum():,} ({out['surv_event'].mean()*100:.2f}%)")
    print(f"  censored (alive): {(out['surv_event']==0).sum():,}")
    print(f"  median duration (h): {out['surv_time_hours'].median():.1f}")
    for s in ["train", "val", "test"]:
        sub = out[out["split"] == s]
        print(f"  {s:5s}: n={len(sub):,} | event_rate={sub['surv_event'].mean()*100:.2f}% | "
              f"median_dur={sub['surv_time_hours'].median():.1f}h")


if __name__ == "__main__":
    main()
