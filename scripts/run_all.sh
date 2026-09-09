#!/usr/bin/env bash
# Run the full preprocessing pipeline end-to-end.
# Edit configs/preprocess.yaml first to point paths.mimic_root + paths.clif_root.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "[$(date +%H:%M:%S)] 01 — cohort"
python3 src/01_build_cohort.py

echo "[$(date +%H:%M:%S)] 02 — events"
python3 src/02_extract_events.py

echo "[$(date +%H:%M:%S)] 03 — vocab + bin edges"
python3 src/03_build_vocab.py

echo "[$(date +%H:%M:%S)] 04 — tokenize"
python3 src/04_tokenize.py

echo "[$(date +%H:%M:%S)] 05 — chunk"
python3 src/05_chunk.py

echo "[$(date +%H:%M:%S)] DONE — chunks under outputs/chunks/{train,val,test}"
