#!/usr/bin/env bash
# Launch EHRFormer pretraining on the H100 server.
#
# Assumes:
#   - conda env 'ehrformer' set up (via scripts/setup_server_env.sh)
#   - EHRFormer repo cloned at $EHRFORMER_DIR (default: $REPO/EHRFormer)
#   - Preprocessing artifacts under outputs/{cohort,vocab,chunks}
#   - configs/pretrain_mimic.json filled in (by scripts/adapt_config.py)
#
# Usage:
#   bash scripts/launch_pretrain.sh                   # uses CUDA_VISIBLE_DEVICES from env, or all GPUs
#   CUDA_VISIBLE_DEVICES=3,4,5,6 bash scripts/launch_pretrain.sh

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

ENV_NAME="${ENV_NAME:-ehrformer}"
EHRFORMER_DIR="${EHRFORMER_DIR:-$REPO/EHRFormer}"
CONFIG_JSON="${CONFIG_JSON:-$REPO/configs/pretrain_mimic.json}"

# --- 0. Bootstrap conda for non-interactive shells ---
if ! command -v conda >/dev/null 2>&1; then
  for CONDA_HOME in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      source "$CONDA_HOME/etc/profile.d/conda.sh"
      break
    fi
  done
fi
command -v conda >/dev/null 2>&1 || { echo "[launch] FATAL: conda not found"; exit 1; }

# --- 1. Sanity: env + repo + config ---
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[launch] env $ENV_NAME missing; run scripts/setup_server_env.sh first" >&2
  exit 1
fi
conda activate "$ENV_NAME"

if [ ! -d "$EHRFORMER_DIR" ]; then
  echo "[launch] cloning EHRFormer into $EHRFORMER_DIR"
  git clone https://github.com/kaiwang13/EHRFormer.git "$EHRFORMER_DIR"
fi

if [ ! -f "$CONFIG_JSON" ]; then
  echo "[launch] missing $CONFIG_JSON; run scripts/adapt_config.py first" >&2
  exit 1
fi

# --- 2. Decide GPU set ---
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_IDS="$CUDA_VISIBLE_DEVICES"
else
  GPU_IDS=$(python - <<'PY'
import json, torch, os
cfg = json.load(open(os.environ['CONFIG_JSON']))
ids = cfg.get('n_gpus', [])
if isinstance(ids, int):
    ids = list(range(ids))
print(','.join(str(i) for i in ids))
PY
)
fi
N_GPU=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | tr -d ' ')
echo "[launch] using GPUs: $GPU_IDS  (N=$N_GPU)"

# Patch n_gpus in the JSON to match selected devices (PL maps onto visible devices)
python - <<PY
import json, sys
p = "$CONFIG_JSON"
ids = [int(x) for x in "$GPU_IDS".split(',') if x != '']
cfg = json.load(open(p))
# After CUDA_VISIBLE_DEVICES is set, devices are reindexed 0..N-1
cfg['n_gpus'] = list(range(len(ids)))
json.dump(cfg, open(p, 'w'), indent=2)
PY

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# --- 3. Stage our adapted config + feat_info into the EHRFormer repo ---
mkdir -p "$EHRFORMER_DIR/configs"
cp -v "$CONFIG_JSON" "$EHRFORMER_DIR/configs/pretrain.json"

# Make the chunked data + feat_info reachable via the relative paths in the JSON
cd "$EHRFORMER_DIR"

LOG="$REPO/outputs/pretrain.log"
mkdir -p "$(dirname "$LOG")"
echo "[launch] log → $LOG"

# --- 4. Launch (PyTorch Lightning handles DDP internally based on n_gpus) ---
python pretrain.py 2>&1 | tee "$LOG"
