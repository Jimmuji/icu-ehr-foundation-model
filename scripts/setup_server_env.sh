#!/usr/bin/env bash
# Create a clean conda environment for EHRFormer pretraining on the H100 server.
# Idempotent: safe to re-run.

set -euo pipefail

ENV_NAME="${ENV_NAME:-ehrformer}"
PY_VER="${PY_VER:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu124}"   # H100 wants cu12.x

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
command -v conda >/dev/null 2>&1 || { echo "[setup] FATAL: conda not found"; exit 1; }

echo "[setup] target env: $ENV_NAME (python=$PY_VER, cuda=$CUDA_TAG)"
echo "[setup] conda: $(which conda)"

# --- 1. conda env ---
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] env $ENV_NAME exists, reusing"
else
  conda create -y -n "$ENV_NAME" python="$PY_VER"
fi
conda activate "$ENV_NAME"

# --- 2. PyTorch (CUDA build) ---
pip install --upgrade pip
pip install "torch==2.4.*" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# --- 3. EHRFormer deps ---
pip install \
  "pytorch-lightning>=2.2,<2.5" \
  "transformers>=4.40,<4.50" \
  einops \
  pandas \
  pyarrow \
  numpy \
  tqdm \
  wandb \
  scikit-learn \
  torchmetrics \
  pyyaml

# cosine_annealing_warmup is only on GitHub, not PyPI
pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup.git

# --- 4. Optional: flash-attn (massive speedup on H100). Skip if it fails. ---
pip install ninja packaging
pip install flash-attn==2.6.3 --no-build-isolation || echo "[setup] flash-attn install failed (optional, continuing)"

# --- 5. Sanity check ---
python - <<'PY'
import torch, pytorch_lightning, transformers, einops
print(f"  torch              : {torch.__version__} (cuda={torch.version.cuda})")
print(f"  pytorch-lightning  : {pytorch_lightning.__version__}")
print(f"  transformers       : {transformers.__version__}")
print(f"  cuda available     : {torch.cuda.is_available()}")
print(f"  device count       : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu {i}: {p.name}, {p.total_memory/1e9:.0f} GB")
try:
    import flash_attn
    print(f"  flash-attn         : {flash_attn.__version__}")
except ImportError:
    print("  flash-attn         : (not installed)")
PY

echo "[setup] done. Activate with: conda activate $ENV_NAME"
