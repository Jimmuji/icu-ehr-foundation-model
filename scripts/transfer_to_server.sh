#!/usr/bin/env bash
# rsync chunked tensors + vocab artifacts to the H100 server for pretraining.
# Configure SERVER_HOST and REMOTE_DIR below.

set -euo pipefail
cd "$(dirname "$0")/.."

SERVER_HOST="${SERVER_HOST:-user@host}"        # e.g. user@gpu-host
REMOTE_DIR="${REMOTE_DIR:-/data/ehrformer}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=accept-new}"

# Only ship what training needs: vocab artifacts + chunked .pt files.
# Skip raw events/tokens (large, regeneratable).

rsync -avzP -e "ssh $SSH_OPTS" \
  --include 'vocab/' --include 'vocab/**' \
  --include 'cohort/' --include 'cohort/**' \
  --include 'chunks/' --include 'chunks/**' \
  --exclude '*' \
  outputs/ "${SERVER_HOST}:${REMOTE_DIR}/"

echo "Transfer complete to ${SERVER_HOST}:${REMOTE_DIR}/"
