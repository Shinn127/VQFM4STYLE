#!/usr/bin/env bash
set -euo pipefail

# Optional overrides:
#   DB=... VQVAE_CKPT=... NORM_STATS=... OUT=... CHUNK_ROWS=4096 DEVICE=cuda \
#   ./NewCode/run_precompute_vq_targets.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${SCRIPT_DIR}/precompute_vq_targets.py"

DB="${DB:-${SCRIPT_DIR}/database_100sty_375_memmap}"
VQVAE_CKPT="${VQVAE_CKPT:-${SCRIPT_DIR}/checkpoints/vqvae_375to291_fast_ema/best.pt}"
NORM_STATS="${NORM_STATS:-${SCRIPT_DIR}/checkpoints/vqvae_375to291_fast_ema/norm_stats.npz}"
OUT="${OUT:-${SCRIPT_DIR}/cache/vq_targets_vqvae_375to291_fast_ema}"
CHUNK_ROWS="${CHUNK_ROWS:-4096}"
DEVICE="${DEVICE:-cuda}"

ru_require_file "${RUN_PY}" "precompute script"
ru_require_python_modules "${PYTHON_BIN}" numpy torch

cmd=(
  "${PYTHON_BIN}" "${RUN_PY}"
  --db "${DB}"
  --vqvae-ckpt "${VQVAE_CKPT}"
  --norm-stats "${NORM_STATS}"
  --out-dir "${OUT}"
  --chunk-rows "${CHUNK_ROWS}"
  --device "${DEVICE}"
)

cmd+=("$@")
"${cmd[@]}"
