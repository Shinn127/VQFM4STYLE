#!/usr/bin/env bash
set -euo pipefail

# Optional overrides:
#   DB=... OUT=... EPOCHS=... BATCH_SIZE=... SEQ_LEN=... NUM_WORKERS=... \
#   GRAD_ACCUM_STEPS=... PREFETCH_FACTOR=... VAL_RATIO=... TEST_RATIO=... \
#   TEST_FRACTION=... LAT_ACC_WEIGHT=... \
#   COMPILE=1 LOAD_INTO_MEMORY=1 \
#   ./NewCode/run_train_vqvae_375to291_fast_ema.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${SCRIPT_DIR}/train_vqvae_375to291.py"

DB="${DB:-${SCRIPT_DIR}/database_100sty_375_memmap}"
OUT="${OUT:-${SCRIPT_DIR}/checkpoints/vqvae_375to291_fast_ema}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEQ_LEN="${SEQ_LEN:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
DEVICE="${DEVICE:-cuda}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
COMPILE="${COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-max-autotune}"
LOAD_INTO_MEMORY="${LOAD_INTO_MEMORY:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-1.0}"
LAT_ACC_WEIGHT="${LAT_ACC_WEIGHT:-0.5}"

ru_require_file "${TRAIN_PY}" "train script"
ru_require_python_modules "${PYTHON_BIN}" numpy torch

cmd=(
  "${PYTHON_BIN}" "${TRAIN_PY}"
  --db "${DB}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --seq-len "${SEQ_LEN}"
  --stride 8
  --val-ratio "${VAL_RATIO}"
  --test-ratio "${TEST_RATIO}"
  --test-fraction "${TEST_FRACTION}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --persistent-workers
  --prefetch-factor "${PREFETCH_FACTOR}"
  --pin-memory
  --normalize
  --max-windows-for-stats 0
  --lr 2e-4
  --min-lr 1e-5
  --weight-decay 1e-4
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --grad-clip 1.0
  --amp
  --amp-dtype "${AMP_DTYPE}"
  --tf32
  --fused-adamw
  --recon-type mse
  --temporal-loss-type smooth_l1
  --recon-weight 1.0
  --vq-weight 1.0
  --vel-weight 1.0
  --acc-weight 1.0
  --lat-acc-weight "${LAT_ACC_WEIGHT}"
  --use-ema-codebook
  --ema-decay 0.99
  --ema-eps 1e-5
  --save-every 10
  --val-every 5
  --log-interval 100
)

ru_append_if_enabled cmd "${LOAD_INTO_MEMORY}" --load-into-memory
ru_append_if_enabled cmd "${COMPILE}" --compile --compile-mode "${COMPILE_MODE}"

cmd+=("$@")
"${cmd[@]}"
