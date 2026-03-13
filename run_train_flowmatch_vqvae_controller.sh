#!/usr/bin/env bash
set -euo pipefail

# FlowMatchVQVAEController training runner.
#
# Optional overrides:
#   DB=... VQVAE_CKPT=... OUT=... \
#   USE_VQ_TARGET_CACHE=auto|1|0 VQ_TARGET_CACHE=... NORM_STATS=... \
#   EPOCHS=... BATCH_SIZE=... SEQ_LEN=... STRIDE=... \
#   NUM_WORKERS=... PREFETCH_FACTOR=... GRAD_ACCUM_STEPS=... \
#   VAL_RATIO=... TEST_RATIO=... TRAIN_FRACTION=... VAL_FRACTION=... TEST_FRACTION=... \
#   LR=... MIN_LR=... BETA1=... BETA2=... WEIGHT_DECAY=... GRAD_CLIP=... \
#   NUM_STYLES=... CONTENT_DIM=... STYLE_DIM=... TIME_EMBED_DIM=... \
#   ENCODER_HIDDEN_DIM=... FLOW_HIDDEN_DIM=... FLOW_LAYERS=... DROPOUT=... \
#   FLOW_LOSS_TYPE=... RECON_TYPE=... FLOW_WEIGHT=... RECON_WEIGHT=... NOISE_STD=... \
#   SOLVER=euler|midpoint|heun SOLVER_STEPS=... \
#   SEED=... SUBSET_SEED=... MAX_WINDOWS_FOR_STATS=... STATS_WINDOWS_CHUNK_SIZE=... \
#   AMP=1 TF32=1 CUDNN_BENCHMARK=1 FUSED_ADAMW=1 MATMUL_PRECISION=high \
#   PIN_MEMORY=1 PERSISTENT_WORKERS=1 NORMALIZE=1 LOAD_INTO_MEMORY=0 \
#   COMPILE=0 COMPILE_MODE=max-autotune COMPILE_FULLGRAPH=0 RESUME=... RESET_OPTIMIZER=0 \
#   SAVE_EVERY=10 VAL_EVERY=5 LOG_INTERVAL=100 \
#   ./NewCode/run_train_flowmatch_vqvae_controller.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/run_utils.sh"

append_bool_arg() {
  local -n arr_ref="$1"
  local enabled="${2:-0}"
  local flag="$3"

  if ru_is_enabled "${enabled}"; then
    arr_ref+=("--${flag}")
  else
    arr_ref+=("--no-${flag}")
  fi
}

print_info() {
  printf '[Info] %-15s: %s\n' "$1" "$2"
}

infer_default_vq_target_cache() {
  local vqvae_ckpt="$1"
  local ckpt_dir_name
  ckpt_dir_name="$(basename "$(dirname "${vqvae_ckpt}")")"
  local candidate="${SCRIPT_DIR}/cache/vq_targets_${ckpt_dir_name}"
  if [[ -d "${candidate}" ]]; then
    printf '%s' "${candidate}"
  fi
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${SCRIPT_DIR}/train_flowmatch_vqvae_controller.py"

DB="${DB:-${SCRIPT_DIR}/database_100sty_375_memmap}"
VQVAE_CKPT="${VQVAE_CKPT:-${SCRIPT_DIR}/checkpoints/vqvae_375to291_fast_ema/best.pt}"
OUT="${OUT:-${SCRIPT_DIR}/checkpoints/flowmatch_vqvae_controller}"

USE_VQ_TARGET_CACHE="${USE_VQ_TARGET_CACHE:-auto}"
VQ_TARGET_CACHE="${VQ_TARGET_CACHE:-}"
NORM_STATS="${NORM_STATS:-}"
RESUME="${RESUME:-}"
RESET_OPTIMIZER="${RESET_OPTIMIZER:-0}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SEQ_LEN="${SEQ_LEN:-32}"
STRIDE="${STRIDE:-24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
VAL_FRACTION="${VAL_FRACTION:-1.0}"
TEST_FRACTION="${TEST_FRACTION:-1.0}"
SUBSET_SEED="${SUBSET_SEED:-123}"

LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-1e-5}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.99}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

NUM_STYLES="${NUM_STYLES:-0}"
CONTENT_DIM="${CONTENT_DIM:-256}"
STYLE_DIM="${STYLE_DIM:-128}"
TIME_EMBED_DIM="${TIME_EMBED_DIM:-64}"
ENCODER_HIDDEN_DIM="${ENCODER_HIDDEN_DIM:-512}"
FLOW_HIDDEN_DIM="${FLOW_HIDDEN_DIM:-512}"
FLOW_LAYERS="${FLOW_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.1}"

FLOW_LOSS_TYPE="${FLOW_LOSS_TYPE:-mse}"
RECON_TYPE="${RECON_TYPE:-mse}"
FLOW_WEIGHT="${FLOW_WEIGHT:-1.0}"
RECON_WEIGHT="${RECON_WEIGHT:-0.25}"
NOISE_STD="${NOISE_STD:-1.0}"
SOLVER="${SOLVER:-euler}"
SOLVER_STEPS="${SOLVER_STEPS:-4}"

SEED="${SEED:-42}"
MAX_WINDOWS_FOR_STATS="${MAX_WINDOWS_FOR_STATS:-0}"
STATS_WINDOWS_CHUNK_SIZE="${STATS_WINDOWS_CHUNK_SIZE:-4096}"

DEVICE="${DEVICE:-cuda}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
AMP="${AMP:-1}"
TF32="${TF32:-1}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"
FUSED_ADAMW="${FUSED_ADAMW:-1}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
NORMALIZE="${NORMALIZE:-1}"
LOAD_INTO_MEMORY="${LOAD_INTO_MEMORY:-0}"
COMPILE="${COMPILE:-0}"
COMPILE_MODE="${COMPILE_MODE:-max-autotune}"
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"

SAVE_EVERY="${SAVE_EVERY:-10}"
VAL_EVERY="${VAL_EVERY:-5}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

ru_require_file "${TRAIN_PY}" "train script"
ru_require_file "${VQVAE_CKPT}" "VQ-VAE checkpoint"
ru_require_python_modules "${PYTHON_BIN}" numpy torch
[[ -d "${DB}" ]] || ru_die "db directory not found: ${DB}"

case "${SOLVER}" in
  euler|midpoint|heun)
    ;;
  *)
    ru_die "SOLVER must be one of: euler, midpoint, heun (got ${SOLVER})"
    ;;
esac

case "${USE_VQ_TARGET_CACHE,,}" in
  auto|1|true|yes|on|0|false|no|off)
    ;;
  *)
    ru_die "USE_VQ_TARGET_CACHE must be auto|1|0, got: ${USE_VQ_TARGET_CACHE}"
    ;;
esac

if ru_is_enabled "${COMPILE_FULLGRAPH}" && ! ru_is_enabled "${COMPILE}"; then
  ru_die "COMPILE_FULLGRAPH=1 requires COMPILE=1"
fi

if ru_is_enabled "${NORMALIZE}"; then
  if [[ -z "${VQ_TARGET_CACHE}" ]]; then
    case "${USE_VQ_TARGET_CACHE,,}" in
      auto)
        VQ_TARGET_CACHE="$(infer_default_vq_target_cache "${VQVAE_CKPT}")"
        ;;
      1|true|yes|on)
        VQ_TARGET_CACHE="$(infer_default_vq_target_cache "${VQVAE_CKPT}")"
        [[ -n "${VQ_TARGET_CACHE}" ]] || ru_die \
          "USE_VQ_TARGET_CACHE=${USE_VQ_TARGET_CACHE} but no default cache found. Run ./NewCode/run_precompute_vq_targets.sh first or set VQ_TARGET_CACHE explicitly."
        ;;
      0|false|no|off)
        ;;
    esac
  fi
else
  case "${USE_VQ_TARGET_CACHE,,}" in
    1|true|yes|on)
      ru_die "USE_VQ_TARGET_CACHE=${USE_VQ_TARGET_CACHE} requires NORMALIZE=1"
      ;;
  esac
fi

if ! ru_is_enabled "${NORMALIZE}"; then
  [[ -z "${VQ_TARGET_CACHE}" ]] || ru_die "VQ_TARGET_CACHE requires NORMALIZE=1"
  [[ -z "${NORM_STATS}" ]] || ru_die "NORM_STATS requires NORMALIZE=1"
fi

if [[ -n "${VQ_TARGET_CACHE}" ]]; then
  [[ -d "${VQ_TARGET_CACHE}" ]] || ru_die "VQ target cache directory not found: ${VQ_TARGET_CACHE}"
fi

if ru_is_enabled "${NORMALIZE}" && [[ -z "${NORM_STATS}" && -z "${VQ_TARGET_CACHE}" ]]; then
  default_norm_stats="$(dirname "${VQVAE_CKPT}")/norm_stats.npz"
  if [[ -f "${default_norm_stats}" ]]; then
    NORM_STATS="${default_norm_stats}"
  fi
fi
if [[ -n "${NORM_STATS}" ]]; then
  ru_require_file "${NORM_STATS}" "norm stats"
fi

cmd=(
  "${PYTHON_BIN}" "${TRAIN_PY}"
  --db "${DB}"
  --vqvae-ckpt "${VQVAE_CKPT}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --seq-len "${SEQ_LEN}"
  --stride "${STRIDE}"
  --val-ratio "${VAL_RATIO}"
  --test-ratio "${TEST_RATIO}"
  --train-fraction "${TRAIN_FRACTION}"
  --val-fraction "${VAL_FRACTION}"
  --test-fraction "${TEST_FRACTION}"
  --subset-seed "${SUBSET_SEED}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --max-windows-for-stats "${MAX_WINDOWS_FOR_STATS}"
  --stats-windows-chunk-size "${STATS_WINDOWS_CHUNK_SIZE}"
  --lr "${LR}"
  --min-lr "${MIN_LR}"
  --beta1 "${BETA1}"
  --beta2 "${BETA2}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --grad-clip "${GRAD_CLIP}"
  --seed "${SEED}"
  --matmul-precision "${MATMUL_PRECISION}"
  --amp-dtype "${AMP_DTYPE}"
  --num-styles "${NUM_STYLES}"
  --content-dim "${CONTENT_DIM}"
  --style-dim "${STYLE_DIM}"
  --time-embed-dim "${TIME_EMBED_DIM}"
  --encoder-hidden-dim "${ENCODER_HIDDEN_DIM}"
  --flow-hidden-dim "${FLOW_HIDDEN_DIM}"
  --flow-layers "${FLOW_LAYERS}"
  --dropout "${DROPOUT}"
  --flow-loss-type "${FLOW_LOSS_TYPE}"
  --recon-type "${RECON_TYPE}"
  --flow-weight "${FLOW_WEIGHT}"
  --recon-weight "${RECON_WEIGHT}"
  --noise-std "${NOISE_STD}"
  --solver "${SOLVER}"
  --solver-steps "${SOLVER_STEPS}"
  --save-every "${SAVE_EVERY}"
  --val-every "${VAL_EVERY}"
  --log-interval "${LOG_INTERVAL}"
)

append_bool_arg cmd "${PERSISTENT_WORKERS}" persistent-workers
append_bool_arg cmd "${PIN_MEMORY}" pin-memory
append_bool_arg cmd "${NORMALIZE}" normalize
append_bool_arg cmd "${AMP}" amp
append_bool_arg cmd "${TF32}" tf32
append_bool_arg cmd "${CUDNN_BENCHMARK}" cudnn-benchmark
append_bool_arg cmd "${FUSED_ADAMW}" fused-adamw

ru_append_if_nonempty cmd --vq-target-cache "${VQ_TARGET_CACHE}"
ru_append_if_nonempty cmd --norm-stats "${NORM_STATS}"
ru_append_if_nonempty cmd --resume "${RESUME}"
ru_append_if_enabled cmd "${RESET_OPTIMIZER}" --reset-optimizer
ru_append_if_enabled cmd "${LOAD_INTO_MEMORY}" --load-into-memory
ru_append_if_enabled cmd "${COMPILE}" --compile --compile-mode "${COMPILE_MODE}"
ru_append_if_enabled cmd "${COMPILE_FULLGRAPH}" --compile-fullgraph

print_info "db" "${DB}"
print_info "vqvae ckpt" "${VQVAE_CKPT}"
print_info "out dir" "${OUT}"
print_info "cache mode" "${USE_VQ_TARGET_CACHE}"
if [[ -n "${VQ_TARGET_CACHE}" ]]; then
  print_info "vq target cache" "${VQ_TARGET_CACHE}"
else
  print_info "vq target cache" "disabled"
fi
if [[ -n "${NORM_STATS}" ]]; then
  print_info "norm stats" "${NORM_STATS}"
elif [[ -n "${VQ_TARGET_CACHE}" ]]; then
  print_info "norm stats" "infer from cache metadata"
else
  print_info "norm stats" "estimate from train split"
fi
print_info "solver" "${SOLVER}"
print_info "solver steps" "${SOLVER_STEPS}"
print_info "compile" "${COMPILE} (${COMPILE_MODE}, fullgraph=${COMPILE_FULLGRAPH})"

cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
