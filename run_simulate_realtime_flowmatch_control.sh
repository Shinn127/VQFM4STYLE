#!/usr/bin/env bash
set -euo pipefail

# FlowMatchVQVAEController realtime rollout runner.
#
# Optional overrides:
#   DB=... FLOW_CKPT=... OUT=... PREFIX=... \
#   VQVAE_CKPT=... NORM_STATS=... \
#   NUM_FRAMES=300 SEED=... SOURCE_STYLE= TARGET_STYLE= START_INDEX=... \
#   SOLVER=euler|midpoint|heun ODE_STEPS=... NOISE_STD=1.0 DEVICE=cuda \
#   BENCHMARK=1 WARMUP_FRAMES=5 \
#   VISUALIZE=1 VIS_BACKEND=genoview LOCAL_ONLY=0 FPS=60 DT=0.016666666666666666 \
#   GENO_GAP=2.4 DATA_ROOT=... BVH_REF=... \
#   ./NewCode/run_simulate_realtime_flowmatch_control.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/run_utils.sh"

print_info() {
  printf '[Info] %-15s: %s\n' "$1" "$2"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${SCRIPT_DIR}/simulate_realtime_flowmatch_control.py"

DB="${DB:-${SCRIPT_DIR}/database_100sty_375_memmap}"
FLOW_CKPT="${FLOW_CKPT:-${SCRIPT_DIR}/checkpoints/flowmatch_vqvae_controller/best.pt}"
VQVAE_CKPT="${VQVAE_CKPT:-}"
NORM_STATS="${NORM_STATS:-}"
OUT="${OUT:-${SCRIPT_DIR}/rollout_vis}"
PREFIX="${PREFIX:-flowmatch_realtime_300}"

NUM_FRAMES="${NUM_FRAMES:-300}"
SEED="${SEED:-}"
SOURCE_STYLE="${SOURCE_STYLE:-}"
TARGET_STYLE="${TARGET_STYLE:-}"
START_INDEX="${START_INDEX:-}"

SOLVER="${SOLVER:-}"
ODE_STEPS="${ODE_STEPS:-}"
NOISE_STD="${NOISE_STD:-1.0}"
DEVICE="${DEVICE:-cuda}"
BENCHMARK="${BENCHMARK:-0}"
WARMUP_FRAMES="${WARMUP_FRAMES:-5}"

VISUALIZE="${VISUALIZE:-1}"
VIS_BACKEND="${VIS_BACKEND:-genoview}"
LOCAL_ONLY="${LOCAL_ONLY:-0}"
FPS="${FPS:-60}"
DT="${DT:-0.016666666666666666}"
GENO_GAP="${GENO_GAP:-2.4}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data/100sty}"
BVH_REF="${BVH_REF:-}"

ru_require_file "${RUN_PY}" "simulate script"
ru_require_file "${FLOW_CKPT}" "flow checkpoint"
ru_require_python_modules "${PYTHON_BIN}" numpy torch h5py
if ru_is_enabled "${VISUALIZE}" && [[ "${VIS_BACKEND}" == "genoview" ]]; then
  ru_require_python_modules "${PYTHON_BIN}" raylib
fi
[[ -e "${DB}" ]] || ru_die "db path not found: ${DB}"

if [[ -n "${VQVAE_CKPT}" ]]; then
  ru_require_file "${VQVAE_CKPT}" "VQ-VAE checkpoint"
fi
if [[ -n "${NORM_STATS}" ]]; then
  ru_require_file "${NORM_STATS}" "norm stats"
fi
if [[ -n "${BVH_REF}" ]]; then
  ru_require_file "${BVH_REF}" "BVH ref"
fi
if [[ -n "${SOLVER}" ]]; then
  case "${SOLVER}" in
    euler|midpoint|heun)
      ;;
    *)
      ru_die "SOLVER must be one of: euler, midpoint, heun (got ${SOLVER})"
      ;;
  esac
fi
case "${VIS_BACKEND}" in
  genoview|matplotlib)
    ;;
  *)
    ru_die "VIS_BACKEND must be one of: genoview, matplotlib (got ${VIS_BACKEND})"
    ;;
esac

cmd=(
  "${PYTHON_BIN}" "${RUN_PY}"
  --db "${DB}"
  --flow-ckpt "${FLOW_CKPT}"
  --out-dir "${OUT}"
  --prefix "${PREFIX}"
  --num-frames "${NUM_FRAMES}"
  --noise-std "${NOISE_STD}"
  --device "${DEVICE}"
  --warmup-frames "${WARMUP_FRAMES}"
  --fps "${FPS}"
  --dt "${DT}"
  --vis-backend "${VIS_BACKEND}"
  --geno-gap "${GENO_GAP}"
  --data-root "${DATA_ROOT}"
)

ru_append_if_nonempty cmd --seed "${SEED}"
ru_append_if_nonempty cmd --vqvae-ckpt "${VQVAE_CKPT}"
ru_append_if_nonempty cmd --norm-stats "${NORM_STATS}"
ru_append_if_nonempty cmd --source-style "${SOURCE_STYLE}"
ru_append_if_nonempty cmd --target-style "${TARGET_STYLE}"
ru_append_if_nonempty cmd --start-index "${START_INDEX}"
ru_append_if_nonempty cmd --solver "${SOLVER}"
ru_append_if_nonempty cmd --ode-steps "${ODE_STEPS}"
ru_append_if_nonempty cmd --bvh-ref "${BVH_REF}"
ru_append_if_enabled cmd "${LOCAL_ONLY}" --local-only
ru_append_if_enabled cmd "${BENCHMARK}" --benchmark
if ! ru_is_enabled "${VISUALIZE}"; then
  cmd+=(--no-visualize)
fi

print_info "db" "${DB}"
print_info "flow ckpt" "${FLOW_CKPT}"
if [[ -n "${VQVAE_CKPT}" ]]; then
  print_info "vqvae ckpt" "${VQVAE_CKPT}"
else
  print_info "vqvae ckpt" "infer from flow checkpoint args"
fi
if [[ -n "${NORM_STATS}" ]]; then
  print_info "norm stats" "${NORM_STATS}"
else
  print_info "norm stats" "infer from flow checkpoint / VQ-VAE"
fi
print_info "out dir" "${OUT}"
print_info "prefix" "${PREFIX}"
print_info "visualize" "${VISUALIZE}"
print_info "vis backend" "${VIS_BACKEND}"
print_info "geno gap" "${GENO_GAP}"
print_info "vis mode" "left=reference | right=prediction"
print_info "benchmark" "${BENCHMARK}"
if ru_is_enabled "${BENCHMARK}"; then
  print_info "warmup frames" "${WARMUP_FRAMES}"
fi
if [[ -n "${SEED}" ]]; then
  print_info "seed" "${SEED}"
else
  print_info "seed" "random each run"
fi
if [[ -n "${SOLVER}" ]]; then
  print_info "solver" "${SOLVER}"
else
  print_info "solver" "inherit from flow checkpoint"
fi
if [[ -n "${ODE_STEPS}" ]]; then
  print_info "ode steps" "${ODE_STEPS}"
else
  print_info "ode steps" "inherit from flow checkpoint"
fi

cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
