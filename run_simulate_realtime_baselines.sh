#!/usr/bin/env bash
set -euo pipefail

# Unified baseline simulation runner.
#
# Optional overrides:
#   MODEL=vqvae|lmm|mvae MODE=rollout|reconstruct \
#   DB=... CKPT=... NORM_STATS=... OUT=... PREFIX=... \
#   NUM_FRAMES=300 STYLE=Zombie START_INDEX=1234 SEED=1234 DEVICE=cuda NOISE_STD=1.0 \
#   VISUALIZE=1 LOCAL_ONLY=0 DATA_ROOT=... BVH_REF=... \
#   ./NewCode/run_simulate_realtime_baselines.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${SCRIPT_DIR}/simulate_realtime_baselines.py"

MODEL="${MODEL:-vqvae}"
MODE="${MODE:-rollout}"
DB="${DB:-${SCRIPT_DIR}/database_100sty_375_memmap}"
CKPT="${CKPT:-}"
NORM_STATS="${NORM_STATS:-}"
OUT="${OUT:-${SCRIPT_DIR}/rollout_vis}"
PREFIX="${PREFIX:-}"
NUM_FRAMES="${NUM_FRAMES:-300}"
STYLE="${STYLE:-}"
START_INDEX="${START_INDEX:-}"
SEED="${SEED:-3907}"
DEVICE="${DEVICE:-cuda}"
NOISE_STD="${NOISE_STD:-1.0}"
FPS="${FPS:-60}"
DT="${DT:-0.016666666666666666}"
VISUALIZE="${VISUALIZE:-1}"
LOCAL_ONLY="${LOCAL_ONLY:-0}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data/100sty}"
BVH_REF="${BVH_REF:-}"

ru_require_file "${RUN_PY}" "unified simulate script"
ru_require_python_modules "${PYTHON_BIN}" numpy torch h5py

# Prefer explicit CKPT; otherwise auto-pick a sensible default for vqvae that
# matches either old or new training output directories.
if [[ -z "${CKPT}" && "${MODEL}" == "vqvae" ]]; then
  candidate_fast="${SCRIPT_DIR}/checkpoints/vqvae_375to291_fast_ema/best.pt"
  candidate_plain="${SCRIPT_DIR}/checkpoints/vqvae_375to291/best.pt"
  if [[ -f "${candidate_fast}" ]]; then
    CKPT="${candidate_fast}"
  elif [[ -f "${candidate_plain}" ]]; then
    CKPT="${candidate_plain}"
  fi
fi

cmd=(
  "${PYTHON_BIN}" "${RUN_PY}"
  --model "${MODEL}"
  --mode "${MODE}"
  --db "${DB}"
  --num-frames "${NUM_FRAMES}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --noise-std "${NOISE_STD}"
  --fps "${FPS}"
  --dt "${DT}"
  --out-dir "${OUT}"
)

ru_append_if_nonempty cmd --ckpt "${CKPT}"
ru_append_if_nonempty cmd --norm-stats "${NORM_STATS}"
ru_append_if_nonempty cmd --prefix "${PREFIX}"
ru_append_if_nonempty cmd --style "${STYLE}"
ru_append_if_nonempty cmd --start-index "${START_INDEX}"
ru_append_if_nonempty cmd --data-root "${DATA_ROOT}"
ru_append_if_nonempty cmd --bvh-ref "${BVH_REF}"
ru_append_if_enabled cmd "${LOCAL_ONLY}" --local-only

if ! ru_is_enabled "${VISUALIZE}"; then
  cmd+=(--no-visualize)
fi

cmd+=("$@")
"${cmd[@]}"
