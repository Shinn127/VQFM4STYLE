#!/usr/bin/env bash
# shellcheck shell=bash

ru_die() {
  echo "[Error] $*" >&2
  exit 1
}

ru_require_file() {
  local path="$1"
  local label="${2:-file}"
  [[ -f "${path}" ]] || ru_die "${label} not found: ${path}"
}

ru_require_python_modules() {
  local python_bin="$1"
  shift
  local modules=("$@")

  local imports
  imports="$(IFS=,; echo "${modules[*]}")"

  if ! "${python_bin}" -c "import ${imports}" >/dev/null 2>&1; then
    local modules_text
    modules_text="$(IFS=', '; echo "${modules[*]}")"
    echo "[Error] Missing python deps in current env (need at least ${modules_text})." >&2
    echo "        Try: conda activate <your_env>  (or install deps in this python)." >&2
    exit 1
  fi
}

ru_is_enabled() {
  local value="${1:-0}"
  case "${value,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

ru_append_if_nonempty() {
  local -n arr_ref="$1"
  local flag="$2"
  local value="${3:-}"

  if [[ -n "${value}" ]]; then
    arr_ref+=("${flag}" "${value}")
  fi
}

ru_append_if_enabled() {
  local -n arr_ref="$1"
  local enabled="${2:-0}"
  shift 2

  if ru_is_enabled "${enabled}"; then
    arr_ref+=("$@")
  fi
}
