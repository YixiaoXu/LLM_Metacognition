#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CONFIG_FILE="${CONFIG_FILE:-configs/ultrachat_external_semantic_continuous_m4_v1.env}"
export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export FORCE="${FORCE:-0}"
export FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="runs/ultrachat_external_semantic_m4_multimodel_$(date +%Y%m%d_%H%M%S)"
  export RUN_ROOT
fi

mkdir -p "$RUN_ROOT"
printf '[%s] UltraChat full multimodel experiment\n' "$(date '+%F %T')"
printf '  config=%s\n  run_root=%s\n  gpu0=%s gpu1=%s\n' \
  "$CONFIG_FILE" "$RUN_ROOT" "$GPU0" "$GPU1"
printf '  stages=data -> target activations -> semantic activations -> decoupling -> modules -> baseline -> intervention\n'

exec bash scripts/run_ultrachat_experiment.sh
