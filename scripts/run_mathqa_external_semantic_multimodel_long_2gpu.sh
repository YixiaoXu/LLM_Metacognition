#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CONFIG_FILE="${CONFIG_FILE:-configs/mathqa_external_semantic_continuous_m8_long_v2.env}"
export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export FORCE="${FORCE:-0}"
export FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

[[ -s "$CONFIG_FILE" ]] || { echo "Missing config: $CONFIG_FILE" >&2; exit 2; }
mkdir -p "${RUN_ROOT:-runs}"
printf '[%s] MathQA external-semantic continuous multi-model experiment\n' "$(date '+%F %T')"
printf '  config=%s\n  gpu0=%s gpu1=%s\n' "$CONFIG_FILE" "$GPU0" "$GPU1"

exec bash scripts/run_generated_step_experiment.sh
