#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${RUN_ROOT:?Set RUN_ROOT to runs/<new-tag>}"
CONFIG_FILE="${CONFIG_FILE:-configs/neuron_report_search_v1.json}"
GPU_IDS="${GPU_IDS:-0 1}"
read -r -a GPUS <<< "$GPU_IDS"
ARGS=(--config "$CONFIG_FILE" --output-dir "$RUN_ROOT" --gpus "${GPUS[@]}")
[[ -n "${SOURCE_ROOT:-}" ]] && ARGS+=(--source-root "$SOURCE_ROOT")
[[ -n "${PAIR_FILTER:-}" ]] && for pair in $PAIR_FILTER; do ARGS+=(--only "$pair"); done
[[ "${PLAN_ONLY:-0}" == 1 ]] && ARGS+=(--plan-only)
exec "${PYTHON_BIN:-python}" scripts/run_neuron_report_search.py "${ARGS[@]}"
