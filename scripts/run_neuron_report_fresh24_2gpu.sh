#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONFIG_FILE="${CONFIG_FILE:-configs/neuron_report_search_fresh24_v3.json}"
RUN_ROOT="${RUN_ROOT:-runs/neuron_report_fresh24_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0 1}"
read -r -a GPUS <<< "$GPU_IDS"
mkdir -p "$RUN_ROOT"
ARGS=(--config "$CONFIG_FILE" --output-dir "$RUN_ROOT" --gpus "${GPUS[@]}"
      --require-fresh-measurement --reuse-root "")
[[ -n "${SOURCE_ROOT:-}" ]] && ARGS+=(--source-root "$SOURCE_ROOT")
[[ "${PLAN_ONLY:-0}" == 1 ]] && ARGS+=(--plan-only)
exec "${PYTHON_BIN:-python}" -u scripts/run_neuron_report_search.py "${ARGS[@]}"
