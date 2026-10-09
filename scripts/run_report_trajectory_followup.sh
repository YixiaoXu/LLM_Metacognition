#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${RUN_ROOT:?Set RUN_ROOT to runs/<new-tag>}"
CONFIG_FILE="${CONFIG_FILE:-configs/report_trajectory_followup_v1.json}"
GPU_IDS="${GPU_IDS:-0 1}"
read -r -a GPUS <<< "$GPU_IDS"
mkdir -p "$RUN_ROOT"
ARGS=(--config "$CONFIG_FILE" --output-dir "$RUN_ROOT" --gpus "${GPUS[@]}")
[[ -n "${SOURCE_ROOT:-}" ]] && ARGS+=(--source-root "$SOURCE_ROOT")
[[ "${PLAN_ONLY:-0}" == 1 ]] && ARGS+=(--plan-only)
exec "${PYTHON_BIN:-python}" scripts/run_report_trajectory_followup.py "${ARGS[@]}"
