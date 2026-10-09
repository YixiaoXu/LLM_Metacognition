#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${RUN_ROOT:?Set RUN_ROOT to a new runs/<tag> directory}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/strict_main24_direct_three_ring_20260923_140753}"
GPU_IDS="${GPU_IDS:-0 1}"
CONFIG_FILE="${CONFIG_FILE:-configs/strict_followup_group_v1.json}"
PLAN_ONLY="${PLAN_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

read -r -a GPUS <<< "$GPU_IDS"
mkdir -p "$RUN_ROOT"
ARGS=(--config "$CONFIG_FILE" --source-root "$SOURCE_ROOT" --output-dir "$RUN_ROOT" --gpus "${GPUS[@]}")
[[ "$PLAN_ONLY" == 1 ]] && ARGS+=(--plan-only)
exec "$PYTHON_BIN" scripts/run_strict_followup_group.py "${ARGS[@]}"
