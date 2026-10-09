#!/usr/bin/env bash
set -euo pipefail

AUDIT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AUDIT_REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_FILE="${CONFIG_FILE:-configs/semantic_leakage_audit_v1.json}"
RUN_ROOT="${RUN_ROOT:-runs/semantic_leakage_audit_$(date +%Y%m%d_%H%M%S)}"
read -r -a audit_gpus <<< "${GPUS:-0 1}"
audit_args=(--config "$CONFIG_FILE" --output-dir "$RUN_ROOT" --gpus "${audit_gpus[@]}")
if [[ -n "${SOURCE_ROOT:-}" ]]; then
  audit_args+=(--source-root "$SOURCE_ROOT")
fi
if [[ -n "${TASKS:-}" ]]; then
  read -r -a audit_tasks <<< "$TASKS"
  audit_args+=(--tasks "${audit_tasks[@]}")
fi
[[ "${PLAN_ONLY:-0}" != 1 ]] || audit_args+=(--plan-only)
[[ "${SMOKE:-0}" != 1 ]] || audit_args+=(--smoke)
exec "$PYTHON_BIN" -u scripts/run_semantic_leakage_audit.py "${audit_args[@]}"
