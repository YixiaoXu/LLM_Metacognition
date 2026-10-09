#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_FILE="${CONFIG_FILE:-configs/module_self_report_pilot_v1.json}"
RUN_ROOT="${RUN_ROOT:-runs/module_self_report_pilot_$(date +%Y%m%d_%H%M%S)}"
read -r -a GPUS <<< "${GPU_IDS:-0 1}"
ARGS=(--config "$CONFIG_FILE" --output-dir "$RUN_ROOT" --gpus "${GPUS[@]}")
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
  ARGS+=(--plan-only)
fi
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  ARGS+=(--preflight-only)
fi
if [[ -n "${ONLY:-}" ]]; then
  ARGS+=(--only "$ONLY")
fi
if [[ -n "${N_PROMPTS:-}" ]]; then
  ARGS+=(--n-prompts "$N_PROMPTS")
fi
python -u scripts/run_module_self_report_pilot.py "${ARGS[@]}"
