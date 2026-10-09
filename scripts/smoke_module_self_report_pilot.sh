#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m py_compile \
  scripts/audit_module_self_report.py \
  scripts/run_module_self_report_pilot.py
bash -n scripts/run_module_self_report_pilot_2gpu.sh
PLAN_ONLY=1 RUN_ROOT="${PLAN_ROOT:-runs/module_self_report_plan_smoke_$$}" \
  bash scripts/run_module_self_report_pilot_2gpu.sh

if [[ "${RUN_MODEL:-0}" == 1 ]]; then
  GPU_IDS="${GPU:-0}" ONLY="${ONLY:-mathqa/llama2_7b}" \
    N_PROMPTS=8 RUN_ROOT="${MODEL_ROOT:-runs/module_self_report_model_smoke_$$}" \
    bash scripts/run_module_self_report_pilot_2gpu.sh
fi
