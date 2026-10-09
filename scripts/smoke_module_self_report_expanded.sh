#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m py_compile \
  metacog/models/chat_tokens.py \
  scripts/audit_module_self_report.py \
  scripts/run_module_self_report_pilot.py
bash -n scripts/run_module_self_report_pilot_2gpu.sh \
  scripts/run_module_self_report_expanded_2gpu.sh

RUN_ROOT="${SMOKE_ROOT:-runs/module_self_report_expanded_smoke_$(date +%Y%m%d_%H%M%S)}"
PLAN_ONLY=1 RUN_ROOT="$RUN_ROOT" bash scripts/run_module_self_report_expanded_2gpu.sh
PREFLIGHT_ONLY=1 RUN_ROOT="$RUN_ROOT" bash scripts/run_module_self_report_expanded_2gpu.sh

if [[ "${RUN_MODEL:-0}" == 1 ]]; then
  for condition in \
    mathqa/llama31_8b \
    ultrachat/qwen3_8b \
    ultrachat/deepseek_qwen7b \
    beavertails/llama2_7b; do
    name="${condition//\//_}"
    GPU_IDS="${GPU:-0}" ONLY="$condition" N_PROMPTS=8 \
      RUN_ROOT="${RUN_ROOT}_${name}" bash scripts/run_module_self_report_expanded_2gpu.sh
  done
fi
