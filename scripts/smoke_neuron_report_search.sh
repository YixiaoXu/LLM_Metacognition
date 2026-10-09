#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
"${PYTHON_BIN:-python}" -m py_compile \
  scripts/audit_neuron_report_search.py scripts/run_neuron_report_search.py
bash -n scripts/run_neuron_report_search_2gpu.sh
PLAN_ONLY=1 RUN_ROOT="${RUN_ROOT:-runs/neuron_report_search_plan_smoke_v1_21pairs}" \
  bash scripts/run_neuron_report_search_2gpu.sh
if "${PYTHON_BIN:-python}" -c 'import torch, scipy, sklearn' >/dev/null 2>&1; then
  "${PYTHON_BIN:-python}" scripts/audit_neuron_report_search.py --help >/dev/null
  "${PYTHON_BIN:-python}" scripts/run_neuron_report_search.py --help >/dev/null
  if [[ "${RUN_MODEL:-0}" == 1 ]]; then
    PLAN="${RUN_ROOT:-runs/neuron_report_search_plan_smoke_v1_21pairs}/frozen_plan.json"
    PAIR="${SMOKE_PAIR:-mathqa/llama2_7b}"
    IFS=$'\t' read -r TARGET LAYER ACTIVATION SEMANTIC AXIS < <(
      jq -r --arg pair "$PAIR" '.conditions[] |
        select((.dataset + "/" + .target) == $pair) |
        [.target, .layer_i, .activation_dir, .semantic_cache, .axis_csv] | @tsv' "$PLAN")
    [[ -n "${TARGET:-}" && -n "${AXIS:-}" ]] || { echo "[smoke] unknown pair: $PAIR" >&2; exit 2; }
    CUDA_VISIBLE_DEVICES="${GPU:-0}" "${PYTHON_BIN:-python}" scripts/audit_neuron_report_search.py \
      --target "$TARGET" --layer-id "$LAYER" --activation-dir "$ACTIVATION" \
      --semantic-cache "$SEMANTIC" --axis-csv "$AXIS" \
      --output-dir "${RUN_ROOT:-runs/neuron_report_search_plan_smoke_v1_21pairs}/model_smoke/${PAIR//\//__}" \
      --semantic-fit-rows 200 --screen-a-pool 32 --candidate-neurons 8 \
      --semantic-bins 2 --screen-a-rows 8 --screen-b-candidates 4 \
      --screen-b-rows 16 --finalists 2 --evaluation-pool-rows 96 \
      --confirm-rows 40 --flip-rows 4 --bootstrap-samples 20
  fi
else
  if [[ "${RUN_MODEL:-0}" == 1 ]]; then
    echo "[smoke] torch/scipy/sklearn are required for RUN_MODEL=1" >&2
    exit 2
  fi
  echo "[smoke] torch/scipy/sklearn unavailable locally; model CLI deferred to server"
fi
echo "[smoke] report-only plan and syntax passed"
