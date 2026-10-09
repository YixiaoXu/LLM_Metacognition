#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_FILE="${CONFIG_FILE:-configs/semantic_witness_v2.json}"
RUN_ROOT="${RUN_ROOT:-runs/semantic_witness_$(date +%Y%m%d_%H%M%S)}"
read -r -a witness_gpus <<< "${GPUS:-${GPU:-0}}"
args=(--config "$CONFIG_FILE" --output-dir "$RUN_ROOT" --gpus "${witness_gpus[@]}")
[[ -z "${PILOT_ROOT:-}" ]] || args+=(--pilot-root "$PILOT_ROOT")
[[ -z "${WITNESS_COUNT:-}" ]] || args+=(--witness-count "$WITNESS_COUNT")
[[ "${ALL_CANDIDATES:-0}" != 1 ]] || args+=(--all-candidates)
if [[ -n "${WITNESSES:-}" ]]; then
  read -r -a exact_witnesses <<< "$WITNESSES"
  args+=(--witness "${exact_witnesses[@]}")
fi
if [[ -n "${TASKS:-}" ]]; then
  read -r -a patterns <<< "$TASKS"
  args+=(--tasks "${patterns[@]}")
fi
[[ "${PLAN_ONLY:-0}" != 1 ]] || args+=(--plan-only)
[[ "${SMOKE:-0}" != 1 ]] || args+=(--smoke)
exec "$PYTHON_BIN" -u scripts/run_semantic_witness.py "${args[@]}"
