#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CONFIG_FILE="${CONFIG_FILE:-configs/ultrachat_ablation_small_v1.env}"
export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export FORCE="${FORCE:-0}"
export FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="runs/ultrachat_ablation_small_$(date +%Y%m%d_%H%M%S)"
  export RUN_ROOT
fi

mkdir -p "$RUN_ROOT"
printf '[%s] UltraChat small-model ablation suite\n' "$(date '+%F %T')"
printf '  config=%s\n  run_root=%s\n  gpu0=%s gpu1=%s\n' \
  "$CONFIG_FILE" "$RUN_ROOT" "$GPU0" "$GPU1"
printf '  panels=5a external-reference, 5b single-vs-persistent, 5c random-control, 6a-6d dose/critical\n'
printf '  semantic references=qwen3_0p6b,qwen3_1p7b,gemma3_1b_it,qwen3_4b\n'

exec bash scripts/run_generated_step_experiment.sh
