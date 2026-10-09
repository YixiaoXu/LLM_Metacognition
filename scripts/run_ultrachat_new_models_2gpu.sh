#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CONFIG_FILE="${CONFIG_FILE:-configs/ultrachat_external_semantic_new_models_v1.env}"
export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export FORCE="${FORCE:-0}"
export FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="runs/ultrachat_external_semantic_new_models_$(date +%Y%m%d_%H%M%S)"
  export RUN_ROOT
fi

mkdir -p "$RUN_ROOT"
printf '[%s] UltraChat new-model external-semantic experiment\n' "$(date '+%F %T')"
printf '  config=%s\n  run_root=%s\n  gpu0=%s gpu1=%s\n' \
  "$CONFIG_FILE" "$RUN_ROOT" "$GPU0" "$GPU1"
printf '  targets=Qwen3-0.6B,1.7B,14B,32B; Gemma3-1B,4B,12B\n'
printf '  semantic_reference=Qwen3-4B layer=16\n'
printf '  target_inference_batch=1 train_batch=512 probe_encode_batch=1024\n'

exec bash scripts/run_generated_step_experiment.sh
