#!/usr/bin/env bash
set -euo pipefail

# Train independently sampled same-size support modules and run the unchanged
# held-out continuous downstream pipeline. This is a module-selection null:
# it is different from random hidden-direction dosing.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CONFIG_FILE="${CONFIG_FILE:-configs/ultrachat_random_support_null_v1.env}"
export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export FORCE="${FORCE:-0}"
export FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"
export SHARED_BASELINE_ROOT="${SHARED_BASELINE_ROOT:-}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="runs/ultrachat_random_support_null_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_ROOT
mkdir -p "$RUN_ROOT"

exec > >(tee -a "$RUN_ROOT/random_support_launcher.log") 2>&1
printf '[%s] UltraChat random-support module null\n' "$(date '+%F %T')"
printf '  config=%s\n  run_root=%s\n  gpu0=%s gpu1=%s\n  source_baseline_root=%s\n' \
  "$CONFIG_FILE" "$RUN_ROOT" "$GPU0" "$GPU1" "${SHARED_BASELINE_ROOT:-none}"
printf '  null=uniform candidate-pool support; mode=random_support seed=%s\n' \
  "${RANDOM_SUPPORT_SEED:-777}"

exec bash scripts/run_generated_step_experiment.sh
