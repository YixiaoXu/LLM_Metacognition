#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_FILE="${CONFIG_FILE:-configs/mathqa_proxy_confidence_qwen3_4b_v1.env}"
RUN_ROOT="${RUN_ROOT:-runs/mathqa_proxy_confidence_qwen3_4b_$(date +%Y%m%d_%H%M%S)}"
FORCE="${FORCE:-0}"
FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"

[[ -s "$CONFIG_FILE" ]] || { echo "Missing config: $CONFIG_FILE" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

"$PYTHON_BIN" -m py_compile \
  scripts/cluster_intervention.py \
  scripts/audit_continuous_module_dose.py \
  scripts/analyze_proxy_confidence.py
bash -n scripts/run_generated_step_experiment.sh \
  scripts/run_model_pair.sh scripts/run_continuous_module.sh

printf '[%s] MathQA proxy-confidence experiment\n' "$(date '+%F %T')"
printf '  target=Qwen3-4B semantic_reference=Llama-3.1-8B layer=16\n'
printf '  primary_proxy=mean negative entropy over the first 16 generated tokens\n'
printf '  run_root=%s gpu=%s config=%s\n' "$RUN_ROOT" "$GPU" "$CONFIG_FILE"

env CONFIG_FILE="$CONFIG_FILE" GPU0="$GPU" GPU1="$GPU" \
  RUN_ROOT="$RUN_ROOT" FORCE="$FORCE" FORCE_ACTIVATIONS="$FORCE_ACTIVATIONS" \
  bash scripts/run_generated_step_experiment.sh \
  2>&1 | tee "$RUN_ROOT/task.log"

printf '[%s] COMPLETE: %s\n' "$(date '+%F %T')" "$RUN_ROOT"
