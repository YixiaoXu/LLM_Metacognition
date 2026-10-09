#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-$GPU0}"
FORCE="${FORCE:-0}"
FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/mathqa_single_pair_full_${STAMP}}"
CONFIG_FILE="${CONFIG_FILE:-configs/mathqa_external_semantic_single_pair_full.env}"

[[ -s "$CONFIG_FILE" ]] || { echo "Missing config: $CONFIG_FILE" >&2; exit 2; }
python -m py_compile \
  scripts/cluster_intervention.py \
  scripts/audit_continuous_module_dose.py \
  scripts/train_decoupler_joint_v2.py
bash -n scripts/run_mathqa_experiment.sh scripts/run_model_pair.sh scripts/run_continuous_module.sh

mkdir -p "$RUN_ROOT"
echo "[$(date '+%F %T')] Single-pair full MathQA experiment"
echo "  config=$CONFIG_FILE"
echo "  pair=qwen3_4b -> semantic reference llama31_8b (layer 16)"
echo "  modules=1 support=4 gpu0=$GPU0 gpu1=$GPU1"
echo "  run_root=$RUN_ROOT"

env \
  CONFIG_FILE="$CONFIG_FILE" \
  GPU0="$GPU0" GPU1="$GPU1" \
  RUN_ROOT="$RUN_ROOT" FORCE="$FORCE" FORCE_ACTIVATIONS="$FORCE_ACTIVATIONS" \
  bash scripts/run_mathqa_experiment.sh \
  2>&1 | tee "$RUN_ROOT/task.log"

echo "[$(date '+%F %T')] COMPLETE: $RUN_ROOT"
