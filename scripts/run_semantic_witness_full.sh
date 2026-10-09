#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
FULL_RUN_ROOT="${RUN_ROOT:-runs/semantic_witness_full_$(date +%Y%m%d_%H%M%S)}"
PILOT_RUN_ROOT="$FULL_RUN_ROOT/pilot"
FINAL_RUN_ROOT="$FULL_RUN_ROOT/final"
PILOT_CONFIG="${PILOT_CONFIG:-configs/semantic_witness_pilot_v2.json}"
FINAL_CONFIG="${FINAL_CONFIG:-configs/semantic_witness_v2.json}"
RUN_GPUS="${GPUS:-${GPU:-0}}"

mkdir -p "$FULL_RUN_ROOT"
exec > >(tee -a "$FULL_RUN_ROOT/task.log") 2>&1

echo "[$(date '+%F %T')] Frozen-witness semantic audit"
echo "  run_root=$FULL_RUN_ROOT"
echo "  source_root=${SOURCE_ROOT:-from $PILOT_CONFIG}"
echo "  gpus=$RUN_GPUS smoke=${SMOKE:-0}"

"$PYTHON_BIN" scripts/smoke_semantic_witness.py

echo "[$(date '+%F %T')] Stage 1/2: independent pilot and global witness freeze"
env CONFIG_FILE="$PILOT_CONFIG" RUN_ROOT="$PILOT_RUN_ROOT" GPUS="$RUN_GPUS" \
  SOURCE_ROOT="${SOURCE_ROOT:-}" TASKS="${TASKS:-}" SMOKE="${SMOKE:-0}" \
  PLAN_ONLY="${PLAN_ONLY:-0}" bash scripts/run_semantic_leakage_audit.sh

if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
  echo "[$(date '+%F %T')] Pilot preflight complete. Re-run without PLAN_ONLY using the same RUN_ROOT."
  exit 0
fi

echo "[$(date '+%F %T')] Stage 2/2: frozen predictors and untouched final prompts"
env CONFIG_FILE="$FINAL_CONFIG" PILOT_ROOT="$PILOT_RUN_ROOT" RUN_ROOT="$FINAL_RUN_ROOT" \
  GPUS="$RUN_GPUS" WITNESS_COUNT="${WITNESS_COUNT:-1}" SMOKE="${SMOKE:-0}" \
  bash scripts/run_semantic_witness.sh

echo "[$(date '+%F %T')] Complete: $FULL_RUN_ROOT"
echo "  final_summary=$FINAL_RUN_ROOT/existence_summary.json"
