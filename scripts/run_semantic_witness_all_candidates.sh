#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PILOT_ROOT:?Set PILOT_ROOT to the completed pilot directory containing audit_manifest.json}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_FILE="${CONFIG_FILE:-configs/semantic_witness_v2.json}"
RUN_ROOT="${RUN_ROOT:-runs/semantic_witness_all_candidates_$(date +%Y%m%d_%H%M%S)}"
RUN_GPUS="${GPUS:-${GPU:-0}}"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/task.log") 2>&1

echo "[$(date '+%F %T')] All-candidate frozen semantic witness evaluation"
echo "  pilot_root=$PILOT_ROOT"
echo "  run_root=$RUN_ROOT"
echo "  gpus=$RUN_GPUS"
echo "  multiplicity=Bonferroni across every completed finite-score pilot candidate"

"$PYTHON_BIN" scripts/smoke_semantic_witness.py

env CONFIG_FILE="$CONFIG_FILE" PILOT_ROOT="$PILOT_ROOT" RUN_ROOT="$RUN_ROOT" \
  GPUS="$RUN_GPUS" ALL_CANDIDATES=1 PLAN_ONLY="${PLAN_ONLY:-0}" \
  bash scripts/run_semantic_witness.sh

if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
  echo "[$(date '+%F %T')] Preflight complete. Re-run without PLAN_ONLY using the same RUN_ROOT."
else
  echo "[$(date '+%F %T')] Complete: $RUN_ROOT"
  echo "  summary=$RUN_ROOT/existence_summary.json"
fi
