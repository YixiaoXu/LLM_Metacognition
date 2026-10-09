#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORCE="${FORCE:-0}"
FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"
FORCE_BASELINE="${FORCE_BASELINE:-0}"
FORCE_ASSOCIATION="${FORCE_ASSOCIATION:-1}"
FORCE_INTERVENTION="${FORCE_INTERVENTION:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

# These are the completed upstream runs named in the experiment plan.  Override
# a root when the runs live elsewhere; the launcher reuses all existing
# activation, decoupler, module, and baseline artifacts under that root.
ULTRACHAT_ROOT="${ULTRACHAT_ROOT:-runs/ultrachat_external_semantic_m4_long_20260811_001253}"
BEAVERTAILS_ROOT="${BEAVERTAILS_ROOT:-runs/beavertails_external_semantic_support128_m8_20260812_164440}"
MATHQA_ROOT="${MATHQA_ROOT:-runs/mathqa_external_semantic_m8_long_20260813_024122}"

mkdir -p runs
printf '[meta-main] reuse roots:\n  ultrachat=%s\n  beavertails=%s\n  mathqa=%s\n' \
  "$ULTRACHAT_ROOT" "$BEAVERTAILS_ROOT" "$MATHQA_ROOT"

run_one() {
  local label="$1" config="$2" root="$3"
  [[ -s "$config" ]] || { echo "Missing config: $config" >&2; return 2; }
  [[ -d "$root" ]] || { echo "Missing upstream run root: $root" >&2; return 2; }
  local log="$root/metacognitive_behavior_main_${STAMP}.log"
  echo "[$(date '+%F %T')] START $label config=$config root=$root"
  env \
    CONFIG_FILE="$config" RUN_ROOT="$root" \
    GPU0="$GPU0" GPU1="$GPU1" \
    FORCE="$FORCE" FORCE_ACTIVATIONS="$FORCE_ACTIVATIONS" \
    FORCE_BASELINE="$FORCE_BASELINE" FORCE_ASSOCIATION="$FORCE_ASSOCIATION" \
    FORCE_INTERVENTION="$FORCE_INTERVENTION" \
    bash "$ROOT_DIR/scripts/run_generated_step_experiment.sh" \
    > "$log" 2>&1
  echo "[$(date '+%F %T')] COMPLETE $label log=$log"
}

failures=0
run_one ultrachat \
  configs/ultrachat_metacognitive_behavior_main_v1.env \
  "$ULTRACHAT_ROOT" || failures=$((failures + 1))
run_one beavertails \
  configs/beavertails_metacognitive_behavior_main_v1.env \
  "$BEAVERTAILS_ROOT" || failures=$((failures + 1))
run_one mathqa \
  configs/mathqa_metacognitive_behavior_main_v1.env \
  "$MATHQA_ROOT" || failures=$((failures + 1))

echo "[meta-main] completed with failures=$failures"
exit "$failures"
