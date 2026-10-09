#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CONFIG_FILE="${CONFIG_FILE:-configs/beavertails_external_semantic_continuous_m4_long_v2.env}"
export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export FORCE="${FORCE:-0}"
export FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

[[ -s "$CONFIG_FILE" ]] || { echo "Missing config: $CONFIG_FILE" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

[[ "$DATASET_PROFILE" == "beavertails" ]] || {
  echo "Expected DATASET_PROFILE=beavertails, got $DATASET_PROFILE" >&2
  exit 2
}
[[ -n "${BASE_DATA_PATH:-}" && -n "${DATA_SOURCE:-}" ]] || {
  echo "Config must define BASE_DATA_PATH and DATA_SOURCE" >&2
  exit 2
}

if [[ "$FORCE" == 1 || ! -s "$BASE_DATA_PATH" ]]; then
  echo "[data] prepare natural-neutral BeaverTails prompts"
  mkdir -p "$(dirname "$BASE_DATA_PATH")"
  "$PYTHON_BIN" scripts/prepare_beavertails_neutral.py \
    --input "$DATA_SOURCE" \
    --output "$BASE_DATA_PATH" \
    --max-samples "${DATA_MAX_SAMPLES:-0}" \
    --sampling balanced \
    --seed "$SEED"
else
  echo "[data] reuse $BASE_DATA_PATH"
fi

"$PYTHON_BIN" -m metacog.cli dataset validate beavertails \
  "$BASE_DATA_PATH" --min-samples "$TARGET_PROMPT_MAX_SAMPLES"

if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="runs/beavertails_external_semantic_m4_long_$(date +%Y%m%d_%H%M%S)"
  export RUN_ROOT
fi
mkdir -p "$RUN_ROOT"
printf '[%s] BeaverTails external-semantic continuous experiment\n' "$(date '+%F %T')"
printf '  config=%s\n  run_root=%s\n  gpu0=%s gpu1=%s\n  pairs=%s\n' \
  "$CONFIG_FILE" "$RUN_ROOT" "$GPU0" "$GPU1" "$PAIR_SPECS"

exec bash scripts/run_generated_step_experiment.sh
