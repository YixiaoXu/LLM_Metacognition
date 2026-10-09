#!/usr/bin/env bash
set -euo pipefail

# Re-run only the downstream controls of an existing UltraChat ablation run.
# Upstream activations, decouplers, modules, deterministic baselines, axes and
# next-token prototypes are reused. The random control is isotropic in code
# space, orthogonal to the real module axis, and matched on hidden/semantic
# drift rather than on the real code displacement.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/task_lock.sh"

: "${RUN_ROOT:?Set RUN_ROOT to an existing UltraChat ablation run}"
CONFIG_FILE="${CONFIG_FILE:-$RUN_ROOT/input_config.env}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORCE_INTERVENTION="${FORCE_INTERVENTION:-1}"

[[ -d "$RUN_ROOT" ]] || { echo "Missing RUN_ROOT: $RUN_ROOT" >&2; exit 2; }
[[ -s "$CONFIG_FILE" ]] || { echo "Missing config: $CONFIG_FILE" >&2; exit 2; }

set -a
# shellcheck source=/dev/null
source "$CONFIG_FILE"
set +a
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ALL_HELDOUT_MODULES=1

RETEST_ROOT="${RETEST_ROOT:-$RUN_ROOT/random_orthogonal_retest}"
if ! acquire_task_lock "$RETEST_ROOT/.locks/retest.lock" "random-orthogonal:$RETEST_ROOT"; then
  echo "[retest] duplicate invocation refused: $RETEST_ROOT" >&2
  exit 17
fi
trap release_task_locks EXIT INT TERM
mkdir -p "$RETEST_ROOT/status"
exec > >(tee -a "$RETEST_ROOT/task.log") 2>&1

echo "[$(date '+%F %T')] strict orthogonal-random downstream retest"
echo "  run_root=$RUN_ROOT"
echo "  output=$RETEST_ROOT"
echo "  gpu0=$GPU0 gpu1=$GPU1"

model_field() { "$PYTHON_BIN" -m metacog.cli model get "$1" "$2"; }

config_field() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json, sys
path, key = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    value = json.load(handle).get(key)
if value is None:
    raise SystemExit(1)
print(value)
PY
}

abspath_from_root() {
  local path="$1"
  if [[ "$path" = /* ]]; then printf '%s\n' "$path"; else printf '%s/%s\n' "$ROOT_DIR" "$path"; fi
}

pair_job_config() {
  find "$1/jobs" -path '*/decoupler_joint_v2/config.json' -type f -print -quit 2>/dev/null || true
}

link_if_present() {
  local source="$1" target="$2"
  [[ -e "$source" ]] || return 0
  if [[ -L "$target" ]]; then
    rm "$target"
  elif [[ -e "$target" ]]; then
    echo "Refusing to replace existing non-symlink: $target" >&2
    return 1
  fi
  ln -s "$source" "$target"
}

run_pair() {
  local gpu="$1" spec="$2"
  local target reference semantic_layer pair_name pair_root job_config
  local activation_dir decoupler_dir data_path model_path selected shared_baseline
  IFS=: read -r target reference semantic_layer <<< "$spec"
  pair_name="${target}__semantic_${reference}_l${semantic_layer}"
  pair_root="$RUN_ROOT/$pair_name"
  job_config="$(pair_job_config "$pair_root")"
  [[ -n "$job_config" ]] || { echo "Missing job config: $pair_root" >&2; return 1; }

  activation_dir="$(abspath_from_root "$(config_field "$job_config" activation_dir)")"
  decoupler_dir="$(abspath_from_root "$(config_field "$job_config" output_dir)")"
  data_path="$activation_dir/expanded_generation_data.jsonl"
  model_path="$(model_field "$target" path)"
  selected="$pair_root/adaptive_selection/selected_adaptive_modules.tsv"
  [[ -s "$selected" ]] || { echo "Missing selected modules: $selected" >&2; return 1; }

  shared_baseline="$(find "$pair_root/continuous_modules" -type f \
    -path '*/baseline_*/baseline_generations.jsonl' -size +0c -print -quit 2>/dev/null || true)"
  echo "[$(date '+%F %T')] pair=$pair_name gpu=$gpu baseline=${shared_baseline:-none}"

  local failed=0 rank profile support module module_dir output old_output axis_dir
  while IFS=$'\t' read -r rank profile support module module_dir rest; do
    [[ "$rank" == rank || -z "$rank" ]] && continue
    module_dir="$(abspath_from_root "$module_dir")"
    output="$RETEST_ROOT/$pair_name/rank_${rank}_k${support}_m${module}"
    old_output="$pair_root/continuous_modules/rank_${rank}_k${support}_m${module}"
    mkdir -p "$output"

    # Reuse the exact axis, deterministic baseline and prototype from the
    # original module run. Only causal intervention files are regenerated.
    axis_dir="$old_output/continuous_axis"
    link_if_present "$axis_dir" "$output/continuous_axis"
    link_if_present "$old_output/baseline_association_confirmatory" "$output/baseline_association_confirmatory"
    link_if_present "$old_output/baseline_behavior" "$output/baseline_behavior"
    link_if_present "$old_output/prototype_discovery" "$output/prototype_discovery"

    set +e
    env \
      GPU="$gpu" MODULE_DIR="$module_dir" OUTPUT_DIR="$output" \
      MODEL_PATH="$model_path" DATA_PATH="$data_path" \
      ACTIVATION_DIR="$activation_dir" DECOUPLER_DIR="$(cd "$module_dir/../.." && pwd)" \
      FORCE=0 FORCE_AXIS=0 FORCE_BASELINE=0 FORCE_ASSOCIATION=0 \
      FORCE_PROTOTYPE=0 FORCE_INTERVENTION="$FORCE_INTERVENTION" \
      RUN_PERSISTENT_TRAJECTORY=1 PERSISTENT_POPULATION=critical \
      CONTINUOUS_CONTROLS="opposite_direction random_direction" \
      REFINED_CODE_DOSE_MATCH_CONTROLS=1 \
      REFINED_CODE_RANDOM_DIRECTION_MODE=isotropic \
      RANDOM_DIRECTION_DOSE_MODE=hidden_semantic \
      RANDOM_DIRECTION_AXIS_PENALTY=10.0 \
      RANDOM_DIRECTION_MAX_REAL_AXIS_COSINE=0.25 \
      RANDOM_DIRECTION_HIDDEN_TOLERANCE=0.50 \
      RANDOM_DIRECTION_SEMANTIC_TOLERANCE=0.50 \
      RANDOM_DIRECTION_MIN_COSINE=0.50 \
      SHARED_BASELINE_FILE="${shared_baseline:-}" \
      bash scripts/run_continuous_module.sh
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' \
      "$([[ "$rc" == 0 ]] && echo complete || echo failed)" "$gpu" "$rank" "$rc" \
      > "$output/status.tsv"
    [[ "$rc" == 0 ]] || failed=1
  done < "$selected"
  return "$failed"
}

read -r -a PAIRS <<< "${PAIR_SPECS:?PAIR_SPECS is missing from $CONFIG_FILE}"
worker() {
  local gpu="$1" parity="$2" index spec rc failed=0
  for index in "${!PAIRS[@]}"; do
    (( index % 2 == parity )) || continue
    spec="${PAIRS[$index]}"
    set +e
    run_pair "$gpu" "$spec"
    rc=$?
    set -e
    [[ "$rc" == 0 ]] || failed=1
  done
  return "$failed"
}

worker "$GPU0" 0 & pid0=$!
worker "$GPU1" 1 & pid1=$!
status0=0; status1=0
wait "$pid0" || status0=1
wait "$pid1" || status1=1

echo "[$(date '+%F %T')] retest complete: gpu0_status=$status0 gpu1_status=$status1"
(( status0 == 0 && status1 == 0 ))
