#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${CONFIG_FILE:-configs/ultrachat_layer_topology_supplement_v1.env}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-runs/ultrachat_layer_topology_supplement_$(date +%Y%m%d_%H%M%S)}"
FORCE="${FORCE:-0}"
FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"

mkdir -p "$RUN_ROOT"
printf '[supplement] run_root=%s config=%s gpu0=%s gpu1=%s\n' \
  "$RUN_ROOT" "$CONFIG_FILE" "$GPU0" "$GPU1"

# A rerun with the same RUN_ROOT used to launch every condition again.  This
# is particularly dangerous when the first launcher is still alive.  The
# launcher lock is atomic across shells; condition locks below additionally
# protect against duplicated entries inside a resumed run.
LAUNCHER_LOCK="$RUN_ROOT/.supplement_launcher.lock"
LOCK_HELD=0
cleanup_launcher_lock() {
  if [[ "$LOCK_HELD" == 1 ]]; then
    rm -f "$LAUNCHER_LOCK/pid" "$LAUNCHER_LOCK/host" "$LAUNCHER_LOCK/started_at" 2>/dev/null || true
    rmdir "$LAUNCHER_LOCK" 2>/dev/null || true
  fi
}
trap cleanup_launcher_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! mkdir "$LAUNCHER_LOCK" 2>/dev/null; then
  lock_pid="$(cat "$LAUNCHER_LOCK/pid" 2>/dev/null || true)"
  lock_host="$(cat "$LAUNCHER_LOCK/host" 2>/dev/null || true)"
  if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
    echo "[supplement] another launcher is already active: pid=$lock_pid host=$lock_host root=$RUN_ROOT" >&2
    exit 17
  fi
  if [[ "${BREAK_STALE_LOCK:-0}" != 1 ]]; then
    echo "[supplement] stale launcher lock found: $LAUNCHER_LOCK" >&2
    echo "[supplement] inspect the old process, then rerun with BREAK_STALE_LOCK=1 if it is gone" >&2
    exit 17
  fi
  rm -f "$LAUNCHER_LOCK/pid" "$LAUNCHER_LOCK/host" "$LAUNCHER_LOCK/started_at" 2>/dev/null || true
  rmdir "$LAUNCHER_LOCK" 2>/dev/null || {
    echo "[supplement] cannot clear launcher lock: $LAUNCHER_LOCK" >&2
    exit 17
  }
  mkdir "$LAUNCHER_LOCK"
fi
printf '%s\n' "$$" > "$LAUNCHER_LOCK/pid"
hostname > "$LAUNCHER_LOCK/host"
date '+%F %T %z' > "$LAUNCHER_LOCK/started_at"
LOCK_HELD=1

# Each entry is (name, target layer fractions, matched external semantic layer).
TOPOLOGIES=(
  "early_local|0.25 0.3125 0.375|9"
  "mid_local|0.40625 0.4375 0.50|12"
  "mid_to_late|0.46875 0.50 0.875|14"
  "late_local|0.6875 0.75 0.875|21"
)

run_one() {
  local gpu="$1" name="$2" fracs="$3" semantic_layer="$4" supports="$5" group="$6"
  local semantic_head="${7:-mlp}" semantic_head_depth="${8:-4}"
  local cache_tag="${9:-${group}_${name}}"
  local activation_force="$FORCE_ACTIVATIONS"
  [[ "$group" == "semantic_head" ]] && activation_force=0
  local out="$RUN_ROOT/$name"
  local task_lock="$out/.task.lock"
  local done_marker="$out/.task.complete"
  mkdir -p "$out"

  task_has_completed_output() {
    [[ -s "$out/final_status.tsv" ]] || return 1
    awk -F '\t' 'NR > 1 && $1 == "complete" && $3 == "0" { found = 1 } END { exit(found ? 0 : 1) }' \
      "$out/final_status.tsv"
  }
  if [[ "$FORCE" != 1 ]]; then
    if [[ -s "$done_marker" ]] || task_has_completed_output; then
      [[ -s "$done_marker" ]] || printf '%s\n' "recovered from final_status.tsv" > "$done_marker"
      printf '[%s] SKIP complete %s\n' "$group" "$name" | tee -a "$RUN_ROOT/${group}_sweep.log"
      return 0
    fi
  fi
  if ! mkdir "$task_lock" 2>/dev/null; then
    local task_pid="$(cat "$task_lock/pid" 2>/dev/null || true)"
    if [[ -n "$task_pid" ]] && kill -0 "$task_pid" 2>/dev/null; then
      printf '[%s] SKIP already running %s pid=%s\n' "$group" "$name" "$task_pid" \
        | tee -a "$RUN_ROOT/${group}_sweep.log"
      return 0
    fi
    if [[ "${BREAK_STALE_LOCK:-0}" != 1 ]]; then
      printf '[%s] BLOCK stale task lock %s; use BREAK_STALE_LOCK=1 after inspection\n' \
        "$group" "$name" | tee -a "$RUN_ROOT/${group}_sweep.log" >&2
      return 17
    fi
    rm -f "$task_lock/pid" 2>/dev/null || true
    rmdir "$task_lock" 2>/dev/null || return 17
    mkdir "$task_lock"
  fi
  printf '%s\n' "$$" > "$task_lock/pid"
  printf '[%s] START %s gpu=%s fracs="%s" semantic_layer=%s supports=%s head=%s depth=%s\n' \
    "$group" "$name" "$gpu" "$fracs" "$semantic_layer" "$supports" \
    "$semantic_head" "$semantic_head_depth" \
    | tee -a "$RUN_ROOT/${group}_sweep.log"
  set +e
  env GPU0="$gpu" GPU1="$gpu" FORCE="$FORCE" FORCE_ACTIVATIONS="$activation_force" \
    CONFIG_FILE="$CONFIG_FILE" RUN_ROOT="$out" \
    PAIR_SPECS_OVERRIDE="llama32_3b:qwen3_0p6b:${semantic_layer}" \
    LAYER_FRACS_OVERRIDE="$fracs" SUPPORTS_OVERRIDE="$supports" \
    EXTERNAL_SEMANTIC_HEAD_TYPE_OVERRIDE="$semantic_head" \
    EXTERNAL_SEMANTIC_HEAD_DEPTH_OVERRIDE="$semantic_head_depth" \
    ACTIVATION_CACHE_TAG="$cache_tag" \
    bash scripts/run_generated_step_experiment.sh \
    > "$out/task.log" 2>&1
  local rc=$?
  set -e
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$gpu" "$rc" "$fracs" "$semantic_layer" "$supports" \
    >> "$RUN_ROOT/${group}_sweep_status.tsv"
  if [[ "$rc" == 0 ]]; then
    printf '%s\n' "$(date '+%F %T %z')" > "$done_marker"
    printf '[%s] COMPLETE %s\n' "$group" "$name" | tee -a "$RUN_ROOT/${group}_sweep.log"
  else
    printf '[%s] FAILED %s rc=%s\n' "$group" "$name" "$rc" \
      | tee -a "$RUN_ROOT/${group}_sweep.log" >&2
  fi
  rm -f "$task_lock/pid" 2>/dev/null || true
  rmdir "$task_lock" 2>/dev/null || true
  return "$rc"
}

printf 'condition\tgpu\texit_code\tlayer_fracs\tsemantic_layer\tsupports\n' \
  > "$RUN_ROOT/layer_sweep_status.tsv"
printf 'topology\tlayer_fracs\tsemantic_layer\tsupports\n' \
  > "$RUN_ROOT/layer_topologies.tsv"
for item in "${TOPOLOGIES[@]}"; do
  IFS='|' read -r name fracs semantic_layer <<< "$item"
  printf '%s\t%s\t%s\t128\n' "$name" "$fracs" "$semantic_layer" \
    >> "$RUN_ROOT/layer_topologies.tsv"
done

failed=0
for ((i=0; i<${#TOPOLOGIES[@]}; i+=2)); do
  pids=()
  for offset in 0 1; do
    j=$((i + offset))
    (( j < ${#TOPOLOGIES[@]} )) || continue
    IFS='|' read -r name fracs semantic_layer <<< "${TOPOLOGIES[$j]}"
    gpu="$GPU0"; (( offset == 1 )) && gpu="$GPU1"
    run_one "$gpu" "$name" "$fracs" "$semantic_layer" "128" "layer" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
done

printf '[layer] aggregate supplementary figures\n' | tee -a "$RUN_ROOT/layer_sweep.log"
"$PYTHON_BIN" scripts/summarize_layer_topology_supplement.py \
  --run-root "$RUN_ROOT" --output-dir "$RUN_ROOT/supplementary_figures" \
  --alpha 0.05 || failed=1

# ---------------------------------------------------------------------------
# Independent support-size sweep. The reference layer topology is fixed and
# only SUPPORTS changes, so this effect is not confounded with layer choice.
# This group starts only after the layer sweep has finished.
# ---------------------------------------------------------------------------
SUPPORT_SWEEP_FRACS="${SUPPORT_SWEEP_FRACS:-0.46875 0.50 0.875}"
SUPPORT_SWEEP_SEMANTIC_LAYER="${SUPPORT_SWEEP_SEMANTIC_LAYER:-14}"
read -r -a SUPPORT_VALUES <<< "${SUPPORT_VALUES_OVERRIDE:-32 64 128}"
printf '[support] fixed fracs="%s" semantic_layer=%s supports=%s\n' \
  "$SUPPORT_SWEEP_FRACS" "$SUPPORT_SWEEP_SEMANTIC_LAYER" "${SUPPORT_VALUES[*]}" \
  | tee "$RUN_ROOT/support_sweep.log"
printf 'condition\tgpu\texit_code\tlayer_fracs\tsemantic_layer\tsupports\n' \
  > "$RUN_ROOT/support_sweep_status.tsv"
printf 'support\tlayer_fracs\tsemantic_layer\n' > "$RUN_ROOT/support_conditions.tsv"
for support in "${SUPPORT_VALUES[@]}"; do
  printf '%s\t%s\t%s\n' "$support" "$SUPPORT_SWEEP_FRACS" "$SUPPORT_SWEEP_SEMANTIC_LAYER" \
    >> "$RUN_ROOT/support_conditions.tsv"
done

support_failed=0
for ((i=0; i<${#SUPPORT_VALUES[@]}; i+=2)); do
  pids=()
  for offset in 0 1; do
    j=$((i + offset))
    (( j < ${#SUPPORT_VALUES[@]} )) || continue
    support="${SUPPORT_VALUES[$j]}"
    gpu="$GPU0"; (( offset == 1 )) && gpu="$GPU1"
    run_one "$gpu" "support_${support}" "$SUPPORT_SWEEP_FRACS" \
      "$SUPPORT_SWEEP_SEMANTIC_LAYER" "$support" "support" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || support_failed=1
  done
done

printf '[support] aggregate supplementary figures\n' | tee -a "$RUN_ROOT/support_sweep.log"
"$PYTHON_BIN" scripts/summarize_support_supplement.py \
  --run-root "$RUN_ROOT" --output-dir "$RUN_ROOT/support_supplementary_figures" \
  --alpha 0.05 || support_failed=1

# ---------------------------------------------------------------------------
# Semantic-head capacity ablation. This starts after the topology and support
# sweeps and reuses the support-128 activation cache. All scientific settings
# remain fixed; only the frozen Z1 -> external-F head architecture changes.
# ---------------------------------------------------------------------------
HEAD_SWEEP_FRACS="${HEAD_SWEEP_FRACS:-$SUPPORT_SWEEP_FRACS}"
HEAD_SWEEP_SEMANTIC_LAYER="${HEAD_SWEEP_SEMANTIC_LAYER:-$SUPPORT_SWEEP_SEMANTIC_LAYER}"
HEAD_SWEEP_SUPPORT="${HEAD_SWEEP_SUPPORT:-128}"
HEAD_SHARED_CACHE_TAG="${HEAD_SHARED_CACHE_TAG:-support_support_128}"
HEAD_RUN_CONDITIONS=(
  "linear|linear|1"
  "deep_residual|deep_residual|4"
  "ensemble|ensemble|4"
)
printf '[semantic-head] fixed fracs="%s" semantic_layer=%s support=%s\n' \
  "$HEAD_SWEEP_FRACS" "$HEAD_SWEEP_SEMANTIC_LAYER" "$HEAD_SWEEP_SUPPORT" \
  | tee "$RUN_ROOT/semantic_head_sweep.log"
printf 'condition\tgpu\texit_code\tlayer_fracs\tsemantic_layer\tsupports\n' \
  > "$RUN_ROOT/semantic_head_sweep_status.tsv"
printf 'condition\thead_type\thead_depth\tlayer_fracs\tsemantic_layer\tsupport\toutput_dir\n' \
  > "$RUN_ROOT/semantic_head_conditions.tsv"
head_failed=0
MLP_BASELINE_DIR="$RUN_ROOT/support_${HEAD_SWEEP_SUPPORT}"
if find "$MLP_BASELINE_DIR" -path '*/decoupler_joint_v2/analysis_summary.json' \
    -type f -size +0c -print -quit 2>/dev/null | grep -q .; then
  printf '[semantic-head] reuse support sweep as MLP baseline: %s\n' \
    "$MLP_BASELINE_DIR" | tee -a "$RUN_ROOT/semantic_head_sweep.log"
  printf 'mlp\treused\t0\t%s\t%s\t%s\n' \
    "$HEAD_SWEEP_FRACS" "$HEAD_SWEEP_SEMANTIC_LAYER" "$HEAD_SWEEP_SUPPORT" \
    >> "$RUN_ROOT/semantic_head_sweep_status.tsv"
else
  MLP_BASELINE_DIR="$RUN_ROOT/semantic_head_mlp"
  run_one "$GPU0" "semantic_head_mlp" "$HEAD_SWEEP_FRACS" \
    "$HEAD_SWEEP_SEMANTIC_LAYER" "$HEAD_SWEEP_SUPPORT" "semantic_head" \
    "mlp" "2" "$HEAD_SHARED_CACHE_TAG" || head_failed=1
fi
printf 'mlp\tmlp\t2\t%s\t%s\t%s\t%s\n' \
  "$HEAD_SWEEP_FRACS" "$HEAD_SWEEP_SEMANTIC_LAYER" "$HEAD_SWEEP_SUPPORT" \
  "$MLP_BASELINE_DIR" >> "$RUN_ROOT/semantic_head_conditions.tsv"
for item in "${HEAD_RUN_CONDITIONS[@]}"; do
  IFS='|' read -r name head_type head_depth <<< "$item"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$head_type" "$head_depth" "$HEAD_SWEEP_FRACS" \
    "$HEAD_SWEEP_SEMANTIC_LAYER" "$HEAD_SWEEP_SUPPORT" \
    "$RUN_ROOT/semantic_head_${name}" \
    >> "$RUN_ROOT/semantic_head_conditions.tsv"
done

for ((i=0; i<${#HEAD_RUN_CONDITIONS[@]}; i+=2)); do
  pids=()
  for offset in 0 1; do
    j=$((i + offset))
    (( j < ${#HEAD_RUN_CONDITIONS[@]} )) || continue
    IFS='|' read -r name head_type head_depth <<< "${HEAD_RUN_CONDITIONS[$j]}"
    gpu="$GPU0"; (( offset == 1 )) && gpu="$GPU1"
    run_one "$gpu" "semantic_head_${name}" "$HEAD_SWEEP_FRACS" \
      "$HEAD_SWEEP_SEMANTIC_LAYER" "$HEAD_SWEEP_SUPPORT" "semantic_head" \
      "$head_type" "$head_depth" "$HEAD_SHARED_CACHE_TAG" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || head_failed=1
  done
done

printf '[semantic-head] aggregate ablation\n' | tee -a "$RUN_ROOT/semantic_head_sweep.log"
"$PYTHON_BIN" scripts/summarize_semantic_head_ablation.py \
  --conditions "$RUN_ROOT/semantic_head_conditions.tsv" \
  --output-dir "$RUN_ROOT/semantic_head_supplementary_figures" \
  || head_failed=1

(( support_failed == 0 && failed == 0 && head_failed == 0 )) || exit 1
printf '[supplement] complete: %s\n' "$RUN_ROOT"
