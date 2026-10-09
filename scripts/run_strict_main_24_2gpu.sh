#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ROOT="${RUN_ROOT:-runs/strict_main24_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0 1}"
FORCE="${FORCE:-0}"
FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"
RUN_PERSISTENT_TRAJECTORY_OVERRIDE="${RUN_PERSISTENT_TRAJECTORY_OVERRIDE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
POLL_SECONDS="${POLL_SECONDS:-15}"
if [[ "$RUN_PERSISTENT_TRAJECTORY_OVERRIDE" == 1 ]]; then
  PHASE_LABEL=trajectory
else
  PHASE_LABEL=screen
fi

TARGETS=(
  llama2_7b llama31_8b llama32_3b qwen25_7b
  qwen3_4b qwen3_8b deepseek_llama8b deepseek_qwen7b
)
DATASETS=(ultrachat beavertails mathqa)
REFERENCE_POOL=(llama31_8b:16 qwen3_8b:18 qwen25_7b:14 qwen3_4b:18)

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/logs"

semantic_specs() {
  local target="$1" item profile selected=0 output=""
  for item in "${REFERENCE_POOL[@]}"; do
    IFS=: read -r profile _ <<< "$item"
    [[ "$profile" == "$target" ]] && continue
    output+="${output:+ }$item"
    selected=$((selected + 1))
    (( selected == 3 )) && break
  done
  (( selected == 3 )) || {
    echo "Unable to choose three non-self semantic references for $target" >&2
    return 2
  }
  printf '%s\n' "$output"
}

dataset_config() {
  case "$1" in
    ultrachat) echo "configs/ultrachat_strict_chain_main_v2.env" ;;
    beavertails) echo "configs/beavertails_strict_chain_main_v2.env" ;;
    mathqa) echo "configs/mathqa_strict_chain_main_v2.env" ;;
    *) echo "Unknown dataset: $1" >&2; return 2 ;;
  esac
}

declare -a TASK_DATASETS=() TASK_TARGETS=()
for dataset in "${DATASETS[@]}"; do
  for target in "${TARGETS[@]}"; do
    TASK_DATASETS+=("$dataset")
    TASK_TARGETS+=("$target")
  done
done

printf '[strict-main24] root=%s tasks=%s gpus=%s persistent=%s\n' \
  "$RUN_ROOT" "${#TASK_TARGETS[@]}" "$GPU_IDS" \
  "$RUN_PERSISTENT_TRAJECTORY_OVERRIDE"
printf 'index\tdataset\ttarget\tsemantic_references\n' > "$RUN_ROOT/task_plan.tsv"
for index in "${!TASK_TARGETS[@]}"; do
  dataset="${TASK_DATASETS[$index]}"
  target="${TASK_TARGETS[$index]}"
  printf '%s\t%s\t%s\t%s\n' "$index" "$dataset" "$target" \
    "$(semantic_specs "$target")" >> "$RUN_ROOT/task_plan.tsv"
done

if [[ "$PLAN_ONLY" == 1 ]]; then
  cat "$RUN_ROOT/task_plan.tsv"
  exit 0
fi

launch_task() {
  local index="$1" gpu="$2" dataset target references config condition_root log
  dataset="${TASK_DATASETS[$index]}"
  target="${TASK_TARGETS[$index]}"
  references="$(semantic_specs "$target")"
  config="$(dataset_config "$dataset")"
  condition_root="$RUN_ROOT/${dataset}/${target}"
  log="$RUN_ROOT/logs/${dataset}__${target}__${PHASE_LABEL}.launcher.log"
  mkdir -p "$condition_root"
  echo "[$(date '+%F %T')] [queue] start index=$index dataset=$dataset target=$target gpu=$gpu refs=$references"
  (
    env CONFIG_FILE="$config" RUN_ROOT="$condition_root" \
      GPU0="$gpu" GPU1="$gpu" FORCE="$FORCE" \
      FORCE_ACTIVATIONS="$FORCE_ACTIVATIONS" \
      TARGET_PROFILE_OVERRIDE="$target" \
      SEMANTIC_REFERENCE_SPECS_OVERRIDE="$references" \
      RUN_PERSISTENT_TRAJECTORY_OVERRIDE="$RUN_PERSISTENT_TRAJECTORY_OVERRIDE" \
      bash scripts/run_mathqa_strict_chain_deepseek_qwen7b.sh
  ) > "$log" 2>&1 &
  LAST_PID=$!
}

read -r -a GPUS <<< "$GPU_IDS"
(( ${#GPUS[@]} > 0 )) || { echo "GPU_IDS is empty" >&2; exit 2; }
declare -a SLOT_PIDS=() SLOT_TASKS=()
next_index=0
failed=0

for slot in "${!GPUS[@]}"; do
  gpu="${GPUS[$slot]}"
  (( next_index < ${#TASK_TARGETS[@]} )) || break
  launch_task "$next_index" "$gpu"
  SLOT_PIDS[$slot]="$LAST_PID"
  SLOT_TASKS[$slot]="$next_index"
  next_index=$((next_index + 1))
done

active_slots=${#SLOT_PIDS[@]}
while (( active_slots > 0 )); do
  progressed=0
  for slot in "${!SLOT_PIDS[@]}"; do
    pid="${SLOT_PIDS[$slot]:-}"
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    progressed=1
    gpu="${GPUS[$slot]}"
    index="${SLOT_TASKS[$slot]}"
    dataset="${TASK_DATASETS[$index]}"
    target="${TASK_TARGETS[$index]}"
    if wait "$pid"; then
      status=complete
      code=0
    else
      status=failed
      code=$?
      failed=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$status" "$gpu" "$dataset" "$target" "$code" \
      > "$RUN_ROOT/status/${dataset}__${target}__${PHASE_LABEL}.tsv"
    echo "[$(date '+%F %T')] [queue] $status index=$index dataset=$dataset target=$target gpu=$gpu exit=$code"
    SLOT_PIDS[$slot]=""
    SLOT_TASKS[$slot]=""
    active_slots=$((active_slots - 1))
    if (( next_index < ${#TASK_TARGETS[@]} )); then
      launch_task "$next_index" "$gpu"
      SLOT_PIDS[$slot]="$LAST_PID"
      SLOT_TASKS[$slot]="$next_index"
      active_slots=$((active_slots + 1))
      next_index=$((next_index + 1))
    fi
  done
  (( progressed == 1 )) || sleep "$POLL_SECONDS"
done

phase_status="$RUN_ROOT/final_status_${PHASE_LABEL}.tsv"
printf 'status\tgpu\tdataset\ttarget\texit_code\n' > "$phase_status"
for file in "$RUN_ROOT"/status/*"__${PHASE_LABEL}.tsv"; do
  [[ -s "$file" ]] || continue
  cat "$file" >> "$phase_status"
done
cp "$phase_status" "$RUN_ROOT/final_status.tsv"
cat "$phase_status"
(( failed == 0 ))
