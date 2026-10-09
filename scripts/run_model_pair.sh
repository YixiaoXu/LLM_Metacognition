#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/task_lock.sh"

: "${PAIR_ROOT:?}"
: "${TARGET_PROFILE:?}"
: "${TARGET_MODEL_PATH:?}"
: "${TARGET_ACTIVATION_DIR:?}"
: "${SEMANTIC_PROFILE:?}"
: "${SEMANTIC_ACTIVATION_DIR:?}"
: "${SUPPORTS:?}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! acquire_task_lock "$PAIR_ROOT/.locks/pair.lock" \
    "pair:${TARGET_PROFILE}__semantic_${SEMANTIC_PROFILE}"; then
  echo "[pair] duplicate invocation refused: $PAIR_ROOT" >&2
  exit 17
fi
trap release_task_locks EXIT INT TERM

mkdir -p "$PAIR_ROOT/jobs" "$PAIR_ROOT/support_summary" \
  "$PAIR_ROOT/adaptive_selection" "$PAIR_ROOT/continuous_modules"

stage() {
  printf '\n[%s] ===== %s =====\n' "$(date '+%F %T')" "$1"
}

stage "task start: target=${TARGET_PROFILE}, semantic_reference=${SEMANTIC_PROFILE}, gpu=${GPU}"

for support in $SUPPORTS; do
  job_root="$PAIR_ROOT/jobs/support_${support}"
  mkdir -p "$job_root"
  stage "train external-semantic modules: support=${support}"
  env GPU="$GPU" JOB_ROOT="$job_root" MODULE_SUPPORT="$support" FORCE="$FORCE" \
    bash scripts/train_external_semantic_modules.sh
done

stage "summarize support sensitivity"
"$PYTHON_BIN" scripts/summarize_joint_v2_support_sensitivity.py \
  --run-root "$PAIR_ROOT" --output-dir "$PAIR_ROOT/support_summary"

stage "select continuous modules"
SELECTION_ARGS=(
  --max-modules "$MAX_SELECTED_MODULES" --max-per-profile "$MAX_PER_SUPPORT"
)
if [[ "${ADAPTIVE_FILL_TO_MAX:-0}" == 1 ]]; then
  SELECTION_ARGS+=(--fill-to-max)
  echo "[selection] freeze a fixed family of up to $MAX_SELECTED_MODULES modules"
fi
if [[ "${RANDOM_SUPPORT_NULL:-0}" == 1 ]]; then
  SELECTION_ARGS+=(--all-candidates)
  echo "[selection] random-support null: exporting every pre-registered random module"
fi
if [[ "${ALL_HELDOUT_MODULES:-0}" == 1 ]]; then
  SELECTION_ARGS+=(--all-heldout-gate)
  echo "[selection] exporting every module passing the held-out gate"
fi
"$PYTHON_BIN" scripts/select_adaptive_module_sizes.py \
  --diagnostics-csv "$PAIR_ROOT/support_summary/module_support_diagnostics.csv" \
  --output-dir "$PAIR_ROOT/adaptive_selection" \
  "${SELECTION_ARGS[@]}" \
  --evidence-split "${ADAPTIVE_EVIDENCE_SPLIT:-heldout}" \
  --min-heldout-rf "$ADAPTIVE_MIN_RF" \
  --min-gain-ci-low "$ADAPTIVE_MIN_GAIN_CI_LOW" \
  --max-continuous-semantic-r2 "$ADAPTIVE_MAX_SEMANTIC_R2" \
  --size-penalty "$ADAPTIVE_SIZE_PENALTY"

selected="$PAIR_ROOT/adaptive_selection/selected_adaptive_modules.tsv"
[[ -s "$selected" ]] || { echo "No selected modules: $selected" >&2; exit 3; }
downstream_failed=0
declare -A seen_module_outputs=()
declare -a MODULE_RANKS=()
declare -a MODULE_PROFILES=()
declare -a MODULE_SUPPORTS=()
declare -a MODULE_IDS=()
declare -a MODULE_DIRS=()
declare -a MODULE_OUTPUTS=()
declare -a ACTIVE_MODULE_INDICES=()
declare -A PRIMARY_CONSTRUCT_BY_INDEX=()
declare -A PRIMARY_DIRECTION_BY_INDEX=()

while IFS=$'\t' read -r rank profile support module module_dir rest; do
  [[ "$rank" == rank || -z "$rank" ]] && continue
  output="$PAIR_ROOT/continuous_modules/rank_${rank}_k${support}_m${module}"
  if [[ -n "${seen_module_outputs[$output]:-}" ]]; then
    echo "[queue] skip duplicate module row: $output" >&2
    continue
  fi
  seen_module_outputs[$output]=1
  mkdir -p "$output"
  MODULE_RANKS+=("$rank")
  MODULE_PROFILES+=("$profile")
  MODULE_SUPPORTS+=("$support")
  MODULE_IDS+=("$module")
  MODULE_DIRS+=("$module_dir")
  MODULE_OUTPUTS+=("$output")
done < "$selected"

module_count=${#MODULE_RANKS[@]}
(( module_count > 0 )) || { echo "No unique selected modules: $selected" >&2; exit 3; }
if [[ -n "${STRICT_EXPECTED_FROZEN_MODULES:-}" ]] && \
   (( module_count != STRICT_EXPECTED_FROZEN_MODULES )); then
  echo "Strict family requires ${STRICT_EXPECTED_FROZEN_MODULES} frozen modules; got $module_count" >&2
  exit 3
fi
for ((index=0; index<module_count; index++)); do
  ACTIVE_MODULE_INDICES+=("$index")
done
echo "[plan] selected_modules=$module_count supports=${SUPPORTS} downstream_gpus=${DOWNSTREAM_GPUS:-$GPU}"
echo "[plan] stages=axis -> shared_baseline -> association -> next_token_A/B -> critical_trajectory"
echo "[plan] barrier=all stages-1-2 passers must finish next_token_A/B before any critical trajectory starts"
echo "[plan] trajectory_gate=heldout transmission AND positive-information directional association with separate five-item behavior/monitoring Holm correction; next-token results do not select trajectories"
echo "[plan] persistent_population=${PERSISTENT_POPULATION:-critical} trajectory_max_samples=${CRITICAL_MAX_SAMPLES:-all} max_new_tokens=${GENERATION_MAX_NEW_TOKENS:-unset}"

all_axis_artifacts_present() {
  local index output missing=0
  for ((index=0; index<module_count; index++)); do
    output="${MODULE_OUTPUTS[$index]}"
    if [[ ! -s "$output/continuous_axis/continuous_axis.pt" || \
          ! -s "$output/continuous_axis/continuous_axis_summary.json" ]]; then
      echo "[axis] incomplete rank=${MODULE_RANKS[$index]} output=$output" >&2
      missing=1
    fi
  done
  (( missing == 0 ))
}

read -r -a requested_downstream_gpus <<< "${DOWNSTREAM_GPUS:-$GPU}"
declare -A seen_downstream_gpus=()
declare -a DOWNSTREAM_GPU_IDS=()
for downstream_gpu in "${requested_downstream_gpus[@]}"; do
  [[ -n "$downstream_gpu" ]] || continue
  if [[ -z "${seen_downstream_gpus[$downstream_gpu]:-}" ]]; then
    DOWNSTREAM_GPU_IDS+=("$downstream_gpu")
    seen_downstream_gpus[$downstream_gpu]=1
  fi
done
(( ${#DOWNSTREAM_GPU_IDS[@]} > 0 )) || DOWNSTREAM_GPU_IDS+=("$GPU")

gpu_free_mb() {
  local gpu_id="$1" free_mb
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo -1
    return 0
  fi
  free_mb="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/[[:space:]]/, ""); print; exit}')"
  [[ "$free_mb" =~ ^[0-9]+$ ]] || free_mb=-1
  echo "$free_mb"
}

declare -a PHASE_GPU_WORKERS=()
build_gpu_worker_list() {
  local desired_per_gpu="$1" estimated_worker_mb="$2" reserve_mb="$3" label="$4"
  local gpu_id free_mb slots slot
  (( desired_per_gpu > 0 )) || { echo "$label desired worker count must be positive" >&2; return 2; }
  (( estimated_worker_mb > 0 )) || { echo "$label estimated worker memory must be positive" >&2; return 2; }
  (( reserve_mb >= 0 )) || { echo "$label GPU reserve must be non-negative" >&2; return 2; }
  PHASE_GPU_WORKERS=()
  for gpu_id in "${DOWNSTREAM_GPU_IDS[@]}"; do
    free_mb="$(gpu_free_mb "$gpu_id")"
    if (( free_mb < 0 )); then
      slots=1
      echo "[scheduler] $label gpu=$gpu_id free=unknown; use conservative slots=1"
    else
      slots=$(( (free_mb - reserve_mb) / estimated_worker_mb ))
      (( slots > desired_per_gpu )) && slots=$desired_per_gpu
      (( slots < 0 )) && slots=0
      echo "[scheduler] $label gpu=$gpu_id free=${free_mb}MiB reserve=${reserve_mb}MiB estimated_worker=${estimated_worker_mb}MiB slots=$slots"
    fi
    for ((slot=0; slot<slots; slot++)); do
      PHASE_GPU_WORKERS+=("$gpu_id")
    done
  done
  if (( ${#PHASE_GPU_WORKERS[@]} == 0 )); then
    echo "[scheduler] no GPU has enough free memory for $label; no waiting or oversubscription attempted" >&2
    return 1
  fi
}

run_module_stage() {
  local index="$1" gpu_id="$2" phase="$3"
  local rank="${MODULE_RANKS[$index]}"
  local support="${MODULE_SUPPORTS[$index]}"
  local module="${MODULE_IDS[$index]}"
  local module_dir="${MODULE_DIRS[$index]}"
  local output="${MODULE_OUTPUTS[$index]}"
  local decoupler_dir force_axis=0 force_baseline=0 force_association=0
  local force_prototype=0 force_intervention=0
  local -a worker_env=()
  decoupler_dir="$(cd "$module_dir/../.." && pwd)"
  case "$phase" in
    axis)
      force_axis="${FORCE_AXIS:-$FORCE}"
      worker_env+=(
        "OMP_NUM_THREADS=${DOWNSTREAM_AXIS_CPU_THREADS_PER_WORKER:-2}"
        "MKL_NUM_THREADS=${DOWNSTREAM_AXIS_CPU_THREADS_PER_WORKER:-2}"
        "OPENBLAS_NUM_THREADS=${DOWNSTREAM_AXIS_CPU_THREADS_PER_WORKER:-2}"
        "NUMEXPR_NUM_THREADS=${DOWNSTREAM_AXIS_CPU_THREADS_PER_WORKER:-2}"
        "CONTINUOUS_AXIS_CPU_THREADS=${DOWNSTREAM_AXIS_CPU_THREADS_PER_WORKER:-2}"
      )
      ;;
    baseline_plan) ;;
    baseline)
      force_baseline="$FORCE"
      worker_env+=("BASELINE_VALIDATE_ONLY=1")
      ;;
    association)
      force_association="$FORCE"
      worker_env+=(
        "CUDA_VISIBLE_DEVICES=$gpu_id"
        "OMP_NUM_THREADS=${DOWNSTREAM_ASSOCIATION_CPU_THREADS_PER_WORKER:-4}"
        "MKL_NUM_THREADS=${DOWNSTREAM_ASSOCIATION_CPU_THREADS_PER_WORKER:-4}"
        "OPENBLAS_NUM_THREADS=${DOWNSTREAM_ASSOCIATION_CPU_THREADS_PER_WORKER:-4}"
        "NUMEXPR_NUM_THREADS=${DOWNSTREAM_ASSOCIATION_CPU_THREADS_PER_WORKER:-4}"
        "CONTINUOUS_BEHAVIOR_CPU_THREADS=${DOWNSTREAM_ASSOCIATION_CPU_THREADS_PER_WORKER:-4}"
        "CONTINUOUS_BEHAVIOR_DEVICE=${DOWNSTREAM_ASSOCIATION_DEVICE:-cuda:0}"
      )
      ;;
    intervention|next_token|trajectory)
      force_prototype="$FORCE"
      force_intervention="$FORCE"
      ;;
    *) echo "Unknown downstream phase: $phase" >&2; return 2 ;;
  esac
  if [[ -n "${SHARED_BASELINE_ID_FILE:-}" && -s "$SHARED_BASELINE_ID_FILE" ]]; then
    worker_env+=("BASELINE_EVALUATION_ID_FILE=$SHARED_BASELINE_ID_FILE")
  fi
  echo "[$(date '+%F %T')] [scheduler] start phase=$phase rank=$rank support=$support module=$module gpu=$gpu_id"
  if env "${worker_env[@]}" GPU="$gpu_id" MODULE_DIR="$module_dir" OUTPUT_DIR="$output" \
      DECOUPLER_DIR="$decoupler_dir" MODEL_PATH="$TARGET_MODEL_PATH" \
      ACTIVATION_DIR="$TARGET_ACTIVATION_DIR" FORCE="$FORCE" \
      FORCE_AXIS="$force_axis" FORCE_BASELINE="$force_baseline" \
      FORCE_ASSOCIATION="$force_association" FORCE_PROTOTYPE="$force_prototype" \
      FORCE_INTERVENTION="$force_intervention" CONTINUOUS_STAGE="$phase" \
      PRIMARY_BEHAVIOR_CONSTRUCT="${PRIMARY_CONSTRUCT_BY_INDEX[$index]:-}" \
      PRIMARY_BEHAVIOR_DIRECTION="${PRIMARY_DIRECTION_BY_INDEX[$index]:-}" \
      SHARED_BASELINE_FILE="${SHARED_BASELINE_FILE:-$PAIR_ROOT/shared_association_baseline/baseline_generations.jsonl}" \
      bash scripts/run_continuous_module.sh 2>&1 | \
      "$PYTHON_BIN" -u scripts/log_stream.py \
        --prefix "[${phase}:r${rank}:g${gpu_id}]" \
        --progress-interval "${LOG_PROGRESS_INTERVAL_SECONDS:-60}"; then
    echo "[$(date '+%F %T')] [scheduler] complete phase=$phase rank=$rank gpu=$gpu_id"
    return 0
  fi
  echo "[$(date '+%F %T')] [scheduler] failed phase=$phase rank=$rank gpu=$gpu_id" >&2
  return 1
}

declare -a ACTIVE_WORKER_GPUS=()
module_phase_worker() {
  local phase="$1" worker_index="$2" worker_count="$3" gpu_id="$4"
  local position index failed=0 active_count=${#ACTIVE_MODULE_INDICES[@]}
  for ((position=worker_index; position<active_count; position+=worker_count)); do
    index="${ACTIVE_MODULE_INDICES[$position]}"
    run_module_stage "$index" "$gpu_id" "$phase" || failed=1
  done
  return "$failed"
}

run_parallel_phase() {
  local phase="$1" worker_count worker pid failed=0 active_count
  local -a pids=()
  active_count=${#ACTIVE_MODULE_INDICES[@]}
  (( active_count > 0 )) || {
    echo "[$(date '+%F %T')] [scheduler] phase=$phase has no eligible modules; skip"
    return 0
  }
  worker_count=${#ACTIVE_WORKER_GPUS[@]}
  (( worker_count > active_count )) && worker_count=$active_count
  (( worker_count > 0 )) || { echo "No workers configured for phase=$phase" >&2; return 2; }
  stage "parallel downstream phase=${phase}, modules=${active_count}, workers=${worker_count}"
  for ((worker=0; worker<worker_count; worker++)); do
    module_phase_worker "$phase" "$worker" "$worker_count" \
      "${ACTIVE_WORKER_GPUS[$worker]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

run_baseline_shard() {
  local shard_index="$1" gpu_id="$2" id_file="$3" shard_output="$4"
  local baseline_batch_size="${5:-${BASELINE_GENERATION_BATCH_SIZE:-$GENERATION_BATCH_SIZE}}"
  local module_dir="${MODULE_DIRS[0]}"
  local module_output="${MODULE_OUTPUTS[0]}"
  local decoupler_dir
  decoupler_dir="$(cd "$module_dir/../.." && pwd)"
  mkdir -p "$shard_output"
  echo "[$(date '+%F %T')] [baseline-shard] start shard=$shard_index gpu=$gpu_id batch=$baseline_batch_size ids=$id_file"
  if env GPU="$gpu_id" MODULE_DIR="$module_dir" OUTPUT_DIR="$shard_output" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
      GENERATION_BATCH_SIZE="$baseline_batch_size" \
      CONTINUOUS_AXIS_DIR="$module_output/continuous_axis" \
      DECOUPLER_DIR="$decoupler_dir" MODEL_PATH="$TARGET_MODEL_PATH" \
      ACTIVATION_DIR="$TARGET_ACTIVATION_DIR" FORCE="$FORCE" \
      FORCE_AXIS=0 FORCE_BASELINE="$FORCE" FORCE_ASSOCIATION=0 \
      FORCE_PROTOTYPE=0 FORCE_INTERVENTION=0 CONTINUOUS_STAGE=baseline \
      BASELINE_EVALUATION_ID_FILE="$id_file" BASELINE_MAX_SAMPLES_OVERRIDE=0 \
      PUBLISH_SHARED_BASELINE=0 \
      SHARED_BASELINE_FILE="$shard_output/disabled_shared_baseline.jsonl" \
      bash scripts/run_continuous_module.sh 2>&1 | \
      "$PYTHON_BIN" -u scripts/log_stream.py \
        --prefix "[baseline:s${shard_index}:g${gpu_id}]" \
        --progress-interval "${LOG_PROGRESS_INTERVAL_SECONDS:-60}"; then
    echo "[$(date '+%F %T')] [baseline-shard] complete shard=$shard_index gpu=$gpu_id"
    return 0
  fi
  echo "[$(date '+%F %T')] [baseline-shard] failed shard=$shard_index gpu=$gpu_id" >&2
  return 1
}

generate_parallel_shared_baseline() {
  local shared_baseline="$1"
  local requested_shards="${DOWNSTREAM_BASELINE_SHARDS:-2}"
  local plan_dir assignments adaptive_root state_file pool_count selected_count
  local initial_count min_valid_count topup_step valid_count missing_count
  local round_dir plan_summary plan_hash shard_root manifest run_root
  local shard_count shard_index gpu_id id_file shard_output shard_file pid failed=0
  local final_ids_file="${shared_baseline}.ids.txt"
  local -a baseline_gpus=() pids=() shard_files=() failed_shards=()
  local -a history_args=() contract_args=() required_args=() merge_args=()

  (( requested_shards > 0 )) || {
    echo "DOWNSTREAM_BASELINE_SHARDS must be positive" >&2
    return 2
  }

  # Freeze the largest permitted population once.  Smaller adaptive rounds are
  # always prefixes of this order, so top-up never changes earlier membership.
  if ! run_module_stage 0 "${DOWNSTREAM_GPU_IDS[0]}" baseline_plan; then
    echo "[$(date '+%F %T')] baseline id planning failed" >&2
    return 1
  fi
  plan_dir="${MODULE_OUTPUTS[0]}/baseline_plan"
  assignments="$plan_dir/cluster_assignments.csv"
  [[ -s "$assignments" ]] || {
    echo "Missing frozen baseline assignments: $assignments" >&2
    return 1
  }
  pool_count=$(($(wc -l < "$assignments") - 1))
  (( pool_count > 0 )) || { echo "Frozen baseline pool is empty" >&2; return 1; }
  initial_count="${BASELINE_INITIAL_SAMPLES:-1600}"
  min_valid_count="${BASELINE_MIN_UNTRUNCATED_SAMPLES:-1300}"
  topup_step="${BASELINE_TOPUP_STEP:-400}"
  (( initial_count > pool_count )) && initial_count=$pool_count
  (( min_valid_count > pool_count )) && min_valid_count=$pool_count
  (( initial_count > 0 && min_valid_count > 0 && topup_step > 0 )) || {
    echo "Adaptive baseline counts must be positive" >&2
    return 2
  }

  adaptive_root="$(dirname "$shared_baseline")/adaptive_cache"
  state_file="$adaptive_root/baseline_state.jsonl"
  mkdir -p "$adaptive_root"
  read -r -a history_roots <<< "${HISTORICAL_BASELINE_SEARCH_ROOTS:-runs}"
  for history_root in "${history_roots[@]}"; do
    history_args+=(--search-root "$history_root")
  done
  required_args=(
    --required-field generated_text
    --required-field generated_tokens
    --required-field style_truncated
    --required-field prompt_end_entropy
    --required-field prompt_end_normalized_entropy
    --required-field prompt_end_top1_top2_logit_margin
    --required-field generated_mean_logprob
    --required-field generated_min_logprob
    --required-field generated_prefix16_entropy_mean
    --required-field generated_prefix16_negative_entropy_mean
  )
  contract_args=(
    --model-path "$TARGET_MODEL_PATH"
    --metric-profile "$METRIC_PROFILE"
    --answer-extraction "$ANSWER_EXTRACTION"
    --data-marker "${DATASET_PROFILE:-$METRIC_PROFILE}"
    --prompt-style data
    --thinking "${CHAT_TEMPLATE_ENABLE_THINKING:-auto}"
    --system-prompt "${SYSTEM_PROMPT:-}"
    --max-length "${DOWNSTREAM_MAX_LENGTH:-1024}"
    --max-new-tokens "$BASELINE_MAX_NEW_TOKENS"
  )
  if [[ "${DOWNSTREAM_USE_CHAT_TEMPLATE:-1}" == "1" ]]; then
    contract_args+=(--use-chat-template)
  fi

  selected_count=$initial_count
  while true; do
    round_dir="$adaptive_root/round_${selected_count}"
    mkdir -p "$round_dir"
    "$PYTHON_BIN" scripts/adaptive_baseline_cache.py plan \
      --assignments "$assignments" --limit "$selected_count" \
      --state-file "$state_file" --output-dir "$round_dir" \
      --exclude-root "$PAIR_ROOT" \
      "${history_args[@]}" "${contract_args[@]}" "${required_args[@]}"
    plan_summary="$round_dir/plan_summary.json"
    read -r missing_count plan_hash < <("$PYTHON_BIN" - "$plan_summary" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
print(value["missing_rows"], value["plan_sha256"])
PY
)
    shard_files=()
    failed_shards=()
    pids=()
    failed=0
    if (( missing_count > 0 )); then
      if ! build_gpu_worker_list 1 \
          "${DOWNSTREAM_INTERVENTION_WORKER_GPU_MB:-15000}" \
          "${DOWNSTREAM_GPU_RESERVE_MB:-0}" baseline_shards; then
        return 1
      fi
      baseline_gpus=("${PHASE_GPU_WORKERS[@]}")
      shard_count=${#baseline_gpus[@]}
      (( shard_count > requested_shards )) && shard_count=$requested_shards
      (( shard_count > missing_count )) && shard_count=$missing_count
      (( shard_count > 0 )) || return 1
      baseline_gpus=("${baseline_gpus[@]:0:shard_count}")
      shard_root="$round_dir/shards"
      mkdir -p "$shard_root"
      "$PYTHON_BIN" scripts/baseline_shards.py split \
        --assignments "$round_dir/missing_assignments.csv" \
        --output-dir "$shard_root" --num-shards "$shard_count" \
        --contract "adaptive_plan=$plan_hash" \
        --contract "model=$TARGET_MODEL_PATH" \
        --contract "max_new_tokens=$BASELINE_MAX_NEW_TOKENS"
      manifest="$shard_root/baseline_shard_manifest.json"
      run_root="$shard_root/runs/$plan_hash"
      stage "adaptive shared baseline: selected=$selected_count cached=$((selected_count - missing_count)) missing=$missing_count shards=$shard_count"
      for ((shard_index=0; shard_index<shard_count; shard_index++)); do
        gpu_id="${baseline_gpus[$shard_index]}"
        id_file="$shard_root/shard_$(printf '%02d' "$shard_index").txt"
        shard_output="$run_root/shard_$(printf '%02d' "$shard_index")"
        shard_file="$shard_output/baseline_${ASSOCIATION_EVALUATION_ROLE:-association_confirmatory}/baseline_generations.jsonl"
        shard_files+=("$shard_file")
        run_baseline_shard "$shard_index" "$gpu_id" "$id_file" "$shard_output" &
        pids+=("$!")
      done
      for ((shard_index=0; shard_index<shard_count; shard_index++)); do
        pid="${pids[$shard_index]}"
        wait "$pid" || failed_shards+=("$shard_index")
      done
      for shard_index in "${failed_shards[@]}"; do
        gpu_id="${baseline_gpus[$((shard_index % shard_count))]}"
        id_file="$shard_root/shard_$(printf '%02d' "$shard_index").txt"
        shard_output="$run_root/shard_$(printf '%02d' "$shard_index")"
        echo "[$(date '+%F %T')] retry adaptive baseline shard=$shard_index gpu=$gpu_id batch=${BASELINE_OOM_RETRY_BATCH_SIZE:-2}" >&2
        run_baseline_shard "$shard_index" "$gpu_id" "$id_file" "$shard_output" \
          "${BASELINE_OOM_RETRY_BATCH_SIZE:-2}" || failed=1
      done
      (( failed == 0 )) || return 1
    else
      echo "[$(date '+%F %T')] [baseline-cache] all selected rows recovered without generation"
    fi

    merge_args=(
      merge --selected-ids "$round_dir/selected_ids.txt"
      --state-file "$state_file"
      --summary "$round_dir/merge_summary.json"
    )
    merge_args+=("${required_args[@]}")
    for shard_file in "${shard_files[@]}"; do
      merge_args+=(--generated-file "$shard_file")
    done
    "$PYTHON_BIN" scripts/adaptive_baseline_cache.py "${merge_args[@]}"
    valid_count="$($PYTHON_BIN - "$round_dir/merge_summary.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["untruncated_rows"])
PY
)"
    echo "[$(date '+%F %T')] [baseline-cache] selected=$selected_count untruncated=$valid_count target=$min_valid_count pool=$pool_count"
    if (( valid_count >= min_valid_count || selected_count >= pool_count )); then
      if (( valid_count < min_valid_count )); then
        echo "[$(date '+%F %T')] [baseline-cache] warning: exhausted pool below requested usable target" >&2
      fi
      mkdir -p "$(dirname "$shared_baseline")"
      cp "$state_file" "${shared_baseline}.tmp.$$"
      mv "${shared_baseline}.tmp.$$" "$shared_baseline"
      cp "$round_dir/selected_ids.txt" "${final_ids_file}.tmp.$$"
      mv "${final_ids_file}.tmp.$$" "$final_ids_file"
      cp "$round_dir/merge_summary.json" "${shared_baseline}.adaptive.json"
      break
    fi
    selected_count=$((selected_count + topup_step))
    (( selected_count > pool_count )) && selected_count=$pool_count
  done
  echo "[$(date '+%F %T')] [baseline-cache] published rows=$(wc -l < "$shared_baseline") valid=$valid_count file=$shared_baseline"
}

# Axis fitting uses small probe networks. It can safely use several workers
# per device, but still respects current free memory and a reserve for other
# project tasks. On resume, trust the complete artifact set rather than a
# stale non-zero worker status left by an interrupted earlier invocation.
if all_axis_artifacts_present; then
  echo "[$(date '+%F %T')] [reuse] all ${module_count} continuous axes are complete"
else
  if ! build_gpu_worker_list \
      "${DOWNSTREAM_AXIS_WORKERS_PER_GPU:-4}" \
      "${DOWNSTREAM_AXIS_WORKER_GPU_MB:-4000}" \
      "${DOWNSTREAM_AXIS_GPU_RESERVE_MB:-8000}" axis; then
    exit 4
  fi
  ACTIVE_WORKER_GPUS=("${PHASE_GPU_WORKERS[@]}")
  if ! run_parallel_phase axis; then
    if all_axis_artifacts_present; then
      echo "[$(date '+%F %T')] [axis] workers returned non-zero, but all final artifacts are complete; continue"
    else
      echo "[$(date '+%F %T')] axis phase failed; retry once with one worker per available GPU" >&2
      if ! build_gpu_worker_list 1 \
          "${DOWNSTREAM_AXIS_WORKER_GPU_MB:-4000}" \
          "${DOWNSTREAM_AXIS_GPU_RESERVE_MB:-8000}" axis_retry; then
        exit 4
      fi
      ACTIVE_WORKER_GPUS=("${PHASE_GPU_WORKERS[@]}")
      if ! run_parallel_phase axis; then
        if all_axis_artifacts_present; then
          echo "[$(date '+%F %T')] [axis] conservative retry returned non-zero, but all final artifacts are complete; continue"
        else
          echo "[$(date '+%F %T')] conservative axis retry failed" >&2
          exit 4
        fi
      fi
    fi
  fi
fi

# Build exactly one deterministic shared baseline before association workers
# start. If no compatible local result exists, freeze the original population,
# generate disjoint shards on up to two GPUs, and publish only after strict
# coverage validation.
shared_baseline="${SHARED_BASELINE_FILE:-$PAIR_ROOT/shared_association_baseline/baseline_generations.jsonl}"
module_zero_baseline="${MODULE_OUTPUTS[0]}/baseline_${ASSOCIATION_EVALUATION_ROLE:-association_confirmatory}/baseline_generations.jsonl"
if [[ ! -s "$shared_baseline" && -s "$module_zero_baseline" ]]; then
  mkdir -p "$(dirname "$shared_baseline")"
  cp "$module_zero_baseline" "${shared_baseline}.tmp.$$"
  mv "${shared_baseline}.tmp.$$" "$shared_baseline"
  echo "[$(date '+%F %T')] seeded shared baseline from completed local result: $module_zero_baseline"
fi
if [[ ! -s "$shared_baseline" ]]; then
  if ! generate_parallel_shared_baseline "$shared_baseline"; then
    echo "[$(date '+%F %T')] parallel shared baseline failed" >&2
    exit 4
  fi
fi
shared_baseline_ids="${SHARED_BASELINE_ID_FILE:-${shared_baseline}.ids.txt}"
if [[ ! -s "$shared_baseline_ids" ]]; then
  "$PYTHON_BIN" - "$shared_baseline" "$shared_baseline_ids" <<'PY'
import json
import os
import sys

source, output = sys.argv[1:]
ids = []
seen = set()
with open(source, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        sample_id = str(json.loads(line).get("id", "")).strip()
        if sample_id and sample_id not in seen:
            seen.add(sample_id)
            ids.append(sample_id)
temporary = f"{output}.tmp.{os.getpid()}"
with open(temporary, "w", encoding="utf-8") as handle:
    handle.writelines(f"{sample_id}\n" for sample_id in ids)
os.replace(temporary, output)
PY
fi
SHARED_BASELINE_ID_FILE="$shared_baseline_ids"
# Reusing a merged baseline returns from the core runner before model loading.
# This is a CPU coverage/field validation step and must not be blocked by the
# full-model intervention memory budget.  The GPU id is retained only because
# the common module-stage interface requires one.
baseline_validation_gpu="${DOWNSTREAM_GPU_IDS[0]}"
echo "[$(date '+%F %T')] [scheduler] baseline validation is CPU-only; bypass GPU memory gate"
if ! run_module_stage 0 "$baseline_validation_gpu" baseline; then
  echo "[$(date '+%F %T')] merged shared baseline validation failed" >&2
  exit 4
fi

# Association readouts use lightweight GPU linear algebra once axes and the
# shared deterministic baseline exist. Bootstrap and reporting remain on CPU.
association_workers="${DOWNSTREAM_ASSOCIATION_WORKERS:-8}"
(( association_workers > 0 )) || { echo "DOWNSTREAM_ASSOCIATION_WORKERS must be positive" >&2; exit 2; }
(( association_workers > module_count )) && association_workers=$module_count
ACTIVE_WORKER_GPUS=()
for ((worker=0; worker<association_workers; worker++)); do
  ACTIVE_WORKER_GPUS+=("${DOWNSTREAM_GPU_IDS[$((worker % ${#DOWNSTREAM_GPU_IDS[@]}))]}")
done
if ! run_parallel_phase association; then
  downstream_failed=1
fi

STRICT_CHAIN_DIR="${STRICT_CHAIN_DIR:-$PAIR_ROOT/strict_module_chain}"
stage "freeze stages 1-2 and persistent-trajectory eligibility"
"$PYTHON_BIN" scripts/summarize_strict_module_chain.py \
  --pair-root "$PAIR_ROOT" --output-dir "$STRICT_CHAIN_DIR" \
  --metric-profile "$METRIC_PROFILE" --phase eligibility \
  --alpha "${STRICT_CHAIN_ALPHA:-0.05}"
ELIGIBILITY_FILE="$STRICT_CHAIN_DIR/trajectory_eligible_modules.tsv"

# Re-read free memory after CPU analysis. The independent A/B next-token screens
# form ring 3 and run only for modules that passed rings 1-2 on their held-out
# data. A module passes the chain when all three prespecified rings pass; no
# additional cross-module test follows. Persistent trajectories use this same
# eligible set after the next-token barrier.
next_token_failed=0
ACTIVE_MODULE_INDICES=()
PRIMARY_CONSTRUCT_BY_INDEX=()
PRIMARY_DIRECTION_BY_INDEX=()
if [[ -s "$ELIGIBILITY_FILE" ]]; then
  while IFS=$'\t' read -r module_index module_key rank support module module_dir output_dir primary_construct component_metrics primary_direction rest; do
    [[ "$module_index" == module_index || -z "$module_index" ]] && continue
    ACTIVE_MODULE_INDICES+=("$module_index")
    PRIMARY_CONSTRUCT_BY_INDEX[$module_index]="$primary_construct"
    PRIMARY_DIRECTION_BY_INDEX[$module_index]="$primary_direction"
  done < "$ELIGIBILITY_FILE"
fi
stage "schedule next-token A/B for stages-1-2 passers (${#ACTIVE_MODULE_INDICES[@]}/${module_count})"
if (( ${#ACTIVE_MODULE_INDICES[@]} == 0 )); then
  echo "[$(date '+%F %T')] no module passed stages 1-2; next-token and persistent intervention skipped"
elif build_gpu_worker_list \
    "${DOWNSTREAM_INTERVENTION_WORKERS_PER_GPU:-2}" \
    "${DOWNSTREAM_INTERVENTION_WORKER_GPU_MB:-26000}" \
    "${DOWNSTREAM_GPU_RESERVE_MB:-12000}" next_token; then
  ACTIVE_WORKER_GPUS=("${PHASE_GPU_WORKERS[@]}")
  if ! run_parallel_phase next_token; then
    echo "[$(date '+%F %T')] next-token phase failed; retry once with one full-model worker per available GPU" >&2
    if build_gpu_worker_list 1 \
        "${DOWNSTREAM_INTERVENTION_WORKER_GPU_MB:-26000}" \
        "${DOWNSTREAM_GPU_RESERVE_MB:-12000}" next_token_retry; then
      ACTIVE_WORKER_GPUS=("${PHASE_GPU_WORKERS[@]}")
      run_parallel_phase next_token || next_token_failed=1
    else
      next_token_failed=1
    fi
  fi
else
  next_token_failed=1
fi

if (( next_token_failed != 0 )); then
  echo "[$(date '+%F %T')] next-token screens are incomplete; critical trajectories will not start" >&2
  downstream_failed=1
elif [[ "${RUN_PERSISTENT_TRAJECTORY:-1}" == 1 ]]; then
  stage "eligible next-token screens complete; start critical trajectories (${#ACTIVE_MODULE_INDICES[@]}/${module_count})"
  if (( ${#ACTIVE_MODULE_INDICES[@]} == 0 )); then
    echo "[$(date '+%F %T')] no module passed stages 1-2; persistent generation skipped"
  elif build_gpu_worker_list \
      "${DOWNSTREAM_TRAJECTORY_WORKERS_PER_GPU:-${DOWNSTREAM_INTERVENTION_WORKERS_PER_GPU:-2}}" \
      "${DOWNSTREAM_TRAJECTORY_WORKER_GPU_MB:-${DOWNSTREAM_INTERVENTION_WORKER_GPU_MB:-26000}}" \
      "${DOWNSTREAM_GPU_RESERVE_MB:-12000}" trajectory; then
    ACTIVE_WORKER_GPUS=("${PHASE_GPU_WORKERS[@]}")
    if ! run_parallel_phase trajectory; then
      echo "[$(date '+%F %T')] trajectory phase failed; retry once with one full-model worker per available GPU" >&2
      if build_gpu_worker_list 1 \
          "${DOWNSTREAM_TRAJECTORY_WORKER_GPU_MB:-${DOWNSTREAM_INTERVENTION_WORKER_GPU_MB:-26000}}" \
          "${DOWNSTREAM_GPU_RESERVE_MB:-12000}" trajectory_retry; then
        ACTIVE_WORKER_GPUS=("${PHASE_GPU_WORKERS[@]}")
        run_parallel_phase trajectory || downstream_failed=1
      else
        downstream_failed=1
      fi
    fi
  else
    downstream_failed=1
  fi
else
  echo "[$(date '+%F %T')] persistent trajectory phase disabled by RUN_PERSISTENT_TRAJECTORY=0"
fi

stage "summarize strict three-stage module chain"
"$PYTHON_BIN" scripts/summarize_strict_module_chain.py \
  --pair-root "$PAIR_ROOT" --output-dir "$STRICT_CHAIN_DIR" \
  --metric-profile "$METRIC_PROFILE" --phase final \
  --alpha "${STRICT_CHAIN_ALPHA:-0.05}"

if [[ "${PROXY_CONFIDENCE_EVALUATION:-0}" == 1 ]]; then
  stage "summarize module-specific proxy-confidence results"
  "$PYTHON_BIN" scripts/summarize_proxy_confidence_modules.py \
    --pair-root "$PAIR_ROOT" --output-dir "$PAIR_ROOT/proxy_confidence_summary"
fi

stage "summarize module-specific behavior map"
"$PYTHON_BIN" scripts/summarize_module_behavior_map.py \
  --run-root "$PAIR_ROOT" \
  --output-dir "$PAIR_ROOT/module_behavior_map" \
  --alpha "${MODULE_BEHAVIOR_ALPHA:-0.05}"

stage "task complete"
if [[ "$downstream_failed" != 0 ]]; then
  echo "[$(date '+%F %T')] task failed: one or more continuous modules failed" >&2
  exit 4
fi
