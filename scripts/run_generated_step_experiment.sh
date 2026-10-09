#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/task_lock.sh"

CONFIG_FILE="${CONFIG_FILE:?Set CONFIG_FILE to an experiment .env file}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
FORCE="${FORCE:-0}"
FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

[[ -s "$CONFIG_FILE" ]] || { echo "Missing config: $CONFIG_FILE" >&2; exit 2; }
set -a
# shellcheck source=/dev/null
source "$CONFIG_FILE"
set +a

# Sweep launchers may override only the pair and layer topology while keeping
# the scientific base configuration in one versioned env file.
[[ -n "${PAIR_SPECS_OVERRIDE:-}" ]] && PAIR_SPECS="$PAIR_SPECS_OVERRIDE"
[[ -n "${LAYER_FRACS_OVERRIDE:-}" ]] && LAYER_FRACS="$LAYER_FRACS_OVERRIDE"
[[ -n "${TARGET_CACHE_LAYER_FRACS_OVERRIDE:-}" ]] && \
  TARGET_CACHE_LAYER_FRACS="$TARGET_CACHE_LAYER_FRACS_OVERRIDE"
[[ -n "${SUPPORTS_OVERRIDE:-}" ]] && SUPPORTS="$SUPPORTS_OVERRIDE"
[[ -n "${EXTERNAL_SEMANTIC_HEAD_TYPE_OVERRIDE:-}" ]] && \
  EXTERNAL_SEMANTIC_HEAD_TYPE="$EXTERNAL_SEMANTIC_HEAD_TYPE_OVERRIDE"
[[ -n "${EXTERNAL_SEMANTIC_HEAD_DEPTH_OVERRIDE:-}" ]] && \
  EXTERNAL_SEMANTIC_HEAD_DEPTH="$EXTERNAL_SEMANTIC_HEAD_DEPTH_OVERRIDE"
[[ -n "${TARGET_EXTRACTION_BATCH_SIZE_OVERRIDE:-}" ]] && \
  TARGET_EXTRACTION_BATCH_SIZE="$TARGET_EXTRACTION_BATCH_SIZE_OVERRIDE"
[[ -n "${GENERATION_BATCH_SIZE_OVERRIDE:-}" ]] && \
  GENERATION_BATCH_SIZE="$GENERATION_BATCH_SIZE_OVERRIDE"
[[ -n "${RUN_SINGLE_STEP_TRAJECTORY_OVERRIDE:-}" ]] && \
  RUN_SINGLE_STEP_TRAJECTORY="$RUN_SINGLE_STEP_TRAJECTORY_OVERRIDE"
[[ -n "${MODULE_DIRECTION_MODE_OVERRIDE:-}" ]] && \
  MODULE_DIRECTION_MODE="$MODULE_DIRECTION_MODE_OVERRIDE"
[[ -n "${RANDOM_SUPPORT_NULL_OVERRIDE:-}" ]] && \
  RANDOM_SUPPORT_NULL="$RANDOM_SUPPORT_NULL_OVERRIDE"
[[ -n "${RANDOM_SUPPORT_SEED_OVERRIDE:-}" ]] && \
  RANDOM_SUPPORT_SEED="$RANDOM_SUPPORT_SEED_OVERRIDE"
[[ -n "${CONTINUOUS_CONTROLS_OVERRIDE:-}" ]] && \
  CONTINUOUS_CONTROLS="$CONTINUOUS_CONTROLS_OVERRIDE"
[[ -n "${REFINED_CODE_DOSE_MATCH_CONTROLS_OVERRIDE:-}" ]] && \
  REFINED_CODE_DOSE_MATCH_CONTROLS="$REFINED_CODE_DOSE_MATCH_CONTROLS_OVERRIDE"

required=(EXPERIMENT_PROFILE DATASET_PROFILE BASE_DATA_PATH DATA_SOURCE DATA_PREPARED_PATH PAIR_SPECS GENERATION_RECORD_STEPS SUPPORTS METRIC_PROFILE ANSWER_EXTRACTION)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "Missing config field: $name" >&2; exit 2; }
done
expected_metric="$($PYTHON_BIN -m metacog.cli dataset get "$DATASET_PROFILE" metric_profile)"
expected_answer="$($PYTHON_BIN -m metacog.cli dataset get "$DATASET_PROFILE" answer_extraction)"
[[ "$METRIC_PROFILE" == "$expected_metric" ]] || {
  echo "$DATASET_PROFILE requires METRIC_PROFILE=$expected_metric" >&2; exit 2;
}
[[ "$ANSWER_EXTRACTION" == "$expected_answer" ]] || {
  echo "$DATASET_PROFILE requires ANSWER_EXTRACTION=$expected_answer" >&2; exit 2;
}
RUN_ROOT="${RUN_ROOT:-runs/${DATASET_PROFILE}_external_semantic_continuous_${STAMP}}"

# Refuse a second launcher/resume process for the same run root. This protects
# activation-cache replacement and downstream GPU scheduling as one unit.
PIPELINE_LOCK="$RUN_ROOT/.locks/pipeline.lock"
if ! acquire_task_lock "$PIPELINE_LOCK" "pipeline:$RUN_ROOT"; then
  echo "[pipeline] duplicate launch refused: $RUN_ROOT" >&2
  exit 17
fi
trap release_task_locks EXIT INT TERM

mkdir -p "$RUN_ROOT/status"
cp "$CONFIG_FILE" "$RUN_ROOT/input_config.env"
shasum -a 256 "$CONFIG_FILE" > "$RUN_ROOT/config.sha256"
if [[ -n "${BASE_CONFIG_FILE:-}" ]]; then
  [[ -s "$BASE_CONFIG_FILE" ]] || {
    echo "Missing inherited base config: $BASE_CONFIG_FILE" >&2; exit 2;
  }
  cp "$BASE_CONFIG_FILE" "$RUN_ROOT/base_config.env"
  shasum -a 256 "$BASE_CONFIG_FILE" > "$RUN_ROOT/base_config.sha256"
fi

# Persist and print the parameters most likely to change the scientific result
# or runtime. This is deliberately resolved after all config inheritance.
resolved_config="$RUN_ROOT/resolved_config.env"
{
  printf 'EXPERIMENT_PROFILE=%q\n' "$EXPERIMENT_PROFILE"
  printf 'PAIR_SPECS=%q\n' "$PAIR_SPECS"
  printf 'LAYER_FRACS=%q\n' "$LAYER_FRACS"
  printf 'TARGET_CACHE_LAYER_FRACS=%q\n' "${TARGET_CACHE_LAYER_FRACS:-$LAYER_FRACS}"
  printf 'SUPPORTS=%q\n' "$SUPPORTS"
  printf 'EXTERNAL_SEMANTIC_HEAD_TYPE=%q\n' "${EXTERNAL_SEMANTIC_HEAD_TYPE:-mlp}"
  printf 'EXTERNAL_SEMANTIC_HEAD_DEPTH=%q\n' "${EXTERNAL_SEMANTIC_HEAD_DEPTH:-4}"
  printf 'NUM_MODULES=%q\n' "$NUM_MODULES"
  printf 'MAX_SELECTED_MODULES=%q\n' "$MAX_SELECTED_MODULES"
  printf 'MAX_PER_SUPPORT=%q\n' "$MAX_PER_SUPPORT"
  printf 'MODULE_DIRECTION_MODE=%q\n' "${MODULE_DIRECTION_MODE:-learned}"
  printf 'RANDOM_SUPPORT_SEED=%q\n' "${RANDOM_SUPPORT_SEED:-$SEED}"
  printf 'ALL_HELDOUT_MODULES=%q\n' "${ALL_HELDOUT_MODULES:-0}"
  printf 'MAIN_EPOCHS=%q\n' "$MAIN_EPOCHS"
  printf 'TARGET_PROMPT_MAX_SAMPLES=%q\n' "$TARGET_PROMPT_MAX_SAMPLES"
  printf 'GENERATION_RECORD_STEPS=%q\n' "$GENERATION_RECORD_STEPS"
  printf 'AXIS_PROTOTYPE_FRACTION=%q\n' "$AXIS_PROTOTYPE_FRACTION"
  printf 'AXIS_CAUSAL_FRACTION=%q\n' "$AXIS_CAUSAL_FRACTION"
  printf 'BASELINE_MAX_SAMPLES=%q\n' "$BASELINE_MAX_SAMPLES"
  printf 'BASELINE_MAX_NEW_TOKENS=%q\n' "$BASELINE_MAX_NEW_TOKENS"
  printf 'GENERATION_MAX_NEW_TOKENS=%q\n' "$GENERATION_MAX_NEW_TOKENS"
  printf 'GENERATION_BATCH_SIZE=%q\n' "$GENERATION_BATCH_SIZE"
  printf 'RUN_SINGLE_STEP_TRAJECTORY=%q\n' "${RUN_SINGLE_STEP_TRAJECTORY:-0}"
  printf 'CONTINUOUS_CONTROLS=%q\n' "${CONTINUOUS_CONTROLS:-opposite_direction}"
  printf 'REFINED_CODE_DOSE_MATCH_CONTROLS=%q\n' "${REFINED_CODE_DOSE_MATCH_CONTROLS:-0}"
  printf 'MEMORY_EFFICIENT_EXTRACTION=%q\n' "${MEMORY_EFFICIENT_EXTRACTION:-0}"
  printf 'TARGET_EXTRACTION_BATCH_SIZE=%q\n' "$TARGET_EXTRACTION_BATCH_SIZE"
  printf 'SEMANTIC_EXTRACTION_BATCH_SIZE=%q\n' "$SEMANTIC_EXTRACTION_BATCH_SIZE"
  printf 'ACTIVATION_CACHE_TAG=%q\n' "${ACTIVATION_CACHE_TAG:-}"
  printf 'PROXY_CONFIDENCE_EVALUATION=%q\n' "${PROXY_CONFIDENCE_EVALUATION:-0}"
  printf 'PROXY_CONFIDENCE_PRIMARY=%q\n' "${PROXY_CONFIDENCE_PRIMARY:-}"
  printf 'PROXY_CONFIDENCE_BOOTSTRAP_SAMPLES=%q\n' "${PROXY_CONFIDENCE_BOOTSTRAP_SAMPLES:-0}"
  printf 'PROXY_CONFIDENCE_PERMUTATION_TESTS=%q\n' "${PROXY_CONFIDENCE_PERMUTATION_TESTS:-0}"
  printf 'META_BEHAVIOR_EVALUATION=%q\n' "${META_BEHAVIOR_EVALUATION:-0}"
  printf 'META_BEHAVIOR_BOOTSTRAP_SAMPLES=%q\n' "${META_BEHAVIOR_BOOTSTRAP_SAMPLES:-0}"
  printf 'META_BEHAVIOR_PERMUTATION_TESTS=%q\n' "${META_BEHAVIOR_PERMUTATION_TESTS:-0}"
  printf 'META_BEHAVIOR_INCLUDE_SECONDARY=%q\n' "${META_BEHAVIOR_INCLUDE_SECONDARY:-1}"
} > "$resolved_config"
printf '[resolved-config] layers=%s supports=%s modules=%s selected=%s epochs=%s baseline_tokens=%s persistent_tokens=%s memory_efficient=%s\n' \
  "$LAYER_FRACS" "$SUPPORTS" "$NUM_MODULES" "$MAX_SELECTED_MODULES" "$MAIN_EPOCHS" \
  "$BASELINE_MAX_NEW_TOKENS" "$GENERATION_MAX_NEW_TOKENS" \
  "${MEMORY_EFFICIENT_EXTRACTION:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

model_field() { "$PYTHON_BIN" -m metacog.cli model get "$1" "$2"; }
model_path() { model_field "$1" path; }
model_dtype() { model_field "$1" dtype; }
model_trust() { model_field "$1" trust_remote_code; }
MEMORY_EXTRACTION_ARGS=()
MEMORY_EXTRACTION_EFFECTIVE=0
if [[ "${MEMORY_EFFICIENT_EXTRACTION:-0}" == 1 ]]; then
  # The optimization is optional.  Keep the launcher compatible with older
  # server checkouts whose extractor CLI predates this flag.
  if "$PYTHON_BIN" scripts/extract_generation_step_activations.py --help 2>&1 | grep -q -- "--memory-efficient" && \
     "$PYTHON_BIN" scripts/extract_activations.py --help 2>&1 | grep -q -- "--memory-efficient"; then
    MEMORY_EXTRACTION_ARGS+=(--memory-efficient)
    MEMORY_EXTRACTION_EFFECTIVE=1
  else
    echo "[extract] warning: extractor CLI lacks --memory-efficient; continuing without optional optimization"
  fi
fi
printf 'MEMORY_EXTRACTION_EFFECTIVE=%q\n' "$MEMORY_EXTRACTION_EFFECTIVE" >> "$resolved_config"

slug_steps="${GENERATION_RECORD_STEPS//,/-}"
cache_suffix=""
if [[ -n "${ACTIVATION_CACHE_TAG:-}" ]]; then
  safe_cache_tag="${ACTIVATION_CACHE_TAG//[^A-Za-z0-9_.-]/_}"
  cache_suffix="_${safe_cache_tag}"
fi
target_cache() {
  echo "activations/${1}_${DATASET_PROFILE}_gensteps_n${TARGET_PROMPT_MAX_SAMPLES}_steps${slug_steps}_s${SEED}${cache_suffix}"
}
semantic_cache() {
  echo "activations/${1}_on_${2}_${DATASET_PROFILE}_gensteps_n${TARGET_PROMPT_MAX_SAMPLES}_steps${slug_steps}_s${SEED}_l${3}${cache_suffix}"
}

prepare_data() {
  if [[ -s "$BASE_DATA_PATH" ]]; then
    echo "[data] reuse $BASE_DATA_PATH"
    return
  fi
  read -r -a data_sources <<< "$DATA_SOURCE"
  "$PYTHON_BIN" -m metacog.cli dataset prepare "$DATASET_PROFILE" \
    --input "${data_sources[@]}" --output "$DATA_PREPARED_PATH" \
    --max-samples "${DATA_MAX_SAMPLES:-$TARGET_PROMPT_MAX_SAMPLES}" \
    --split-mode "${DATA_SPLIT_MODE:-resplit}" \
    --train-size "${DATA_TRAIN_SIZE:-22000}" \
    --validation-size "${DATA_VALIDATION_SIZE:-7000}" --seed "$SEED"
}

validate_data() {
  "$PYTHON_BIN" -m metacog.cli dataset validate "$DATASET_PROFILE" \
    "$BASE_DATA_PATH" --min-samples "$TARGET_PROMPT_MAX_SAMPLES"
}

resolve_target_layers() {
  local path="$1"
  shift
  "$PYTHON_BIN" - "$path" "$@" <<'PY'
import math, sys
from transformers import AutoConfig
path, *fractions = sys.argv[1:]
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
n = int(getattr(config, "num_hidden_layers"))
layers = [max(0, min(n - 1, int(math.floor(float(f) * n + 0.5)))) for f in fractions]
print(" ".join(map(str, layers)))
PY
}

validate_target_cache() {
  local cache="$1" expected_model="$2" expected_data="$3" expected_layers="$4"
  "$PYTHON_BIN" - "$cache/manifest.json" "$GENERATION_RECORD_STEPS" \
    "$TARGET_PROMPT_MAX_SAMPLES" "$expected_model" "$expected_data" "$expected_layers" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected_steps = [int(x) for x in sys.argv[2].split(",")]
expected_n = int(sys.argv[3])
expected_model, expected_data = sys.argv[4:6]
expected_layers = [int(value) for value in sys.argv[6].split()]
assert manifest.get("record_type") == "generated_token_states", manifest.get("record_type")
assert manifest.get("record_steps") == expected_steps, (manifest.get("record_steps"), expected_steps)
if expected_n > 0:
    assert int(manifest.get("num_prompt_examples", -1)) == expected_n
assert manifest.get("model_path") == expected_model, (manifest.get("model_path"), expected_model)
assert manifest.get("data") == expected_data, (manifest.get("data"), expected_data)
actual_layers = {int(value) for value in manifest.get("layers", [])}
assert set(expected_layers).issubset(actual_layers), (sorted(actual_layers), expected_layers)
assert manifest.get("expanded_data_is_fully_rendered") is True
assert manifest.get("expanded_data_has_exact_prompt_token_ids") is True
assert manifest.get("external_semantic_data_uses_raw_prompt_plus_assistant_prefix") is True
PY
}

find_compatible_target_cache() {
  local target="$1" preferred="$2" expected_model="$3" expected_data="$4" expected_layers="$5"
  local candidate
  for candidate in "$preferred" \
      activations/${target}_${DATASET_PROFILE}_gensteps_n${TARGET_PROMPT_MAX_SAMPLES}_steps${slug_steps}_s${SEED}*; do
    [[ -s "$candidate/manifest.json" ]] || continue
    if validate_target_cache "$candidate" "$expected_model" "$expected_data" "$expected_layers" \
        >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

validate_semantic_cache() {
  local cache="$1" expected_model="$2" target_dir="$3" target_layer="$4" semantic_layer="$5"
  "$PYTHON_BIN" - "$cache/manifest.json" "$expected_model" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest.get("model_path") == sys.argv[2], (manifest.get("model_path"), sys.argv[2])
PY
  "$PYTHON_BIN" scripts/validate_activation_alignment.py \
    --target-dir "$target_dir" --semantic-dir "$cache" \
    --target-layer "$target_layer" --semantic-layer "$semantic_layer" >/dev/null
}

find_compatible_semantic_cache() {
  local reference="$1" target="$2" semantic_layer="$3" preferred="$4"
  local expected_model="$5" target_dir="$6" target_layer="$7" candidate
  for candidate in "$preferred" \
      activations/${reference}_on_${target}_${DATASET_PROFILE}_gensteps_n${TARGET_PROMPT_MAX_SAMPLES}_steps${slug_steps}_s${SEED}_l${semantic_layer}*; do
    [[ -s "$candidate/manifest.json" ]] || continue
    if validate_semantic_cache "$candidate" "$expected_model" "$target_dir" \
        "$target_layer" "$semantic_layer" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

quarantine_cache() {
  local path="$1"
  if [[ -d "$path" ]]; then
    local backup="${path}.stale_$(date +%Y%m%d_%H%M%S)"
    echo "[cache] move stale/forced cache to $backup"
    mv "$path" "$backup"
  fi
}

run_pair() {
  local gpu="$1" spec="$2"
  local target reference semantic_layer name pair_root target_model reference_model target_dir semantic_dir
  local semantic_layer_count preferred_target_dir preferred_semantic_dir compatible_cache historical_baseline
  local -a target_trust_args=() reference_trust_args=() historical_confidence_args=()
  local -a layers=() capture_layers=()
  local target_cache_ok semantic_cache_ok target_cache_built=0
  IFS=: read -r target reference semantic_layer <<< "$spec"
  [[ "$target" != "$reference" ]] || { echo "Self semantic references are disabled: $spec" >&2; return 2; }
  name="${target}__semantic_${reference}_l${semantic_layer}"
  pair_root="$RUN_ROOT/$name"
  mkdir -p "$pair_root"
  exec > >(tee -a "$pair_root/task.log") 2>&1
  echo "[$(date '+%F %T')] START pair=$name gpu=$gpu config=$EXPERIMENT_PROFILE launch_id=${TASK_LAUNCH_ID:-standalone} pid=$BASHPID ppid=$PPID"

  target_model="$(model_path "$target")"
  reference_model="$(model_path "$reference")"
  [[ "$(model_trust "$target")" == 1 ]] && target_trust_args+=(--trust-remote-code)
  [[ "$(model_trust "$reference")" == 1 ]] && reference_trust_args+=(--trust-remote-code)
  preferred_target_dir="$(target_cache "$target")"
  preferred_semantic_dir="$(semantic_cache "$reference" "$target" "$semantic_layer")"
  target_dir="$preferred_target_dir"
  semantic_dir="$preferred_semantic_dir"
  read -r -a layers <<< "$(resolve_target_layers "$target_model" $LAYER_FRACS)"
  read -r -a capture_layers <<< "$(resolve_target_layers "$target_model" ${TARGET_CACHE_LAYER_FRACS:-$LAYER_FRACS})"
  semantic_layer_count="$($PYTHON_BIN - "$reference_model" <<'PY'
import sys
from transformers import AutoConfig
config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
print(int(getattr(config, "num_hidden_layers")))
PY
)"
  if (( semantic_layer < 0 || semantic_layer >= semantic_layer_count )); then
    echo "Semantic layer $semantic_layer outside reference depth $semantic_layer_count: $reference" >&2
    return 2
  fi
  echo "[contract] target_layers=${layers[*]} cache_layers=${capture_layers[*]} semantic_layer=$semantic_layer steps=$GENERATION_RECORD_STEPS"

  target_cache_ok=0
  compatible_cache=""
  if [[ "$FORCE_ACTIVATIONS" != 1 ]]; then
    compatible_cache="$(find_compatible_target_cache "$target" "$preferred_target_dir" \
      "$target_model" "$BASE_DATA_PATH" "${layers[*]}" || true)"
  fi
  if [[ -n "$compatible_cache" ]]; then
    target_dir="$compatible_cache"
    target_cache_ok=1
  fi
  if [[ "$FORCE_ACTIVATIONS" == 1 || "$target_cache_ok" != 1 ]]; then
    target_dir="$preferred_target_dir"
    quarantine_cache "$target_dir"
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/extract_generation_step_activations.py \
      --model-path "$target_model" --data "$BASE_DATA_PATH" --layers "${capture_layers[@]}" \
      --output-dir "$target_dir" --batch-size "$TARGET_EXTRACTION_BATCH_SIZE" \
      --max-samples "$TARGET_PROMPT_MAX_SAMPLES" --sample-strategy random --seed "$SEED" \
      --max-length "$EXTRACTION_MAX_LENGTH" --max-generation-steps "$MAX_GENERATION_STEPS" \
      --record-steps "$GENERATION_RECORD_STEPS" --temperature 0.0 \
      --dtype "$(model_dtype "$target")" --device-map single --prompt-style data \
      --use-chat-template --chat-template-enable-thinking "${CHAT_TEMPLATE_ENABLE_THINKING:-auto}" \
      --shard-size "$EXTRACTION_SHARD_SIZE" "${MEMORY_EXTRACTION_ARGS[@]}" \
      "${target_trust_args[@]}"; then
      echo "[cache] target activation extraction failed: $target_dir" >&2
      return 1
    fi
    target_cache_built=1
  else
    echo "[reuse] compatible target generated-step cache: $target_dir"
  fi
  if ! validate_target_cache "$target_dir" "$target_model" "$BASE_DATA_PATH" "${layers[*]}"; then
    echo "[cache] target activation cache is missing or incompatible: $target_dir" >&2
    return 1
  fi
  if [[ "$target_cache_built" == 1 ]] && \
     ! validate_target_cache "$target_dir" "$target_model" "$BASE_DATA_PATH" "${capture_layers[*]}"; then
    echo "[cache] newly extracted target cache does not contain the shared layer union" >&2
    return 1
  fi

  semantic_cache_ok=0
  compatible_cache=""
  if [[ "$FORCE_ACTIVATIONS" != 1 ]]; then
    compatible_cache="$(find_compatible_semantic_cache "$reference" "$target" "$semantic_layer" \
      "$preferred_semantic_dir" "$reference_model" "$target_dir" "${layers[1]}" || true)"
  fi
  if [[ -n "$compatible_cache" ]]; then
    semantic_dir="$compatible_cache"
    semantic_cache_ok=1
  fi
  if [[ "$FORCE_ACTIVATIONS" == 1 || "$semantic_cache_ok" != 1 ]]; then
    semantic_dir="$preferred_semantic_dir"
    quarantine_cache "$semantic_dir"
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/extract_activations.py \
      --model-path "$reference_model" --data "$target_dir/external_semantic_data.jsonl" \
      --layers "$semantic_layer" --output-dir "$semantic_dir" \
      --batch-size "$SEMANTIC_EXTRACTION_BATCH_SIZE" --max-length "$EXTERNAL_SEMANTIC_MAX_LENGTH" \
      --pooling last_token --dtype "$(model_dtype "$reference")" --device-map single \
      --prompt-style data --use-chat-template \
      --chat-template-enable-thinking "${CHAT_TEMPLATE_ENABLE_THINKING:-auto}" \
      --shard-size "$EXTRACTION_SHARD_SIZE" "${MEMORY_EXTRACTION_ARGS[@]}" \
      "${reference_trust_args[@]}"; then
      echo "[cache] external semantic extraction failed: $semantic_dir" >&2
      return 1
    fi
  else
    echo "[reuse] compatible external semantic cache: $semantic_dir"
  fi
  if ! "$PYTHON_BIN" scripts/validate_activation_alignment.py \
    --target-dir "$target_dir" --semantic-dir "$semantic_dir" \
    --target-layer "${layers[1]}" --semantic-layer "$semantic_layer"; then
    echo "[cache] target/external-semantic activation alignment failed" >&2
    return 1
  fi

  shared_baseline=""
  if [[ -n "${SHARED_BASELINE_ROOT:-}" ]]; then
    shared_baseline="${SHARED_BASELINE_ROOT}/${DATASET_PROFILE}/${target}/baseline_generations.jsonl"
    if [[ ! -s "$shared_baseline" && "${REUSE_HISTORICAL_BASELINES:-1}" == 1 ]]; then
      if [[ "${PROXY_CONFIDENCE_EVALUATION:-0}" == 1 || "${META_BEHAVIOR_EVALUATION:-0}" == 1 ]]; then
        historical_confidence_args+=(--require-confidence)
      fi
      historical_baseline="$("$PYTHON_BIN" scripts/find_reusable_baseline.py \
        --search-root runs --exclude-root "$RUN_ROOT" --model-path "$target_model" \
        --dataset-profile "$DATASET_PROFILE" --metric-profile "$METRIC_PROFILE" \
        --answer-extraction "$ANSWER_EXTRACTION" \
        --max-new-tokens "$BASELINE_MAX_NEW_TOKENS" --seed "$SEED" \
        "${historical_confidence_args[@]}" 2>/dev/null || true)"
      if [[ -n "$historical_baseline" && -s "$historical_baseline" ]]; then
        mkdir -p "$(dirname "$shared_baseline")"
        cp "$historical_baseline" "${shared_baseline}.tmp.$$"
        mv "${shared_baseline}.tmp.$$" "$shared_baseline"
        echo "[baseline] seeded cross-condition cache from historical result: $historical_baseline"
      fi
    fi
    if [[ -s "$shared_baseline" ]]; then
      echo "[baseline] reuse cross-condition deterministic baseline: $shared_baseline"
    else
      echo "[baseline] initialize cross-condition cache: $shared_baseline"
    fi
  fi
  if ! env GPU="$gpu" PAIR_ROOT="$pair_root" TARGET_PROFILE="$target" \
    TARGET_MODEL_PATH="$target_model" TARGET_ACTIVATION_DIR="$target_dir" \
    SEMANTIC_PROFILE="$reference" SEMANTIC_ACTIVATION_DIR="$semantic_dir" \
    EXTERNAL_SEMANTIC_LAYER="$semantic_layer" \
    DATA_PATH="$target_dir/expanded_generation_data.jsonl" \
    MODEL_DTYPE="$(model_dtype "$target")" TRUST_REMOTE_CODE="$(model_trust "$target")" \
    SHARED_BASELINE_FILE="$shared_baseline" \
    RANDOM_SUPPORT_NULL="${RANDOM_SUPPORT_NULL:-0}" \
    FORCE="$FORCE" \
    bash scripts/run_model_pair.sh; then
    echo "[$(date '+%F %T')] downstream failed pair=$name" >&2
    return 1
  fi
  echo "[$(date '+%F %T')] COMPLETE pair=$name"
}

prepare_data
validate_data
read -r -a raw_pairs <<< "$PAIR_SPECS"
declare -A seen_pairs=()
pairs=()
for spec in "${raw_pairs[@]}"; do
  IFS=: read -r target reference semantic_layer extra <<< "$spec"
  pair_key="${target}__semantic_${reference}_l${semantic_layer}"
  if [[ -n "${seen_pairs[$pair_key]:-}" ]]; then
    echo "[queue] skip duplicate pair spec: $spec" >&2
    continue
  fi
  seen_pairs[$pair_key]=1
  pairs+=("$spec")
done
[[ "${#pairs[@]}" -gt 0 ]] || { echo "PAIR_SPECS is empty" >&2; exit 2; }
if [[ -n "${TASK_LAUNCH_ID:-}" ]]; then
  (( ${#pairs[@]} == 1 )) || {
    echo "[queue] canonical master task must contain exactly one pair; got ${#pairs[@]} launch_id=$TASK_LAUNCH_ID" >&2
    exit 2
  }
  [[ "$GPU0" == "$GPU1" ]] || {
    echo "[queue] canonical master task must bind one GPU; got GPU0=$GPU0 GPU1=$GPU1 launch_id=$TASK_LAUNCH_ID" >&2
    exit 2
  }
fi
declare -A seen_targets=()
declare -A target_gpu=()
next_gpu=0
for spec in "${pairs[@]}"; do
  IFS=: read -r target reference semantic_layer extra <<< "$spec"
  [[ -n "$target" && -n "$reference" && "$semantic_layer" =~ ^[0-9]+$ && -z "${extra:-}" ]] || {
    echo "Invalid pair spec (expected target:reference:layer): $spec" >&2; exit 2;
  }
  model_path "$target" >/dev/null
  model_path "$reference" >/dev/null
  if [[ -z "${seen_targets[$target]:-}" ]]; then
    seen_targets[$target]=1
    if (( next_gpu == 0 )); then
      target_gpu[$target]="$GPU0"
    else
      target_gpu[$target]="$GPU1"
    fi
    next_gpu=$(( (next_gpu + 1) % 2 ))
  fi
done
printf 'status\tgpu\tpair\tlog\n' > "$RUN_ROOT/task_logs.tsv"
for index in "${!pairs[@]}"; do
  gpu="$GPU0"; ((index % 2 == 1)) && gpu="$GPU1"
  IFS=: read -r target reference semantic_layer <<< "${pairs[$index]}"
  name="${target}__semantic_${reference}_l${semantic_layer}"
  gpu="${target_gpu[$target]}"
  printf 'pending\t%s\t%s\t%s\n' "$gpu" "$name" "$RUN_ROOT/$name/task.log" >> "$RUN_ROOT/task_logs.tsv"
done

worker() {
  local gpu="$1" index spec target reference semantic_layer name rc failed=0
  # This background subshell does not own the pipeline lifecycle. Keep the
  # parent EXIT trap from releasing the pipeline lock while another worker is
  # still running.
  trap - EXIT INT TERM
  for index in "${!pairs[@]}"; do
    spec="${pairs[$index]}"
    IFS=: read -r target reference semantic_layer <<< "$spec"
    [[ "${target_gpu[$target]}" == "$gpu" ]] || continue
    name="${target}__semantic_${reference}_l${semantic_layer}"
    set +e
    (run_pair "$gpu" "$spec")
    rc=$?
    set -e
    [[ "$rc" == 0 ]] || failed=1
    printf '%s\t%s\t%s\n' "$([[ "$rc" == 0 ]] && echo complete || echo failed)" "$gpu" "$rc" \
      > "$RUN_ROOT/status/${name}.status"
  done
  return "$failed"
}

worker_status=0
worker_pids=()
if (( ${#pairs[@]} == 1 )); then
  # The full experiment launcher submits exactly one pair per canonical task.
  # Execute it synchronously: there is no second-level scheduler and therefore
  # no second path that can claim the same pair.
  spec="${pairs[0]}"
  IFS=: read -r target reference semantic_layer <<< "$spec"
  name="${target}__semantic_${reference}_l${semantic_layer}"
  echo "[queue] direct single-pair execution: pair=$name gpu=$GPU0 launch_id=${TASK_LAUNCH_ID:-standalone}"
  set +e
  (run_pair "$GPU0" "$spec")
  rc=$?
  set -e
  [[ "$rc" == 0 ]] || worker_status=1
  printf '%s\t%s\t%s\n' "$([[ "$rc" == 0 ]] && echo complete || echo failed)" "$GPU0" "$rc" \
    > "$RUN_ROOT/status/${name}.status"
else
  worker_gpu_list="$GPU0"
  worker "$GPU0" & worker_pids+=("$!")
  if [[ "$GPU1" != "$GPU0" ]]; then
    worker_gpu_list="$worker_gpu_list $GPU1"
    worker "$GPU1" & worker_pids+=("$!")
  else
    echo "[queue] GPU0 and GPU1 both resolve to $GPU0; start one worker for this device"
  fi
  echo "[queue] started ${#worker_pids[@]} unique GPU worker(s): $worker_gpu_list"
  for worker_pid in "${worker_pids[@]}"; do
    wait "$worker_pid" || worker_status=1
  done
fi

if find "$RUN_ROOT" -path '*/continuous_behavior/continuous_behavior_summary.json' \
    -type f -print -quit | grep -q .; then
  "$PYTHON_BIN" scripts/summarize_module_behavior_map.py \
    --run-root "$RUN_ROOT" --output-dir "$RUN_ROOT/module_behavior_map" \
    --alpha "$MODULE_BEHAVIOR_ALPHA"
fi

printf 'status\tgpu\texit_code\tpair\n' > "$RUN_ROOT/final_status.tsv"
for file in "$RUN_ROOT"/status/*.status; do
  [[ -e "$file" ]] || continue
  read -r status gpu code < "$file"
  printf '%s\t%s\t%s\t%s\n' "$status" "$gpu" "$code" "$(basename "$file" .status)" >> "$RUN_ROOT/final_status.tsv"
done
cat "$RUN_ROOT/final_status.tsv"
[[ "$worker_status" == 0 ]]
