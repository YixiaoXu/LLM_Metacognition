#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/task_lock.sh"

CONFIG_FILE="${CONFIG_FILE:-configs/beavertails330k_external_semantic_continuous_v1.env}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
PREP_GPU="${PREP_GPU:-$GPU0}"
FORCE="${FORCE:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/beavertails330k_external_semantic_continuous_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

[[ -s "$CONFIG_FILE" ]] || { echo "Missing experiment config: $CONFIG_FILE" >&2; exit 2; }
set -a
# shellcheck source=/dev/null
source "$CONFIG_FILE"
set +a

if ! acquire_task_lock "$RUN_ROOT/.locks/pipeline.lock" "pipeline:$RUN_ROOT"; then
  echo "[pipeline] duplicate launch refused: $RUN_ROOT" >&2
  exit 17
fi
trap release_task_locks EXIT INT TERM

required=(EXPERIMENT_PROFILE DATASET_PROFILE DATA_PATH PAIR_SPECS SUPPORTS CANDIDATE_TOP_K NUM_MODULES CONTINUOUS_DOSES METRIC_PROFILE ANSWER_EXTRACTION ACTIVATION_DATA_TAG)
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

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/activation_prep"
cp "$CONFIG_FILE" "$RUN_ROOT/resolved_config.env"
shasum -a 256 "$CONFIG_FILE" > "$RUN_ROOT/config.sha256"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

model_field() { "$PYTHON_BIN" -m metacog.cli model get "$1" "$2"; }
model_value() { model_field "$1" path; }
model_dtype() { model_field "$1" dtype; }
extraction_batch() { model_field "$1" extraction_batch_size; }
model_trust() { model_field "$1" trust_remote_code; }

activation_value() {
  echo "${ACTIVATION_ROOT:-activations}/${1}_${ACTIVATION_DATA_TAG}"
}

prepare_data() {
  [[ -s "$DATA_PATH" ]] && return 0
  [[ -s "$SOURCE_DATA" ]] || { echo "Missing BeaverTails source: $SOURCE_DATA" >&2; exit 2; }
  "$PYTHON_BIN" scripts/prepare_beavertails_neutral.py \
    --input "$SOURCE_DATA" --output "$DATA_PATH" \
    --max-samples "$DATA_MAX_SAMPLES" --sampling proportional --seed "$SEED"
}

ensure_activation() {
  local profile="$1" model out log_file
  model="$(model_value "$profile")"
  out="$(activation_value "$profile")"
  log_file="$RUN_ROOT/activation_prep/${profile}.log"
  if [[ -s "$out/manifest.json" ]]; then
    printf '[%s] reuse activation cache: %s\n' "$(date '+%F %T')" "$out" >> "$log_file"
    return 0
  fi
  local trust=()
  [[ "$(model_trust "$profile")" == 1 ]] && trust+=(--trust-remote-code)
  echo "[$(date '+%F %T')] extract activations profile=$profile gpu=$PREP_GPU"
  CUDA_VISIBLE_DEVICES="$PREP_GPU" "$PYTHON_BIN" scripts/extract_activations.py \
    --model-path "$model" --data "$DATA_PATH" --layers all --output-dir "$out" \
    --batch-size "$(extraction_batch "$profile")" --max-length "$EXTRACTION_MAX_LENGTH" \
    --pooling last_token --dtype "$(model_dtype "$profile")" --device-map single \
    --prompt-style data --use-chat-template --chat-template-enable-thinking auto \
    --shard-size "$EXTRACTION_SHARD_SIZE" "${trust[@]}" \
    >> "$log_file" 2>&1
}

run_pair() {
  local gpu="$1" spec="$2" target reference name pair_root code
  target="${spec%%:*}"
  reference="${spec##*:}"
  name="${target}__semantic_${reference}"
  pair_root="$RUN_ROOT/$name"
  mkdir -p "$pair_root"
  echo "[$(date '+%F %T')] pair=$name gpu=$gpu log=$pair_root/task.log"
  set +e
  env GPU="$gpu" PAIR_ROOT="$pair_root" TARGET_PROFILE="$target" \
    TARGET_MODEL_PATH="$(model_value "$target")" \
    TARGET_ACTIVATION_DIR="$(activation_value "$target")" \
    SEMANTIC_PROFILE="$reference" \
    SEMANTIC_ACTIVATION_DIR="$(activation_value "$reference")" \
    TRUST_REMOTE_CODE="$(model_trust "$target")" \
    FORCE="$FORCE" bash scripts/run_model_pair.sh \
    >> "$pair_root/task.log" 2>&1
  code=$?
  set -e
  printf '%s\t%s\t%s\t%s\n' "$([[ "$code" == 0 ]] && echo complete || echo failed)" "$gpu" "$target" "$reference" \
    > "$RUN_ROOT/status/${name}.status"
  return "$code"
}

prepare_data
"$PYTHON_BIN" -m metacog.cli dataset validate "$DATASET_PROFILE" "$DATA_PATH"
declare -A profiles=()
read -r -a raw_pairs <<< "$PAIR_SPECS"
declare -A seen_pairs=()
pairs=()
for spec in "${raw_pairs[@]}"; do
  pair_key="$spec"
  if [[ -n "${seen_pairs[$pair_key]:-}" ]]; then
    echo "[queue] skip duplicate pair spec: $spec" >&2
    continue
  fi
  seen_pairs[$pair_key]=1
  pairs+=("$spec")
  profiles["${spec%%:*}"]=1
  profiles["${spec##*:}"]=1
done
[[ "${#pairs[@]}" -gt 0 ]] || { echo "PAIR_SPECS is empty" >&2; exit 2; }

printf 'task_type\ttask\tgpu\tlog\n' > "$RUN_ROOT/task_logs.tsv"
for profile in "${!profiles[@]}"; do
  printf 'activation\t%s\t%s\t%s\n' "$profile" "$PREP_GPU" \
    "$RUN_ROOT/activation_prep/${profile}.log" >> "$RUN_ROOT/task_logs.tsv"
done
for index in "${!pairs[@]}"; do
  spec="${pairs[$index]}"
  target="${spec%%:*}"
  reference="${spec##*:}"
  gpu="$GPU0"
  (( index % 2 == 1 )) && gpu="$GPU1"
  printf 'model_pair\t%s__semantic_%s\t%s\t%s\n' \
    "$target" "$reference" "$gpu" \
    "$RUN_ROOT/${target}__semantic_${reference}/task.log" >> "$RUN_ROOT/task_logs.tsv"
done

echo "Task log index: $RUN_ROOT/task_logs.tsv"
for profile in "${!profiles[@]}"; do ensure_activation "$profile"; done

queue0=(); queue1=()
for index in "${!pairs[@]}"; do
  (( index % 2 == 0 )) && queue0+=("${pairs[$index]}") || queue1+=("${pairs[$index]}")
done

run_queue() {
  local gpu="$1"; shift
  local failed=0
  for spec in "$@"; do run_pair "$gpu" "$spec" || failed=1; done
  return "$failed"
}

status0=0; status1=0
run_queue "$GPU0" "${queue0[@]}" & pid0=$!
run_queue "$GPU1" "${queue1[@]}" & pid1=$!
wait "$pid0" || status0=1
wait "$pid1" || status1=1

if find "$RUN_ROOT" -path '*/continuous_behavior/continuous_behavior_summary.json' \
    -type f -print -quit | grep -q .; then
  "$PYTHON_BIN" scripts/summarize_module_behavior_map.py \
    --run-root "$RUN_ROOT" \
    --output-dir "$RUN_ROOT/module_behavior_map" \
    --alpha "${MODULE_BEHAVIOR_ALPHA:-0.05}"
fi

printf 'status\tgpu\ttarget\tsemantic_reference\n' > "$RUN_ROOT/final_status.tsv"
for status_file in "$RUN_ROOT"/status/*.status; do
  [[ -e "$status_file" ]] && cat "$status_file" >> "$RUN_ROOT/final_status.tsv"
done
cat "$RUN_ROOT/final_status.tsv"
[[ "$status0" == 0 && "$status1" == 0 ]]
