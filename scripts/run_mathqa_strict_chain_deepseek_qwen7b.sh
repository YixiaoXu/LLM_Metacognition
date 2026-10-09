#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${CONFIG_FILE:-configs/mathqa_strict_chain_deepseek_qwen7b_v1.env}"
RUN_ROOT="${RUN_ROOT:-runs/mathqa_strict_chain_deepseek_qwen7b_$(date +%Y%m%d_%H%M%S)}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
FORCE="${FORCE:-0}"
FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
requested_target_profile="${TARGET_PROFILE_OVERRIDE:-}"
requested_reference_specs="${SEMANTIC_REFERENCE_SPECS_OVERRIDE:-}"
requested_persistent="${RUN_PERSISTENT_TRAJECTORY_OVERRIDE:-}"

[[ -s "$CONFIG_FILE" ]] || { echo "Missing config: $CONFIG_FILE" >&2; exit 2; }
set -a
# shellcheck source=/dev/null
source "$CONFIG_FILE"
set +a
[[ -n "$requested_target_profile" ]] && TARGET_PROFILE="$requested_target_profile"
[[ -n "$requested_reference_specs" ]] && SEMANTIC_REFERENCE_SPECS="$requested_reference_specs"
[[ -n "$requested_persistent" ]] && RUN_PERSISTENT_TRAJECTORY="$requested_persistent"

mkdir -p "$RUN_ROOT"
exec > >(
  "$PYTHON_BIN" -u scripts/log_stream.py \
    --log-file "$RUN_ROOT/task.log" \
    --progress-interval "${LOG_PROGRESS_INTERVAL_SECONDS:-60}"
) 2>&1
cp "$CONFIG_FILE" "$RUN_ROOT/input_config.env"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$CONFIG_FILE" > "$RUN_ROOT/config.sha256"
else
  shasum -a 256 "$CONFIG_FILE" > "$RUN_ROOT/config.sha256"
fi
{
  printf 'DATASET_PROFILE=%q\n' "$DATASET_PROFILE"
  printf 'TARGET_PROFILE=%q\n' "$TARGET_PROFILE"
  printf 'SEMANTIC_REFERENCE_SPECS=%q\n' "$SEMANTIC_REFERENCE_SPECS"
  printf 'TARGET_PROMPT_MAX_SAMPLES=%q\n' "$TARGET_PROMPT_MAX_SAMPLES"
  printf 'SUPPORTS=%q\n' "$SUPPORTS"
  printf 'NUM_MODULES=%q\n' "$NUM_MODULES"
  printf 'MAX_SELECTED_MODULES=%q\n' "$MAX_SELECTED_MODULES"
  printf 'AXIS_PROTOTYPE_FRACTION=%q\n' "$AXIS_PROTOTYPE_FRACTION"
  printf 'AXIS_CAUSAL_FRACTION=%q\n' "$AXIS_CAUSAL_FRACTION"
  printf 'AXIS_ASSOCIATION_FRACTION=%q\n' "${AXIS_ASSOCIATION_FRACTION:-}"
  printf 'AXIS_TRAJECTORY_FRACTION=%q\n' "${AXIS_TRAJECTORY_FRACTION:-}"
  printf 'BASELINE_MAX_SAMPLES=%q\n' "$BASELINE_MAX_SAMPLES"
  printf 'BASELINE_INITIAL_SAMPLES=%q\n' "${BASELINE_INITIAL_SAMPLES:-$BASELINE_MAX_SAMPLES}"
  printf 'BASELINE_MIN_UNTRUNCATED_SAMPLES=%q\n' "${BASELINE_MIN_UNTRUNCATED_SAMPLES:-0}"
  printf 'BASELINE_TOPUP_STEP=%q\n' "${BASELINE_TOPUP_STEP:-0}"
  printf 'HISTORICAL_BASELINE_SEARCH_ROOTS=%q\n' "${HISTORICAL_BASELINE_SEARCH_ROOTS:-runs}"
  printf 'RUN_PERSISTENT_TRAJECTORY=%q\n' "$RUN_PERSISTENT_TRAJECTORY"
} > "$RUN_ROOT/resolved_config.env"

stage() { printf '\n[%s] ===== %s =====\n' "$(date '+%F %T')" "$1"; }
model_field() { "$PYTHON_BIN" -m metacog.cli model get "$1" "$2"; }
model_path() { model_field "$1" path; }
model_dtype() { model_field "$1" dtype; }
model_trust() { model_field "$1" trust_remote_code; }
model_thinking() { model_field "$1" chat_template_enable_thinking; }

printf '[run] root=%s profile=%s target=%s gpus=%s,%s force=%s force_activations=%s\n' \
  "$RUN_ROOT" "$EXPERIMENT_PROFILE" "$TARGET_PROFILE" "$GPU0" "$GPU1" \
  "$FORCE" "$FORCE_ACTIVATIONS"
printf '[run] modules=%s selected=%s support=%s persistent_population=%s trajectory_max_samples=%s\n' \
  "$NUM_MODULES" "$MAX_SELECTED_MODULES" "$SUPPORTS" \
  "$PERSISTENT_POPULATION" "${CRITICAL_MAX_SAMPLES:-all}"
printf '[run] baseline initial=%s valid_target=%s ceiling=%s topup=%s history=%s\n' \
  "${BASELINE_INITIAL_SAMPLES:-$BASELINE_MAX_SAMPLES}" \
  "${BASELINE_MIN_UNTRUNCATED_SAMPLES:-0}" "$BASELINE_MAX_SAMPLES" \
  "${BASELINE_TOPUP_STEP:-0}" "${HISTORICAL_BASELINE_SEARCH_ROOTS:-runs}"

resolve_layers() {
  "$PYTHON_BIN" - "$1" ${LAYER_FRACS} <<'PY'
import math, sys
from transformers import AutoConfig
path, *fractions = sys.argv[1:]
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
n = int(getattr(cfg, "num_hidden_layers"))
print(" ".join(str(max(0, min(n - 1, int(math.floor(float(f) * n + 0.5))))) for f in fractions))
PY
}

memory_args=()
if [[ "${MEMORY_EFFICIENT_EXTRACTION:-0}" == 1 ]] && \
   "$PYTHON_BIN" scripts/extract_generation_step_activations.py --help 2>&1 | grep -q -- '--memory-efficient' && \
   "$PYTHON_BIN" scripts/extract_activations.py --help 2>&1 | grep -q -- '--memory-efficient'; then
  memory_args+=(--memory-efficient)
fi

stage "prepare and validate ${DATASET_PROFILE}"
if [[ ! -s "$BASE_DATA_PATH" ]]; then
  read -r -a data_sources <<< "$DATA_SOURCE"
  "$PYTHON_BIN" -m metacog.cli dataset prepare "$DATASET_PROFILE" \
    --input "${data_sources[@]}" --output "$DATA_PREPARED_PATH" \
    --max-samples "${DATA_MAX_SAMPLES:-0}" --split-mode "${DATA_SPLIT_MODE:-resplit}" \
    --train-size "${DATA_TRAIN_SIZE:-22000}" \
    --validation-size "${DATA_VALIDATION_SIZE:-7000}" --seed "$SEED"
fi
"$PYTHON_BIN" -m metacog.cli dataset validate "$DATASET_PROFILE" \
  "$BASE_DATA_PATH" --min-samples "$TARGET_PROMPT_MAX_SAMPLES"

target_model="$(model_path "$TARGET_PROFILE")"
read -r -a target_layers <<< "$(resolve_layers "$target_model")"
steps_slug="${GENERATION_RECORD_STEPS//,/-}"
target_dir="activations/${TARGET_PROFILE}_${DATASET_PROFILE}_strict_chain_n${TARGET_PROMPT_MAX_SAMPLES}_steps${steps_slug}_s${SEED}_${ACTIVATION_CACHE_TAG:-v2}"

stage "extract target generated-step activations: ${TARGET_PROFILE} layers=${target_layers[*]}"
if [[ "$FORCE_ACTIVATIONS" == 1 || ! -s "$target_dir/manifest.json" || \
      ! -s "$target_dir/external_semantic_data.jsonl" || \
      ! -s "$target_dir/expanded_generation_data.jsonl" ]]; then
  trust_args=()
  [[ "$(model_trust "$TARGET_PROFILE")" == 1 ]] && trust_args+=(--trust-remote-code)
  CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON_BIN" scripts/extract_generation_step_activations.py \
    --model-path "$target_model" --data "$BASE_DATA_PATH" \
    --layers "${target_layers[@]}" --output-dir "$target_dir" \
    --batch-size "$TARGET_EXTRACTION_BATCH_SIZE" \
    --max-samples "$TARGET_PROMPT_MAX_SAMPLES" --sample-strategy random --seed "$SEED" \
    --max-length "$EXTRACTION_MAX_LENGTH" --max-generation-steps "$MAX_GENERATION_STEPS" \
    --record-steps "$GENERATION_RECORD_STEPS" --temperature 0 \
    --dtype "$(model_dtype "$TARGET_PROFILE")" --device-map single \
    --prompt-style data --use-chat-template \
    --chat-template-enable-thinking "$(model_thinking "$TARGET_PROFILE")" \
    --shard-size "$EXTRACTION_SHARD_SIZE" "${memory_args[@]}" "${trust_args[@]}"
else
  echo "[reuse] $target_dir"
fi

extract_reference() {
  local spec="$1" gpu="$2" profile layer model semantic_dir
  IFS=: read -r profile layer <<< "$spec"
  model="$(model_path "$profile")"
  semantic_dir="activations/${profile}_on_${TARGET_PROFILE}_${DATASET_PROFILE}_strict_chain_n${TARGET_PROMPT_MAX_SAMPLES}_steps${steps_slug}_s${SEED}_l${layer}_${ACTIVATION_CACHE_TAG:-v2}"
  if [[ "$FORCE_ACTIVATIONS" != 1 && -s "$semantic_dir/manifest.json" && \
        -s "$semantic_dir/layer_$(printf '%03d' "$layer").pt" ]]; then
    echo "[reuse] $semantic_dir"
    return 0
  fi
  local -a trust_args=()
  [[ "$(model_trust "$profile")" == 1 ]] && trust_args+=(--trust-remote-code)
  echo "[extract-reference] profile=$profile layer=$layer gpu=$gpu output=$semantic_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/extract_activations.py \
    --model-path "$model" --data "$target_dir/external_semantic_data.jsonl" \
    --layers "$layer" --output-dir "$semantic_dir" \
    --batch-size "$SEMANTIC_EXTRACTION_BATCH_SIZE" \
    --max-length "$EXTERNAL_SEMANTIC_MAX_LENGTH" --pooling last_token \
    --dtype "$(model_dtype "$profile")" --device-map single --prompt-style data \
    --use-chat-template --chat-template-enable-thinking "$(model_thinking "$profile")" \
    --shard-size "$EXTRACTION_SHARD_SIZE" "${memory_args[@]}" "${trust_args[@]}"
}

stage "extract three semantic references"
read -r -a reference_specs <<< "$SEMANTIC_REFERENCE_SPECS"
reference_dirs=()
for spec in "${reference_specs[@]}"; do
  IFS=: read -r profile layer <<< "$spec"
  reference_dirs+=("activations/${profile}_on_${TARGET_PROFILE}_${DATASET_PROFILE}_strict_chain_n${TARGET_PROMPT_MAX_SAMPLES}_steps${steps_slug}_s${SEED}_l${layer}_${ACTIVATION_CACHE_TAG:-v2}:${layer}:${profile}")
done

if [[ "$GPU0" == "$GPU1" ]]; then
  for spec in "${reference_specs[@]}"; do
    extract_reference "$spec" "$GPU0"
  done
else
  # Two references at a time, one process per GPU.
  for ((i=0; i<${#reference_specs[@]}; i+=2)); do
    extract_reference "${reference_specs[$i]}" "$GPU0" & p0=$!
    p1=""
    if (( i + 1 < ${#reference_specs[@]} )); then
      extract_reference "${reference_specs[$((i + 1))]}" "$GPU1" & p1=$!
    fi
    wait "$p0"
    [[ -z "$p1" ]] || wait "$p1"
  done
fi

stage "build train-only PCA semantic ensemble"
ensemble_dir="$RUN_ROOT/semantic_ensemble_cache"
ensemble_args=()
for value in "${reference_dirs[@]}"; do ensemble_args+=(--source "$value"); done
ensemble_contract_ok=0
if [[ -s "$ensemble_dir/manifest.json" ]]; then
  if "$PYTHON_BIN" - "$ensemble_dir/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
valid = (
    manifest.get("pooling") == "last_token"
    and manifest.get("use_chat_template") is True
    and int(manifest.get("semantic_reference_assistant_prefix_rows", -1))
        == int(manifest.get("num_examples", -2))
)
raise SystemExit(0 if valid else 1)
PY
  then
    ensemble_contract_ok=1
  else
    echo "[semantic-ensemble] legacy manifest detected; rebuild PCA cache only"
  fi
fi
if [[ "$FORCE" == 1 || "$ensemble_contract_ok" != 1 || \
      ! -s "$ensemble_dir/layer_$(printf '%03d' "$SEMANTIC_ENSEMBLE_LAYER").pt" ]]; then
  "$PYTHON_BIN" scripts/build_semantic_ensemble_cache.py \
    --target-dir "$target_dir" --target-layer "${target_layers[1]}" \
    "${ensemble_args[@]}" --output-dir "$ensemble_dir" \
    --output-layer "$SEMANTIC_ENSEMBLE_LAYER" \
    --components-per-source "$SEMANTIC_ENSEMBLE_COMPONENTS_PER_REFERENCE" \
    --pca-fit-max-rows "$SEMANTIC_ENSEMBLE_PCA_FIT_ROWS" \
    --val-ratio "$VAL_RATIO" --test-ratio "$TEST_RATIO" --seed "$SEED" \
    --base-id-step-pattern "$BASE_ID_STEP_PATTERN"
else
  echo "[reuse] $ensemble_dir"
fi
"$PYTHON_BIN" scripts/validate_activation_alignment.py \
  --target-dir "$target_dir" --semantic-dir "$ensemble_dir" \
  --target-layer "${target_layers[1]}" --semantic-layer "$SEMANTIC_ENSEMBLE_LAYER"

stage "train modules and run the same frozen ids through all downstream stages"
pair_name="${TARGET_PROFILE}__semantic_ensemble_l${SEMANTIC_ENSEMBLE_LAYER}"
pair_root="$RUN_ROOT/$pair_name"
env GPU="$GPU0" PAIR_ROOT="$pair_root" TARGET_PROFILE="$TARGET_PROFILE" \
  DOWNSTREAM_GPUS="$GPU0 $GPU1" \
  TARGET_MODEL_PATH="$target_model" TARGET_ACTIVATION_DIR="$target_dir" \
  SEMANTIC_PROFILE=semantic_ensemble SEMANTIC_ACTIVATION_DIR="$ensemble_dir" \
  EXTERNAL_SEMANTIC_LAYER="$SEMANTIC_ENSEMBLE_LAYER" \
  DATA_PATH="$target_dir/expanded_generation_data.jsonl" \
  MODEL_DTYPE="$(model_dtype "$TARGET_PROFILE")" \
  TRUST_REMOTE_CODE="$(model_trust "$TARGET_PROFILE")" \
  SHARED_BASELINE_FILE="$RUN_ROOT/shared_baseline/baseline_generations.jsonl" \
  FORCE="$FORCE" bash scripts/run_model_pair.sh

stage "strict-chain summary"
"$PYTHON_BIN" scripts/summarize_strict_module_chain.py \
  --pair-root "$pair_root" --output-dir "$RUN_ROOT/strict_chain_summary" \
  --metric-profile "$METRIC_PROFILE" --phase final --alpha 0.05

stage "complete"
echo "Run root: $RUN_ROOT"
