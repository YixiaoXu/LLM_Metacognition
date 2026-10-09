#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/task_lock.sh"

: "${MODULE_DIR:?}"
: "${OUTPUT_DIR:?}"
: "${MODEL_PATH:?}"
: "${DATA_PATH:?}"
: "${ACTIVATION_DIR:?}"
: "${DECOUPLER_DIR:?}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"
FORCE_AXIS="${FORCE_AXIS:-$FORCE}"
FORCE_BASELINE="${FORCE_BASELINE:-$FORCE}"
FORCE_ASSOCIATION="${FORCE_ASSOCIATION:-$FORCE}"
FORCE_PROTOTYPE="${FORCE_PROTOTYPE:-$FORCE}"
FORCE_INTERVENTION="${FORCE_INTERVENTION:-$FORCE}"
RUN_PERSISTENT_TRAJECTORY="${RUN_PERSISTENT_TRAJECTORY:-1}"
RUN_SINGLE_STEP_TRAJECTORY="${RUN_SINGLE_STEP_TRAJECTORY:-0}"
PERSISTENT_POPULATION="${PERSISTENT_POPULATION:-critical}"
PERSISTENT_EVALUATION_ROLE="${PERSISTENT_EVALUATION_ROLE:-trajectory_confirmatory}"
CONTINUOUS_STAGE="${CONTINUOUS_STAGE:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$CONTINUOUS_STAGE" in
  all|axis|baseline_plan|baseline|association|intervention|next_token|trajectory) ;;
  *)
    echo "Unknown CONTINUOUS_STAGE=$CONTINUOUS_STAGE" >&2
    exit 2
    ;;
esac

if ! acquire_task_lock "$OUTPUT_DIR/.locks/module.lock" "module:$OUTPUT_DIR"; then
  echo "[module] duplicate invocation refused: $OUTPUT_DIR" >&2
  exit 17
fi
trap release_task_locks EXIT INT TERM
METRIC_PROFILE="${METRIC_PROFILE:-safety}"
ANSWER_EXTRACTION="${ANSWER_EXTRACTION:-none}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
EVALUATION_ID_REGEX="${EVALUATION_ID_REGEX:-}"
DIRECT_TRUST_REGION_STEPS="${DIRECT_TRUST_REGION_STEPS:-16}"
DIRECT_TRUST_REGION_STEP_SCALE="${DIRECT_TRUST_REGION_STEP_SCALE:-1.5}"
DIRECT_TRUST_REGION_HIDDEN_PENALTY="${DIRECT_TRUST_REGION_HIDDEN_PENALTY:-0.05}"
RANDOM_DIRECTION_DOSE_MODE="${RANDOM_DIRECTION_DOSE_MODE:-strict_code}"
RANDOM_DIRECTION_AXIS_PENALTY="${RANDOM_DIRECTION_AXIS_PENALTY:-10.0}"
RANDOM_DIRECTION_MAX_REAL_AXIS_COSINE="${RANDOM_DIRECTION_MAX_REAL_AXIS_COSINE:-0.25}"
RANDOM_DIRECTION_CODE_TOLERANCE="${RANDOM_DIRECTION_CODE_TOLERANCE:-0.25}"
RANDOM_DIRECTION_HIDDEN_TOLERANCE="${RANDOM_DIRECTION_HIDDEN_TOLERANCE:-0.5}"
RANDOM_DIRECTION_MIN_COSINE="${RANDOM_DIRECTION_MIN_COSINE:-0.5}"
RANDOM_DIRECTION_SEMANTIC_TOLERANCE="${RANDOM_DIRECTION_SEMANTIC_TOLERANCE:-0.5}"
RANDOM_HIDDEN_CANDIDATES="${RANDOM_HIDDEN_CANDIDATES:-8}"
PROXY_CONFIDENCE_EVALUATION="${PROXY_CONFIDENCE_EVALUATION:-0}"
PROXY_CONFIDENCE_PRIMARY="${PROXY_CONFIDENCE_PRIMARY:-generated_prefix16_negative_entropy}"
PROXY_CONFIDENCE_BOOTSTRAP_SAMPLES="${PROXY_CONFIDENCE_BOOTSTRAP_SAMPLES:-2000}"
PROXY_CONFIDENCE_PERMUTATION_TESTS="${PROXY_CONFIDENCE_PERMUTATION_TESTS:-2000}"
META_BEHAVIOR_EVALUATION="${META_BEHAVIOR_EVALUATION:-0}"
META_BEHAVIOR_BOOTSTRAP_SAMPLES="${META_BEHAVIOR_BOOTSTRAP_SAMPLES:-2000}"
META_BEHAVIOR_PERMUTATION_TESTS="${META_BEHAVIOR_PERMUTATION_TESTS:-2000}"
META_BEHAVIOR_INCLUDE_SECONDARY="${META_BEHAVIOR_INCLUDE_SECONDARY:-1}"
PRIMARY_BEHAVIOR_CONSTRUCT="${PRIMARY_BEHAVIOR_CONSTRUCT:-}"
PRIMARY_BEHAVIOR_DIRECTION="${PRIMARY_BEHAVIOR_DIRECTION:-}"
mkdir -p "$OUTPUT_DIR/.mplconfig"
export MPLCONFIGDIR="$OUTPUT_DIR/.mplconfig"

stage() {
  printf '\n[%s] ----- %s -----\n' "$(date '+%F %T')" "$1"
}

read -r -a DOSES <<< "$CONTINUOUS_DOSES"
read -r -a CRITICAL_FRACTION_VALUES <<< "$CRITICAL_FRACTIONS"
read -r -a CRITICAL_WEIGHT_VALUES <<< "$CRITICAL_WEIGHTS"
read -r -a CONTINUOUS_CONTROL_VALUES <<< "${CONTINUOUS_CONTROLS:-opposite_direction}"
ASSOCIATION_EVALUATION_ROLE="${ASSOCIATION_EVALUATION_ROLE:-association_confirmatory}"
PROMPT_ARGS=(--prompt-style data)
if [[ "${DOWNSTREAM_USE_CHAT_TEMPLATE:-1}" == "1" ]]; then
  PROMPT_ARGS+=(--use-chat-template --chat-template-enable-thinking "${CHAT_TEMPLATE_ENABLE_THINKING:-auto}")
fi
ID_ARGS=()
[[ -n "$EVALUATION_ID_REGEX" ]] && ID_ARGS+=(--id-regex "$EVALUATION_ID_REGEX")
TRUST_ARGS=()
[[ "${TRUST_REMOTE_CODE:-1}" == "1" ]] && TRUST_ARGS+=(--trust-remote-code)

AXIS_DIR="${CONTINUOUS_AXIS_DIR:-$OUTPUT_DIR/continuous_axis}"
AXIS_FILE="$AXIS_DIR/continuous_axis.pt"
# A resumed downstream run may have linked this directory to an incomplete
# upstream artifact.  pathlib.mkdir(..., exist_ok=True) does not treat an
# existing symlink as a reusable directory, so remove only the symlink before
# rebuilding the axis.  Never remove a real output directory here.
if [[ -L "$AXIS_DIR" && ( "$FORCE_AXIS" == 1 || ! -s "$AXIS_FILE" ) ]]; then
  echo "[axis] incomplete symlink; replacing with a real output directory: $AXIS_DIR"
  rm "$AXIS_DIR"
fi
stage "prepare continuous module axis"
if [[ "$FORCE_AXIS" == 1 || ! -s "$AXIS_FILE" ]]; then
  AXIS_ROLE_ARGS=()
  if [[ -n "${AXIS_ASSOCIATION_FRACTION:-}" || -n "${AXIS_TRAJECTORY_FRACTION:-}" ]]; then
    : "${AXIS_ASSOCIATION_FRACTION:?Set both explicit confirmatory role fractions}"
    : "${AXIS_TRAJECTORY_FRACTION:?Set both explicit confirmatory role fractions}"
    AXIS_ROLE_ARGS+=(
      --association-fraction "$AXIS_ASSOCIATION_FRACTION"
      --trajectory-fraction "$AXIS_TRAJECTORY_FRACTION"
    )
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/prepare_continuous_module_axis.py \
    --module-dir "$MODULE_DIR" --output-dir "$AXIS_DIR" \
    --tail-fraction "$AXIS_TAIL_FRACTION" --prototype-fraction "$AXIS_PROTOTYPE_FRACTION" \
    --causal-fraction "$AXIS_CAUSAL_FRACTION" --semantic-teacher strongest \
    "${AXIS_ROLE_ARGS[@]}" \
    --semantic-folds "$AXIS_SEMANTIC_FOLDS" \
    --semantic-mlp-hidden-dim "$AXIS_SEMANTIC_HIDDEN_DIM" \
    --semantic-mlp-epochs "$AXIS_SEMANTIC_MLP_EPOCHS" \
    --semantic-mlp-repeats "$AXIS_SEMANTIC_MLP_REPEATS" \
    --base-id-step-pattern "${BASE_ID_STEP_PATTERN:-}" \
    --batch-size "$ENCODE_BATCH_SIZE" --device cuda --seed "$MEASUREMENT_SEED"
else
  echo "[reuse] $AXIS_FILE"
fi

TARGET_FILE="$MODULE_DIR/soft_residual_intervention_targets.pt"
ASSIGNMENTS="$AXIS_DIR/continuous_axis_all_assignments.csv"
FEATURES="$AXIS_DIR/continuous_axis_all_features.pt"
for path in "$TARGET_FILE" "$ASSIGNMENTS" "$FEATURES" "$MODULE_DIR/module_refiner.pt"; do
  [[ -s "$path" ]] || { echo "Missing continuous artifact: $path" >&2; exit 3; }
done

if [[ "$CONTINUOUS_STAGE" == axis ]]; then
  stage "continuous axis complete"
  exit 0
fi

if [[ "$CONTINUOUS_STAGE" == baseline_plan ]]; then
  BASELINE_PLAN_DIR="${BASELINE_PLAN_DIR:-$OUTPUT_DIR/baseline_plan}"
  BASELINE_PLAN_ASSIGNMENTS="$BASELINE_PLAN_DIR/cluster_assignments.csv"
  stage "freeze baseline evaluation ids"
  if [[ "$FORCE_BASELINE" == 1 || ! -s "$BASELINE_PLAN_ASSIGNMENTS" ]]; then
    mkdir -p "$BASELINE_PLAN_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/audit_continuous_module_dose.py \
      --continuous-axis-file "$AXIS_FILE" \
      --continuous-evaluation-role "$ASSOCIATION_EVALUATION_ROLE" \
      --output-dir "$BASELINE_PLAN_DIR" \
      --data "$DATA_PATH" --target-file "$TARGET_FILE" \
      --activation-dir "$ACTIVATION_DIR" \
      --cluster-source soft_residual --fixed-cluster-ids-only --num-clusters 2 \
      --eval-scope all_cached --cluster-sample-strategy first \
      --max-length "${DOWNSTREAM_MAX_LENGTH:-1024}" \
      "${PROMPT_ARGS[@]}" "${ID_ARGS[@]}" \
      --answer-extraction "$ANSWER_EXTRACTION" --metric-profile "$METRIC_PROFILE" \
      --temperature 0.0 --record-layers none --device-map single --dtype "$MODEL_DTYPE" \
      "${TRUST_ARGS[@]}" --seed "$SEED" \
      --fixed-cluster-assignments "$ASSIGNMENTS" \
      --fixed-cluster-features "$FEATURES" \
      --baseline-only --max-samples "$BASELINE_MAX_SAMPLES" \
      --max-samples-per-cluster 0 --no-plots
  else
    echo "[reuse] $BASELINE_PLAN_ASSIGNMENTS"
  fi
  [[ -s "$BASELINE_PLAN_ASSIGNMENTS" ]] || {
    echo "Baseline planning did not produce assignments: $BASELINE_PLAN_ASSIGNMENTS" >&2
    exit 3
  }
  stage "baseline evaluation ids frozen"
  exit 0
fi

COMMON=(
  --model-path "$MODEL_PATH" --data "$DATA_PATH" --target-file "$TARGET_FILE"
  --activation-dir "$ACTIVATION_DIR" --decoupler-dir "$DECOUPLER_DIR"
  --cluster-source soft_residual --fixed-cluster-ids-only --num-clusters 2
  --eval-scope all_cached --cluster-sample-strategy first
  --max-length "${DOWNSTREAM_MAX_LENGTH:-1024}"
  "${PROMPT_ARGS[@]}" "${ID_ARGS[@]}"
  --answer-extraction "$ANSWER_EXTRACTION" --metric-profile "$METRIC_PROFILE"
  --temperature 0.0 --record-layers none --device-map single --dtype "$MODEL_DTYPE"
  --generation-progress-seconds "${GENERATION_PROGRESS_SECONDS:-30}"
  "${TRUST_ARGS[@]}"
  --seed "$SEED"
  --fixed-cluster-assignments "$ASSIGNMENTS" --fixed-cluster-features "$FEATURES"
)
INTERVENTION=(
  --intervention-space refined_residual --refined-module-dir "$MODULE_DIR"
  --refined-direct-steps "$DIRECT_TRUST_REGION_STEPS"
  --refined-direct-step-scale "$DIRECT_TRUST_REGION_STEP_SCALE"
  --refined-direct-hidden-penalty "$DIRECT_TRUST_REGION_HIDDEN_PENALTY"
  --refined-code-random-dose-mode "$RANDOM_DIRECTION_DOSE_MODE"
  --refined-code-random-axis-penalty "$RANDOM_DIRECTION_AXIS_PENALTY"
  --refined-code-random-max-real-axis-cosine "$RANDOM_DIRECTION_MAX_REAL_AXIS_COSINE"
  --refined-code-random-hidden-candidates "$RANDOM_HIDDEN_CANDIDATES"
  --refined-code-dose-match-tolerance "$RANDOM_DIRECTION_CODE_TOLERANCE"
  --refined-code-dose-hidden-tolerance "$RANDOM_DIRECTION_HIDDEN_TOLERANCE"
  --refined-code-dose-min-direction-cosine "$RANDOM_DIRECTION_MIN_COSINE"
  --refined-code-dose-semantic-tolerance "$RANDOM_DIRECTION_SEMANTIC_TOLERANCE"
  --refined-code-semantic-penalty "$REFINED_CODE_SEMANTIC_PENALTY"
  --refined-code-z1-penalty "$REFINED_CODE_Z1_PENALTY"
  --refined-code-max-semantic-rel-delta "$MAX_SEMANTIC_REL_DELTA"
  --max-delta-rel-norm "$MAX_HIDDEN_REL_NORM" --clear-cuda-cache-between-modes
)
if [[ "${REFINED_CODE_DOSE_MATCH_CONTROLS:-0}" == "1" ]]; then
  INTERVENTION+=(--refined-code-dose-match-controls)
fi

BASELINE_DIR="$OUTPUT_DIR/baseline_${ASSOCIATION_EVALUATION_ROLE}"
BASELINE_FILE="$BASELINE_DIR/baseline_generations.jsonl"
BASELINE_CHECKPOINT_FILE="$BASELINE_DIR/baseline_generation_checkpoint.jsonl"
BASELINE_EFFECTIVE_MAX_SAMPLES="${BASELINE_MAX_SAMPLES_OVERRIDE:-$BASELINE_MAX_SAMPLES}"
BASELINE_ID_ARGS=()
if [[ -n "${BASELINE_EVALUATION_ID_FILE:-}" ]]; then
  [[ -s "$BASELINE_EVALUATION_ID_FILE" ]] || {
    echo "Missing baseline evaluation-id file: $BASELINE_EVALUATION_ID_FILE" >&2
    exit 3
  }
  BASELINE_ID_ARGS+=(--evaluation-id-file "$BASELINE_EVALUATION_ID_FILE")
fi
baseline_has_proxy_confidence() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    row = json.loads(next(line for line in handle if line.strip()))
required = {
    "prompt_end_entropy",
    "prompt_end_top1_top2_logit_margin",
    "generated_mean_logprob",
    "generated_prefix16_negative_entropy_mean",
}
raise SystemExit(0 if required.issubset(row) else 1)
PY
}
generation_has_meta_confidence() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    row = json.loads(next(line for line in handle if line.strip()))
required = {
    "prompt_end_entropy",
    "prompt_end_normalized_entropy",
    "prompt_end_top1_top2_logit_margin",
    "generated_prefix16_entropy_mean",
    "generated_mean_logprob",
}
raise SystemExit(0 if required.issubset(row) else 1)
PY
}
analysis_summary_excludes_truncated() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
raise SystemExit(0 if summary.get("exclude_truncated") is True else 1)
PY
}
association_has_primary_construct_schema() {
  local path="$1" profile="$2"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" "$profile" <<'PY'
import csv
import sys
from metacog.evaluation import metric_families

path, profile = sys.argv[1:]
with open(path, encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
signed = [row for row in rows if row.get("meta_predictor") == "signed_direction"]
expected = set(metric_families(profile))
actual = {row.get("metric_family") for row in signed}
required = {
    "aligned_standardized_slope",
    "aligned_slope_signflip_two_sided_p",
}
fields = set(rows[0]) if rows else set()
raise SystemExit(0 if actual == expected and required.issubset(fields) else 1)
PY
}

safety_behavior_has_primary_fields() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    fields = set((csv.DictReader(handle).fieldnames or []))
required = {
    "compliance_score",
    "prompt_end_entropy",
    "prompt_end_max_probability",
    "generated_prefix16_entropy_mean",
    "generated_mean_logprob",
    "generated_min_logprob",
}
raise SystemExit(0 if required.issubset(fields) else 1)
PY
}
confidence_fields_required() {
  # Confidence and uncertainty form four of the ten primary constructs for
  # every task profile, so these fields are no longer optional diagnostics.
  return 0
}

if [[ "$CONTINUOUS_STAGE" != intervention && \
      "$CONTINUOUS_STAGE" != next_token && \
      "$CONTINUOUS_STAGE" != trajectory ]]; then
BASELINE_REUSE_ARGS=()
if [[ -n "${SHARED_BASELINE_FILE:-}" && -s "$SHARED_BASELINE_FILE" ]] && \
   { ! confidence_fields_required || generation_has_meta_confidence "$SHARED_BASELINE_FILE"; }; then
  BASELINE_REUSE_ARGS+=(--baseline-file "$SHARED_BASELINE_FILE")
fi
BASELINE_CONFIDENCE_ARGS=()
if confidence_fields_required; then
  BASELINE_CONFIDENCE_ARGS+=(--record-prompt-confidence --save-generated-token-logprobs)
fi
BASELINE_RUNTIME_ARGS=(
  --generation-checkpoint-file "$BASELINE_CHECKPOINT_FILE"
  --clear-cuda-cache-every "${BASELINE_CLEAR_CUDA_CACHE_EVERY:-4}"
)
if [[ "${BASELINE_LOG_CUDA_MEMORY:-0}" == "1" ]]; then
  BASELINE_RUNTIME_ARGS+=(--log-cuda-memory)
fi
stage "generate ${ASSOCIATION_EVALUATION_ROLE} baseline"
if [[ -L "$BASELINE_DIR" && ! -s "$BASELINE_FILE" ]]; then
  echo "[baseline] incomplete symlink; replacing with a real output directory: $BASELINE_DIR"
  rm "$BASELINE_DIR"
fi
baseline_needs_refresh=0
baseline_reuse_rejected=0
if [[ "$FORCE_BASELINE" == 1 || ! -s "$BASELINE_FILE" ]]; then
  baseline_needs_refresh=1
elif confidence_fields_required && ! generation_has_meta_confidence "$BASELINE_FILE"; then
  echo "[baseline] existing file lacks proxy-confidence fields; regenerate"
  baseline_needs_refresh=1
fi
if [[ "$baseline_needs_refresh" == 1 ]]; then
  mkdir -p "$BASELINE_DIR"
  if [[ "$FORCE_BASELINE" == 1 ]]; then
    rm -f "$BASELINE_CHECKPOINT_FILE"
  fi
  if ! CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/audit_continuous_module_dose.py \
      --continuous-axis-file "$AXIS_FILE" --continuous-evaluation-role "$ASSOCIATION_EVALUATION_ROLE" \
      --output-dir "$BASELINE_DIR" "${COMMON[@]}" "${BASELINE_ID_ARGS[@]}" \
      "${BASELINE_REUSE_ARGS[@]}" \
      "${BASELINE_CONFIDENCE_ARGS[@]}" \
      "${BASELINE_RUNTIME_ARGS[@]}" \
      --baseline-only --max-samples "$BASELINE_EFFECTIVE_MAX_SAMPLES" --max-samples-per-cluster 0 \
      --batch-size "$GENERATION_BATCH_SIZE" --max-new-tokens "$BASELINE_MAX_NEW_TOKENS" \
      --style-untruncated-only; then
    if (( ${#BASELINE_REUSE_ARGS[@]} == 0 )); then
      echo "[baseline] fresh deterministic generation failed" >&2
      exit 1
    fi
    if [[ "${BASELINE_VALIDATE_ONLY:-0}" == 1 ]]; then
      echo "[baseline] shared baseline validation failed; regeneration is disabled in validation-only mode" >&2
      exit 1
    fi
    failed_baseline_dir="${BASELINE_DIR}.incompatible_reuse_$(date +%Y%m%d_%H%M%S)"
    echo "[baseline] cached rows do not satisfy the current held-out IDs; preserve at $failed_baseline_dir and regenerate"
    mv "$BASELINE_DIR" "$failed_baseline_dir"
    mkdir -p "$BASELINE_DIR"
    BASELINE_REUSE_ARGS=()
    baseline_reuse_rejected=1
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/audit_continuous_module_dose.py \
      --continuous-axis-file "$AXIS_FILE" --continuous-evaluation-role "$ASSOCIATION_EVALUATION_ROLE" \
      --output-dir "$BASELINE_DIR" "${COMMON[@]}" "${BASELINE_ID_ARGS[@]}" \
      "${BASELINE_CONFIDENCE_ARGS[@]}" \
      "${BASELINE_RUNTIME_ARGS[@]}" \
      --baseline-only --max-samples "$BASELINE_EFFECTIVE_MAX_SAMPLES" --max-samples-per-cluster 0 \
      --batch-size "$GENERATION_BATCH_SIZE" --max-new-tokens "$BASELINE_MAX_NEW_TOKENS" \
      --style-untruncated-only
  fi
else
  echo "[reuse] $BASELINE_FILE"
fi
publish_shared_baseline=0
if [[ "$baseline_reuse_rejected" == 1 ]]; then
  publish_shared_baseline=1
elif [[ -n "${SHARED_BASELINE_FILE:-}" && ! -s "$SHARED_BASELINE_FILE" ]]; then
  publish_shared_baseline=1
elif [[ -n "${SHARED_BASELINE_FILE:-}" ]] && confidence_fields_required && \
     ! generation_has_meta_confidence "$SHARED_BASELINE_FILE"; then
  publish_shared_baseline=1
fi
if [[ "$publish_shared_baseline" == 1 && "${PUBLISH_SHARED_BASELINE:-1}" == 1 ]]; then
  mkdir -p "$(dirname "$SHARED_BASELINE_FILE")"
  temporary_baseline="${SHARED_BASELINE_FILE}.tmp.$$"
  cp "$BASELINE_FILE" "$temporary_baseline"
  mv "$temporary_baseline" "$SHARED_BASELINE_FILE"
  echo "[baseline] published shared deterministic baseline: $SHARED_BASELINE_FILE"
fi

if [[ "$CONTINUOUS_STAGE" == baseline ]]; then
  stage "shared baseline complete"
  exit 0
fi

BEHAVIOR_DIR="$OUTPUT_DIR/baseline_behavior"
BEHAVIOR_EXPECTED="$BEHAVIOR_DIR/safety_behavior_by_sample.csv"
if [[ -L "$BEHAVIOR_DIR" && "$METRIC_PROFILE" == "safety" && ! -s "$BEHAVIOR_EXPECTED" ]]; then
  rm "$BEHAVIOR_DIR"
fi
if [[ "$METRIC_PROFILE" == "safety" ]]; then
  stage "extract safety behavior outcomes"
  BEHAVIOR_FILE="$BEHAVIOR_DIR/safety_behavior_by_sample.csv"
  BEHAVIOR_SUMMARY="$BEHAVIOR_DIR/safety_behavior_summary.json"
  if [[ "$FORCE_BASELINE" == 1 || ! -s "$BEHAVIOR_FILE" ]] || \
      ! analysis_summary_excludes_truncated "$BEHAVIOR_SUMMARY" || \
      ! safety_behavior_has_primary_fields "$BEHAVIOR_FILE"; then
    "$PYTHON_BIN" scripts/analyze_safety_baseline.py --baseline-file "$BASELINE_FILE" \
      --data "$DATA_PATH" --output-dir "$BEHAVIOR_DIR"
  else
    echo "[reuse] $BEHAVIOR_FILE"
  fi
else
  stage "use generated style metrics as behavior outcomes (${METRIC_PROFILE})"
  BEHAVIOR_FILE="$BASELINE_FILE"
fi

ASSOCIATION_DIR="$OUTPUT_DIR/continuous_behavior"
stage "test continuous baseline associations (${ASSOCIATION_EVALUATION_ROLE})"
if [[ -L "$ASSOCIATION_DIR" && ! -s "$ASSOCIATION_DIR/continuous_behavior_summary.json" ]]; then
  rm "$ASSOCIATION_DIR"
fi
if [[ "$FORCE_ASSOCIATION" == 1 || \
      ! -s "$ASSOCIATION_DIR/continuous_behavior_summary.json" ]] || \
    ! analysis_summary_excludes_truncated "$ASSOCIATION_DIR/continuous_behavior_summary.json" || \
    ! association_has_primary_construct_schema \
      "$ASSOCIATION_DIR/continuous_behavior_family_omnibus.csv" "$METRIC_PROFILE"; then
  "$PYTHON_BIN" scripts/analyze_continuous_meta_behavior.py \
    --continuous-axis-file "$AXIS_FILE" \
    --behavior-file "$BEHAVIOR_FILE" --metric-profile "$METRIC_PROFILE" \
    --output-dir "$ASSOCIATION_DIR" --evaluation-role "$ASSOCIATION_EVALUATION_ROLE" \
    --semantic-ridge 0.10 --crossfit-folds 5 \
    --bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" \
    --permutation-tests "$INTERVENTION_PERMUTATION_TESTS" \
    --semantic-r2-equivalence-margin 0.20 --max-new-tokens "$BASELINE_MAX_NEW_TOKENS" \
    --seed "$MEASUREMENT_SEED"
else
  echo "[reuse] $ASSOCIATION_DIR/continuous_behavior_summary.json"
fi

if [[ "$METRIC_PROFILE" == math && "$PROXY_CONFIDENCE_EVALUATION" == 1 ]]; then
  PROXY_CONFIDENCE_DIR="$OUTPUT_DIR/proxy_confidence"
  stage "evaluate calibrated confidence proxies and incremental meta information"
  if [[ "$FORCE_ASSOCIATION" == 1 || \
        ! -s "$PROXY_CONFIDENCE_DIR/proxy_confidence_summary.json" ]] || \
      ! analysis_summary_excludes_truncated "$PROXY_CONFIDENCE_DIR/proxy_confidence_summary.json"; then
    "$PYTHON_BIN" scripts/analyze_proxy_confidence.py \
      --continuous-axis-file "$AXIS_FILE" --generation-file "$BASELINE_FILE" \
      --output-dir "$PROXY_CONFIDENCE_DIR" \
      --evaluation-role "$ASSOCIATION_EVALUATION_ROLE" \
      --primary-proxy "$PROXY_CONFIDENCE_PRIMARY" \
      --crossfit-folds 5 --ridge 0.10 \
      --bootstrap-samples "$PROXY_CONFIDENCE_BOOTSTRAP_SAMPLES" \
      --permutation-tests "$PROXY_CONFIDENCE_PERMUTATION_TESTS" \
      --exclude-truncated --seed "$MEASUREMENT_SEED"
  else
    echo "[reuse] $PROXY_CONFIDENCE_DIR/proxy_confidence_summary.json"
  fi
fi

META_BEHAVIOR_DIR="$OUTPUT_DIR/metacognitive_behavior"
if [[ "$META_BEHAVIOR_EVALUATION" == 1 ]]; then
  stage "evaluate held-out confidence and uncertainty behavior"
  if [[ "$FORCE_ASSOCIATION" == 1 || ! -s "$META_BEHAVIOR_DIR/metacognitive_behavior_summary.json" || \
        ! -s "$META_BEHAVIOR_DIR/metacognitive_baseline_associations.csv" ]] || \
      ! analysis_summary_excludes_truncated "$META_BEHAVIOR_DIR/metacognitive_behavior_summary.json"; then
    META_ARGS=(
      --continuous-axis-file "$AXIS_FILE" --baseline-file "$BASELINE_FILE"
      --output-dir "$META_BEHAVIOR_DIR" --evaluation-role "$ASSOCIATION_EVALUATION_ROLE"
      --crossfit-folds 5 --ridge 0.10
      --bootstrap-samples "$META_BEHAVIOR_BOOTSTRAP_SAMPLES"
      --permutation-tests "$META_BEHAVIOR_PERMUTATION_TESTS"
      --max-new-tokens "$BASELINE_MAX_NEW_TOKENS"
      --seed "$MEASUREMENT_SEED"
    )
    [[ "${META_BEHAVIOR_EXCLUDE_TRUNCATED:-1}" == 1 ]] && META_ARGS+=(--exclude-truncated)
    if [[ "$META_BEHAVIOR_INCLUDE_SECONDARY" == 1 ]]; then
      META_ARGS+=(--include-secondary)
    else
      META_ARGS+=(--no-include-secondary)
    fi
    "$PYTHON_BIN" scripts/analyze_metacognitive_behavior.py "${META_ARGS[@]}"
  else
    echo "[reuse] $META_BEHAVIOR_DIR/metacognitive_behavior_summary.json"
  fi
fi
fi

if [[ "$CONTINUOUS_STAGE" == association ]]; then
  stage "continuous association complete"
  exit 0
fi

PROTOTYPE_DIR="$OUTPUT_DIR/prototype_discovery"
PROTOTYPE_FILE="$PROTOTYPE_DIR/next_token_logit_prototypes.pt"
run_screen() {
  local role="$1" output="$2"
  stage "continuous next-token intervention: ${role}"
  if [[ "$FORCE_INTERVENTION" != 1 && -s "$output/continuous_dose_response.csv" ]]; then
    echo "[reuse] $output/continuous_dose_response.csv"
    return 0
  fi
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/audit_continuous_module_dose.py \
    --continuous-axis-file "$AXIS_FILE" --continuous-evaluation-role "$role" \
    --continuous-directions "$CONTINUOUS_DIRECTION" --output-dir "$output" \
    --continuous-controls "${CONTINUOUS_CONTROL_VALUES[@]}" \
    "${COMMON[@]}" "${INTERVENTION[@]}" --alphas "${DOSES[@]}" \
    --max-samples 0 --max-samples-per-cluster 0 --batch-size "$GENERATION_BATCH_SIZE" \
    --max-new-tokens 1 --next-token-audit --next-token-top-k "$NEXT_TOKEN_TOP_K" \
    --next-token-prototype-file "$PROTOTYPE_FILE" \
    --next-token-bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" \
    --next-token-permutation-tests "$INTERVENTION_PERMUTATION_TESTS"
}

SCREEN_A="$OUTPUT_DIR/causal_screen_a"
SCREEN_B="$OUTPUT_DIR/causal_screen_b"

if [[ "$CONTINUOUS_STAGE" == trajectory ]]; then
  # A trajectory worker must never back-fill a missing screen.  This keeps the
  # global execution contract strict: every selected module completes both
  # independent next-token screens before any expensive full generation starts.
  for path in \
      "$PROTOTYPE_FILE" \
      "$SCREEN_A/continuous_dose_response.csv" \
      "$SCREEN_B/continuous_dose_response.csv"; do
    [[ -s "$path" ]] || {
      echo "[trajectory] prerequisite next-token artifact is missing: $path" >&2
      exit 3
    }
  done
  stage "validated completed next-token screens"
  [[ -n "$PRIMARY_BEHAVIOR_CONSTRUCT" ]] || {
    echo "[trajectory] missing frozen PRIMARY_BEHAVIOR_CONSTRUCT" >&2
    exit 3
  }
  [[ "$PRIMARY_BEHAVIOR_DIRECTION" == positive || "$PRIMARY_BEHAVIOR_DIRECTION" == negative ]] || {
    echo "[trajectory] invalid PRIMARY_BEHAVIOR_DIRECTION=$PRIMARY_BEHAVIOR_DIRECTION" >&2
    exit 3
  }
else
  stage "discover continuous next-token prototypes"
  if [[ -L "$PROTOTYPE_DIR" && ! -s "$PROTOTYPE_FILE" ]]; then
    echo "[prototype] incomplete symlink; replacing with a real output directory: $PROTOTYPE_DIR"
    rm "$PROTOTYPE_DIR"
  fi
  if [[ "$FORCE_PROTOTYPE" == 1 || ! -s "$PROTOTYPE_FILE" ]]; then
    mkdir -p "$PROTOTYPE_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/audit_continuous_module_dose.py \
      --continuous-axis-file "$AXIS_FILE" --continuous-evaluation-role prototype_discovery \
      --continuous-logit-prototype --output-dir "$PROTOTYPE_DIR" \
      "${COMMON[@]}" --baseline-only --max-samples 0 --max-samples-per-cluster 0 \
      --batch-size "$GENERATION_BATCH_SIZE" --max-new-tokens 1 --next-token-audit \
      --next-token-top-k "$NEXT_TOKEN_TOP_K" \
      --next-token-auto-logit-top-k "$NEXT_TOKEN_AUTO_LOGIT_TOP_K" \
      --next-token-auto-min-mean-prob "$NEXT_TOKEN_AUTO_MIN_MEAN_PROB" \
      --next-token-auto-min-selected "$NEXT_TOKEN_AUTO_MIN_SELECTED"
  else
    echo "[reuse] $PROTOTYPE_FILE"
  fi

  run_screen causal_screen_a "$SCREEN_A"
  run_screen causal_screen_b "$SCREEN_B"
fi

if [[ "$CONTINUOUS_STAGE" == next_token ]]; then
  stage "next-token screens complete"
  exit 0
fi

if [[ "$RUN_PERSISTENT_TRAJECTORY" != 1 ]]; then
  stage "screen-only direct intervention complete"
  echo "[direct-screen] skipped critical selection and persistent generation"
  exit 0
fi

CRITICAL_DIR=""
TRAJECTORY_ROLE="$PERSISTENT_EVALUATION_ROLE"
TRAJECTORY_ID_ARGS=()
if [[ "$PERSISTENT_POPULATION" == critical ]]; then
  CRITICAL_DIR="$OUTPUT_DIR/critical_heldout"
  mkdir -p "$CRITICAL_DIR"
  stage "select critical held-out samples"
  "$PYTHON_BIN" scripts/analyze_continuous_critical_heldout.py \
    --screen-a-dir "$SCREEN_A" --screen-b-dir "$SCREEN_B" \
    --continuous-axis-file "$AXIS_FILE" --output-dir "$CRITICAL_DIR" \
    --direction "$CONTINUOUS_DIRECTION" --dose 1.0 \
    --candidate-fractions "${CRITICAL_FRACTION_VALUES[@]}" \
    --weights "${CRITICAL_WEIGHT_VALUES[@]}" \
    --trajectory-max-samples "$CRITICAL_MAX_SAMPLES" \
    --bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" --seed "$MEASUREMENT_SEED"
  TRAJECTORY_IDS="$CRITICAL_DIR/trajectory_ids.txt"
  [[ -s "$TRAJECTORY_IDS" ]] || { echo "No critical trajectory ids; module complete without full generation"; exit 0; }
  TRAJECTORY_ROLE=causal_screen_b
  TRAJECTORY_ID_ARGS=(--evaluation-id-file "$TRAJECTORY_IDS")
elif [[ "$PERSISTENT_POPULATION" != all ]]; then
  echo "Unknown PERSISTENT_POPULATION=$PERSISTENT_POPULATION (expected critical or all)" >&2
  exit 2
fi

TRAJECTORY_DIR="$OUTPUT_DIR/persistent_trajectory"
trajectory_has_all_doses() {
  local directory="$1" dose encoded
  [[ -s "$directory/baseline_generations.jsonl" ]] || return 1
  for dose in "${DOSES[@]}"; do
    encoded="${dose//./p}"
    [[ -s "$directory/continuous_dose_0_to_1_main_a${encoded}_generations.jsonl" ]] || return 1
    [[ -s "$directory/continuous_dose_0_to_1_opposite_a${encoded}_generations.jsonl" ]] || return 1
  done
  return 0
}
TRAJECTORY_CONFIDENCE_ARGS=()
if confidence_fields_required; then
  TRAJECTORY_CONFIDENCE_ARGS+=(--record-prompt-confidence)
fi
stage "generate persistent positive/negative trajectories (${PERSISTENT_POPULATION}:${TRAJECTORY_ROLE})"
trajectory_needs_refresh=0
if [[ "$FORCE_INTERVENTION" == 1 || \
      ! -s "$TRAJECTORY_DIR/cluster_intervention_summary.json" ]] || \
   ! trajectory_has_all_doses "$TRAJECTORY_DIR"; then
  trajectory_needs_refresh=1
elif confidence_fields_required; then
  trajectory_main_file="$(find "$TRAJECTORY_DIR" -maxdepth 1 -type f -name '*main_a1p0_generations.jsonl' -print -quit 2>/dev/null || true)"
  if [[ -z "$trajectory_main_file" ]] || ! generation_has_meta_confidence "$trajectory_main_file"; then
    echo "[trajectory] existing generations lack confidence fields; regenerate"
    trajectory_needs_refresh=1
  fi
fi
if [[ "$trajectory_needs_refresh" == 1 ]]; then
  mkdir -p "$TRAJECTORY_DIR"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/audit_continuous_module_dose.py \
    --continuous-axis-file "$AXIS_FILE" --continuous-evaluation-role "$TRAJECTORY_ROLE" \
    --continuous-directions "$CONTINUOUS_DIRECTION" --output-dir "$TRAJECTORY_DIR" \
    --continuous-controls "${CONTINUOUS_CONTROL_VALUES[@]}" \
    "${COMMON[@]}" "${INTERVENTION[@]}" "${TRAJECTORY_ID_ARGS[@]}" \
    "${TRAJECTORY_CONFIDENCE_ARGS[@]}" \
    --alphas "${DOSES[@]}" --max-samples 0 --max-samples-per-cluster 0 \
    --batch-size "$GENERATION_BATCH_SIZE" --max-new-tokens "$GENERATION_MAX_NEW_TOKENS" \
    --generated-patch-steps "$GENERATION_PATCH_STEPS" --save-generated-token-ids \
    --save-generated-token-logprobs \
    --max-truncated-rate "${PERSISTENT_MAX_TRUNCATED_RATE:-0.80}" \
    --style-untruncated-only \
    --style-bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" \
    --style-permutation-tests "$INTERVENTION_PERMUTATION_TESTS"
else
  echo "[reuse] $TRAJECTORY_DIR/cluster_intervention_summary.json"
fi

stage "analyze persistent trajectories"
TRAJECTORY_ANALYSIS_ARGS=()
[[ -n "$CRITICAL_DIR" ]] && TRAJECTORY_ANALYSIS_ARGS+=(--critical-dir "$CRITICAL_DIR")
"$PYTHON_BIN" scripts/analyze_continuous_trajectory.py \
  "${TRAJECTORY_ANALYSIS_ARGS[@]}" --generation-dir "$TRAJECTORY_DIR" \
  --output-dir "$OUTPUT_DIR/trajectory_analysis" --dose 1.0 \
  --metric-profile "$METRIC_PROFILE" \
  --bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" --example-count 24 \
  --seed "$MEASUREMENT_SEED"

stage "test frozen behavior construct across persistent doses"
"$PYTHON_BIN" scripts/analyze_construct_trajectory.py \
  --generation-dir "$TRAJECTORY_DIR" \
  --output-dir "$OUTPUT_DIR/trajectory_analysis" \
  --metric-profile "$METRIC_PROFILE" \
  --construct "$PRIMARY_BEHAVIOR_CONSTRUCT" \
  --expected-direction "$PRIMARY_BEHAVIOR_DIRECTION" \
  --doses "${DOSES[@]}" \
  --max-new-tokens "$GENERATION_MAX_NEW_TOKENS" \
  --bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" \
  --permutation-tests "$INTERVENTION_PERMUTATION_TESTS" \
  --seed "$MEASUREMENT_SEED"

if [[ "$RUN_SINGLE_STEP_TRAJECTORY" == 1 ]]; then
  SINGLE_STEP_DIR="$OUTPUT_DIR/single_step_trajectory"
  stage "generate matched single-step intervention trajectories (${PERSISTENT_POPULATION}:${TRAJECTORY_ROLE})"
  single_step_needs_refresh=0
  if [[ "$FORCE_INTERVENTION" == 1 || ! -s "$SINGLE_STEP_DIR/cluster_intervention_summary.json" ]]; then
    single_step_needs_refresh=1
  fi
  if [[ "$single_step_needs_refresh" == 1 ]]; then
    mkdir -p "$SINGLE_STEP_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/audit_continuous_module_dose.py \
      --continuous-axis-file "$AXIS_FILE" --continuous-evaluation-role "$TRAJECTORY_ROLE" \
      --continuous-directions "$CONTINUOUS_DIRECTION" --output-dir "$SINGLE_STEP_DIR" \
      --continuous-controls "${CONTINUOUS_CONTROL_VALUES[@]}" \
      "${COMMON[@]}" "${INTERVENTION[@]}" "${TRAJECTORY_ID_ARGS[@]}" \
      "${TRAJECTORY_CONFIDENCE_ARGS[@]}" \
      --baseline-file "$TRAJECTORY_DIR/baseline_generations.jsonl" \
      --alphas 1.0 --max-samples 0 --max-samples-per-cluster 0 \
      --batch-size "$GENERATION_BATCH_SIZE" --max-new-tokens "$GENERATION_MAX_NEW_TOKENS" \
      --generated-patch-steps 1 --save-generated-token-ids --save-generated-token-logprobs \
      --max-truncated-rate "${PERSISTENT_MAX_TRUNCATED_RATE:-0.80}" \
      --style-untruncated-only \
      --style-bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" \
      --style-permutation-tests "$INTERVENTION_PERMUTATION_TESTS"
  else
    echo "[reuse] $SINGLE_STEP_DIR/cluster_intervention_summary.json"
  fi

  stage "analyze matched single-step trajectories"
  "$PYTHON_BIN" scripts/analyze_continuous_trajectory.py \
    "${TRAJECTORY_ANALYSIS_ARGS[@]}" --generation-dir "$SINGLE_STEP_DIR" \
    --output-dir "$OUTPUT_DIR/single_step_trajectory_analysis" --dose 1.0 \
    --metric-profile "$METRIC_PROFILE" \
    --bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" --example-count 24 \
    --seed "$MEASUREMENT_SEED"

  "$PYTHON_BIN" scripts/compare_intervention_horizons.py \
    --single-step-dir "$OUTPUT_DIR/single_step_trajectory_analysis" \
    --continuous-dir "$OUTPUT_DIR/trajectory_analysis" \
    --output-dir "$OUTPUT_DIR/intervention_horizon_ablation"
fi

if [[ "$META_BEHAVIOR_EVALUATION" == 1 ]]; then
  stage "evaluate persistent confidence and uncertainty behavior"
  trajectory_meta_dir="$OUTPUT_DIR/metacognitive_behavior_trajectory"
  META_TRAJECTORY_ARGS=(
    --continuous-axis-file "$AXIS_FILE"
    --baseline-file "$TRAJECTORY_DIR/baseline_generations.jsonl"
    --trajectory-dir "$TRAJECTORY_DIR"
    --output-dir "$trajectory_meta_dir"
    --evaluation-role "$TRAJECTORY_ROLE"
    --crossfit-folds 5 --ridge 0.10
    --dose 1.0
    --bootstrap-samples "$META_BEHAVIOR_BOOTSTRAP_SAMPLES"
    --permutation-tests "$META_BEHAVIOR_PERMUTATION_TESTS"
    --max-new-tokens "$GENERATION_MAX_NEW_TOKENS"
    --seed "$MEASUREMENT_SEED"
  )
  [[ "${META_BEHAVIOR_EXCLUDE_TRUNCATED:-1}" == 1 ]] && META_TRAJECTORY_ARGS+=(--exclude-truncated)
  if [[ "$META_BEHAVIOR_INCLUDE_SECONDARY" == 1 ]]; then
    META_TRAJECTORY_ARGS+=(--include-secondary)
  else
    META_TRAJECTORY_ARGS+=(--no-include-secondary)
  fi
  "$PYTHON_BIN" scripts/analyze_metacognitive_behavior.py "${META_TRAJECTORY_ARGS[@]}"
fi

if [[ " ${CONTINUOUS_CONTROL_VALUES[*]} " == *" random_direction "* ||
      " ${CONTINUOUS_CONTROL_VALUES[*]} " == *" random_hidden_direction "* ]]; then
  stage "paired persistent real-vs-orthogonal-random analysis"
  "$PYTHON_BIN" scripts/analyze_random_orthogonal_trajectory.py \
    --generation-dir "$TRAJECTORY_DIR" \
    --output-dir "$OUTPUT_DIR/random_orthogonal_trajectory_analysis" \
    --dose 1.0 --metric-profile "$METRIC_PROFILE" \
    --bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" \
    --permutation-tests "$INTERVENTION_PERMUTATION_TESTS" \
    --seed "$MEASUREMENT_SEED"
fi
