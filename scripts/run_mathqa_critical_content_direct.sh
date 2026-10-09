#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_FILE="${CONFIG_FILE:-configs/mathqa_critical_content_direct.env}"
RUN_ROOT="${RUN_ROOT:-runs/mathqa_critical_content_direct_$(date +%Y%m%d_%H%M%S)}"
FORCE="${FORCE:-0}"
: "${SOURCE_MODULE_OUTPUT:?Set SOURCE_MODULE_OUTPUT to a completed continuous-module output directory}"
: "${MODULE_DIR:?Set MODULE_DIR to the selected module directory}"
: "${DECOUPLER_DIR:?Set DECOUPLER_DIR to its decoupler_joint_v2 directory}"

set -a
# shellcheck source=/dev/null
source "$CONFIG_FILE"
set +a

slug_steps="${GENERATION_RECORD_STEPS//,/-}"
ACTIVATION_DIR="${ACTIVATION_DIR:-activations/qwen3_4b_mathqa_gensteps_n${TARGET_PROMPT_MAX_SAMPLES}_steps${slug_steps}_s${SEED}}"
DATA_PATH="${DATA_PATH:-$ACTIVATION_DIR/expanded_generation_data.jsonl}"
MODEL_PATH="${MODEL_PATH:-$QWEN3_4B_MODEL_PATH}"
AXIS_FILE="$SOURCE_MODULE_OUTPUT/continuous_axis/continuous_axis.pt"
ASSIGNMENTS="$SOURCE_MODULE_OUTPUT/continuous_axis/continuous_axis_all_assignments.csv"
FEATURES="$SOURCE_MODULE_OUTPUT/continuous_axis/continuous_axis_all_features.pt"
TARGET_FILE="$MODULE_DIR/soft_residual_intervention_targets.pt"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/.mplconfig"
export MPLCONFIGDIR="$RUN_ROOT/.mplconfig"

stage() {
  printf '\n[%s] ----- %s -----\n' "$(date '+%F %T')" "$1"
}

for path in "$MODEL_PATH" "$DATA_PATH" "$AXIS_FILE" "$ASSIGNMENTS" "$FEATURES" \
  "$TARGET_FILE" "$MODULE_DIR/module_refiner.pt"; do
  [[ -e "$path" ]] || { echo "Missing required artifact: $path" >&2; exit 2; }
done

COMMON=(
  --model-path "$MODEL_PATH" --data "$DATA_PATH" --target-file "$TARGET_FILE"
  --activation-dir "$ACTIVATION_DIR" --decoupler-dir "$DECOUPLER_DIR"
  --cluster-source soft_residual --fixed-cluster-ids-only --num-clusters 2
  --eval-scope all_cached --cluster-sample-strategy first
  --max-length 4096 --prompt-style data --answer-extraction mathqa_choice
  --metric-profile math --temperature 0.0 --record-layers none
  --device-map single --dtype bfloat16 --trust-remote-code --seed "$SEED"
  --id-regex '::step000$'
  --fixed-cluster-assignments "$ASSIGNMENTS" --fixed-cluster-features "$FEATURES"
)

INTERVENTION=(
  --intervention-space refined_residual --refined-module-dir "$MODULE_DIR"
  --refined-direct-steps "$DIRECT_TRUST_REGION_STEPS"
  --refined-direct-step-scale "$DIRECT_TRUST_REGION_STEP_SCALE"
  --refined-direct-hidden-penalty "$DIRECT_TRUST_REGION_HIDDEN_PENALTY"
  --refined-code-semantic-penalty "$REFINED_CODE_SEMANTIC_PENALTY"
  --refined-code-z1-penalty "$REFINED_CODE_Z1_PENALTY"
  --refined-code-max-semantic-rel-delta "$MAX_SEMANTIC_REL_DELTA"
  --max-delta-rel-norm "$MAX_HIDDEN_REL_NORM" --clear-cuda-cache-between-modes
)

run_early_baseline() {
  local role="$1" output="$2"
  if [[ "$FORCE" != 1 && -s "$output/baseline_generations.jsonl" ]]; then
    echo "[reuse] $output/baseline_generations.jsonl"
    return
  fi
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/audit_continuous_module_dose.py \
    --continuous-axis-file "$AXIS_FILE" --continuous-evaluation-role "$role" \
    --output-dir "$output" "${COMMON[@]}" --baseline-only \
    --max-samples 0 --max-samples-per-cluster 0 --batch-size "$GENERATION_BATCH_SIZE" \
    --max-new-tokens "$CRITICAL_BASELINE_NEW_TOKENS" \
    --save-generated-token-ids --save-generated-token-logprobs \
    --max-truncated-rate 1.0 --style-untruncated-only
}

CALIBRATION_BASELINE="$RUN_ROOT/calibration_baseline"
EVALUATION_BASELINE="$RUN_ROOT/evaluation_baseline"
stage "baseline-only criticality calibration on causal_screen_a"
run_early_baseline causal_screen_a "$CALIBRATION_BASELINE"
stage "baseline-only criticality measurement on independent style_confirmatory"
run_early_baseline style_confirmatory "$EVALUATION_BASELINE"

SELECTION_DIR="$RUN_ROOT/critical_selection"
stage "freeze content-token critical threshold and match high-confidence controls"
if [[ "$FORCE" == 1 || ! -s "$SELECTION_DIR/critical_selection_summary.json" ]]; then
  mkdir -p "$SELECTION_DIR"
  "$PYTHON_BIN" scripts/select_content_token_critical_samples.py \
    --calibration-baseline "$CALIBRATION_BASELINE/baseline_generations.jsonl" \
    --evaluation-baseline "$EVALUATION_BASELINE/baseline_generations.jsonl" \
    --continuous-axis-file "$AXIS_FILE" --model-path "$MODEL_PATH" \
    --output-dir "$SELECTION_DIR" --critical-fraction "$CRITICAL_FRACTION" \
    --content-window "$CRITICAL_CONTENT_WINDOW" \
    --surprisal-top-k "$CRITICAL_SURPRISAL_TOP_K" \
    --max-generated-token-index "$CRITICAL_MAX_GENERATED_TOKEN_INDEX" \
    --control-max-quantile "$CRITICAL_CONTROL_MAX_QUANTILE" \
    --max-pairs "$CRITICAL_MAX_PAIRS" --min-pairs "$CRITICAL_MIN_PAIRS" \
    --trust-remote-code
else
  echo "[reuse] $SELECTION_DIR/critical_selection_summary.json"
fi

FULL_BASELINE="$RUN_ROOT/confirmatory_full_baseline"
stage "generate one shared full baseline for selected critical/control samples"
if [[ "$FORCE" == 1 || ! -s "$FULL_BASELINE/baseline_generations.jsonl" ]]; then
  mkdir -p "$FULL_BASELINE"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/audit_continuous_module_dose.py \
    --continuous-axis-file "$AXIS_FILE" --continuous-evaluation-role style_confirmatory \
    --output-dir "$FULL_BASELINE" "${COMMON[@]}" --baseline-only \
    --evaluation-id-file "$SELECTION_DIR/trajectory_union_ids.txt" \
    --max-samples 0 --max-samples-per-cluster 0 --batch-size "$GENERATION_BATCH_SIZE" \
    --max-new-tokens "$GENERATION_MAX_NEW_TOKENS" \
    --save-generated-token-ids --save-generated-token-logprobs \
    --max-truncated-rate 1.0 --style-untruncated-only
else
  echo "[reuse] $FULL_BASELINE/baseline_generations.jsonl"
fi

run_group() {
  local name="$1" id_file="$2" output="$RUN_ROOT/$name"
  stage "persistent signed-dose intervention: $name"
  if [[ "$FORCE" != 1 && -s "$output/cluster_intervention_summary.json" ]]; then
    echo "[reuse] $output/cluster_intervention_summary.json"
  else
    mkdir -p "$output"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/audit_continuous_module_dose.py \
      --continuous-axis-file "$AXIS_FILE" --continuous-evaluation-role style_confirmatory \
      --continuous-directions 0:1 --output-dir "$output" \
      "${COMMON[@]}" "${INTERVENTION[@]}" \
      --evaluation-id-file "$id_file" \
      --baseline-file "$FULL_BASELINE/baseline_generations.jsonl" \
      --alphas 1.0 --max-samples 0 --max-samples-per-cluster 0 \
      --batch-size "$GENERATION_BATCH_SIZE" --max-new-tokens "$GENERATION_MAX_NEW_TOKENS" \
      --generated-patch-steps "$GENERATION_PATCH_STEPS" \
      --save-generated-token-ids --save-generated-token-logprobs \
      --max-truncated-rate "$PERSISTENT_MAX_TRUNCATED_RATE" --style-untruncated-only \
      --style-bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" \
      --style-permutation-tests "$INTERVENTION_PERMUTATION_TESTS"
  fi
  "$PYTHON_BIN" scripts/analyze_continuous_trajectory.py \
    --generation-dir "$output" --output-dir "$output/trajectory_analysis" \
    --dose 1.0 --metric-profile math \
    --bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" \
    --example-count 16 --seed "$MEASUREMENT_SEED"
}

run_group critical_confirmatory "$SELECTION_DIR/critical_ids.txt"
run_group matched_high_confidence "$SELECTION_DIR/matched_control_ids.txt"

stage "paired criticality-by-intervention interaction"
"$PYTHON_BIN" scripts/analyze_critical_meta_interaction.py \
  --critical-generation-dir "$RUN_ROOT/critical_confirmatory" \
  --control-generation-dir "$RUN_ROOT/matched_high_confidence" \
  --selection-dir "$SELECTION_DIR" --output-dir "$RUN_ROOT/critical_interaction" \
  --bootstrap-samples "$INTERVENTION_BOOTSTRAP_SAMPLES" \
  --example-count 16 --seed "$MEASUREMENT_SEED"

echo "[$(date '+%F %T')] COMPLETE: $RUN_ROOT"
echo "selection=$SELECTION_DIR/critical_selection_summary.json"
echo "interaction=$RUN_ROOT/critical_interaction/critical_interaction_summary.json"
