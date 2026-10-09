#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${JOB_ROOT:?}"
: "${MODULE_SUPPORT:?}"
: "${TARGET_PROFILE:?}"
: "${TARGET_ACTIVATION_DIR:?}"
: "${SEMANTIC_ACTIVATION_DIR:?}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DECOUPLER_DIR="$JOB_ROOT/decoupler_joint_v2"
mkdir -p "$JOB_ROOT"
read -r -a LAYER_FRAC_VALUES <<< "$LAYER_FRACS"
read -r -a SEMANTIC_ANCHOR_VALUES <<< "$SEMANTIC_ANCHOR_OFFSETS"
SPLIT_ARGS=()
if [[ "${SPLIT_BY_BASE_ID:-0}" == "1" ]]; then
  SPLIT_ARGS+=(--split-by-base-id --base-id-step-pattern "${BASE_ID_STEP_PATTERN:-::step\\d+$}")
fi

if [[ "$FORCE" != 1 && -s "$DECOUPLER_DIR/analysis_summary.json" ]]; then
  echo "[train] reuse $DECOUPLER_DIR"
  exit 0
fi

REUSE_MAIN_ARGS=()
if [[ "$FORCE" != 1 && ! -s "$DECOUPLER_DIR/analysis_summary.json" && \
      -s "$DECOUPLER_DIR/best_model.pt" && \
      -s "$DECOUPLER_DIR/joint_module_heads.pt" && \
      -s "$DECOUPLER_DIR/config.json" ]]; then
  REUSE_MAIN_ARGS+=(--reuse-main-dir "$DECOUPLER_DIR")
  echo "[train] resume module export from frozen first-stage optimum: $DECOUPLER_DIR"
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/train_decoupler_joint_v2.py \
  --activation-dir "$TARGET_ACTIVATION_DIR" \
  --external-semantic-activation-dir "$SEMANTIC_ACTIVATION_DIR" \
  --external-semantic-layer "$EXTERNAL_SEMANTIC_LAYER" \
  --external-semantic-max-dim "$EXTERNAL_SEMANTIC_MAX_DIM" \
  --external-semantic-ridge "$EXTERNAL_SEMANTIC_RIDGE" \
  --external-semantic-head-type "${EXTERNAL_SEMANTIC_HEAD_TYPE:-mlp}" \
  --external-semantic-head-depth "${EXTERNAL_SEMANTIC_HEAD_DEPTH:-4}" \
  --output-dir "$DECOUPLER_DIR" \
  --model-label "${TARGET_PROFILE}_${TASK_LABEL:-external_semantic}" \
  --layer-fracs "${LAYER_FRAC_VALUES[@]}" \
  --semantic-anchor-offsets "${SEMANTIC_ANCHOR_VALUES[@]}" \
  --latent-dim "$LATENT_DIM" --hidden-dim "$HIDDEN_DIM" --dropout "$TRAIN_DROPOUT" \
  --batch-size "$TRAIN_BATCH_SIZE" --encode-batch-size "$ENCODE_BATCH_SIZE" \
  --epochs "$MAIN_EPOCHS" --lr "$TRAIN_LR" --weight-decay "$TRAIN_WEIGHT_DECAY" \
  --val-ratio "$VAL_RATIO" --test-ratio "$TEST_RATIO" \
  "${SPLIT_ARGS[@]}" \
  --candidate-top-k "$CANDIDATE_TOP_K" --candidate-min-rate "$CANDIDATE_MIN_RATE" \
  --candidate-max-rate "$CANDIDATE_MAX_RATE" --candidate-max-rate-drift "$CANDIDATE_MAX_RATE_DRIFT" \
  --binary-quantile "$BINARY_QUANTILE" --num-modules "$NUM_MODULES" \
  --module-support "$MODULE_SUPPORT" --module-max-memberships "$MODULE_MAX_MEMBERSHIPS" \
  --module-direction-mode "${MODULE_DIRECTION_MODE:-learned}" \
  --random-support-seed "${RANDOM_SUPPORT_SEED:-$SEED}" \
  --direction-ridge "$DIRECTION_RIDGE" \
  --direction-semantic-penalty "$DIRECTION_SEMANTIC_PENALTY" --direction-ema "$DIRECTION_EMA" \
  --direction-refresh-start "$MAIN_DIRECTION_REFRESH_START" \
  --direction-refresh-every "$MAIN_DIRECTION_REFRESH_EVERY" \
  --direction-freeze-epoch "$MAIN_DIRECTION_FREEZE" \
  --direction-max-samples "$DIRECTION_MAX_SAMPLES" --direction-min-cosine "$DIRECTION_MIN_COSINE" \
  --lambda-next "$LAMBDA_NEXT" --lambda-prev "$LAMBDA_PREV" --lambda-orth "$LAMBDA_ORTH" \
  --lambda-e2-prev-cov "$LAMBDA_E2_PREV_COV" \
  --lambda-semantic-candidate "$LAMBDA_SEMANTIC_CANDIDATE" \
  --lambda-semantic-module "$LAMBDA_SEMANTIC_MODULE" \
  --lambda-residual-module "$LAMBDA_RESIDUAL_MODULE" \
  --lambda-joint-module "$LAMBDA_JOINT_MODULE" --semantic-candidate-source z1 \
  --semantic-probe-mode staged_z1_aligned \
  --semantic-encoder-warmup-epochs "$SEMANTIC_ENCODER_WARMUP_EPOCHS" \
  --semantic-probe-hidden-dim "$SEMANTIC_PROBE_HIDDEN_DIM" \
  --semantic-probe-epochs "$SEMANTIC_PROBE_EPOCHS" \
  --semantic-probe-lr "$SEMANTIC_PROBE_LR" \
  --semantic-probe-weight-decay "$SEMANTIC_PROBE_WEIGHT_DECAY" \
  --lambda-var "$LAMBDA_VAR" --meta-start-epoch "$MAIN_META_START" \
  --eval-every 5 --checkpoint-start-epoch "$MAIN_CHECKPOINT_START" \
  --checkpoint-residual-gain-weight "$CHECKPOINT_RESIDUAL_GAIN_WEIGHT" \
  --checkpoint-total-gain-weight "$CHECKPOINT_TOTAL_GAIN_WEIGHT" \
  --checkpoint-residual-fraction-weight "$CHECKPOINT_RF_WEIGHT" \
  --checkpoint-positive-module-weight "$CHECKPOINT_POSITIVE_MODULE_WEIGHT" \
  --second-stage main_direct --module-code-dim "$MODULE_CODE_DIM" \
  --module-refiner-code-mode latent --module-refiner-hidden-dim "$MODULE_REFINER_HIDDEN_DIM" \
  --module-refiner-epochs "$MODULE_REFINER_EPOCHS" --module-refiner-lr "$MODULE_REFINER_LR" \
  --module-refiner-workers "${MODULE_REFINER_WORKERS:-1}" \
  --module-contribution-mode residual_target \
  --module-semantic-probe-profile legacy_posthoc \
  --module-selection-residual-gain-weight "$MODULE_SELECTION_RESIDUAL_GAIN_WEIGHT" \
  --module-selection-total-gain-weight "$MODULE_SELECTION_TOTAL_GAIN_WEIGHT" \
  --module-selection-residual-fraction-weight "$MODULE_SELECTION_RF_WEIGHT" \
  --module-selection-continuous-semantic-r2-penalty "$MODULE_SELECTION_CONTINUOUS_SEMANTIC_R2_PENALTY" \
  --module-selection-max-continuous-semantic-r2 "$MODULE_SELECTION_MAX_CONTINUOUS_SEMANTIC_R2" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" --permutation-tests "$PERMUTATION_TESTS" \
  --export-min-residual-fraction 0.0 \
  --device cuda --seed "$SEED" --split-seed "$SEED" \
  "${REUSE_MAIN_ARGS[@]}"
