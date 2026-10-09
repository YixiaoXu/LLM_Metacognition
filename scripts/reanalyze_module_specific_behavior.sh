#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${RUN_ROOT:?Set RUN_ROOT to an existing continuous experiment.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-3000}"
PERMUTATION_TESTS="${PERMUTATION_TESTS:-3000}"
CROSSFIT_FOLDS="${CROSSFIT_FOLDS:-5}"
SEMANTIC_RIDGE="${SEMANTIC_RIDGE:-0.10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
MEASUREMENT_SEED="${MEASUREMENT_SEED:-42017}"
MODULE_BEHAVIOR_ALPHA="${MODULE_BEHAVIOR_ALPHA:-0.05}"

mapfile -t module_dirs < <(
  find "$RUN_ROOT" -path '*/continuous_modules/rank_*' -type d | sort
)
if [[ "${#module_dirs[@]}" -eq 0 ]]; then
  echo "No continuous module directories found below $RUN_ROOT" >&2
  exit 2
fi

completed=0
skipped=0
for module_dir in "${module_dirs[@]}"; do
  axis_file="$module_dir/continuous_axis/continuous_axis.pt"
  behavior_file="$module_dir/baseline_behavior/safety_behavior_by_sample.csv"
  output_dir="$module_dir/continuous_behavior"
  if [[ ! -s "$axis_file" || ! -s "$behavior_file" ]]; then
    printf '[skip] %s: missing axis or behavior file\n' "$module_dir"
    skipped=$((skipped + 1))
    continue
  fi
  printf '[module-specific] %s\n' "$module_dir"
  "$PYTHON_BIN" scripts/analyze_continuous_meta_behavior.py \
    --continuous-axis-file "$axis_file" \
    --behavior-file "$behavior_file" \
    --output-dir "$output_dir" \
    --evaluation-role association_confirmatory \
    --semantic-ridge "$SEMANTIC_RIDGE" \
    --crossfit-folds "$CROSSFIT_FOLDS" \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --permutation-tests "$PERMUTATION_TESTS" \
    --semantic-r2-equivalence-margin 0.20 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --seed "$MEASUREMENT_SEED"
  completed=$((completed + 1))
done

"$PYTHON_BIN" scripts/summarize_module_behavior_map.py \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/module_behavior_map" \
  --alpha "$MODULE_BEHAVIOR_ALPHA"

printf 'Module-specific behavior reanalysis complete: completed=%d skipped=%d\n' \
  "$completed" "$skipped"
