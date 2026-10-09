#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/mathqa_continuous_smoke_${STAMP}}"

echo "[smoke] static Python and Bash validation"
"$PYTHON_BIN" -m py_compile \
  scripts/prepare_mathqa_prompts.py \
  scripts/extract_generation_step_activations.py \
  scripts/validate_activation_alignment.py \
  scripts/train_decoupler_joint_v2.py \
  scripts/prepare_continuous_module_axis.py \
  scripts/analyze_continuous_meta_behavior.py \
  scripts/audit_continuous_module_dose.py \
  scripts/analyze_continuous_trajectory.py
for script in \
  scripts/run_mathqa_experiment.sh \
  scripts/run_model_pair.sh \
  scripts/train_external_semantic_modules.sh \
  scripts/run_continuous_module.sh; do
  bash -n "$script"
done

echo "[smoke] command-line interface validation"
for script in \
  scripts/extract_generation_step_activations.py \
  scripts/validate_activation_alignment.py \
  scripts/train_decoupler_joint_v2.py \
  scripts/prepare_continuous_module_axis.py \
  scripts/analyze_continuous_meta_behavior.py \
  scripts/audit_continuous_module_dose.py; do
  "$PYTHON_BIN" "$script" --help >/dev/null
done

echo "[smoke] tiny end-to-end run on GPU $GPU"
env CONFIG_FILE=configs/mathqa_external_semantic_continuous_smoke.env \
  GPU0="$GPU" GPU1="$GPU" RUN_ROOT="$RUN_ROOT" FORCE="${FORCE:-1}" \
  bash scripts/run_mathqa_experiment.sh

echo "[smoke] verifying required artifacts"
find "$RUN_ROOT" -path '*/continuous_axis/continuous_axis.pt' -print -quit | grep -q .
find "$RUN_ROOT" -path '*/continuous_behavior/continuous_behavior_summary.json' -print -quit | grep -q .
find "$RUN_ROOT" -path '*/causal_screen_a/continuous_signed_dose_response.csv' -print -quit | grep -q .
find "$RUN_ROOT" -path '*/critical_heldout/critical_heldout_summary.json' -print -quit | grep -q .
if find "$RUN_ROOT" -path '*/critical_heldout/trajectory_ids.txt' -size +0c -print -quit | grep -q .; then
  find "$RUN_ROOT" -path '*/trajectory_analysis/trajectory_summary.json' -print -quit | grep -q .
fi
echo "[smoke] PASS: $RUN_ROOT"
