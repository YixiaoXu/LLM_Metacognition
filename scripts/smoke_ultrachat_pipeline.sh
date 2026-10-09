#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[smoke] compile maintained Python package"
"$PYTHON_BIN" -m compileall -q metacog
for script in scripts/*.sh; do
  bash -n "$script"
done

echo "[smoke] prepare and validate 32 real UltraChat prompts"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
"$PYTHON_BIN" -m metacog.cli dataset prepare ultrachat \
  --input 'ultrachat_200k/data/train_sft-*.parquet' \
  --output "$temporary/ultrachat.jsonl" --max-samples 32 --seed 42
"$PYTHON_BIN" -m metacog.cli dataset validate ultrachat \
  "$temporary/ultrachat.jsonl" --min-samples 32

echo "[smoke] validate model registry and conversation metrics"
"$PYTHON_BIN" -m metacog.cli model get qwen3_4b dtype >/dev/null
"$PYTHON_BIN" - <<'PY'
from metacog.evaluation import compute_style_metrics, metrics_for_profile

values = compute_style_metrics("Here is an example:\n\n- one\n- two", 8, 64)
assert values["style_paragraph_count"] == 2
assert set(metrics_for_profile("conversation")) <= set(values)
PY

if [[ "${RUN_FULL:-0}" == 1 ]]; then
  echo "[smoke] start reduced GPU pipeline"
  CONFIG_FILE=configs/ultrachat_external_semantic_continuous_smoke.env \
    bash scripts/run_ultrachat_experiment.sh
else
  echo "[smoke] interface/data smoke complete; set RUN_FULL=1 for the GPU path"
fi
