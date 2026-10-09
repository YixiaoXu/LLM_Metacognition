#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[smoke] Python syntax"
"$PYTHON_BIN" -m py_compile \
  metacog/evaluation/profiles.py \
  metacog/evaluation/styles.py \
  scripts/adaptive_baseline_cache.py \
  scripts/cluster_intervention.py \
  scripts/analyze_continuous_meta_behavior.py \
  scripts/analyze_construct_trajectory.py \
  scripts/prepare_continuous_module_axis.py \
  scripts/select_adaptive_module_sizes.py \
  scripts/summarize_strict_module_chain.py

echo "[smoke] Bash syntax"
bash -n scripts/run_continuous_module.sh
bash -n scripts/run_model_pair.sh
bash -n scripts/run_mathqa_strict_chain_deepseek_qwen7b.sh
bash -n scripts/run_strict_main_24_2gpu.sh

echo "[smoke] ten-construct profiles"
"$PYTHON_BIN" - <<'PY'
from metacog.evaluation import (
    metrics_for_profile,
    primary_construct_count,
    primary_construct_groups,
)
from scripts.summarize_strict_module_chain import holm

for profile in ("conversation", "safety", "math"):
    assert primary_construct_count(profile) == 10, profile
    groups = primary_construct_groups(profile)
    assert list(groups) == ["behavior", "monitoring"], profile
    assert all(len(constructs) == 5 for constructs in groups.values()), profile
    assert len({item for values in groups.values() for item in values}) == 10, profile
assert "correct" not in metrics_for_profile("math")
assert "correct" in metrics_for_profile("math", include_diagnostics=True)
assert holm([0.001, 0.02, 0.20]) == [0.003, 0.04, 0.2]
print("profiles=conversation,safety,math constructs=5 behavior + 5 monitoring")
PY

echo "[smoke] command-line interfaces"
"$PYTHON_BIN" scripts/prepare_continuous_module_axis.py --help \
  | grep -q -- '--association-fraction'
"$PYTHON_BIN" scripts/select_adaptive_module_sizes.py --help \
  | grep -q -- '--fill-to-max'
"$PYTHON_BIN" scripts/summarize_strict_module_chain.py --help \
  | grep -q -- '--metric-profile'
"$PYTHON_BIN" scripts/analyze_construct_trajectory.py --help \
  | grep -q -- '--expected-direction'
"$PYTHON_BIN" scripts/adaptive_baseline_cache.py plan --help \
  | grep -q -- '--state-file'

echo "[smoke] 24-condition plan"
plan_root="${SMOKE_PLAN_ROOT:-/tmp/metacog_strict_main24_plan_$$}"
PLAN_ONLY=1 RUN_ROOT="$plan_root" bash scripts/run_strict_main_24_2gpu.sh >/dev/null
[[ "$(($(wc -l < "$plan_root/task_plan.tsv") - 1))" == 24 ]]

echo "[smoke] passed"
