#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_ROOT="${RUN_ROOT:-runs/report_trajectory_followup_smoke_$(date +%Y%m%d_%H%M%S)}"
CONFIG_FILE=configs/report_trajectory_followup_smoke.json
python -m py_compile scripts/audit_neuron_report_groups.py scripts/run_report_trajectory_followup.py \
  scripts/run_strict_followup_group.py
bash -n scripts/run_report_trajectory_followup.sh
echo "[smoke] plan-only run_root=$RUN_ROOT"
PLAN_ONLY=1 RUN_ROOT="$RUN_ROOT" CONFIG_FILE="$CONFIG_FILE" GPU_IDS="${GPU_IDS:-0 1}" \
  bash scripts/run_report_trajectory_followup.sh
echo "[smoke] model and trajectory check"
RUN_ROOT="$RUN_ROOT" CONFIG_FILE="$CONFIG_FILE" GPU_IDS="${GPU_IDS:-0 1}" \
  bash scripts/run_report_trajectory_followup.sh
python - "$RUN_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
report = json.loads((root / "direct_report/ultrachat/llama2_7b/report_group_summary.json").read_text())
assert report["n_evaluation"] >= 100
assert report["selected"], "No candidate reached held-out reporting; inspect discovery selection"
assert (root / "group_complete.json").is_file()
print("[smoke] passed; selected_neurons=", len(report["selected"]))
PY
