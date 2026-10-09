#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
SMOKE_ROOT="${SMOKE_ROOT:-runs/neuron_report_complete24_smoke_$(date +%Y%m%d_%H%M%S)}"
export CONFIG_FILE="${CONFIG_FILE:-configs/neuron_report_search_complete24_v2.json}"

echo "[smoke] Python and Bash syntax"
"$PYTHON_BIN" -m py_compile \
  metacog/audits/neuron_report_replay.py \
  metacog/audits/neuron_report_measurement.py \
  scripts/audit_internal_activation_report.py \
  scripts/audit_neuron_report_groups.py \
  scripts/audit_neuron_report_search.py scripts/run_neuron_report_search.py
bash -n scripts/run_neuron_report_search_24_2gpu.sh
bash -n scripts/run_neuron_report_fresh24_2gpu.sh
bash -n scripts/smoke_neuron_report_complete24.sh
"$PYTHON_BIN" scripts/run_neuron_report_search.py --help >/dev/null
[[ "${STATIC_ONLY:-0}" == 1 ]] && { echo "[smoke] static checks passed"; exit 0; }

echo "[smoke] 24-condition plan, compatible completed results and remaining data pools"
PLAN_ONLY=1 RUN_ROOT="$SMOKE_ROOT" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_neuron_report_search_24_2gpu.sh
"$PYTHON_BIN" scripts/audit_neuron_report_search.py --help >/dev/null
if [[ "${RUN_MODEL:-0}" == 1 ]]; then
  PAIR="${SMOKE_PAIR:-mathqa/qwen3_8b}"
  echo "[smoke] actual generated-token replay, reports and matched flips: $PAIR"
  CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PYTHON_BIN" -u - \
    "$SMOKE_ROOT/frozen_plan.json" "$PAIR" "$SMOKE_ROOT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text())
pair = sys.argv[2]
items = [item for item in plan["conditions"]
         if item["dataset"] + "/" + item["target"] == pair]
if len(items) != 1:
    raise SystemExit("Unknown smoke pair: " + pair)
item = items[0]
command = [sys.executable, "-u", "scripts/audit_neuron_report_search.py",
           "--target", item["target"], "--layer-id", str(item["layer_i"]),
           "--activation-dir", item["activation_dir"], "--semantic-cache", item["semantic_cache"],
           "--axis-csv", item["axis_csv"], "--output-dir",
           str(Path(sys.argv[3]) / "model_smoke" / pair.replace("/", "__")),
           "--semantic-fit-rows", "200", "--screen-a-pool", "32",
           "--candidate-neurons", "8", "--semantic-bins", "2", "--screen-a-rows", "8",
           "--screen-b-candidates", "4", "--screen-b-rows", "16", "--finalists", "2",
           "--evaluation-pool-rows", "96", "--confirm-rows", "40", "--flip-rows", "4",
           "--bootstrap-samples", "20", "--replay-mode", "generation_step_kv"]
command += ["--activation-source", plan["config"]["report"].get("activation_source", "cache")]
subprocess.run(command, check=True)
result = json.loads((Path(command[command.index("--output-dir") + 1]) /
                     "report_search_summary.json").read_text())
expected_source = plan["config"]["report"].get("activation_source", "cache")
if result.get("activation_source", "cache") != expected_source:
    raise SystemExit("Smoke used a different activation source than the frozen configuration")
if expected_source == "runtime_remeasured":
    neurons = result["winners"] + result["matched_controls"]
    if any(row["replay_label_agreement"] != 1.0 for row in neurons):
        raise SystemExit("Fresh measured labels and reported runtime states still disagree")
    print("[smoke] fresh activation labels agree with runtime reports", flush=True)
PY
fi
echo "[smoke] passed"
