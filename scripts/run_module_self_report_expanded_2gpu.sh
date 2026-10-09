#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CONFIG_FILE="configs/module_self_report_expanded_v1.json"
exec bash scripts/run_module_self_report_pilot_2gpu.sh
