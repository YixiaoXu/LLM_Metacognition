#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export CONFIG_FILE="configs/neuron_report_search_fresh24_v3.json"
export REUSE_ROOT=""
export SMOKE_PAIR="${SMOKE_PAIR:-beavertails/deepseek_qwen7b}"
export RUN_MODEL="${RUN_MODEL:-1}"
export SMOKE_ROOT="${SMOKE_ROOT:-runs/neuron_report_fresh24_smoke_$(date +%Y%m%d_%H%M%S)}"
exec bash scripts/smoke_neuron_report_complete24.sh
