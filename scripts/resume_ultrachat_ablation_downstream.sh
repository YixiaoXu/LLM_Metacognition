#!/usr/bin/env bash
set -euo pipefail

# Resume only downstream work from an existing UltraChat ablation run.
# Upstream activation caches, decouplers, support diagnostics, and existing
# deterministic baselines are reused. Every module passing the held-out gate
# is exported and sent through the continuous downstream pipeline.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/task_lock.sh"

: "${RUN_ROOT:?Set RUN_ROOT to an existing ablation run directory}"
CONFIG_FILE="${CONFIG_FILE:-$RUN_ROOT/input_config.env}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
FORCE="${FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

[[ -d "$RUN_ROOT" ]] || { echo "Missing RUN_ROOT: $RUN_ROOT" >&2; exit 2; }
[[ -s "$CONFIG_FILE" ]] || {
  echo "Missing config: $CONFIG_FILE" >&2
  echo "Pass CONFIG_FILE explicitly when the run was selectively synchronized." >&2
  exit 2
}

if ! acquire_task_lock "$RUN_ROOT/.locks/downstream_resume.lock" "downstream-resume:$RUN_ROOT"; then
  echo "[resume] duplicate downstream resume refused: $RUN_ROOT" >&2
  exit 17
fi
trap release_task_locks EXIT INT TERM

set -a
# shellcheck source=/dev/null
source "$CONFIG_FILE"
set +a
export ALL_HELDOUT_MODULES=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUN_ROOT/status"
exec > >(tee -a "$RUN_ROOT/resume_downstream.log") 2>&1

echo "[$(date '+%F %T')] resume downstream: run_root=$RUN_ROOT"
echo "[$(date '+%F %T')] policy=all modules passing held-out gate; q-values are not gates"

model_field() { "$PYTHON_BIN" -m metacog.cli model get "$1" "$2"; }

pair_job_config() {
  find "$1/jobs" -path '*/decoupler_joint_v2/config.json' -type f -print -quit
}

config_field() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json, sys
path, key = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    value = json.load(handle).get(key)
if value is None:
    raise SystemExit(1)
print(value)
PY
}

abspath_from_root() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT_DIR" "$path"
  fi
}

run_pair() {
  local gpu="$1" spec="$2"
  local target reference semantic_layer pair_name pair_root job_config
  local activation_dir decoupler_dir semantic_dir data_path model_path shared_baseline
  IFS=: read -r target reference semantic_layer <<< "$spec"
  pair_name="${target}__semantic_${reference}_l${semantic_layer}"
  pair_root="$RUN_ROOT/$pair_name"
  job_config="$(pair_job_config "$pair_root" || true)"
  if [[ -z "$job_config" ]]; then
    echo "[$(date '+%F %T')] missing upstream job config: $pair_root" >&2
    return 1
  fi

  activation_dir="$(abspath_from_root "$(config_field "$job_config" activation_dir)")"
  decoupler_dir="$(abspath_from_root "$(config_field "$job_config" output_dir)")"
  semantic_dir="$(abspath_from_root "$(config_field "$job_config" external_semantic_activation_dir)")"
  data_path="$activation_dir/expanded_generation_data.jsonl"
  model_path="$(model_field "$target" path)"
  [[ -s "$data_path" ]] || { echo "Missing expanded data: $data_path" >&2; return 1; }
  [[ -d "$decoupler_dir" ]] || { echo "Missing decoupler: $decoupler_dir" >&2; return 1; }

  shared_baseline="$(find "$pair_root/continuous_modules" -type f \
    -path '*/baseline_*/baseline_generations.jsonl' -size +0c -print -quit 2>/dev/null || true)"
  echo "[$(date '+%F %T')] pair=$pair_name gpu=$gpu"
  echo "  activation_dir=$activation_dir"
  echo "  decoupler_dir=$decoupler_dir"
  echo "  semantic_dir=$semantic_dir"
  echo "  shared_baseline=${shared_baseline:-none}"

  env GPU="$gpu" PAIR_ROOT="$pair_root" TARGET_PROFILE="$target" \
    TARGET_MODEL_PATH="$model_path" TARGET_ACTIVATION_DIR="$activation_dir" \
    SEMANTIC_PROFILE="$reference" SEMANTIC_ACTIVATION_DIR="$semantic_dir" \
    EXTERNAL_SEMANTIC_LAYER="$semantic_layer" DATA_PATH="$data_path" \
    MODEL_DTYPE="$(model_field "$target" dtype)" \
    TRUST_REMOTE_CODE="$(model_field "$target" trust_remote_code)" \
    SHARED_BASELINE_FILE="${shared_baseline:-$pair_root/shared_association_baseline/baseline_generations.jsonl}" \
    FORCE="$FORCE" ALL_HELDOUT_MODULES=1 \
    bash scripts/run_model_pair.sh
}

read -r -a raw_pairs <<< "${PAIR_SPECS:?PAIR_SPECS is missing from config}"
declare -A seen_pairs=()
PAIRS=()
for spec in "${raw_pairs[@]}"; do
  if [[ -n "${seen_pairs[$spec]:-}" ]]; then
    echo "[queue] skip duplicate pair spec: $spec" >&2
    continue
  fi
  seen_pairs[$spec]=1
  PAIRS+=("$spec")
done
[[ "${#PAIRS[@]}" -gt 0 ]] || { echo "PAIR_SPECS is empty" >&2; exit 2; }

status0=0; status1=0
run_worker() {
  local gpu="$1" index spec target
  local failed=0
  for index in "${!PAIRS[@]}"; do
    spec="${PAIRS[$index]}"
    IFS=: read -r target _ _ <<< "$spec"
    # Keep one target model on one GPU at a time; the pair list here uses a
    # single target, but the modulo rule remains safe for future configs.
    if (( index % 2 == 0 )); then
      [[ "$gpu" == "$GPU0" ]] || continue
    else
      [[ "$gpu" == "$GPU1" ]] || continue
    fi
    set +e
    run_pair "$gpu" "$spec"
    rc=$?
    set -e
    printf '%s\t%s\t%s\n' \
      "$([[ "$rc" == 0 ]] && echo complete || echo failed)" "$gpu" "$rc" \
      > "$RUN_ROOT/status/${target}_${index}.status"
    [[ "$rc" == 0 ]] || failed=1
  done
  return "$failed"
}

run_worker "$GPU0" & pid0=$!
run_worker "$GPU1" & pid1=$!
wait "$pid0" || status0=1
wait "$pid1" || status1=1

echo "[$(date '+%F %T')] downstream resume complete: gpu0_status=$status0 gpu1_status=$status1"
if (( status0 != 0 || status1 != 0 )); then
  exit 1
fi
