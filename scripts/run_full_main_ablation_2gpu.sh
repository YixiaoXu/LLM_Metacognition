#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/task_lock.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUN_ROOT="${RUN_ROOT:-runs/full_main_ablation_$(date +%Y%m%d_%H%M%S)}"
FORCE="${FORCE:-0}"
FORCE_ACTIVATIONS="${FORCE_ACTIVATIONS:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
MAX_TASKS="${MAX_TASKS:-0}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PREFLIGHT="${DATA_PREFLIGHT:-1}"
SNAPSHOT_SOURCE="${SNAPSHOT_SOURCE:-1}"
INTERFACE_PREFLIGHT="${INTERFACE_PREFLIGHT:-1}"

RUN_MAIN="${RUN_MAIN:-1}"
RUN_SUPPORT_ABLATION="${RUN_SUPPORT_ABLATION:-1}"
RUN_SEMANTIC_HEAD_ABLATION="${RUN_SEMANTIC_HEAD_ABLATION:-1}"
RUN_LAYER_ABLATION="${RUN_LAYER_ABLATION:-1}"
RUN_SEMANTIC_REFERENCE_ABLATION="${RUN_SEMANTIC_REFERENCE_ABLATION:-1}"
RUN_CAUSAL_ABLATION="${RUN_CAUSAL_ABLATION:-1}"
RUN_SCALE_EXTENSION="${RUN_SCALE_EXTENSION:-1}"
SCALE_DATASETS="${SCALE_DATASETS:-ultrachat}"

ULTRACHAT_CONFIG="${ULTRACHAT_CONFIG:-configs/ultrachat_full_rerun_qwen06_s32_v1.env}"
BEAVERTAILS_CONFIG="${BEAVERTAILS_CONFIG:-configs/beavertails_full_rerun_qwen06_s32_v1.env}"
MATHQA_CONFIG="${MATHQA_CONFIG:-configs/mathqa_full_rerun_qwen06_s32_v1.env}"

[[ "$GPU0" != "$GPU1" ]] || {
  echo "GPU0 and GPU1 must identify two different devices." >&2
  exit 2
}
for config in "$ULTRACHAT_CONFIG" "$BEAVERTAILS_CONFIG" "$MATHQA_CONFIG"; do
  [[ -s "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
done

require_cli_flag() {
  local script="$1" flag="$2" help_text="$3"
  grep -q -- "$flag" <<< "$help_text" || {
    echo "Interface preflight failed: $script does not expose $flag" >&2
    return 2
  }
}

validate_interfaces() {
  local script help_text flag
  bash -n scripts/run_generated_step_experiment.sh
  bash -n scripts/run_model_pair.sh
  bash -n scripts/run_continuous_module.sh
  bash -n scripts/run_full_main_ablation_2gpu.sh
  for script in \
      scripts/extract_generation_step_activations.py \
      scripts/extract_activations.py \
      scripts/audit_continuous_module_dose.py \
      scripts/analyze_continuous_meta_behavior.py \
      scripts/analyze_proxy_confidence.py \
      scripts/analyze_safety_baseline.py \
      scripts/analyze_continuous_trajectory.py \
      scripts/analyze_random_orthogonal_trajectory.py; do
    # Inspect the declared interface without importing heavyweight model
    # dependencies. The actual task environment may not be available on a
    # submission/login node, while source-level option drift is still caught.
    help_text="$(<"$script")"
    case "$script" in
      *extract_generation_step_activations.py)
        for flag in --record-steps --max-generation-steps --chat-template-enable-thinking; do
          require_cli_flag "$script" "$flag" "$help_text"
        done
        ;;
      *extract_activations.py)
        for flag in --pooling --chat-template-enable-thinking; do
          require_cli_flag "$script" "$flag" "$help_text"
        done
        ;;
      *audit_continuous_module_dose.py)
        for flag in --continuous-axis-file --continuous-evaluation-role --baseline-file \
            --baseline-only --style-untruncated-only --refined-code-dose-match-controls; do
          require_cli_flag "$script" "$flag" "$help_text"
        done
        ;;
      *analyze_continuous_meta_behavior.py|*analyze_safety_baseline.py|*analyze_continuous_trajectory.py|*analyze_random_orthogonal_trajectory.py)
        require_cli_flag "$script" --include-truncated "$help_text"
        ;;
      *analyze_proxy_confidence.py)
        require_cli_flag "$script" --exclude-truncated "$help_text"
        require_cli_flag "$script" --include-truncated "$help_text"
        ;;
    esac
  done
  "$PYTHON_BIN" -m py_compile scripts/find_reusable_baseline.py \
    scripts/analyze_continuous_meta_behavior.py scripts/analyze_proxy_confidence.py \
    scripts/analyze_safety_baseline.py scripts/analyze_continuous_trajectory.py \
    scripts/analyze_random_orthogonal_trajectory.py
  echo "[preflight] shell syntax and Python interfaces are aligned"
}

if [[ "$INTERFACE_PREFLIGHT" == 1 ]]; then
  validate_interfaces
fi

mkdir -p "$RUN_ROOT/tasks" "$RUN_ROOT/status" "$RUN_ROOT/running" \
  "$RUN_ROOT/cache_initialized" "$RUN_ROOT/.locks"
MASTER_LOCK="$RUN_ROOT/.locks/master_queue.lock"
if ! acquire_task_lock "$MASTER_LOCK" "full-main-ablation:$RUN_ROOT"; then
  echo "Another launcher already owns this RUN_ROOT: $RUN_ROOT" >&2
  exit 17
fi

worker_pids=()
monitor_pid=""
cleanup() {
  touch "$RUN_ROOT/.queue.stop" 2>/dev/null || true
  local pid
  for pid in "${worker_pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  [[ -z "$monitor_pid" ]] || kill -TERM "$monitor_pid" 2>/dev/null || true
  release_task_locks
}
trap cleanup EXIT INT TERM

MANIFEST="$RUN_ROOT/task_manifest.tsv"
EXECUTION="$RUN_ROOT/execution_tasks.tsv"
QUEUE="$RUN_ROOT/execution_queue.tsv"
ALIASES="$RUN_ROOT/task_aliases.tsv"
EVENTS="$RUN_ROOT/task_events.log"
KEY_MAP="$RUN_ROOT/.canonical_task_keys.tsv"

printf 'logical_id\ttask_type\tdataset\ttarget\tsemantic_reference\tsemantic_layer\tlayer_fracs\tsupport\tsemantic_head\tsemantic_head_depth\tmodule_variant\tphase\tcanonical_task_id\texecution_mode\tgpu\toutput_dir\tconfig_file\tsingle_step_ablation\ttarget_generation_batch\tdownstream_generation_batch\n' > "$MANIFEST"
printf 'canonical_task_id\ttask_type\tdataset\ttarget\tsemantic_reference\tsemantic_layer\tlayer_fracs\tsupport\tsemantic_head\tsemantic_head_depth\tmodule_variant\tphase\tsingle_step_ablation\ttarget_generation_batch\tdownstream_generation_batch\tconfig_file\tgpu\tcache_tag\toutput_dir\n' > "$EXECUTION"
printf 'condition_key\tcanonical_task_id\n' > "$KEY_MAP"

logical_count=0
unique_count=0

target_gpu() {
  case "$1" in
    llama2_7b|llama31_8b|qwen25_7b|qwen3_8b|qwen3_14b) printf '%s\n' "$GPU0" ;;
    llama32_3b|deepseek_llama8b|deepseek_qwen7b|qwen3_4b|qwen3_32b) printf '%s\n' "$GPU1" ;;
    *)
      echo "No GPU assignment for target profile: $1" >&2
      return 2
      ;;
  esac
}

register_condition() {
  local logical_id="$1" task_type="$2" dataset="$3" target="$4"
  local reference="$5" semantic_layer="$6" layer_fracs="$7" support="$8"
  local head_type="$9" head_depth="${10}" config_file="${11}"
  local phase="${12:-core}" variant="${13:-learned}"
  local single_step="${14:-0}" target_batch="${15:-0}" generation_batch="${16:-0}"
  local key canonical mode gpu output cache_tag

  key="${dataset}|${target}|${reference}|${semantic_layer}|${layer_fracs}|${support}|${head_type}|${head_depth}|${variant}"
  gpu="$(target_gpu "$target")"
  logical_count=$((logical_count + 1))
  canonical="$(awk -F '\t' -v wanted="$key" 'NR > 1 && $1 == wanted {print $2; exit}' "$KEY_MAP")"
  if [[ -n "$canonical" ]]; then
    mode="reuse"
  else
    canonical="$logical_id"
    mode="execute"
    printf '%s\t%s\n' "$key" "$canonical" >> "$KEY_MAP"
    unique_count=$((unique_count + 1))
    output="$RUN_ROOT/tasks/$canonical"
    cache_tag="fullrerun_${dataset}_${target}_shared_layers_v1"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$canonical" "$task_type" "$dataset" "$target" "$reference" \
      "$semantic_layer" "$layer_fracs" "$support" "$head_type" \
      "$head_depth" "$variant" "$phase" "$single_step" "$target_batch" \
      "$generation_batch" "$config_file" "$gpu" "$cache_tag" "$output" \
      >> "$EXECUTION"
  fi
  output="$RUN_ROOT/tasks/$canonical"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$logical_id" "$task_type" "$dataset" "$target" "$reference" \
    "$semantic_layer" "$layer_fracs" "$support" "$head_type" \
    "$head_depth" "$variant" "$phase" "$canonical" "$mode" "$gpu" "$output" \
    "$config_file" "$single_step" "$target_batch" "$generation_batch" \
    >> "$MANIFEST"
}

create_source_snapshot() {
  local repro_dir="$RUN_ROOT/reproducibility"
  local snapshot_dir="$repro_dir/code_snapshot"
  local bundle="$repro_dir/source_bundle.tar.gz"
  local staging="$repro_dir/.code_snapshot.tmp.$$"
  local bundle_tmp="$repro_dir/.source_bundle.tar.gz.tmp.$$"
  local item

  mkdir -p "$repro_dir"
  if [[ -s "$snapshot_dir/.snapshot_complete" && -s "$bundle" ]]; then
    echo "[snapshot] reuse immutable source bundle: $bundle"
    return 0
  fi
  if [[ -e "$snapshot_dir" || -e "$bundle" ]]; then
    echo "[snapshot] incomplete prior snapshot found; preserve it with a timestamp" >&2
    [[ ! -e "$snapshot_dir" ]] || mv "$snapshot_dir" "${snapshot_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
    [[ ! -e "$bundle" ]] || mv "$bundle" "${bundle}.incomplete_$(date +%Y%m%d_%H%M%S)"
  fi

  mkdir -p "$staging/run_contract"
  cp -R scripts configs metacog "$staging/"
  for item in README.md requirements.txt pyproject.toml LICENSE CITATION.cff \
      CONTRIBUTING.md .editorconfig .gitignore; do
    [[ -f "$item" ]] && cp "$item" "$staging/"
  done
  cp "$MANIFEST" "$EXECUTION" "$QUEUE" "$ALIASES" \
    "$RUN_ROOT/task_types.tsv" "$RUN_ROOT/task_types.md" "$staging/run_contract/"
  find "$staging" -type d -name __pycache__ -prune -exec rm -rf {} +
  find "$staging" -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete
  {
    printf 'snapshot_created_at=%s\n' "$(date -Is 2>/dev/null || date)"
    printf 'source_root=%s\n' "$ROOT_DIR"
    printf 'hostname=%s\n' "$(hostname 2>/dev/null || echo unknown)"
    printf 'python=%s\n' "$($PYTHON_BIN --version 2>&1)"
    printf 'launcher=%s\n' "scripts/run_full_main_ablation_2gpu.sh"
  } > "$staging/runtime_environment.txt"
  "$PYTHON_BIN" - "$staging" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
excluded = {"SHA256SUMS", ".snapshot_complete"}
files = sorted(
    path for path in root.rglob("*")
    if path.is_file() and path.name not in excluded
)
with (root / "SHA256SUMS").open("w", encoding="utf-8") as output:
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        relative = path.relative_to(root).as_posix()
        output.write(f"{digest.hexdigest()}  {relative}\n")
PY
  printf '%s\n' "complete" > "$staging/.snapshot_complete"
  LC_ALL=C LANG=C tar -czf "$bundle_tmp" -C "$staging" .
  mv "$staging" "$snapshot_dir"
  mv "$bundle_tmp" "$bundle"
  LC_ALL=C LANG=C shasum -a 256 "$bundle" > "$repro_dir/source_bundle.sha256"
  echo "[snapshot] source bundle created: $bundle"
}

MAIN_TARGETS=(
  llama2_7b llama32_3b llama31_8b deepseek_llama8b
  qwen25_7b deepseek_qwen7b qwen3_4b qwen3_8b
)
STANDARD_FRACS="0.46875 0.50 0.875"
# Generated trajectories do not depend on the downstream layer topology.
# Capture this union once per dataset/target; every condition trains on its
# original three-layer subset from the shared cache.
SHARED_CAPTURE_FRACS="0.25 0.3125 0.375 0.40625 0.4375 0.46875 0.50 0.6875 0.75 0.875"
STANDARD_REFERENCE="qwen3_0p6b"
STANDARD_SEMANTIC_LAYER=14
STANDARD_SUPPORT=32
STANDARD_HEAD=mlp
STANDARD_HEAD_DEPTH=2

if [[ "$RUN_MAIN" == 1 ]]; then
  for dataset in ultrachat beavertails mathqa; do
    case "$dataset" in
      ultrachat) config="$ULTRACHAT_CONFIG" ;;
      beavertails) config="$BEAVERTAILS_CONFIG" ;;
      mathqa) config="$MATHQA_CONFIG" ;;
    esac
    for target in "${MAIN_TARGETS[@]}"; do
      single_step=0
      if [[ "$dataset" == ultrachat && ( "$target" == llama31_8b || "$target" == qwen3_4b ) ]]; then
        single_step=1
      fi
      register_condition "main_${dataset}_${target}" main "$dataset" "$target" \
        "$STANDARD_REFERENCE" "$STANDARD_SEMANTIC_LAYER" "$STANDARD_FRACS" \
        "$STANDARD_SUPPORT" "$STANDARD_HEAD" "$STANDARD_HEAD_DEPTH" "$config" \
        core learned "$single_step" 0 0
    done
  done
fi

ABLATION_TARGETS=(llama31_8b qwen3_4b)
if [[ "$RUN_SUPPORT_ABLATION" == 1 ]]; then
  for target in "${ABLATION_TARGETS[@]}"; do
    for support in 8 16 32 64 128; do
      register_condition "support_${target}_k${support}" support ultrachat "$target" \
        "$STANDARD_REFERENCE" "$STANDARD_SEMANTIC_LAYER" "$STANDARD_FRACS" \
        "$support" "$STANDARD_HEAD" "$STANDARD_HEAD_DEPTH" "$ULTRACHAT_CONFIG"
    done
  done
fi

HEADS=(
  "linear|1"
  "mlp|2"
  "deep_residual|4"
  "ensemble|4"
)
if [[ "$RUN_SEMANTIC_HEAD_ABLATION" == 1 ]]; then
  for target in "${ABLATION_TARGETS[@]}"; do
    for item in "${HEADS[@]}"; do
      IFS='|' read -r head_type head_depth <<< "$item"
      register_condition "semantic_head_${target}_${head_type}" semantic_head \
        ultrachat "$target" "$STANDARD_REFERENCE" "$STANDARD_SEMANTIC_LAYER" \
        "$STANDARD_FRACS" "$STANDARD_SUPPORT" "$head_type" "$head_depth" \
        "$ULTRACHAT_CONFIG"
    done
  done
fi

LAYER_TARGETS=(llama32_3b llama31_8b qwen3_4b qwen3_8b)
LAYER_TOPOLOGIES=(
  "early_early|0.25 0.3125 0.375"
  "middle_middle|0.40625 0.4375 0.50"
  "late_late|0.6875 0.75 0.875"
  "early_middle|0.25 0.3125 0.50"
  "middle_late|0.46875 0.50 0.875"
  "early_late|0.25 0.3125 0.875"
)
if [[ "$RUN_LAYER_ABLATION" == 1 ]]; then
  for target in "${LAYER_TARGETS[@]}"; do
    for item in "${LAYER_TOPOLOGIES[@]}"; do
      IFS='|' read -r topology layer_fracs <<< "$item"
      register_condition "layer_${target}_${topology}" layer ultrachat "$target" \
        "$STANDARD_REFERENCE" "$STANDARD_SEMANTIC_LAYER" "$layer_fracs" \
        "$STANDARD_SUPPORT" "$STANDARD_HEAD" "$STANDARD_HEAD_DEPTH" \
        "$ULTRACHAT_CONFIG"
    done
  done
fi

SEMANTIC_REFERENCES=(
  "qwen3_0p6b|14"
  "qwen3_1p7b|14"
  "gemma3_1b_it|13"
  "llama32_3b|14"
)
if [[ "$RUN_SEMANTIC_REFERENCE_ABLATION" == 1 ]]; then
  for target in "${ABLATION_TARGETS[@]}"; do
    for item in "${SEMANTIC_REFERENCES[@]}"; do
      IFS='|' read -r reference semantic_layer <<< "$item"
      register_condition "semantic_reference_${target}_${reference}" \
        semantic_reference ultrachat "$target" "$reference" "$semantic_layer" \
        "$STANDARD_FRACS" "$STANDARD_SUPPORT" "$STANDARD_HEAD" \
        "$STANDARD_HEAD_DEPTH" "$ULTRACHAT_CONFIG"
    done
  done
fi

# Causal ablation. The learned-module rows intentionally alias the two
# UltraChat main tasks, which already run matched single-step and persistent
# trajectories. The random-support rows are new training/intervention tasks.
if [[ "$RUN_CAUSAL_ABLATION" == 1 ]]; then
  for target in "${ABLATION_TARGETS[@]}"; do
    register_condition "causal_real_${target}" causal ultrachat "$target" \
      "$STANDARD_REFERENCE" "$STANDARD_SEMANTIC_LAYER" "$STANDARD_FRACS" \
      "$STANDARD_SUPPORT" "$STANDARD_HEAD" "$STANDARD_HEAD_DEPTH" \
      "$ULTRACHAT_CONFIG" causal learned 1 0 0
    register_condition "causal_random_support_${target}" causal ultrachat "$target" \
      "$STANDARD_REFERENCE" "$STANDARD_SEMANTIC_LAYER" "$STANDARD_FRACS" \
      "$STANDARD_SUPPORT" "$STANDARD_HEAD" "$STANDARD_HEAD_DEPTH" \
      "$ULTRACHAT_CONFIG" causal random_support 1 0 0
  done
fi

# Run the large Qwen scale extension only after all core and causal tasks.
# SCALE_DATASETS defaults to UltraChat; it can be explicitly widened without
# changing the registered model/batch contract.
if [[ "$RUN_SCALE_EXTENSION" == 1 ]]; then
  for dataset in $SCALE_DATASETS; do
    case "$dataset" in
      ultrachat) config="$ULTRACHAT_CONFIG" ;;
      beavertails) config="$BEAVERTAILS_CONFIG" ;;
      mathqa) config="$MATHQA_CONFIG" ;;
      *) echo "Unsupported SCALE_DATASETS entry: $dataset" >&2; exit 2 ;;
    esac
    register_condition "scale_${dataset}_qwen3_14b" scale_extension "$dataset" \
      qwen3_14b "$STANDARD_REFERENCE" "$STANDARD_SEMANTIC_LAYER" "$STANDARD_FRACS" \
      "$STANDARD_SUPPORT" "$STANDARD_HEAD" "$STANDARD_HEAD_DEPTH" "$config" \
      scale learned 0 2 2
    register_condition "scale_${dataset}_qwen3_32b" scale_extension "$dataset" \
      qwen3_32b "$STANDARD_REFERENCE" "$STANDARD_SEMANTIC_LAYER" "$STANDARD_FRACS" \
      "$STANDARD_SUPPORT" "$STANDARD_HEAD" "$STANDARD_HEAD_DEPTH" "$config" \
      scale learned 0 1 1
  done
fi

awk -F '\t' 'NR == 1 || $14 == "reuse"' "$MANIFEST" > "$ALIASES"
cp "$EXECUTION" "$QUEUE"
if [[ "$MAX_TASKS" =~ ^[0-9]+$ ]] && (( MAX_TASKS > 0 && MAX_TASKS < unique_count )); then
  awk -v limit="$MAX_TASKS" 'NR == 1 || (NR > 1 && NR <= limit + 1)' \
    "$EXECUTION" > "$QUEUE"
fi
queue_total=$(( $(wc -l < "$QUEUE") - 1 ))

# Materialize disjoint worker queues before launching anything. Each canonical
# task row exists in exactly one (phase, GPU) shard, so workers never scan or
# filter the same source queue concurrently.
SHARD_DIR="$RUN_ROOT/queue_shards"
mkdir -p "$SHARD_DIR"
for phase_name in core causal scale; do
  for gpu_name in "$GPU0" "$GPU1"; do
    shard="$SHARD_DIR/${phase_name}_gpu_${gpu_name}.tsv"
    awk -F '\t' -v wanted_phase="$phase_name" -v wanted_gpu="$gpu_name" \
      'NR == 1 || ($12 == wanted_phase && $17 == wanted_gpu)' "$QUEUE" > "$shard"
  done
done
"$PYTHON_BIN" - "$QUEUE" "$SHARD_DIR" "$GPU0" "$GPU1" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

queue_path, shard_root, gpu0, gpu1 = sys.argv[1:]
with open(queue_path, encoding="utf-8", newline="") as handle:
    expected = [row["canonical_task_id"] for row in csv.DictReader(handle, delimiter="\t")]
observed = []
for phase in ("core", "causal", "scale"):
    for gpu in (gpu0, gpu1):
        path = Path(shard_root) / f"{phase}_gpu_{gpu}.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in rows:
            assert row["phase"] == phase, (path, row["canonical_task_id"], row["phase"])
            assert row["gpu"] == gpu, (path, row["canonical_task_id"], row["gpu"])
            observed.append(row["canonical_task_id"])
assert Counter(observed) == Counter(expected), {
    "missing": sorted((Counter(expected) - Counter(observed)).elements()),
    "duplicated": sorted((Counter(observed) - Counter(expected)).elements()),
}
assert len(observed) == len(set(observed)), "A canonical task appears in more than one worker shard."
print(f"[queue] exact-once shard validation passed: {len(observed)} tasks")
PY

printf 'task_type\tlogical_conditions\tunique_execution_tasks\tdescription\n' > "$RUN_ROOT/task_types.tsv"
write_task_type() {
  local task_type="$1" description="$2" logical unique
  logical="$(awk -F '\t' -v kind="$task_type" 'NR > 1 && $2 == kind {n++} END {print n+0}' "$MANIFEST")"
  unique="$(awk -F '\t' -v kind="$task_type" 'NR > 1 && $2 == kind {n++} END {print n+0}' "$EXECUTION")"
  printf '%s\t%s\t%s\t%s\n' "$task_type" "$logical" "$unique" "$description" \
    >> "$RUN_ROOT/task_types.tsv"
}
write_task_type main "Three datasets, eight target models, Qwen3-0.6B semantic reference, support 32."
write_task_type support "Two target models with support 8, 16, 32, 64, and 128."
write_task_type semantic_head "Two target models with linear, MLP, deep-residual, and ensemble semantic heads."
write_task_type layer "Four target models with six early/middle/late layer topologies."
write_task_type semantic_reference "Two target models with four external semantic reference models."
write_task_type causal "Matched single-step versus persistent intervention, learned modules versus random-support modules, and norm-matched random directions."
write_task_type scale_extension "Qwen3-14B and Qwen3-32B main conditions, executed after all other phases."
{
  printf '# Experiment task types\n\n'
  printf 'Logical conditions: %s  \nUnique executable tasks after exact baseline reuse: %s  \nQueued in this launch: %s\n\n' \
    "$logical_count" "$unique_count" "$queue_total"
  printf '| Task type | Logical | Unique | Description |\n'
  printf '|---|---:|---:|---|\n'
  tail -n +2 "$RUN_ROOT/task_types.tsv" | while IFS=$'\t' read -r type logical unique description; do
    printf '| %s | %s | %s | %s |\n' "$type" "$logical" "$unique" "$description"
  done
  printf '\n`task_manifest.tsv` records all logical conditions; `task_aliases.tsv` records reused baselines; `execution_queue.tsv` is the actual GPU queue.\n'
} > "$RUN_ROOT/task_types.md"

if [[ "$SNAPSHOT_SOURCE" == 1 ]]; then
  create_source_snapshot
fi

printf '[queue] run_root=%s logical=%s unique=%s queued=%s gpu0=%s gpu1=%s\n' \
  "$RUN_ROOT" "$logical_count" "$unique_count" "$queue_total" "$GPU0" "$GPU1" \
  | tee -a "$EVENTS"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "[queue] DRY_RUN=1; manifests were created and no task was launched."
  exit 0
fi
(( queue_total > 0 )) || { echo "No executable tasks were registered." >&2; exit 2; }

prepare_shared_dataset() (
  local config_file="$1"
  set -a
  # shellcheck source=/dev/null
  source "$config_file"
  set +a
  if [[ ! -s "$BASE_DATA_PATH" ]]; then
    read -r -a data_sources <<< "$DATA_SOURCE"
    echo "[data-preflight] prepare dataset=$DATASET_PROFILE output=$DATA_PREPARED_PATH"
    "$PYTHON_BIN" -m metacog.cli dataset prepare "$DATASET_PROFILE" \
      --input "${data_sources[@]}" --output "$DATA_PREPARED_PATH" \
      --max-samples "${DATA_MAX_SAMPLES:-$TARGET_PROMPT_MAX_SAMPLES}" \
      --split-mode "${DATA_SPLIT_MODE:-resplit}" \
      --train-size "${DATA_TRAIN_SIZE:-22000}" \
      --validation-size "${DATA_VALIDATION_SIZE:-7000}" --seed "$SEED"
  else
    echo "[data-preflight] reuse dataset=$DATASET_PROFILE path=$BASE_DATA_PATH"
  fi
  "$PYTHON_BIN" -m metacog.cli dataset validate "$DATASET_PROFILE" \
    "$BASE_DATA_PATH" --min-samples "$TARGET_PROMPT_MAX_SAMPLES"
)

if [[ "$DATA_PREFLIGHT" == 1 ]]; then
  for dataset in ultrachat beavertails mathqa; do
    if ! awk -F '\t' -v wanted="$dataset" 'NR > 1 && $3 == wanted {found=1} END {exit(found ? 0 : 1)}' "$QUEUE"; then
      continue
    fi
    case "$dataset" in
      ultrachat) prepare_shared_dataset "$ULTRACHAT_CONFIG" ;;
      beavertails) prepare_shared_dataset "$BEAVERTAILS_CONFIG" ;;
      mathqa) prepare_shared_dataset "$MATHQA_CONFIG" ;;
    esac
  done
fi

task_has_completed_output() {
  local output="$1" require_single_step="${2:-0}"
  local selected_file expected actual
  [[ -s "$output/final_status.tsv" ]] || return 1
  awk -F '\t' 'NR > 1 && $1 == "complete" && $3 == "0" {ok=1} END {exit(ok ? 0 : 1)}' \
    "$output/final_status.tsv" || return 1
  if [[ "$require_single_step" == 1 ]]; then
    selected_file="$(find "$output" -path '*/adaptive_selection/selected_adaptive_modules.tsv' -type f -size +0c -print -quit 2>/dev/null || true)"
    [[ -n "$selected_file" ]] || return 1
    expected="$(awk 'NR > 1 {n++} END {print n+0}' "$selected_file")"
    actual="$(find "$output" -path '*/intervention_horizon_ablation/intervention_horizon_summary.json' -type f -size +0c -print 2>/dev/null | wc -l | tr -d ' ')"
    (( expected > 0 && actual >= expected )) || return 1
  fi
}

write_status() {
  local task_id="$1" state="$2" gpu="$3" started="$4" ended="$5"
  local duration="$6" rc="$7" note="$8"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$state" "$gpu" "$started" "$ended" "$duration" "$rc" "$note" "$task_id" \
    > "$RUN_ROOT/status/${task_id}.status.tsv"
}

run_task() {
  local task_id="$1" task_type="$2" dataset="$3" target="$4" reference="$5"
  local semantic_layer="$6" layer_fracs="$7" support="$8" head_type="$9"
  local head_depth="${10}" variant="${11}" phase="${12}" single_step="${13}"
  local target_batch="${14}" generation_batch="${15}" config_file="${16}"
  local gpu="${17}" cache_tag="${18}" output="${19}"
  local started ended duration rc activation_force target_batch_override generation_batch_override
  local module_mode random_support_null controls_override dose_match_override target_cache_fracs
  local running_file="$RUN_ROOT/running/${task_id}.running" launch_id

  mkdir -p "$output"
  if [[ "$FORCE" != 1 ]] && task_has_completed_output "$output" "$single_step"; then
    started="$(date +%s)"
    write_status "$task_id" complete "$gpu" "$started" "$started" 0 0 reused
    printf '[%s] REUSE task=%s gpu=%s output=%s\n' "$(date '+%F %T')" \
      "$task_id" "$gpu" "$output" | tee -a "$EVENTS"
    return 0
  fi

  started="$(date +%s)"
  launch_id="${task_id}__${started}__pid${BASHPID}"
  printf '%s\t%s\t%s\t%s\n' "$gpu" "$started" "$task_type" "$launch_id" > "$running_file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is 2>/dev/null || date)" "$launch_id" "$task_id" "$gpu" "$BASHPID" "$PPID" \
    >> "$RUN_ROOT/task_launches.tsv"
  printf '[%s] START task=%s type=%s gpu=%s launch_id=%s worker_pid=%s\n' "$(date '+%F %T')" \
    "$task_id" "$task_type" "$gpu" "$launch_id" "$BASHPID" | tee -a "$EVENTS"

  activation_force=0
  if [[ "$FORCE_ACTIVATIONS" == 1 && ! -s "$RUN_ROOT/cache_initialized/${cache_tag}.done" ]]; then
    activation_force=1
  fi
  target_batch_override=""
  generation_batch_override=""
  (( target_batch > 0 )) && target_batch_override="$target_batch"
  (( generation_batch > 0 )) && generation_batch_override="$generation_batch"
  module_mode=learned
  random_support_null=0
  if [[ "$variant" == random_support ]]; then
    module_mode=random_support
    random_support_null=1
  fi
  controls_override=""
  dose_match_override=""
  target_cache_fracs="$layer_fracs"
  if [[ "$dataset" == ultrachat ]]; then
    case "$target" in
      llama32_3b|llama31_8b|qwen3_4b|qwen3_8b)
        target_cache_fracs="$SHARED_CAPTURE_FRACS"
        ;;
    esac
  fi
  if [[ "$single_step" == 1 ]]; then
    controls_override="opposite_direction random_hidden_direction"
    dose_match_override=1
  fi
  set +e
  env GPU0="$gpu" GPU1="$gpu" RUN_ROOT="$output" CONFIG_FILE="$config_file" \
    FORCE="$FORCE" FORCE_ACTIVATIONS="$activation_force" \
    PAIR_SPECS_OVERRIDE="${target}:${reference}:${semantic_layer}" \
    LAYER_FRACS_OVERRIDE="$layer_fracs" SUPPORTS_OVERRIDE="$support" \
    TARGET_CACHE_LAYER_FRACS_OVERRIDE="$target_cache_fracs" \
    EXTERNAL_SEMANTIC_HEAD_TYPE_OVERRIDE="$head_type" \
    EXTERNAL_SEMANTIC_HEAD_DEPTH_OVERRIDE="$head_depth" \
    TARGET_EXTRACTION_BATCH_SIZE_OVERRIDE="$target_batch_override" \
    GENERATION_BATCH_SIZE_OVERRIDE="$generation_batch_override" \
    RUN_SINGLE_STEP_TRAJECTORY_OVERRIDE="$single_step" \
    MODULE_DIRECTION_MODE_OVERRIDE="$module_mode" \
    RANDOM_SUPPORT_NULL_OVERRIDE="$random_support_null" \
    RANDOM_SUPPORT_SEED_OVERRIDE=777 \
    CONTINUOUS_CONTROLS_OVERRIDE="$controls_override" \
    REFINED_CODE_DOSE_MATCH_CONTROLS_OVERRIDE="$dose_match_override" \
    ACTIVATION_CACHE_TAG="$cache_tag" \
    SHARED_BASELINE_ROOT="$RUN_ROOT/shared_resources/baselines" \
    REUSE_HISTORICAL_BASELINES=1 \
    TASK_LAUNCH_ID="$launch_id" \
    bash scripts/run_generated_step_experiment.sh \
    > "$output/task.log" 2>&1
  rc=$?
  set -e
  ended="$(date +%s)"
  duration=$((ended - started))
  rm -f "$running_file"
  if [[ "$rc" == 0 ]]; then
    touch "$RUN_ROOT/cache_initialized/${cache_tag}.done"
    write_status "$task_id" complete "$gpu" "$started" "$ended" "$duration" 0 executed
    printf '[%s] COMPLETE task=%s gpu=%s duration=%ss\n' "$(date '+%F %T')" \
      "$task_id" "$gpu" "$duration" | tee -a "$EVENTS"
  else
    write_status "$task_id" failed "$gpu" "$started" "$ended" "$duration" "$rc" executed
    printf '[%s] FAILED task=%s gpu=%s rc=%s duration=%ss log=%s\n' \
      "$(date '+%F %T')" "$task_id" "$gpu" "$rc" "$duration" "$output/task.log" \
      | tee -a "$EVENTS" >&2
  fi
  return 0
}

worker() {
  local assigned_gpu="$1" assigned_phase="$2" shard_file="$3"
  local task_id task_type dataset target reference semantic_layer layer_fracs
  local support head_type head_depth variant phase single_step target_batch
  local generation_batch config_file gpu cache_tag output
  # The master owns the queue lock. Background workers must not inherit the
  # master's cleanup trap and release that lock when their own queue finishes.
  trap - EXIT INT TERM
  while IFS=$'\t' read -r task_id task_type dataset target reference semantic_layer \
      layer_fracs support head_type head_depth variant phase single_step target_batch \
      generation_batch config_file gpu cache_tag output; do
    [[ "$task_id" == canonical_task_id ]] && continue
    [[ "$gpu" == "$assigned_gpu" && "$phase" == "$assigned_phase" ]] || {
      echo "[queue] invalid row in exclusive shard: task=$task_id row_gpu=$gpu row_phase=$phase expected_gpu=$assigned_gpu expected_phase=$assigned_phase" >&2
      return 2
    }
    run_task "$task_id" "$task_type" "$dataset" "$target" "$reference" \
      "$semantic_layer" "$layer_fracs" "$support" "$head_type" "$head_depth" "$variant" \
      "$phase" "$single_step" "$target_batch" "$generation_batch" "$config_file" \
      "$gpu" "$cache_tag" "$output"
  done < "$shard_file"
}

format_duration() {
  local total="$1" days hours minutes
  if (( total < 0 )); then printf 'collecting-runtime-samples'; return; fi
  days=$((total / 86400)); hours=$(((total % 86400) / 3600)); minutes=$(((total % 3600) / 60))
  if (( days > 0 )); then printf '%dd%02dh%02dm' "$days" "$hours" "$minutes"
  elif (( hours > 0 )); then printf '%dh%02dm' "$hours" "$minutes"
  else printf '%dm' "$minutes"
  fi
}

collect_progress() {
  local total="$queue_total" terminal=0 success=0 failed=0 running=0 pending
  local duration_sum=0 duration_n=0 state gpu started ended duration rc note task_id file
  local now elapsed average eta waves active=""
  for file in "$RUN_ROOT"/status/*.status.tsv; do
    [[ -e "$file" ]] || continue
    IFS=$'\t' read -r state gpu started ended duration rc note task_id < "$file"
    terminal=$((terminal + 1))
    if [[ "$state" == complete ]]; then success=$((success + 1)); else failed=$((failed + 1)); fi
    if [[ "$duration" =~ ^[0-9]+$ ]] && (( duration > 0 )); then
      duration_sum=$((duration_sum + duration)); duration_n=$((duration_n + 1))
    fi
  done
  for file in "$RUN_ROOT"/running/*.running; do
    [[ -e "$file" ]] || continue
    running=$((running + 1))
    task_id="$(basename "$file" .running)"
    active="${active}${active:+,}${task_id}"
  done
  pending=$((total - terminal - running)); (( pending < 0 )) && pending=0
  now="$(date +%s)"; elapsed=$((now - QUEUE_STARTED_AT))
  eta=-1
  if (( duration_n > 0 )); then
    average=$((duration_sum / duration_n))
    waves=$(((total - terminal + 1) / 2))
    eta=$((average * waves))
  fi
  progress_line="[$(date '+%F %T')] progress=${terminal}/${total} success=${success} failed=${failed} running=${running} pending=${pending} eta=$(format_duration "$eta") active=${active:-none}"
  printf '%s\n' "$progress_line"
  printf '%s\n' "$progress_line" > "$RUN_ROOT/progress_current.txt"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is 2>/dev/null || date)" "$terminal" "$total" "$success" "$failed" \
    "$running" "$pending" "$eta" "$elapsed" >> "$RUN_ROOT/progress.tsv"
}

monitor_progress() {
  local waited
  trap - EXIT INT TERM
  collect_progress
  while [[ ! -e "$RUN_ROOT/.queue.stop" ]]; do
    waited=0
    while (( waited < PROGRESS_INTERVAL )); do
      [[ -e "$RUN_ROOT/.queue.stop" ]] && break
      sleep 1
      waited=$((waited + 1))
    done
    collect_progress
  done
}

rm -f "$RUN_ROOT/.queue.stop"
printf 'timestamp\tterminal\ttotal\tsuccess\tfailed\trunning\tpending\teta_seconds\telapsed_seconds\n' \
  > "$RUN_ROOT/progress.tsv"
printf 'timestamp\tlaunch_id\ttask_id\tgpu\tworker_pid\tparent_pid\n' \
  > "$RUN_ROOT/task_launches.tsv"
QUEUE_STARTED_AT="$(date +%s)"
export QUEUE_STARTED_AT

monitor_progress & monitor_pid="$!"
for phase in core causal scale; do
  if ! awk -F '\t' -v wanted="$phase" 'NR > 1 && $12 == wanted {found=1} END {exit(found ? 0 : 1)}' "$QUEUE"; then
    continue
  fi
  printf '[%s] PHASE START %s\n' "$(date '+%F %T')" "$phase" | tee -a "$EVENTS"
  worker_pids=()
  shard0="$SHARD_DIR/${phase}_gpu_${GPU0}.tsv"
  shard1="$SHARD_DIR/${phase}_gpu_${GPU1}.tsv"
  if (( $(wc -l < "$shard0") > 1 )); then
    worker "$GPU0" "$phase" "$shard0" & worker_pids+=("$!")
  fi
  if (( $(wc -l < "$shard1") > 1 )); then
    worker "$GPU1" "$phase" "$shard1" & worker_pids+=("$!")
  fi
  for pid in "${worker_pids[@]}"; do
    wait "$pid" || true
  done
  worker_pids=()
  printf '[%s] PHASE COMPLETE %s\n' "$(date '+%F %T')" "$phase" | tee -a "$EVENTS"
done
touch "$RUN_ROOT/.queue.stop"
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=""

printf 'task_id\tstate\tgpu\tstart_epoch\tend_epoch\tduration_seconds\texit_code\tnote\n' \
  > "$RUN_ROOT/final_status.tsv"
for file in "$RUN_ROOT"/status/*.status.tsv; do
  [[ -e "$file" ]] || continue
  IFS=$'\t' read -r state gpu started ended duration rc note task_id < "$file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$task_id" "$state" "$gpu" "$started" "$ended" "$duration" "$rc" "$note" \
    >> "$RUN_ROOT/final_status.tsv"
done
failed_count="$(awk -F '\t' 'NR > 1 && $2 == "failed" {n++} END {print n+0}' "$RUN_ROOT/final_status.tsv")"
complete_count="$(awk -F '\t' 'NR > 1 && $2 == "complete" {n++} END {print n+0}' "$RUN_ROOT/final_status.tsv")"
printf '[queue] finished complete=%s failed=%s total=%s root=%s\n' \
  "$complete_count" "$failed_count" "$queue_total" "$RUN_ROOT" | tee -a "$EVENTS"
(( failed_count == 0 && complete_count == queue_total ))
