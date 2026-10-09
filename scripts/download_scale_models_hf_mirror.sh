#!/usr/bin/env bash
set -Eeuo pipefail

# Resumable serial downloader for the model-scale experiment.
# Run from the repository root or override MODEL_ROOT/HF_ENDPOINT.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$MODEL_ROOT}"
LOG_DIR="${LOG_DIR:-$MODEL_ROOT/logs/model_downloads}"
STATUS_FILE="${STATUS_FILE:-$LOG_DIR/status.tsv}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
FORCE="${FORCE:-0}"

mkdir -p "$DOWNLOAD_ROOT" "$LOG_DIR"
export HF_ENDPOINT

if command -v hf >/dev/null 2>&1; then
  DOWNLOADER="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  DOWNLOADER="huggingface-cli"
else
  echo "Missing Hugging Face CLI. Install with: pip install -U huggingface_hub" >&2
  exit 127
fi

# name|repository|local directory
MODELS=(
  "qwen3_0.6b|Qwen/Qwen3-0.6B|Qwen3-0.6B"
  "qwen3_1.7b|Qwen/Qwen3-1.7B|Qwen3-1.7B"
  "qwen3_4b|Qwen/Qwen3-4B|Qwen3-4B"
  "qwen3_8b|Qwen/Qwen3-8B|Qwen3-8B"
  "qwen3_14b|Qwen/Qwen3-14B|Qwen3-14B"
  "qwen3_32b|Qwen/Qwen3-32B|Qwen3-32B"
  "gemma3_1b_it|unsloth/gemma-3-1b-it|gemma-3-1b-it"
  "gemma3_4b_it|unsloth/gemma-3-4b-it|gemma-3-4b-it"
  "gemma3_12b_it|unsloth/gemma-3-12b-it|gemma-3-12b-it"
)

if [[ ! -f "$STATUS_FILE" || "$FORCE" == "1" ]]; then
  printf 'timestamp\tname\trepository\tstatus\tdestination\n' > "$STATUS_FILE"
fi

download_one() {
  local name="$1"
  local repo="$2"
  local dirname="$3"
  local dest="$DOWNLOAD_ROOT/$dirname"
  local log="$LOG_DIR/${name}.log"
  local marker="$dest/.download_complete"
  local started
  started="$(date '+%Y-%m-%d %H:%M:%S')"

  if [[ "$FORCE" != "1" && -f "$marker" ]]; then
    printf '%s\t%s\t%s\tSKIP\t%s\n' "$started" "$name" "$repo" "$dest" >> "$STATUS_FILE"
    echo "[$started] SKIP $name (completion marker exists)"
    return 0
  fi

  mkdir -p "$dest"
  echo "[$started] START $name repo=$repo dest=$dest endpoint=$HF_ENDPOINT"
  printf '%s\t%s\t%s\tSTART\t%s\n' "$started" "$name" "$repo" "$dest" >> "$STATUS_FILE"

  set +e
  if [[ "$DOWNLOADER" == "hf" ]]; then
    # Current `hf download` rejects --local-dir together with --cache-dir.
    # The local-dir metadata is sufficient for resuming partial downloads.
    hf download "$repo" \
      --local-dir "$dest" \
      > >(tee -a "$log") 2>&1
  else
    huggingface-cli download "$repo" \
      --local-dir "$dest" \
      --resume-download \
      > >(tee -a "$log") 2>&1
  fi
  local rc=$?
  set -e

  local finished
  finished="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ "$rc" -eq 0 ]]; then
    touch "$marker"
    printf '%s\t%s\t%s\tDONE\t%s\n' "$finished" "$name" "$repo" "$dest" >> "$STATUS_FILE"
    echo "[$finished] DONE $name"
  else
    printf '%s\t%s\t%s\tFAILED_${rc}\t%s\n' "$finished" "$name" "$repo" "$dest" >> "$STATUS_FILE"
    echo "[$finished] FAILED $name rc=$rc; partial files are kept for resume"
  fi
  return "$rc"
}

failures=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r name repo dirname <<< "$spec"
  if ! download_one "$name" "$repo" "$dirname"; then
    failures=$((failures + 1))
  fi
done

echo
echo "Download queue finished: failures=$failures"
echo "Status: $STATUS_FILE"
echo "Logs:   $LOG_DIR"
exit "$failures"
