#!/usr/bin/env bash

# Small, dependency-free inter-process lock for the shell launchers.
# mkdir is atomic on the local filesystem, unlike a check-then-create test.

TASK_LOCK_DIRS=()

_task_lock_now() {
  date +%s
}

_task_lock_mtime() {
  local path="$1"
  stat -c %Y "$path" 2>/dev/null || stat -f %m "$path" 2>/dev/null || echo 0
}

_task_lock_live_pid() {
  local path="$1" pid
  [[ -s "$path/owner" ]] || return 1
  pid="$(awk -F= '$1 == "pid" {print $2; exit}' "$path/owner" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

acquire_task_lock() {
  local lock_dir="$1" label="${2:-task}"
  local stale_seconds="${TASK_LOCK_STALE_SECONDS:-300}"
  local mtime now age

  mkdir -p "$(dirname "$lock_dir")"
  if mkdir "$lock_dir" 2>/dev/null; then
    {
      printf 'pid=%s\n' "$$"
      printf 'host=%s\n' "$(hostname 2>/dev/null || echo unknown)"
      printf 'started=%s\n' "$(date -Is 2>/dev/null || date)"
      printf 'label=%s\n' "$label"
    } > "$lock_dir/owner"
    TASK_LOCK_DIRS+=("$lock_dir")
    echo "[lock] acquired $label: $lock_dir"
    return 0
  fi

  if _task_lock_live_pid "$lock_dir"; then
    local owner_pid
    owner_pid="$(awk -F= '$1 == "pid" {print $2; exit}' "$lock_dir/owner" 2>/dev/null || true)"
    echo "[lock] active task already owns $lock_dir (pid=$owner_pid); skip duplicate $label" >&2
    return 1
  fi

  # Do not remove a just-created lock whose owner file has not been written
  # yet. Old locks with dead owners are recoverable after this grace period.
  mtime="$(_task_lock_mtime "$lock_dir")"
  now="$(_task_lock_now)"
  age=$((now - mtime))
  if (( age < stale_seconds )); then
    echo "[lock] lock exists without a confirmed live owner: $lock_dir; skip duplicate $label" >&2
    return 1
  fi
  echo "[lock] removing stale lock (age=${age}s): $lock_dir" >&2
  rm -rf "$lock_dir"
  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "[lock] another process acquired $lock_dir during stale-lock recovery" >&2
    return 1
  fi
  {
    printf 'pid=%s\n' "$$"
    printf 'host=%s\n' "$(hostname 2>/dev/null || echo unknown)"
    printf 'started=%s\n' "$(date -Is 2>/dev/null || date)"
    printf 'label=%s\n' "$label"
  } > "$lock_dir/owner"
  TASK_LOCK_DIRS+=("$lock_dir")
  echo "[lock] acquired after stale recovery $label: $lock_dir"
}

release_task_locks() {
  local lock_dir
  for lock_dir in "${TASK_LOCK_DIRS[@]:-}"; do
    [[ -d "$lock_dir" ]] || continue
    if _task_lock_live_pid "$lock_dir"; then
      rm -rf "$lock_dir"
      echo "[lock] released $lock_dir"
    fi
  done
  TASK_LOCK_DIRS=()
}
