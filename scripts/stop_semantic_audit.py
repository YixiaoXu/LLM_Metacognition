#!/usr/bin/env python
"""Stop only a named semantic-audit run on Linux; dry-run unless --apply."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import signal
import time


SCRIPTS = {"run_semantic_leakage_audit.py", "run_semantic_witness.py"}


def matches_run(argv: list[str], cwd: Path, repo: Path, run_root: Path) -> bool:
    scripts = []
    for arg in argv:
        if Path(arg).name in SCRIPTS:
            scripts.append((cwd / arg).resolve() if not Path(arg).is_absolute() else Path(arg).resolve())
    if not any(script == repo / "scripts" / script.name for script in scripts):
        return False
    if "--output-dir" in argv:
        i = argv.index("--output-dir")
        if i + 1 >= len(argv):
            return False
        value = argv[i + 1]
    else:
        value = next((arg.split("=", 1)[1] for arg in argv if arg.startswith("--output-dir=")), None)
    return value is not None and (cwd / value).resolve() == run_root


def process(pid: int) -> dict | None:
    try:
        directory = Path("/proc") / str(pid)
        if directory.stat().st_uid != os.getuid():
            return None
        # The comm field can contain spaces/parentheses; parse after its final ')'.
        stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {"pid": pid, "ppid": int(stat[1]), "pgrp": int(stat[2]), "start": stat[19],
                "argv": (directory / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0"),
                "cwd": (directory / "cwd").resolve(strict=True)}
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--apply", action="store_true", help="Send SIGTERM after verifying exact script and run path.")
    parser.add_argument("--timeout", type=float, default=30.)
    args = parser.parse_args()
    if not Path("/proc/self/stat").exists():
        parser.error("Run this command on the Linux experiment server, not on the laptop.")
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.run_root).resolve()
    if not root.is_dir() or not any((root / name).exists() for name in ("audit_manifest.json", "witness_manifest.json")):
        parser.error("--run-root must identify an existing semantic audit/witness run with its manifest.")
    if args.timeout < 0:
        parser.error("--timeout must be nonnegative.")
    all_processes = {int(p.name): process(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdigit()}
    all_processes = {pid: p for pid, p in all_processes.items() if p is not None}
    matched = {pid: p for pid, p in all_processes.items() if matches_run(p["argv"], p["cwd"], repo, root)}
    # Snapshot descendants too, so a launcher that exits first cannot leave its
    # extractor/probe process behind. Never signal a shell or an unrelated job.
    selected = dict(matched)
    while True:
        children = {pid: p for pid, p in all_processes.items() if p["ppid"] in selected and pid not in selected}
        if not children:
            break
        selected.update(children)
    for p in selected.values():
        print(f"[stop-audit] {'MATCH' if p['pid'] in matched else 'CHILD'} pid={p['pid']} {shlex.join(p['argv'])}")
    if not selected:
        print(f"[stop-audit] No running processes for {root}")
        return
    if not args.apply:
        print(f"[stop-audit] DRY RUN: {len(selected)} processes; add --apply to send SIGTERM.")
        return
    # Parent launchers stop queueing new work; their own handlers also terminate
    # workers' process groups. PID start times protect against stale PID reuse.
    for pid, old in selected.items():
        current = process(pid)
        if current and current["start"] == old["start"]:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + args.timeout
    while True:
        remaining = [pid for pid, old in selected.items()
                     if (p := process(pid)) is not None and p["start"] == old["start"]]
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(.25)
    if remaining:
        print(f"[stop-audit] Still alive after SIGTERM: {remaining}; inspect before any forced termination.")
        raise SystemExit(1)
    print(f"[stop-audit] Stopped {len(selected)} matched processes. Existing results were not removed.")


if __name__ == "__main__":
    main()
