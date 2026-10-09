#!/usr/bin/env python3
"""Normalize mixed stdout/tqdm streams for readable task logs.

Carriage-return progress bars work in a TTY but become thousands of historical
updates after ``tee`` and worker-prefix pipelines.  This filter treats both
newlines and carriage returns as records, emits ordinary messages immediately,
and throttles progress-bar snapshots to a configurable interval.
"""

from __future__ import annotations

import argparse
import codecs
import os
import re
import sys
import time
from pathlib import Path
from typing import TextIO


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_RE = re.compile(r"(?:^|\s)\d{1,3}%\|.*\|\s*\d+(?:/|\s+of\s+)\d+")


def clean(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\x00", "").strip()


def is_progress(text: str) -> bool:
    return bool(PROGRESS_RE.search(text))


def progress_key(text: str) -> str:
    before_percent = re.split(r"\s*\d{1,3}%\|", text, maxsplit=1)[0]
    return before_percent.rsplit("\n", 1)[-1][-120:]


class StreamWriter:
    def __init__(
        self,
        prefix: str,
        log_file: str,
        progress_interval: float,
    ) -> None:
        self.prefix = prefix.strip()
        self.progress_interval = max(float(progress_interval), 0.0)
        self.last_progress_at: dict[str, float] = {}
        self.last_progress_text: dict[str, str] = {}
        self.log_handle: TextIO | None = None
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.log_handle = path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        if self.log_handle is not None:
            self.log_handle.close()

    def emit(self, text: str, progress: bool = False) -> None:
        text = clean(text)
        if not text:
            return
        if self.prefix:
            text = f"{self.prefix} {text}"
        if progress and not text.startswith("[progress]"):
            text = f"[progress] {text}"
        line = text + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        if self.log_handle is not None:
            self.log_handle.write(line)
            self.log_handle.flush()

    def handle(self, raw: str, final: bool = False) -> None:
        text = clean(raw)
        if not text:
            return
        if not is_progress(text):
            self.emit(text)
            return
        key = progress_key(text)
        now = time.monotonic()
        complete = "100%|" in text
        changed = self.last_progress_text.get(key) != text
        due = now - self.last_progress_at.get(key, -1e12) >= self.progress_interval
        if complete or final or (changed and due):
            self.emit(text, progress=True)
            self.last_progress_at[key] = now
            self.last_progress_text[key] = text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="")
    parser.add_argument("--log-file", default="")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=float(os.environ.get("LOG_PROGRESS_INTERVAL_SECONDS", "60")),
    )
    args = parser.parse_args()

    writer = StreamWriter(args.prefix, args.log_file, args.progress_interval)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    try:
        while True:
            chunk = sys.stdin.buffer.read1(8192)
            if not chunk:
                break
            pending += decoder.decode(chunk)
            start = 0
            for index, char in enumerate(pending):
                if char not in "\r\n":
                    continue
                writer.handle(pending[start:index])
                start = index + 1
            pending = pending[start:]
        pending += decoder.decode(b"", final=True)
        writer.handle(pending, final=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
