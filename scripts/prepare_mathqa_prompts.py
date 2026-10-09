#!/usr/bin/env python3
"""Compatibility CLI for preparing MathQA prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metacog.datasets.mathqa import prepare_mathqa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-mode", choices=["official", "resplit"], default="resplit")
    parser.add_argument("--train-size", type=int, default=22000)
    parser.add_argument("--validation-size", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare_mathqa(
        args.input_dir,
        args.output_dir,
        args.split_mode,
        args.train_size,
        args.validation_size,
        args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
