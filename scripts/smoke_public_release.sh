#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
echo '[smoke] Python source syntax (no model imports)'
"$PYTHON_BIN" - <<'PY'
import ast
from pathlib import Path
files = [p for folder in ('metacog', 'scripts', 'tests') for p in Path(folder).rglob('*.py')]
for file in files:
    ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
print(f'{len(files)} Python files parsed')
PY
echo '[smoke] Bash syntax'
for file in scripts/*.sh; do bash -n "$file"; done
echo '[smoke] main-plan and registry contracts'
"$PYTHON_BIN" - <<'PY'
from metacog.models import get_model
from metacog.evaluation import primary_construct_groups
targets = ('llama2_7b', 'llama31_8b', 'llama32_3b', 'qwen25_7b',
           'qwen3_4b', 'qwen3_8b', 'deepseek_llama8b', 'deepseek_qwen7b')
for target in targets:
    assert get_model(target).path_env
for profile in ('conversation', 'safety', 'math'):
    groups = primary_construct_groups(profile)
    assert len(groups['behavior']) == len(groups['monitoring']) == 5
print('8 model profiles; 3 domains; 5 behavior + 5 monitoring constructs each')
PY
echo '[smoke] passed; GPU/model execution was not performed'
