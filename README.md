# Second-order Internal-state Signals in Large Language Models

Code accompanying **Evidence for Second-order Internal-state Signals in Large Language Models**.

Authors: Yixiao Xu, Mohan Li, Yuan Liu and Zhihong Tian. Correspondence: tianzhihong@gzhu.edu.cn.

This prepared repository contains the maintained experimental code, including the fresh-24 direct-report runtime measurement and prefix-replay fixes. It is not the earlier manuscript code snapshot. No code, dataset or paper DOI has been assigned by this preparation step.

## What is included

- Activation extraction at generation steps 0, 4, 8, 12, 16 and 20.
- External semantic-reference ensembles and joint residual-module discovery.
- Frozen-module transmission, behavioral/monitoring association and next-token tests.
- Search-first neuron-report experiments with runtime-remeasured labels.
- Continuous constrained interventions and critical-sample persistent generation.
- Source-result validation and R rendering of quantitative figures 2--6.

The main study contains 24 conditions (eight targets across three domains), 384 frozen modules, 89 association passers and 46 three-ring passers. The fresh neuron-report study covers all 24 conditions; persistent-generation follow-ups cover all 46 three-ring passers. The source-data deposit, not a README or default config, is authoritative for reported estimates.

## Install

Use Python 3.10 or newer and a clean environment:

```bash
python -m pip install -e '.[dev,data]'
bash scripts/smoke_public_release.sh
```

Full experiments additionally require Linux, compatible CUDA/PyTorch and locally available licensed model weights. This package does not download gated models or provide access tokens. Tested server versions and resource requirements are recorded in `docs/REPRODUCIBILITY.md`.

## Inspect results without a GPU

Download and extract the separate `research_data.zip`, then run:

```bash
python scripts/validate_deposit.py /path/to/research_data
Rscript figures/requirements.R
Rscript figures/draw_figures.R /path/to/research_data
```

Validation uses only the Python standard library. Plotting requires R and the listed R packages, not an LLM. The input data remain unchanged. See `figures/README.md` for font and editable-case-panel details.

## Run experiments

First configure licensed model locations and original dataset paths as described in `docs/REPRODUCIBILITY.md`. Then:

```bash
PLAN_ONLY=1 RUN_ROOT=runs/main_plan bash scripts/run_strict_main_24_2gpu.sh
GPU_IDS='0 1' RUN_ROOT=runs/main bash scripts/run_strict_main_24_2gpu.sh

SOURCE_ROOT=runs/main GPU_IDS='0 1' RUN_ROOT=runs/neuron_report \
  bash scripts/run_neuron_report_fresh24_2gpu.sh
```

The main entry stops after next-token tests by default. Persistent generation must use the frozen main-run configuration and eligible module identities; do not substitute an old grouped report baseline. Detailed follow-up instructions and the distinction between historical result files and rerunnable configurations are in `docs/REPRODUCIBILITY.md`.

## Data and licences

Code retains the existing MIT licence. Original MathQA, BeaverTails and UltraChat data, model weights and their licences are separate. The result deposit contains its own rights statement. Large activation tensors and trained checkpoints are not included; some direct-report per-prompt binary caches were not synchronized locally. See the deposit's `metadata/not_deposited.json` rather than assuming every raw training artifact is available.

## Citation and publication

`CITATION.cff` describes this software and its authors. Public repository URL, DOI, release tag and paper identifier must be filled after publication; none is invented here. Checklist: `docs/PUBLISHING_CHECKLIST.md`.
