# Reproducibility guide

## Three distinct levels

1. **Inspect published results:** Python standard-library validation of the result deposit, complete signed tables and source checksums.
2. **Recreate plots:** R rendering from the frozen exported estimates. This does not retrain predictors or regenerate responses.
3. **Rerun the science:** licensed target/reference weights, original datasets, CUDA and recomputation of activation/training caches. Some binary caches are absent from this deposit; GPU experiments have not been rerun during packaging.

## Recorded execution environment

Author-provided server environment: Ubuntu 22.04.5 LTS; two NVIDIA RTX PRO 5000 72GB Blackwell GPUs; Python 3.10.20; PyTorch 2.12.1+cu130; Transformers 5.12.1. The installed system `nvcc` reported CUDA 12.2, while the PyTorch build reported cu130. These are different version sources, not interchangeable CUDA requirements. Verify the driver/runtime combination before replicating the run; this preparation does not assert GPU compatibility on another machine.

`pyproject.toml` gives supported minimum dependencies, not an exact lockfile for the historical server. The rest of the historical package versions, dataset revisions and exact downloaded model revisions must be supplied by the authors for bitwise reproduction. The main pipeline records its configuration and can resume completed stages.

## Models

Eight targets: `llama2_7b`, `llama31_8b`, `llama32_3b`, `qwen25_7b`, `qwen3_4b`, `qwen3_8b`, `deepseek_llama8b`, `deepseek_qwen7b`. These profiles are in `configs/models.json`; each accepts an environment override such as `LLAMA31_MODEL_PATH`.

Use `configs/local.example.env` as a path template and source your private copy. The runner selects three non-self references from Llama-3.1-8B layer 16, Qwen3-8B layer 18, Qwen2.5-7B layer 14 and Qwen3-4B layer 18. Reference features receive train-only PCA, 512 components per reference. Other profiles remain in the general registry for compatibility; they do not enter this paper's 24-condition plan.

## Original data

Original datasets and input text are not bundled. Obtain them from their original providers under their own terms, then configure:

| Dataset | Expected source configured by the runner | Preparation |
|---|---|---|
| UltraChat | `ultrachat_200k/data/train_sft-*.parquet` | `metacog dataset prepare ultrachat` |
| BeaverTails | ` BeaverTails/round0/330k/train.jsonl` | `metacog dataset prepare beavertails` |
| MathQA | `mathqa/` with the original split files | `metacog dataset prepare mathqa` |

The leading space in the historical BeaverTails directory name is literal; edit `DATA_SOURCE` in `configs/beavertails_external_semantic_continuous_m4_long_v2.env` if your path differs. The dataset-specific source files configure `DATA_SOURCE`, `DATA_PREPARED_PATH` and `BASE_DATA_PATH`; change paths only, not scientific settings. Dataset `prepare --help` documents the supported schema. The main entry prepares data when prepared files are missing.

MathQA and UltraChat request 28,000 prompts. BeaverTails uses 16,140 deduplicated prompts with a 50/20/30 train/selection/evaluation allocation, avoiding the earlier fresh-sample shortage. All sampled states from a prompt remain in one split. The deposited score files preserve held-out IDs and analysis roles, not the absent complete training-ID cache.

## Scientific configuration

Main entry: `scripts/run_strict_main_24_2gpu.sh`; frozen common configuration: `configs/strict_chain_common_v2.env`; dataset configs: `*_strict_chain_main_v2.env`.

- Candidate budget 3,072; support 32; discover 64 modules and freeze 16 per condition.
- Main training 180 epochs; semantic warm-up 80; semantic probe 100; refiner 100.
- Main semantic head: ensemble; latent dimension 384; hidden dimension 1,536.
- Ten constructs per domain, partitioned into five behavioral and five monitoring constructs.
- Ring 2 requires positive conditional information and semantic-residualized direction passing within-group Holm. The final chain is the conjunction of ring decisions, not an additional cross-module significance test.
- All stages-1-2 passers receive independent A/B next-token tests; primary figure eligibility comes from `source_data/modules.csv`.
- Continuous doses 0.25/0.5/1.0, opposite-direction and random-hidden-direction controls.
- Baseline generation uses compatible historical reuse plus deterministic adaptive top-up; defaults alone do not identify a historical sample pool.

`research_data/source_data/configurations.csv` and original `raw_results/.../config.json` retain the actual training settings. They are evidence records containing historical paths, not config files to execute unchanged elsewhere.

## Direct reports

Use `scripts/run_neuron_report_fresh24_2gpu.sh` and `configs/neuron_report_search_fresh24_v3.json` with `SOURCE_ROOT` pointing to your completed main run. This searches 256 candidates, rescreens 24 and evaluates eight selected plus eight matched-control neurons per condition. Runtime-remeasured activation labels and generation-step KV replay are required. No old cache-label report estimates should be mixed with this study.

## Persistent generation

The deposited follow-up selects all 46 frozen three-ring passers, with at most 128 critical samples and doses 0.25/0.5/1.0. Its original `config.json` and frozen plan are in `raw_results/runs/strict_all_passing_trajectory_20261002_175024/`.

The historical runner `run_report_trajectory_followup.py` also schedules an older grouped report task. For a clean science rerun, use the main pipeline's persistent stage rather than launching that legacy report mode as if it were the fresh-24 study. Resume your completed main run with `RUN_PERSISTENT_TRAJECTORY_OVERRIDE=1`; this runner's exact eligibility is stages 1+2, not a guarantee of the paper's narrower 46-module follow-up population. To reproduce precisely the 46-module follow-up, supply its frozen module list to the per-module runner using the actual saved settings. Do not silently substitute one population for the other.

## Data boundaries

- Original JSON/CSV statistics are byte-identical copies with provenance hashes.
- Compressed module score CSVs decompress to the exact original bytes.
- Baseline and trajectory tables remove only four third-party input-text columns (`question`, `options`, `prompt`, `source_prompt`); retained generated text, measurements and IDs are unchanged.
- Six reused trajectories are resolved to their actual earlier source directories. Provenance retains those paths.
- The three extra historical next-token artifacts remain in the full table, but do not enter the eligible 89-module primary population.
- Large weights, activations, report binary caches and all original model/dataset downloads are excluded. See `metadata/not_deposited.json`.
