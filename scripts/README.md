# Script entry points

The supported user-facing launchers are deliberately small:

| Script | Purpose |
|---|---|
| `run_experiment.sh` | BeaverTails single-step pipeline |
| `run_mathqa_experiment.sh` | MathQA generated-step pipeline |
| `run_ultrachat_experiment.sh` | UltraChat generated-step pipeline |
| `run_ultrachat_multimodel_full_2gpu.sh` | Full nine-target UltraChat experiment on two GPUs |
| `smoke_mathqa_pipeline.sh` | MathQA contract and full GPU smoke test |
| `smoke_ultrachat_pipeline.sh` | UltraChat contract and optional GPU smoke test |

`run_generated_step_experiment.sh`, `run_model_pair.sh`, and
`run_continuous_module.sh` are internal stage orchestrators. Python files in this
directory remain compatibility CLIs for existing checkpoints and artifact schemas.
Reusable code belongs in `metacog/`; new scripts should not import implementation
details from another script.

Scientific parameters live in `configs/*.env`. Launchers should contain only runtime
choices such as GPU identifiers, output paths, and force/reuse flags.
