# Contributing

## Development setup

```bash
pip install -e '.[data,dev]'
pytest
ruff check metacog tests
bash -n scripts/*.sh
```

## Pull requests

- Keep scientific defaults in versioned configuration files.
- Keep runtime settings such as GPUs and output directories out of scientific configs.
- Add a unit test for every new model registry field, dataset contract, metric, or artifact
  schema.
- Preserve train/validation/test isolation by base prompt ID for generated-step data.
- Do not import one experiment script from another. Extract shared code to `metacog/`.
- Do not commit model weights, activations, run artifacts, or raw datasets.

Changes to statistical hypotheses or FDR families must be documented as scientific
changes, not hidden inside plotting or reporting code.
