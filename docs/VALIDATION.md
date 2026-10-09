# Preparation checks

Prepared locally on 9 October 2026. This package has not been published or assigned a DOI. Manuscript files and experiment estimates were not changed.

## Code

- 102 Python source files parse successfully; Bash syntax checks pass.
- All eight main model profiles and the three five-behavior/five-monitoring construct profiles validate.
- The main runner's plan-only execution produces exactly 24 conditions and non-self semantic-reference ensembles.
- Lightweight tests: 42 passed, 3 skipped because PyTorch is unavailable locally. Two further PyTorch-dependent test files were explicitly excluded from the lightweight run.
- A stale test was updated to distinguish an empty trajectory list from a missing condition. No training, inference or statistical decision logic changed.
- SciPy and scikit-learn were added to the installation metadata because the existing direct-report implementation imports them.
- CUDA experiments and the full PyTorch-dependent suite were not rerun; verify the complete GPU installation on the server before public release.

## Data

- 10,851 indexed result/sample files validate, including 454 compressed sample tables.
- Coverage: 24 main conditions, 384 modules, 3,840 construct estimates, 24 fresh report conditions and 46 persistent follow-ups.
- The recorded 384/89/46 three-ring counts and all frozen module identities match the deposited estimates.
- All 384 original baseline copies were compared by ID, generated text/counts/log-probabilities, prompt-confidence and style measurements. They reduce to 24 identical measured pools, containing 39,200 distinct condition-level rows. Module-specific cluster labels were not used to establish equivalence.
- Source records were copied byte-identically; module-score compression is lossless. Generation CSV projections remove four input-text fields only.
- Negative estimates, truncated records, controls and earlier reused trajectory sources remain traceable. No simulated values were added.

## Figures

The portable R renderer completed figures 2--6 and its supplementary quantitative exports using only the deposited data. It wrote to a separate test output, not to the manuscript. Full figures and panels were checked for nonempty output. A valid UTF-8 locale was required on the local macOS system. The renderer's Fig. 6 text preview is not a replacement for the native editable case illustration; full case texts are deposited separately.

## Remaining author actions

Review model-generated text and any third-party examples for release suitability, complete exact model/dataset revision identifiers and historical environment details where available, and perform a server GPU smoke test. Add public repository links and DOI identifiers only after the corresponding records are published. Binary activation/checkpoint and per-prompt report caches missing from local synchronization are explicitly listed in the data package.
