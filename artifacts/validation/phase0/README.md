# Phase 0 learning diagnostic

Date: 2026-11-20  
Status: completed, development-only

## Frozen protocol

- Experiment: `B0-phase0-multitask-baseline`
- Config hash: `sha256:4fd720ad7c190f1daece97069810510c7dc43467450efb2c3ee7281262446b85`
- Runs: 18 (`2 feature views × 3 train fractions × 3 seeds`)
- Model: the same two-layer shared MLP and four output heads for every run
- Evaluation: 2,000 Reference validation cases with design weights
- Leakage checks: passed; train/validation candidate overlap was zero; the test split was not accessed
- Full machine-readable result: `../../experiments/phase0_20261120/results.json`

## Results

Values are mean across three seeds. The bottleneck count uses component PR-AUC `< 0.2`.

| Feature view | Train fraction | Support macro PR-AUC | Bottleneck count | Tree soft PR-AUC | Positive-only severity MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `x_design`, train-standardized | 0.25 | 0.470 | 9.33 | 0.671 | 0.321 |
| `x_design`, train-standardized | 0.50 | 0.637 | 2.67 | 0.767 | 0.270 |
| `x_design`, train-standardized | 1.00 | 0.740 | 2.67 | 0.812 | 0.218 |
| `x_normalized` | 0.25 | 0.294 | 25.33 | 0.585 | 0.349 |
| `x_normalized` | 0.50 | 0.371 | 15.33 | 0.619 | 0.330 |
| `x_normalized` | 1.00 | 0.427 | 10.33 | 0.661 | 0.311 |

At full training size, the mean per-component PR-AUC was 0.740 for `x_design` and 0.427 for `x_normalized`. Based on each component's three-seed mean, `x_design` had 3 of 67 components below 0.2; `x_normalized` had 11. The three remaining `x_design` bottlenecks were components 044, 056, and 055, with only 19, 17, and 22 unweighted validation positives.

## Interpretation

The learning problem is responsive to data volume when the learner receives compact sampling coordinates. The original physical pose view improves much less with additional data, so representation is the first bottleneck and rare-component support is the second.

`x_design` contains sampler-specific direction categories and latent quantiles. It is a pre-label diagnostic view, not the production input contract. Its performance supports the Section 9 target-encoder direction, while its sampler dependence rules out direct production promotion. The 16-dimensional first-armour impact-state contract must be evaluated on a newly generated archive.

The evidence establishes software-teacher learnability under the current Reference distribution. It does not establish external physical validity.
