# Proxy–teacher development comparison summary

## Scope and sealing

- Source: `recovery_paired_bound_v3_r8_20261122`.
- Included: 30 train + 15 validation cases, 67 components.
- Excluded: 15 sealed-test cases.
- Test result JSON lines decoded: 0.
- Test metrics computed: false.
- Test open-count increment: 0.

The comparison is development-descriptive. It measures agreement with the current
software teacher, not physical truth.

## Point estimates

| Metric | Estimate | Denominator / scope |
|---|---:|---|
| Identified case–component pairs | 3,013 | 45 × 67 = 3,015 possible |
| Domain-complete cases | 43 | 45 development cases |
| Brier | 0.037655 | 3,013 identified pairs |
| ECE | 0.047849 | 3,013 identified pairs |
| Fixed-bin JS | 0.014182 | 3,013 identified pairs; equal design weights |
| Positive-energy `log1p` W1 | 1.036943 | 189 paired positive observations |
| RBF-MMD² | 0.045656 | 43 complete 67-component vectors |
| Proxy-unreached pairs | 2,779 | identified pairs with proxy `p_hit=0` |
| Teacher-positive among proxy-unreached | 210 | 2,779 proxy-unreached pairs |
| False-unreached rate | 0.075567 | Wilson upper 0.085990 |

The false-unreached rate remains above the pre-existing 3% promotion gate.

## Case-cluster bootstrap uncertainty

Paired nonparametric bootstrap resampled the 45 development cases with all 67
components kept together, 2,000 resamples, seed 2026112201. Intervals are conditional
on this single QMC design; they do not estimate between-scramble design variance.

| Metric | 95% percentile interval |
|---|---:|
| Brier | [0.030909, 0.044648] |
| ECE | [0.040448, 0.055897] |
| Fixed-bin JS | [0.011241, 0.017837] |
| Positive-energy `log1p` W1 | [0.833925, 1.258550] |
| RBF-MMD² | [0.038941, 0.083549] |
| False-unreached rate | [0.063587, 0.088515] |

## Calibration structure

- 2,835 of 3,013 identified pairs fall in the proxy [0, 0.1) bin.
- In that bin, proxy mean is 0.000262 and teacher mean is 0.046120.
- The proxy therefore misses a nontrivial low-probability-hit mass even though the
  aggregate Brier and JS distances are modest.
- The [0.9, 1.0] bin contains 143 pairs; proxy mean is 0.989857 and teacher mean is
  0.940559.

## Stratification

- 14 of 15 region × profile cells appear in the 45 development cases.
- `neg_x::large_front` is absent.
- Per-cell case counts range from 1 to 7, so cell rankings remain exploratory.
- Three one-case cells have constant inputs and undefined correlations; Brier/ECE
  remain computable.

## Reproducibility

- Primary report: `v3_proxy_teacher_distribution.json`.
- Bootstrap report: `v3_proxy_teacher_bootstrap.json`.
- Bootstrap script: `bootstrap_proxy_teacher.py`.
- Relevant target tests: 8 passed.

