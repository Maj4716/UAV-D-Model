### T1. Azimuthal ray count (default seed, 1 realization, historical accumulation)

| config | rays/case | s/case | p_hit ratio Poisson | p_hit ratio union | Brier Poisson | Brier union | train FU (Wilson up) | val FU | RMS Δp_hit vs ref | unreached-status flips vs ref | OOD mass mean | any-OOD cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `az2` | 24 | 0.059* | 0.307 | 0.307 | 0.0580 | 0.0580 | 9.69% (9.86%) | 11.55% | 0.2370 | 15.17% | 0.0667 | 629 |
| `az4` | 48 | 0.052 | 0.497 | 0.504 | 0.0423 | 0.0426 | 7.60% (7.75%) | 9.23% | 0.2022 | 12.38% | 0.0701 | 1094 |
| `az8` | 96 | 0.077 | 0.748 | 0.785 | 0.0205 | 0.0215 | 4.18% (4.29%) | 5.64% | 0.1404 | 8.01% | 0.0697 | 1752 |
| `az16` | 166 | 0.116 | 0.890 | 0.968 | 0.0063 | 0.0077 | 1.79% (1.87%) | 2.91% | 0.0771 | 4.86% | 0.0691 | 2114 |
| `azall` | 170 | 0.122 | 0.896 | 0.978 | 0.0060 | 0.0075 | 1.65% (1.72%) | 2.73% | 0.0752 | 4.61% | 0.0691 | 2124 |

### T2. Layer quadrature (all-fragment rays)

| config | layer_quadrature_samples | layered interactions (all cases) | Brier union | train FU | p_fail ratio |
|---|---:|---:|---:|---:|---:|
| `azall_q4` | 4 | 0 | 0.0075 | 1.65% | 0.717 |
| `azall` | 16 | 0 | 0.0075 | 1.65% | 0.717 |
| `azall_q64` | 64 | 0 | 0.0075 | 1.65% | 0.717 |

### T3. Source seed (single realization)

| rays | seeds | p_hit ratio union (min–max) | Brier union (min–max) | train FU (min–max) | any-OOD cases (min–max) | OOD mass mean (min–max) | mean pairwise RMS Δp_hit between seeds | mean pairwise unreached flips |
|---|---:|---|---|---|---|---|---:|---:|
| az4 | 4 | 0.493–0.504 | 0.0425–0.0427 | 7.60%–7.74% | 1042–1094 | 0.0691–0.0714 | 0.0762 | 1.07% |
| azall | 4 | 0.974–0.978 | 0.0072–0.0077 | 1.65%–1.75% | 2099–2124 | 0.0685–0.0695 | 0.1094 | 2.63% |

### T4. Source-realization averaging (all-fragment rays)

| config | K | s/case | p_hit ratio union | Brier union [95% CI] | Brier Poisson | train FU (Wilson up) | val FU | RMS Δp_hit vs K16 | RMS Δp_fail vs K16 (same options) | any-OOD cases | OOD mass mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `azall` | 1 | 0.122 | 0.978 | 0.0075 [0.0072, 0.0078] | 0.0060 | 1.65% (1.72%) | 2.73% | 0.0752 | 0.0266 | 2124 | 0.0691 |
| `azall_k2` | 2 | 0.470* | 0.978 | 0.0046 [0.0044, 0.0047] | 0.0041 | 1.10% (1.17%) | 1.89% | 0.0519 | 0.0185 | 2184 | 0.0689 |
| `azall_k4` | 4 | 0.401 | 0.978 | 0.0030 [0.0029, 0.0032] | 0.0032 | 0.70% (0.76%) | 1.28% | 0.0338 | 0.0117 | 2225 | 0.0691 |
| `azall_k8` | 8 | 1.856* | 0.978 | 0.0023 [0.0022, 0.0024] | 0.0027 | 0.43% (0.47%) | 0.81% | 0.0197 | 0.0069 | 2255 | 0.0689 |
| `azall_k16` | 16 | 3.834* | 0.978 | 0.0020 [0.0019, 0.0021] | 0.0024 | 0.24% (0.27%) | 0.53% | — | — | 2270 | 0.0688 |
| `az4_k8` | 8 | 0.823* | 0.502 | 0.0399 [0.0385, 0.0412] | 0.0397 | 7.00% (7.14%) | 8.49% | 0.1953 | 0.0903 | 1443 | 0.0701 |
| `azall_reg_dom` | 1 | 0.131 | 0.983 | 0.0075 [0.0072, 0.0078] | 0.0059 | 1.61% (1.68%) | 2.67% | 0.0755 | 0.0338 | 2117 | 0.0002 |
| `azall_k4_reg_dom` | 4 | 0.426 | 0.984 | 0.0030 [0.0029, 0.0031] | 0.0031 | 0.67% (0.72%) | 1.27% | 0.0339 | 0.0149 | 2219 | 0.0002 |
| `azall_k8_reg_dom` | 8 | 0.792 | 0.983 | 0.0023 [0.0022, 0.0024] | 0.0026 | 0.40% (0.44%) | 0.78% | 0.0198 | 0.0086 | 2251 | 0.0002 |
| `azall_k16_reg_dom` | 16 | 1.539 | 0.984 | 0.0019 [0.0018, 0.0020] | 0.0023 | 0.22% (0.25%) | 0.51% | — | — | 2263 | 0.0002 |

Teacher Monte-Carlo noise floor of the p_hit Brier, E[q̂(1−q̂)/(R−1)]: all 0.0013, train R8 0.0014, validation R32 0.0004.

### T5. Damage accumulation: historical per-hit product vs registry spatial clusters

| config | p_deg ratio | p_fail ratio [95% CI] | Brier p_deg | Brier p_fail [95% CI] | failed/hit proxy | failed/hit teacher | degraded/hit proxy | degraded/hit teacher | RMS Δp_fail vs K16 (same accumulation and censoring) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `az4` | 0.528 | 0.499 [0.488, 0.510] | 0.0333 | 0.0159 [0.0152, 0.0166] | 0.405 | 0.495 | 0.763 | 0.812 | 0.1234 |
| `az4_reg` | 0.532 | 0.606 [0.594, 0.618] | 0.0333 | 0.0147 [0.0140, 0.0154] | 0.491 | 0.495 | 0.768 | 0.812 | 0.1190 |
| `azall` | 0.972 | 0.717 [0.710, 0.724] | 0.0061 | 0.0056 [0.0054, 0.0059] | 0.281 | 0.398 | 0.689 | 0.764 | 0.0702 |
| `azall_reg` | 0.977 | 0.952 [0.947, 0.956] | 0.0059 | 0.0016 [0.0015, 0.0016] | 0.373 | 0.398 | 0.694 | 0.764 | 0.0333 |
| `azall_k4` | 0.971 | 0.717 [0.710, 0.723] | 0.0028 | 0.0051 [0.0048, 0.0053] | 0.271 | 0.380 | 0.706 | 0.758 | 0.0659 |
| `azall_k4_reg` | 0.977 | 0.954 [0.951, 0.957] | 0.0026 | 0.0007 [0.0006, 0.0007] | 0.358 | 0.380 | 0.710 | 0.758 | 0.0147 |
| `azall_k8` | 0.971 | 0.717 [0.710, 0.724] | 0.0022 | 0.0050 [0.0048, 0.0052] | 0.269 | 0.375 | 0.712 | 0.757 | 0.0651 |
| `azall_k8_reg` | 0.977 | 0.955 [0.952, 0.957] | 0.0021 | 0.0005 [0.0005, 0.0005] | 0.354 | 0.375 | 0.716 | 0.757 | 0.0086 |
| `azall_k16` | 0.971 | 0.717 [0.710, 0.724] | 0.0020 | 0.0049 [0.0047, 0.0051] | 0.268 | 0.372 | 0.717 | 0.757 | 0.0648 |
| `azall_k16_reg` | 0.977 | 0.956 [0.954, 0.959] | 0.0018 | 0.0004 [0.0004, 0.0005] | 0.353 | 0.372 | 0.721 | 0.757 | — |
| `azall_reg_dom` | 0.983 | 0.958 [0.954, 0.963] | 0.0059 | 0.0015 [0.0015, 0.0016] | 0.374 | 0.397 | 0.694 | 0.764 | 0.0338 |
| `azall_k8_reg_dom` | 0.983 | 0.962 [0.959, 0.964] | 0.0020 | 0.0005 [0.0005, 0.0005] | 0.356 | 0.375 | 0.718 | 0.757 | 0.0086 |
| `azall_k16_reg_dom` | 0.983 | 0.963 [0.961, 0.966] | 0.0017 | 0.0004 [0.0004, 0.0004] | 0.355 | 0.372 | 0.723 | 0.757 | — |

### T5b. Domain censoring: whole packet on any refused branch vs refused quantile mass only

| config | OOD mass fraction mean | any-OOD flag cases | proxy expected refused contacts / case | teacher unresolved impacts / replicate | armor hits proxy/teacher | component hits proxy/teacher | absorbed energy proxy/teacher | p_hit ratio union | Brier union | train FU | p_fail ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `azall` | 0.0691 | 2124 | 10.810 | 0.031 | 0.898 | 0.996 | 0.918 | 0.978 | 0.0075 | 1.65% | 0.717 |
| `azall_dom` | 0.0002 | 2117 | 0.029 | 0.031 | 1.004 | 1.000 | 0.929 | 0.983 | 0.0075 | 1.61% | 0.726 |
| `azall_reg` | 0.0691 | 2124 | 10.810 | 0.031 | 0.898 | 0.996 | 0.918 | 0.978 | 0.0075 | 1.65% | 0.952 |
| `azall_reg_dom` | 0.0002 | 2117 | 0.029 | 0.031 | 1.004 | 1.000 | 0.929 | 0.983 | 0.0075 | 1.61% | 0.958 |
| `azall_k8_reg` | 0.0689 | 2255 | 10.792 | 0.031 | 0.899 | 0.997 | 0.921 | 0.978 | 0.0023 | 0.43% | 0.955 |
| `azall_k8_reg_dom` | 0.0002 | 2251 | 0.030 | 0.031 | 1.006 | 1.001 | 0.931 | 0.983 | 0.0023 | 0.40% | 0.962 |

### T6. Continuity with REVIEW §3.4 (173-case three-way-complete subset)

| config | p_hit ratio (Poisson) | p_hit ratio (union) | p_deg ratio | p_fail ratio | failed/hit proxy (Poisson denominator) | failed/hit teacher |
|---|---:|---:|---:|---:|---:|---:|
| `az4` | 0.533 | 0.547 | 0.583 | 0.477 | 0.246 | 0.329 |
| `azall` | 0.887 | 0.979 | 0.962 | 0.615 | 0.170 | 0.274 |
| `azall_reg` | 0.887 | 0.979 | 0.981 | 0.935 | 0.257 | 0.274 |
| `azall_k16` | 0.928 | 0.989 | 0.972 | 0.614 | 0.156 | 0.251 |
| `azall_k16_reg` | 0.928 | 0.989 | 0.990 | 0.952 | 0.238 | 0.251 |
| `azall_k8_reg_dom` | 0.930 | 0.996 | 0.998 | 0.948 | 0.241 | 0.254 |

### T7. Damage-tree outputs, proxy/teacher mean ratio (teacher-unaffected cases)

| output | teacher mean | `az4` | `azall` | `azall_reg` | `azall_reg_dom` | `azall_k8_reg_dom` | `azall_k16_reg_dom` |
|---|---:|---:|---:|---:|---:|---:|---:|
| effect-c1 | 0.4301 | 0.757 | 0.971 | 0.989 | 0.989 | 0.980 | 0.981 |
| effect-c2 | 0.2060 | 0.373 | 0.394 | 0.951 | 0.951 | 0.950 | 0.954 |
| effect-f1 | 0.1773 | 0.454 | 0.672 | 0.949 | 0.960 | 0.949 | 0.948 |
| effect-k1 | 0.0321 | 0.277 | 0.308 | 0.807 | 0.814 | 0.848 | 0.849 |
| effect-k2 | 0.0118 | 0.247 | 0.426 | 0.788 | 0.802 | 0.810 | 0.811 |
| effect-m1 | 0.2121 | 0.543 | 0.832 | 0.854 | 0.861 | 0.865 | 0.860 |
| effect-m2 | 0.5703 | 0.564 | 0.981 | 0.993 | 0.994 | 0.983 | 0.982 |

Cases: 2087 (teacher affected-replicate fraction = 0; selection uses the teacher side only).

### T8. Promotion gate on train R8 (2,000 cases)

| status | config | identified pairs | proxy-unreached pairs | teacher-positive | FU rate | Wilson 95% upper | case-cluster bootstrap 95% | gate ≤ 3% |
|---|---|---:|---:|---:|---:|---:|---|---|
| pre-declared | `azall` | 133708 | 114806 | 1893 | 1.649% | 1.724% |  [1.56, 1.74] | PASS |
| post-hoc | `azall_reg_dom` | 133708 | 114731 | 1842 | 1.605% | 1.680% |  [1.52, 1.69] | PASS |
| post-hoc | `azall_k8_reg_dom` | 133708 | 109972 | 438 | 0.398% | 0.437% |  [0.36, 0.44] | PASS |

Secondary (unmasked `exposure_audit`, declared config): 1899/115027 = 1.651%, Wilson upper 1.726%.
Validation R32 (supplementary, declared config): 2.728%, Wilson upper 2.983%.

Declared config by profile (train): large_front 1.81% (Wilson 1.95%); medium_front 1.88% (Wilson 2.02%); small_front 1.28% (Wilson 1.39%)

Declared config by region (train): neg_x 1.68% (Wilson 2.02%); neg_y 1.46% (Wilson 1.61%); pos_x 2.22% (Wilson 2.40%); pos_y 0.57% (Wilson 0.69%); top_cap 1.93% (Wilson 2.09%)

