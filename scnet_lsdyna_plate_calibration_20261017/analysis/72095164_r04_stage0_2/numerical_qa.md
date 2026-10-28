# SCNet R04 Stage 0-2 numerical QA

- Job: `72095164`
- Overall: `review_required`
- Final total-energy ratio: `0.977608`
- Total-energy-ratio range: `0.977608–1.001750`
- Maximum |hourglass/internal|: `0`
- Final energy ratio without eroded energy: `0.629447`
- Final eroded-energy fraction of initial: `34.816114%`
- Minimum/maximum time step: `7.56953e-06` / `6.46272e-05` ms

## Gates

| Gate | Status |
|---|---|
| archive_and_input_identity | `pass` |
| solver_normal_termination | `pass` |
| global_total_energy_candidate_band | `pass` |
| hourglass_candidate_gate | `pass` |
| mass_scaling_gate | `configured_off_actual_unavailable` |
| negative_volume_localization | `review_required` |
| binary_database_readability | `not_checked_reader_unavailable` |
| case_level_perforation_and_residual_velocity | `not_assessed_stage_3` |

## Negative-volume localization

`874` unique elements; all mapped to the seven frozen cases.

| Case | Speed m/s | Mesh mm | Plate | Fragment | First time | Last time |
|---|---:|---:|---:|---:|---:|---:|
| `p0_v1200_h0p75` | 1200 | 0.75 | 49 | 69 | 0.0021628 | 0.018942 |
| `p0_v1600_h0p75` | 1600 | 0.75 | 72 | 62 | 0.001721 | 0.015436 |
| `p0_v2000_h0p75` | 2000 | 0.75 | 85 | 52 | 0.0014681 | 0.014323 |
| `p0_v2300_h0p75` | 2300 | 0.75 | 101 | 68 | 0.0013244 | 0.011815 |
| `p0_v2600_h0p75` | 2600 | 0.75 | 72 | 46 | 0.0012192 | 0.010611 |
| `p0_v2000_h1p00` | 2000 | 1.00 | 69 | 59 | 0.0018147 | 0.014026 |
| `p0_v2000_h0p50` | 2000 | 0.50 | 53 | 17 | 0.0013244 | 0.0035618 |

## Boundary

Stage 0-2 establishes archive/input identity, solver completion and global ASCII numerical indicators only. It does not establish case-level numerical qualification, convergence, physical validity, application calibration or research admission.
