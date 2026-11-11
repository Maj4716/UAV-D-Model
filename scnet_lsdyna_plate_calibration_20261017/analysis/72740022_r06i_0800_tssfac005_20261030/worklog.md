# R06i return analysis and R06j release worklog

Date: 2026-10-30

## Immutable source

- Source return: `D:/Desktop/UAV新/uav_hf_r06i_0800_tssfac005`
- Platform job: `72740022`
- Returned deck SHA-256: `e2e8da8e55304b7fe2deb2c1790e00bf7fea9c64145c6c09a954c50492f3dd88`
- The return directory was treated as read-only. The audited deck was copied to
  `immutable_parent_r06i_0800_tssfac005.k` and hash-checked before successor generation.

## Executed evidence

- Solver: normal termination on 32 MPP ranks, empty stderr, 68,795 cycles, final time 0.04 ms.
- No mass scaling was reported; `DT2MS=0`.
- D3PART: 802 states; full D3PLOT: 402 states; all 402 shared states matched for all seven
  compared channels with zero alive-mask and time mismatch.
- MAT_015 HVAR6 was readable and finite for alive elements.
- NCFORC was readable and finite but ragged; it remains mechanism evidence pending semantic validation.
- Energy ratio range: 0.998954 to 1.00009; hourglass/internal maximum: 0.
- One negative-volume element was reported, element 41208 in fragment part 2002.

## Frozen comparison and decision

- R06i deleted-upstream mass fraction: 0.06851311815869016.
- R06h control deleted-upstream mass fraction: 0.07434402197408253.
- Absolute difference: 0.005830903815392377, passing the 0.01 terminal comparison tolerance.
- qM, qP and qE were all zero in both runs; qV was null in both because neither run had confirmed
  capture-plane crossing mass. The preregistered both-null qV rule therefore passed.
- The local 0.1-to-0.05 timestep consistency gate passed, but the response availability gate failed
  because 0.06851311815869016 exceeds the 0.05 deleted-upstream limit.
- Automatic timestep halving stopped. TSSFAC=0.025, 2000 m/s, treatments, engine fitting,
  damage-tree fitting and reference-label promotion remain denied.

## R06j single-factor diagnostic

- Revision: `R06_fragment_erosion_isolation_20261030`.
- Job: `r06j_0800_fragment_erod1`.
- Only changed model field: fragment MAT_015 Card 4 field 3 `EROD`, 0 to 1.
- Plate MAT_015 EROD remains 0; TSSFAC remains 0.05; DT2MS remains 0.
- The diagnostic tests whether failed-fragment erosion causes the unavailable upstream deleted mass.
  EROD=1 responses are mechanism evidence only and cannot be used as calibration or dataset targets.

## Verification

- `python scripts/validate_p0_r06_diagnostics.py --batch-dir manual_upload_20261030_r06j_fragment_erosion_isolation --write-reports`: PASS, one job.
- `python -m pytest tests/test_r06j_fragment_erosion_isolation.py ...`: 5 passed.
- Targeted historical reproducibility plus R06j regression: 9 passed.
- Controlled full test suite excluding three known external-return fixture tests: PASS (exit code 0).
- Direct delivery was staged without ZIP and checked against an exact root/file whitelist and per-file hashes.

## Delivery

- Delivery root: `artifacts/r06j_fragment_erod1_submission_20261030`
- Upload folder: `uav_hf_r06j_0800_fragment_erod1`
- Head deck: `r06j_0800_fragment_erod1.k`
