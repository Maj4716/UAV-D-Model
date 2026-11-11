# R06j return analysis and R06k release worklog

Date: 2026-10-30

## Immutable source

- Source return: `D:/Desktop/UAV新/uav_hf_r06j_0800_fragment_erod1`
- Platform job: `72751725`
- Returned deck SHA-256: `8ec1613324159b3ebc00217871ad29321564423b0a7f73a267207e4952bada77`
- The 112-file, 2,129,312,420-byte return was treated as read-only.
- The audited deck was copied to `immutable_parent_r06j_0800_fragment_erod1.k` and hash-checked.

## Executed evidence

- Normal termination on 32 MPP ranks, empty stderr, 82,993 cycles, final time 0.04 ms.
- No mass scaling; `DT2MS=0`.
- D3PART: 802 states; full D3PLOT: 402 states; all 402 shared states matched across all seven
  channels with zero alive-mask and time mismatch.
- MAT_015 HVAR6 remained readable and finite. Fragment HVAR6 reached 1.0 and final p99 was 1.0,
  confirming that failure/damage evidence remained present with fragment `EROD=1`.
- NCFORC remained readable, finite and ragged; its units/sign/contact-side semantics are still pending.
- Energy ratio range was 0.960449 to 1.00009, technically inside the frozen 0.95 to 1.05 engineering
  gate but materially degraded from the R06i minimum of 0.998954.
- All 35 unique negative-volume elements belonged to fragment part 2002.

## Mechanism result

- R06i control deleted 47 of 686 fragment elements; R06j deleted 35.
- Setting fragment MAT_015 `EROD=1` retained 12 additional elements and reduced deleted-upstream
  mass fraction from 0.06851311815869016 to 0.05102040709405755, a relative reduction of about 25.53%.
- The remaining 35 deleted fragment elements exactly matched the 35 negative-volume fragment elements.
- Solver stdout states that ERODING contact automatically enables a negative-volume failure criterion
  for all solid parts and records each remaining element as eroded due to negative volume.
- Therefore MAT_015 failure erosion is a partial deletion cause; the remaining deletion path is the
  ERODING-contact negative-volume criterion.
- The response remains unavailable because 0.05102040709405755 is still above the 0.05 limit.

## R06k single-factor diagnostic

- Revision: `R06_fragment_negative_volume_isolation_20261030`.
- Job: `r06k_0800_fragment_psfail_plate`.
- Only changed model field: `CONTROL_SOLID` Card 1 field 5 `PSFAIL`, 0 to 910001.
- Set 910001 contains plate part 1002 only, so fragment part 2002 is excluded from the automatically
  enabled negative-volume failure criterion.
- Fragment MAT_015 `EROD=1`, plate `EROD=0`, ERODING contact, TSSFAC=0.05, DT2MS=0, mesh and all
  other initial/boundary/output settings remain frozen.
- Normal completion is not assumed. Severe distortion, timestep collapse or solver termination are
  valid diagnostic evidence and must not be repaired inside this run.

## Verification

- R06k static validation: PASS, one job.
- R06k focused tests: 5 passed.
- R06j/R06k and historical deterministic-generation regression passed except for tests requiring raw
  R06h/R06i returns that the user has moved; their frozen analysis evidence remains available.
- Controlled full suite excluding five known unavailable external-return fixture tests: PASS, exit code 0.
- Direct delivery contains no ZIP and was checked against an exact root/file whitelist and per-file hashes.

## Delivery

- Delivery root: `artifacts/r06k_fragment_psfail_isolation_submission_20261030`
- Upload folder: `uav_hf_r06k_0800_fragment_psfail_plate`
- Head deck: `r06k_0800_fragment_psfail_plate.k`
