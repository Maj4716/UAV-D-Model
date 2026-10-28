# R06f measurement-repair and Wave 0 delivery worklog

Date: 2026-10-27

1. Re-read the frozen Job 72397333 audit and kept `D:\Desktop\UAV新\06` read-only.
2. Confirmed the raw return still has 113 files / 1,932,839,917 bytes and the returned deck SHA-256 remains `47c5fcacbec9ba3edac10d06611b6ba602f58a135b27b62d968290334a763a70`.
3. Froze prospective revision `R06_measurement_contract_repair_20261027` with exactly one job, `r06f_0800_ctrl`.
4. Preserved 800 m/s, 0.75 mm mesh, 0.04 ms termination, TSSFAC=0.8, DT2MS=0, PSFAIL=0, material, EOS, contact, mesh, boundary conditions, IDs, and 1-node/32-rank resource identity.
5. Removed only the unsupported binary INTFOR request and its extent card. The prior Warning 20340 remains the capability reason; missing INTFOR is null/unavailable, not zero.
6. Kept `*DATABASE_NCFORC` with SPR/MPR=1 as a best-effort, nonblocking request because Job 72397333 did not prove structural incompatibility. Missing NCFORC remains null/unavailable, not zero.
7. Added `*DATABASE_EXTENT_D3PART` Card 1 `6,0,3,0,1,1,1,1` and Card 2 `0,0,0,0,0,0,0,0`. D3PART remains the capture-ledger source; D3PART HVAR6 is informative rather than a Wave 0 hard gate.
8. Bound MAT_015 solid damage to full-D3PLOT history variable 6 / array index 5 with mandatory alive masking. Deleted-state values are ignored; non-finite or `abs(value)>=1e29` alive values become null plus reason.
9. Added a self-contained R06 single-job adapter. Capture thresholds are embedded in the R06f contract, so runtime analysis does not silently depend on the historical R05 design.
10. Generated `manual_upload_20261027_r06f_measurement_contract` and statically validated the one-job batch.
11. Verified non-measurement keyword blocks against the exact returned R06e deck. Physics/solver-control parity passed; only title and database-output blocks are allowed to differ.
12. Built the deterministic explicit-whitelist ZIP and independently verified ZIP CRC/readability, safe paths, unique names, exact members, one `.k`, embedded/external manifest equality, and per-member hashes.
13. Rebuilt the ZIP twice in independent temporary directories; byte identity passed.
14. Staged only the canary. The staging directory contains exactly four ordinary files and no extracted subdirectory.
15. `compileall` passed. The R06f-specific suite passed 10/10 tests. The complete discovered suite contains 119 tests and completed with exit code 0.
16. Final lifecycle status is `ready_for_user_submission_not_submitted_not_solver_validated`.

No LS-DYNA executable was available locally. Static and packaging PASS do not establish target-solver parsing, runtime, numerical repeatability, response availability, physical validity, P0/P1 admission, engine fitting, or dataset-label admission.
