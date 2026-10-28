# R06e parser-repair worklog

Date: 2026-10-25

1. Preserved both downloaded R06d ZIPs as read-only evidence and performed ZIP CRC/path/hash audits without modifying them.
2. Bound each returned deck to its ZIP manifest and the locally released R06d deck byte-for-byte.
3. Confirmed solver R15.0.2 MPP double launch, 32 ranks and successful Ansys license checkout for Jobs 72269229 and 72269336.
4. Located the common first failure in `mes0000`: missing mandatory Card 2 of `*DATABASE_EXTENT_BINARY`; classified D3PART KEY+450 as a downstream parser-shift error.
5. Recorded the independent INTFOR KEY+457 warning: the returned R15.0.2 parser accepts four timing-card fields, not five.
6. Added `scripts/analyze_r06_wave1_keyword_failure.py` and generated hash-bound diagnosis, worklog and staged submission plan under `analysis/r06d_wave1_keyword_failure_20261025/`.
7. Added a non-overwriting design builder and froze `design/p0_r06_keyword_parser_repair_design.json` with distinct `r06e_*` identities and a one-job Wave 0 gate.
8. Updated the generator conditionally: historical R06d serialization remains reproducible as evidence; R06e emits the exact two-card EXTENT_BINARY and four-field INTFOR contracts.
9. Updated the validator to fail closed on missing Card 2, non-default parser fields and over-wide INTFOR cards. Default validation now rejects the known-failed R06d revision; forensic replay requires an explicit override.
10. Generalized bundle/release verification to bind the design revision and staged wave order rather than hard-coded R06d job IDs.
11. Generated `manual_upload_20261025_r06e_parser_repair/`, wrote per-job static reports and verified mesh/connectivity parity plus paired-factor isolation.
12. Built four deterministic whitelist ZIPs under `artifacts/r06e_parser_repair_20261025/` and independently verified all internal/external manifests, hashes and historical R05 evidence.
13. Staged only `r06e_0800_ctrl` under `artifacts/r06e_wave0_canary_20261025/` to prevent accidental early submission of later waves.
14. Added regression tests covering deterministic design generation, exact R15 cards, physical-block parity, rejection of both historical card defects, reproduction of both downloaded failures, and Wave 0 staging.
15. Ran the full suite with a workspace-local temporary directory and disabled pytest cache: 109/109 tests passed.

Boundary: local PASS establishes software/package consistency only. Target R15.0.2 parser acceptance, normal runtime, output readability, numerical qualification and physical validity remain external gates.
