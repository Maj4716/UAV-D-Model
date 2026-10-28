# Job 72252362 R05 results audit worklog

Date: 2026-10-23 (Asia/Shanghai)

1. Treated `05/` as immutable returned evidence; no returned file was edited.
2. Inventoried 120 files totaling 3,606,867,357 bytes, including 8 `d3part*`, 19 `d3plot*`, 32 `mes*`, `glstat`, `binout0000`, the solver output, Slurm stdout/stderr, and all eight submitted candidate files.
3. Recomputed the eight candidate-file hashes and compared them with `manual_upload_20261021_r05_candidate/`; all matched byte for byte.
4. Verified Job 72252362 runtime identity: LS-DYNA MPP double R15.0.2, Intel-MPI 2018, 1 node, 32 ranks, 512/256 MB, and no extra command-line arguments.
5. Verified license checkout, keyword read, initialization, 32/32 rank normal termination, 5,344 cycles, and problem time 0.040003 ms.
6. Ran the existing R05 regression suite with workspace-owned TEMP/TMP: 50 tests passed.
7. Ran `parse_lsdyna_ascii.py` against the returned files. The derived `ascii_qa.json` reports normal termination, energy-ratio and hourglass gates passing, but `review_or_fail` because of 628 unique negative-volume elements.
8. Opened the real D3PART family using the project-local `lasso-python==2.0.4`. It contains 802 states, 7,737 nodes, 5,680 fragment solids and covers 0 to 0.0400026376 ms.
9. Ran the frozen `extract_r05_d3part.py` entry point without altering inputs. It failed closed at the maximum-state-interval gate: actual 0.0000646272 ms versus frozen upper bound 0.00005005 ms. No formal extraction payload was written.
10. Performed a diagnostic-only in-memory probe to expose later blockers. After temporarily relaxing only the time gate, the adapter rejected zero-solid plate PIDs in the reader part catalog. After also normalizing only those zero-solid metadata entries, the capture algorithm produced one available case (600 m/s, confirmed nonperforation) and six unavailable cases. These diagnostic values are explicitly non-qualifying and were not bridged to the assessor.
11. Bound all 628 negative-volume events to the seven frozen cases and distinguished them from 1,620 other failed solids. The final total of 2,248 matches the solver output.
12. Confirmed the R15 parser warning that the D3PART card accepts five fields while the frozen deck wrote eight. PSETID in field 5 worked, but the trailing fields were not accepted by this build.
13. Did not submit P1, fit an engine, generate a promoted dataset, train a model, or access validation/test data.

Outputs created by this audit:

- `ascii_qa.json`
- `run_audit_summary.json`
- this worklog
- project-level Chinese audit and next-step report
