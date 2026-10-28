# Stage A worklog

Date: 2026-10-23 (Asia/Shanghai)  
Source: `D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\05`  
Boundary: the source return is read-only; this log records derived evidence and code changes only.

1. Froze Job `72252362`, deck/design/case-map/material identities and the D3PART family identity.
2. Added `build_r05_run_integrity.py` and generated a qualification-relevant 75-file manifest plus run-integrity audit/evidence. The manifest is intentionally not represented as a hash of all 120 returned files.
3. Confirmed 32/32 normal rank terminations, no fatal/error/non-finite tokens, energy ratio `0.968466–1.00216`, zero hourglass-to-peak-internal-energy ratio, no added-mass parts and `dt2ms=0`.
4. Repaired the real D3PART adapter at the software seam: used-solid PID semantics, zero-solid catalog reporting, solver-realizable timeline, termination coverage, and `0.5×mesh` crossing-resolution gate.
5. Exercised the repaired adapter on the real D3PART family: 802 states, 7,737 nodes, 5,680 solids, temporal gate PASS and no alive-to-alive half-mesh overlimit case.
6. Kept the extraction state at `real_d3part_exercised_pending_independent_integration_acceptance`; no assessor bridge or formal case-results payload was generated.
7. Changed the live successor generator/validator contract to the five R15 fields `DT,CYCL,IOOPT,LCDT,PSETID`, without modifying the historical R05 candidate or return.
8. Froze `design/p0_r06_real_output_contract.json` before any R06 deck or submission. No R06 deck was generated or submitted.
9. Final root regression ran 82 `test_r05*.py` tests and 1 `test_r06*.py` test: all 83 passed. The R05 run also completed 55 subtests. Python compilation, strict JSON parsing, Stage B manifest verification and visualization-manifest verification passed.

Admission boundary: Stage A passes software consistency and run-evidence integrity only. Historical R05 keyword compatibility, independent D3PART integration acceptance, case response qualification and physical validity remain separate gates.
