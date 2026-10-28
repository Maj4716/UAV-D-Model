# Job 72617857 R06f 800 m/s control return audit worklog

- 2026-10-28: Began read-only audit of `D:/Desktop/UAV新/uav_hf_r06f_0800_ctrl2`.
- Confirmed submitted deck SHA-256 matches the released R06f control manifest: `111911562b703b8af1e96810b08edd8c31fbeea739ddddd2921b8bcd07c517af`.
- Confirmed scheduler/resource identity: one node, 32 tasks, `mpirun -np 32`, host record `b13r3n01:32`.
- Confirmed LS-DYNA R15 MPP double reached normal termination at problem time 0.040005 ms and cycle 2863; stderr is empty.
- Ran the ASCII QA parser: 32/32 rank logs terminate normally, no numbered error, no mass scaling, energy-ratio range 0.957156--1.00268, hourglass energy zero, and 66 unique negative-volume events.
- Ran the frozen R06 single-job adapter outside the return directory. Capture diagnostics reproduce Job 72397333 to sub-nanometric numerical-report precision; formal primary QoI remains unavailable by the preregistered deleted-mass/tail gates.
- Added and ran `validate_r06f_measurements.py`. D3PART history is `(802,686,1,7)`; all 402 shared states and all seven channels match full D3PLOT exactly on 251,379 alive element-states; HVAR6 is finite and available.
- Independently traversed ragged BINOUT NCFORC. `ncforc/surfa_1` contains 801 finite states with 3365--3427 unique node IDs per state. The existing adapter's NCFORC/SLEOUT-missing statements are therefore invalid; its capture and HVAR6 sections remain valid.
- Applied frozen resource-matched repeatability gates against Job 72397333. qM/qP/qE absolute differences are below `5e-10`; qV/v0 relative difference is below `3e-10`. All four gates pass by very large margins.
- Decision: release exactly one 800 m/s, 0.75 mm, `TSSFAC=0.4`, `DT2MS=0` single-factor diagnostic at 1 node / 32 ranks. Keep 2000 m/s, treatments, P0/P1 admission, engine fitting and dataset-reference promotion frozen.

