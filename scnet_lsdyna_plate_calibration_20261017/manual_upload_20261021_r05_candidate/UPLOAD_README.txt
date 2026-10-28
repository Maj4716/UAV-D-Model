SCNet LS-DYNA P0 R05 candidate upload set

STATUS
  Prepared and statically validated; NOT submitted and NOT solver-validated.
  R04 remains immutable. P0 response calibration and P1 submission remain denied.

SOLVER INPUT (select this file only as the LS-DYNA head/input file)
  p0_velocity_mesh_calibration.k

EXPECTED SOLVER INPUT IDENTITY
  size_bytes=20605206
  sha256=6fbd753a113d0214e7ba4ff5136dc526d151e6f2c52c94e70399b29c4b9d3d0c
  first_line=*KEYWORD LONG=Y
  package_revision=R05_capture_ledger_preregistered

COMPANION AUDIT FILES (do not select as the solver input)
  p0_case_part_map.csv
  p0_cases_r05.json
  jc4340_reference.json
  p1_preregistered_cases.csv
  manifest.json
  static_validation.json

SCNET SETTINGS
  job_name=uav_hf_p0_plate_r05
  work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_p0_plate_r05
  software=ls-dyna_R15_mmp_d
  nodes=1
  ranks=32
  walltime=04:00:00
  head_memory=512MB
  child_memory=256MB
  command_line_parameters=<empty>
  auto_fill_previous_parameters=off

R05 CHANGES RELATIVE TO R04
  1. The 0.75 mm speed scan is 600/800/1000/1200/2000 m/s to improve the
     probability of bracketing the numerical perforation transition. The 2000
     m/s 1.00/0.75/0.50 mm stress test is retained; neither outcome is assumed.
  2. Termination time is 0.04 ms. Window sufficiency is decided only by the
     frozen tail-stability gates.
  3. Full-model D3PLOT remains at 0.0005 ms. Fragment-only D3PART is added at
     0.00005 ms using PSETID=900001 containing fragment PIDs 2001-2007 only.
  4. The primary response is the frozen z=-12 mm initial-reference-mass capture
     ledger (qM/qP/qE/qV_over_v0). qP is outward-positive and qE is element-
     centroid translational energy. qM/qP/qE use the expected case-map mass (and
     associated v0 scales); qV_over_v0=qP/qM. A truncated observed element
     population therefore cannot normalize itself.
  5. Missing/non-finite inputs, mass deficits, ambiguous crossing/deletion, stale
     validation, and identity mixing fail closed. Center NODOUT and terminal
     surviving-part averages remain diagnostic only.

REQUIRED RETURN FILES
  Preserve the submitted .k and all d3hsp, mes*, solver .out, Slurm stdout/stderr,
  glstat, matsum, sleout, nodout, binout*, d3plot*, and d3part* files. D3PART
  readability and the capture ledger cannot be accepted by static validation.

BOUNDARY
  static_validation.json PASS proves only this candidate's serialization,
  identities, hashes, geometry, frozen R15 card contracts, D3PART part-set binding,
  cases, and preregistered thresholds. It does not prove LS-DYNA parser/runtime
  acceptance, D3PART reader integration, response convergence, physical validity,
  P0 calibration admission, or P1 admission.
