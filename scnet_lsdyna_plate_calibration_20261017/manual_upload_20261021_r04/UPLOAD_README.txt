SCNet LS-DYNA P0 R04 upload set

Solver input (select this file only as the LS-DYNA head/input file):
  p0_velocity_mesh_calibration.k

Expected solver input identity:
  size_bytes=20604806
  sha256=312f2ce1c42dd070097dad1036d1d670790704c55c5b534a3b845a0709709e08
  first_line=*KEYWORD LONG=Y
  package_revision=R04_full_keyword_type_contract

Companion audit files (upload to the same work directory, but do not select as solver input):
  p0_case_part_map.csv
  manifest.json
  jc4340_reference.json
  static_validation.json

SCNet settings:
  job_name=uav_hf_p0_plate_r04
  work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_p0_plate_r04
  software=ls-dyna_R15_mmp_d
  nodes=1
  ranks=32
  walltime=04:00:00
  head_memory=512MB
  child_memory=256MB
  command_line_parameters=<empty>
  auto_fill_previous_parameters=off

R04 fixes the only primary R03 parser error: *CONTROL_CONTACT Card 2 ECDT is
an R15 integer field and is now serialized as 0, not 0.000000000e+00. The deck
diff versus the solver-read R03 input is exactly one line and one token.

Static validation is not LS-DYNA execution. Preserve the submitted .k and all
mes*, solver .out, Slurm stdout/stderr, d3hsp, glstat, matsum, sleout, nodout,
and d3plot* files for solver and numerical acceptance.
