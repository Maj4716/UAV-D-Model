SCNet LS-DYNA P0 R06 paired diagnostic job

STATUS
  Prepared and statically validated only after static_validation.json reports PASS.
  NOT submitted, NOT solver-validated, NOT admitted for P0/P1 or training labels.

PAIR / FACTOR
  experiment_id=r06_diag_800_window
  factor=termination_time_ms
  role=paired_control
  termination_time_ms=0.04
  plate_only_psfail=false

SOLVER INPUT (select this file only as the LS-DYNA head/input file)
  r06e_0800_ctrl.k

EXPECTED SOLVER INPUT IDENTITY
  size_bytes=2557671
  sha256:47c5fcacbec9ba3edac10d06611b6ba602f58a135b27b62d968290334a763a70
  first_line=*KEYWORD LONG=Y
  package_revision=R06_keyword_parser_repair_20261025
  job_id=r06e_0800_ctrl

SCNET SETTINGS
  job_name=uav_hf_r06e_0800_c
  work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06e_0800_c
  software=ls-dyna_R15_mmp_d
  nodes=1
  ranks=32
  walltime=04:00:00
  head_memory=512MB
  child_memory=256MB
  command_line_parameters=<empty; INTFOR filename is embedded by *DATABASE_BINARY_INTFOR_FILE>
  auto_fill_previous_parameters=off

UPLOAD SEQUENCE
  wave_0=r06e_0800_ctrl
  wave_1=r06e_2000_ctrl
  wave_2=r06e_0800_tend008,r06e_2000_psfail_plate
  Follow the frozen gates in the packaged design. Do not advance to a later wave
  until the preceding parser/runtime/output-readability gate passes.

REQUIRED RETURN
  Preserve the submitted .k, solver o= output or d3hsp, mes*, glstat, matsum,
  sleout, nodout, ncforc, binout*, d3plot*, d3part*, intfor*, stdout/stderr,
  Slurm script, and resource record. Do not rename or merge job directories.

BOUNDARY
  This is one member of two paired single-factor diagnostics. A static PASS does
  not prove target-solver parsing, output readability, numerical quality, response
  availability, physical validity, P0 admission, or dataset-label admission.
