SCNet LS-DYNA P0 R06 diagnostic job

STATUS
  Prepared and statically validated only after static_validation.json reports PASS.
  NOT submitted, NOT solver-validated, NOT admitted for P0/P1 or training labels.

PAIR / FACTOR
  experiment_id=r06f_timestep_sensitivity_800
  factor=control_timestep_tssfac
  role=no_mass_scaling_timestep_sensitivity
  termination_time_ms=0.04
  plate_only_psfail=false
  tssfac=0.4

SOLVER INPUT (select this file only as the LS-DYNA head/input file)
  r06f_0800_tssfac04.k

EXPECTED SOLVER INPUT IDENTITY
  size_bytes=2557654
  sha256:afb0f34112ccfdfcfd95bf442c9b45e2cb1590fdad8d76e98fbbc3fb5d6baee7
  first_line=*KEYWORD LONG=Y
  package_revision=R06_timestep_sensitivity_20261028
  job_id=r06f_0800_tssfac04

SCNET SETTINGS
  job_name=uav_hf_r06f_0800_ts04
  work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06f_0800_ts04
  software=ls-dyna_R15_mmp_d
  nodes=1
  ranks=32
  walltime=04:00:00
  head_memory=512MB
  child_memory=256MB
  command_line_parameters=<empty; R06f does not request unsupported binary INTFOR>
  auto_fill_previous_parameters=off

UPLOAD SEQUENCE
  wave_0=r06f_0800_tssfac04
  Follow the frozen gates in the packaged design. Do not advance to a later wave
  until the preceding parser/runtime/output-readability gate passes.

REQUIRED RETURN
  Required: submitted .k, solver o= output or d3hsp, mes*, glstat, matsum, sleout, nodout, binout*, d3plot*, d3part*, solver stdout/stderr, Slurm script and resource record.
  Optional but preserve if present: intfor*, ncforc*.
  Do not rename or merge job directories. A missing optional direct-contact file
  is unavailable evidence, never a zero response.

BOUNDARY
  This job follows the frozen design and submission wave. A static PASS does not
  prove target-solver parsing, output readability, numerical quality, response
  availability, physical validity, P0 admission, or dataset-label admission.
