SCNet LS-DYNA P0 R06 diagnostic job

STATUS
  Prepared and statically validated only after static_validation.json reports PASS.
  NOT submitted, NOT solver-validated, NOT admitted for P0/P1 or training labels.

PAIR / FACTOR
  experiment_id=r06k_fragment_negative_volume_psfail_isolation_800
  factor=control_solid_psfail_part_set
  role=fragment_negative_volume_failure_exclusion_diagnostic
  termination_time_ms=0.04
  plate_only_psfail=true
  tssfac=0.05
  fragment_mat015_erod=1

SOLVER INPUT (select this file only as the LS-DYNA head/input file)
  r06k_0800_fragment_psfail_plate.k

EXPECTED SOLVER INPUT IDENTITY
  size_bytes=2557716
  sha256:f78396a9fd15114fe411507a7774d68cecb10a25460413474772f2b99e8a17ca
  first_line=*KEYWORD LONG=Y
  package_revision=R06_fragment_negative_volume_isolation_20261030
  job_id=r06k_0800_fragment_psfail_plate

SCNET SETTINGS
  job_name=uav_hf_r06k_0800_fpsfail
  work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06k_0800_fpsfail
  software=ls-dyna_R15_mmp_d
  nodes=1
  ranks=32
  walltime=04:00:00
  head_memory=512MB
  child_memory=256MB
  command_line_parameters=<empty; timestep-refinement jobs do not request unsupported binary INTFOR>
  auto_fill_previous_parameters=off

UPLOAD SEQUENCE
  wave_0=r06k_0800_fragment_psfail_plate
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
