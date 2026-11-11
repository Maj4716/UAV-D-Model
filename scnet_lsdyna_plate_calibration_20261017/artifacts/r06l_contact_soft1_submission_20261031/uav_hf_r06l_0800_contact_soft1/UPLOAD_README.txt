SCNet LS-DYNA P0 R06 diagnostic job

STATUS
  Prepared and statically validated only after static_validation.json reports PASS.
  NOT submitted, NOT solver-validated, NOT admitted for P0/P1 or training labels.

PAIR / FACTOR
  experiment_id=r06l_contact_soft_isolation_800
  factor=contact_optional_card_a_soft
  role=contact_soft1_numerical_stability_diagnostic
  termination_time_ms=0.04
  plate_only_psfail=true
  tssfac=0.05
  fragment_mat015_erod=1
  contact_optional_card_a_soft=1

SOLVER INPUT (select this file only as the LS-DYNA head/input file)
  r06l_0800_contact_soft1.k

EXPECTED SOLVER INPUT IDENTITY
  size_bytes=2557668
  sha256:73ba78b3ce25dc0b56b8bc2327b7cc0f5374ab35fef3978aa4ba0fdc62606576
  first_line=*KEYWORD LONG=Y
  package_revision=R06_contact_soft1_isolation_20261031
  job_id=r06l_0800_contact_soft1

SCNET SETTINGS
  job_name=uav_hf_r06l_0800_soft1
  work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06l_0800_soft1
  software=ls-dyna_R15_mmp_d
  nodes=1
  ranks=32
  walltime=04:00:00
  head_memory=512MB
  child_memory=256MB
  command_line_parameters=<empty; timestep-refinement jobs do not request unsupported binary INTFOR>
  auto_fill_previous_parameters=off

UPLOAD SEQUENCE
  wave_0=r06l_0800_contact_soft1
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
