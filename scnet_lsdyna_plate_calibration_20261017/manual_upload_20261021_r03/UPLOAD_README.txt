SCNet LS-DYNA P0 manual upload bundle R03

R03 retains *KEYWORD LONG=Y and fixes the R02 required-card layout errors:
- *CONTROL_CONTACT now has two required 8-field cards.
- Each *INITIAL_VELOCITY_GENERATION block now has two 8-field cards.
- PHASE, IRIGID, and the other integer fields are serialized as integers.
- Eroding-contact Card 4 explicitly uses ISYM=1, EROSOP=1, IADJ=1.
- Contact controls use documented R15 defaults except ISLCHK=2 for a full
  initial-penetration diagnostic.

Select p0_velocity_mesh_calibration.k as the only LS-DYNA solver input.
Keep all six files in the same remote work directory for provenance.
Do not select ZIP, CSV, JSON, or TXT files as the solver input.

Recommended submission values:
- job name: uav_hf_p0_plate_r03
- remote directory: /work/home/acd7dt5yv4/LS_DYNA/uav_hf_p0_plate_r03
- software: ls-dyna_R15_mmp_d
- queue: xahcnormal
- 1 node, 32 MPP ranks, 4-hour walltime
- head memory 512 MB, child memory 256 MB
- command-line parameters: empty
- auto-fill previous parameters: off
- no GPU and no restart file

Deck SHA-256:
7063a40d26dc7e044674b446788af348b3ea47beb78368c6be48c7b278e0ef93
Deck size:
20604820 bytes

Static validation status:
PASS, including long-field width, required-card layout, mesh, identity,
integral, and hash checks. This PASS does not establish LS-DYNA parser
acceptance, numerical convergence, or physical validity. R03 still requires
a new platform run.
