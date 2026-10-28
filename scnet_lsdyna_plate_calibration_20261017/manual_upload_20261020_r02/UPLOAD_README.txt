SCNet LS-DYNA P0 manual upload bundle R02

R02 fixes the R01 keyword-reader failure by declaring long-field input with:
*KEYWORD LONG=Y

Select p0_velocity_mesh_calibration.k as the only LS-DYNA solver input.
Keep all six files in the same remote work directory for provenance.
Do not select ZIP, CSV, JSON, or TXT files as the solver input.

Recommended submission values:
- job name: uav_hf_p0_plate_r02
- remote directory: /work/home/acd7dt5yv4/LSDYNA/uav_hf_p0_plate_r02
- software: ls-dyna_R15_mmp_d
- queue: xahcnormal
- 1 node, 32 MPP ranks, 4-hour walltime
- head memory 512 MB, child memory 256 MB
- command-line parameters: empty
- auto-fill previous parameters: off
- no GPU and no restart file

Deck SHA-256:
bdc3cf4accc25044f525b4b6cc89249ce4ef73e168386b0dc145be534bdf28a4
Deck size:
20604830 bytes

Static validation status:
PASS, including long-field declaration and 20-character numeric-width checks.
This PASS does not establish LS-DYNA parser acceptance, numerical convergence,
or physical validity. R02 still requires a new platform run.
