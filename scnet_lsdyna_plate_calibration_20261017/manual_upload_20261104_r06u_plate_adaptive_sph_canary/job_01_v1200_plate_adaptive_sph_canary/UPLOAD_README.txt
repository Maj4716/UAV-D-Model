R06u plate adaptive-SPH capability canary

Select r06u_1200_plate_adaptive_sph_canary.k as the head deck.
Use 1 node, exactly 32 MPP ranks, ls-dyna_R15_mmp_d, head memory 512 MB, child memory 256 MB and no extra command-line parameters.
This branches from completed R06s, restores PSFAIL=0, and adds same-material plate solid-to-SPH conversion part 1002 -> part 1003.
The canary stops at 0.15 ms, beyond R06t's 0.10872 ms failure point. Keep ENDCYC=2500000, speed 1200 m/s, output cadence, materials, mesh and timestep controls frozen.
After completion, download only the Stage-1 small-file set first. Do not download d3plot/d3part/binout until Stage 1 passes.
This is a model-form capability canary, not physical validation or calibration.
