R06o plate MAT EROD=1 isolation

Select r06o_0800_plate_erod1_isolation.k as the head deck.
Required resources: 1 node, exactly 32 MPP ranks, ls-dyna_R15_mmp_d.
The deck changes only the TITLE and plate MAT_JOHNSON_COOK EROD field 0->1 from returned R06n.
All fragment adaptive-SPH, contact, timestep, output and 0.04 ms controls remain unchanged.
This is a numerical isolation canary. Return the complete output even if timestep collapse or cycle-guard exit occurs.
It is not physical, engine or damage-tree calibration.
