R06m adaptive solid-to-SPH capability canary

Head deck: r06m_0800_adaptive_sph_canary.k
Run: one node, 32 MPP ranks, ls-dyna_R15_mmp_d, no extra command-line parameters.
The deck starts from the same post-explosion state as R06j.

This model-form canary converts failed fragment solids (part 2002) into SPH debris (part 2003),
uses eight particles per failed hexahedron, retains SOFT=2 for solid contact, and uses a dedicated
SOFT=1 asymmetric node-to-surface contact only for SPH part 2003 against solid set 920001.

Safety guard: the run must reach problem time 0.04 ms before cycle 829930.
If the cycle limit is reached first, treat the canary as failed and return all partial outputs.

This is not physical validation and not a calibration response. Return the complete job directory.
