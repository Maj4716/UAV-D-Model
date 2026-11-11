R06r 2000 m/s numerical speed upper anchor

Select r06r_2000_speed_upper_anchor.k as the head deck.
Required resources: 1 node, exactly 32 MPP ranks, ls-dyna_R15_mmp_d.
Single physical factor: part 2002 initial speed changes from 800 to 2000 m/s.
D3PLOT/D3PART are measurement controls at 0.0001 ms; ASCII stays 0.00025 ms.
Keep 0.50 ms, TSSFAC=0.05, DT2MS=0 and all material/contact/adaptive-SPH/mesh controls unchanged.
Cycle guard 2500000; if reached, do not restart or tune.
This is one numerical upper anchor, not physical, engine or damage-tree calibration.
