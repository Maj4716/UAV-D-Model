R06s 1200 m/s numerical stability diagnostic

Select r06s_1200_speed_stability_diagnostic.k as the head deck.
Required resources: 1 node, exactly 32 MPP ranks, ls-dyna_R15_mmp_d.
Single physical factor: part 2002 initial speed changes from 2000 to 1200 m/s.
D3PLOT/D3PART are measurement controls at 0.00015 ms; ASCII stays 0.00025 ms.
Keep 0.50 ms, ENDCYC=2500000, TSSFAC=0.05, DT2MS=0 and all physical controls unchanged.
After completion, download only the Stage-1 small-file set first. Do not download d3plot/d3part/binout until Stage 1 passes.
This is a numerical stability diagnostic, not physical, engine or damage-tree calibration.
