"""Review check: independent Taylor (1941) tilt-direction test vs implemented Shapiro sign.

Independent derivation (steady flow in the detonation-wave frame): casing material
enters the wave at speed U along -t (t = wave-travel direction), is turned outward
by theta at constant speed, so lab velocity = U(1-cos theta) t + U sin theta n.
=> the fragment leans TOWARD the wave-travel direction by delta = theta/2,
   sin(delta) = V/(2U).  This sign does not depend on any angle convention.
The script compares the implemented nominal directions with that prediction.
"""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from damage_engine.fragments import (build_taylor_shapiro_ring_layout,
    _sample_taylor_shapiro_directions)
from damage_engine.munition_profiles import apply_munition_profile
from damage_engine.registry import PhysicsRegistry

PHYS = ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
CASE = ROOT / "configs/cases/loitering_impact.fragment_only.taylor_shapiro.obliquity_bound.v3.json"
reg = PhysicsRegistry.from_file(PHYS)
out = {}
for prof in ("small_front", "medium_front", "large_front"):
    case = json.loads(CASE.read_text(encoding="utf-8-sig"))
    apply_munition_profile(case, prof)
    fs = case["warhead"]["fragment_source"]
    src = reg.fragment_source(fs["model_id"])
    lay = build_taylor_shapiro_ring_layout(case, src, fs["total_count"])
    charge = float(case["warhead"]["charge_mass_kg"]); casing = float(case["warhead"]["casing_mass_kg"])
    if src.get("casing_mass_basis") == "excludes_fragments":
        casing += fs["total_count"] * fs["mass_distribution"]["parameters"]["value_kg"]
    v = float(fs["gurney_constant_mps"]) / math.sqrt(casing / charge + 0.5)
    D = float(fs["detonation_velocity_mps"])
    dirs = _sample_taylor_shapiro_directions(lay, np.eye(3), np.random.default_rng(0),
        gurney_speed=v, detonation_velocity_mps=D, spread_sigma_deg=0.0)
    axis = lay.body_axis
    impl_axial = dirs @ axis                       # + = toward nose (+X)
    wave_axial = lay.wave_directions_body @ axis   # wave-travel axial component at each fragment
    # Taylor prediction: lean toward wave travel => sign(axial) == sign(wave axial component)
    agree = np.sign(impl_axial) == np.sign(wave_axial)
    out[prof] = {
        "initiation_point_body_m": src["direction"]["initiation_point_body_m"],
        "casing_axial_offsets_m": sorted(set(np.round(lay.source_offsets_body_m[:, 0], 4).tolist())),
        "gurney_speed_mps": v, "detonation_velocity_mps": D,
        "taylor_bound_deg": math.degrees(math.asin(v / (2 * D))),
        "wave_axial_component_range": [float(wave_axial.min()), float(wave_axial.max())],
        "implemented_tilt_deg_mean": float(np.degrees(np.arcsin(impl_axial)).mean()),
        "implemented_tilt_deg_range": [float(np.degrees(np.arcsin(impl_axial)).min()),
                                       float(np.degrees(np.arcsin(impl_axial)).max())],
        "fraction_consistent_with_taylor_wave_direction": float(agree.mean()),
    }
print(json.dumps(out, indent=2))
(Path(__file__).with_suffix(".json")).write_text(json.dumps(out, indent=2), encoding="utf-8")
