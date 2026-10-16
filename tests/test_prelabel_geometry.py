import math

import pytest

from damage_engine.prelabel_geometry import prelabel_geometry_features


class Registry:
    def fragment_source(self, _model_id):
        return {
            "direction": {
                "kind": "cone", "axis_body": [1.0, 0.0, 0.0],
                "half_angle_deg": 30.0,
            }
        }


def case(position=(-2.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)):
    return {
        "encounter": {
            "detonation_position_m": list(position),
            "velocity_target_mps": list(velocity),
            "body_to_target_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "warhead": {"fragment_source": {"model_id": "fixed"}},
    }


def test_prelabeled_alignment_is_computed_before_any_outcome():
    features = prelabel_geometry_features(case(), Registry())
    assert features["uav_threat_range_m"] == 2.0
    assert features["fragment_cone_axis_to_uav_cosine"] == 1.0
    assert features["fragment_cone_axis_to_uav_angle_deg"] == 0.0
    assert features["inside_nominal_fragment_cone"] is True


def test_prelabeled_alignment_respects_orientation_and_fixed_threat_velocity():
    value = case(position=(0.0, -2.0, 0.0), velocity=(0.0, -3.0, 0.0))
    # +90 degrees around z maps body +x onto target/UAV-frame +y.
    value["encounter"]["body_to_target_orientation_xyzw"] = [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
    features = prelabel_geometry_features(value, Registry())
    assert features["fragment_cone_axis_to_uav_cosine"] == pytest.approx(1.0)
    assert features["uav_radial_velocity_from_fixed_threat_mps"] == pytest.approx(3.0)
