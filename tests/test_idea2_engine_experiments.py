import importlib.util
from pathlib import Path

import numpy as np

from damage_engine.utils import quaternion_matrix_xyzw


_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_idea2_engine_experiments.py"
_SPEC = importlib.util.spec_from_file_location("idea2_engine_experiments", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def _case():
    return {
        "encounter": {
            "body_to_target_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "velocity_target_mps": [100.0, 0.0, 0.0],
            "trajectory_origin_m": [-10.0, 0.0, 0.0],
            "detonation_position_m": [0.0, 0.0, 0.0],
            "collision_distance_m": 10.0,
            "detonation_time_s": 0.1,
        }
    }


def test_roll_and_distance_variants_preserve_ray_and_rotate_transverse_axes():
    rolled = _MODULE._roll_variant(_case(), np.pi / 2.0)
    rotation = quaternion_matrix_xyzw(rolled["encounter"]["body_to_target_orientation_xyzw"])
    assert np.allclose(rotation[:, 0], [1.0, 0.0, 0.0])
    assert np.allclose(rotation[:, 1], [0.0, 0.0, 1.0], atol=1.0e-12)
    shifted = _MODULE._distance_variant(rolled, 6.0)
    assert shifted["encounter"]["trajectory_origin_m"] == [-6.0, 0.0, 0.0]
    assert shifted["encounter"]["detonation_time_s"] == 0.06


def test_e0_preregistered_distance_levels_and_probability_margin_semantics():
    assert _MODULE._E0_DISTANCES_M == (8.0, 10.0, 12.0)
    reference = np.asarray([0.0, 0.1, 0.2, 0.3])
    probability_scale, probability_semantics = _MODULE._e0_metric_normalization(
        "tree:F1", reference
    )
    count_scale, count_semantics = _MODULE._e0_metric_normalization(
        "total_component_hits", reference
    )

    assert probability_scale == 1.0
    assert probability_semantics == "absolute_probability"
    assert count_scale == np.mean(np.abs(reference))
    assert count_semantics == "relative_to_reference_mean_abs"


def test_e1_multitarget_report_masks_sparse_conditionals_and_selects_R():
    means = np.asarray(
        [
            [0.10, np.nan],
            [0.20, 1.00],
            [0.30, np.nan],
            [0.40, 1.20],
        ]
    )
    variances = np.asarray(
        [
            [0.04, np.nan],
            [0.04, 0.01],
            [0.04, np.nan],
            [0.04, 0.01],
        ]
    )
    counts = np.asarray(
        [
            [32.0, 0.0],
            [32.0, 4.0],
            [32.0, 1.0],
            [32.0, 5.0],
        ]
    )

    report = _MODULE._e1_target_report(
        means,
        variances,
        counts,
        ("component-a", "component-b"),
        rmse_tolerance=0.05,
    )

    assert report["evaluated_component_count"] == 2
    assert report["insufficient_component_count"] == 0
    assert report["components"][0]["valid_case_count"] == 4
    assert report["components"][0]["minimum_candidate_R"] == 16
    assert report["components"][1]["valid_case_count"] == 2
    assert report["components"][1]["minimum_candidate_R"] == 4
    assert report["recommended_R_for_all_evaluated_components"] == 16


def test_e1_multitarget_report_refuses_to_invent_sparse_target_noise():
    report = _MODULE._e1_target_report(
        np.asarray([[np.nan], [0.3]]),
        np.asarray([[np.nan], [0.1]]),
        np.asarray([[0.0], [2.0]]),
        ("component-a",),
        rmse_tolerance=0.1,
    )

    assert report["evaluated_component_count"] == 0
    assert report["insufficient_component_count"] == 1
    assert report["components"][0]["status"] == "insufficient_conditioned_cases"
    assert report["recommended_R_for_all_evaluated_components"] is None


def test_e1_conditional_R_scales_current_positive_sample_noise_not_total_R_naively():
    report = _MODULE._e1_target_report(
        np.asarray([[np.nan], [1.0], [np.nan], [1.2]]),
        np.asarray([[np.nan], [0.01], [np.nan], [0.01]]),
        np.asarray([[0.0], [4.0], [1.0], [5.0]]),
        ("component-a",),
        rmse_tolerance=0.05,
        conditional_on_positive=True,
        total_replicates=32,
    )

    component = report["components"][0]
    assert component["minimum_positive_observations"] == 4
    assert component["minimum_candidate_R"] == 32
    assert component["pooled_positive_rate"] == 10.0 / (4.0 * 32.0)
    assert report["target_semantics"] == "conditional_on_positive_replicates"
