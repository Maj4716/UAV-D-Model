from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_r04_stage3_5 import (  # noqa: E402
    alive_mask,
    build_mesh_assessment,
    build_speed_assessment,
    final_window_statistics,
    is_monotonic,
    relative_difference,
    resolve_node_state_coordinates,
)


def row(
    case_id: str,
    speed: float,
    mesh: float,
    survivor: float,
    mass: float,
    available: bool,
    proxy: bool,
) -> dict:
    return {
        "case_id": case_id,
        "impact_speed_mps": speed,
        "mesh_mm": mesh,
        "survivor_average_residual_speed_mps": survivor,
        "fragment_surviving_mass_fraction": mass,
        "center_history_node_qoi_available": available,
        "perforation_proxy": proxy,
    }


class Stage35Tests(unittest.TestCase):
    def test_alive_encoding_is_nonzero_not_boolean_sum(self) -> None:
        encoded = np.asarray([0.0, 1.0, 2.0, 14.0])
        np.testing.assert_array_equal(alive_mask(encoded), [False, True, True, True])

    def test_final_window_statistics(self) -> None:
        time = np.asarray([0.017, 0.018, 0.019, 0.020])
        values = np.asarray([5.0, 10.0, 11.0, 12.0])
        result = final_window_statistics(time, values)
        self.assertEqual(result["sample_count"], 3)
        self.assertAlmostEqual(result["range"], 2.0)
        self.assertAlmostEqual(result["relative_range"], 2.0 / 11.0)

    def test_absolute_coordinate_state_is_not_added_twice(self) -> None:
        reference = np.asarray([[10.0, 20.0, 30.0]])
        states = np.asarray([[[10.0, 20.0, 30.0]], [[11.0, 20.0, 29.0]]])
        coordinates, mode = resolve_node_state_coordinates(reference, states)
        self.assertEqual(mode, "absolute_coordinates")
        np.testing.assert_allclose(coordinates[-1], [[11.0, 20.0, 29.0]])

    def test_zero_initial_state_is_treated_as_displacement(self) -> None:
        reference = np.asarray([[10.0, 20.0, 30.0]])
        states = np.asarray([[[0.0, 0.0, 0.0]], [[1.0, 0.0, -1.0]]])
        coordinates, mode = resolve_node_state_coordinates(reference, states)
        self.assertEqual(mode, "displacement_from_reference")
        np.testing.assert_allclose(coordinates[-1], [[11.0, 20.0, 29.0]])

    def test_mesh_gate_fails_without_band_and_missing_center_qoi(self) -> None:
        rows = [
            row("coarse", 2000.0, 1.0, 320.0, 0.46, False, True),
            row("medium", 2000.0, 0.75, 1300.0, 0.38, False, True),
            row("fine", 2000.0, 0.5, 1850.0, 0.88, True, True),
        ]
        result = build_mesh_assessment(rows)
        self.assertEqual(result["response_consistency_gate"], "fail")
        self.assertEqual(result["center_history_node_gate"], "unavailable")
        self.assertFalse(result["pre_registered_numeric_error_band_found"])
        self.assertGreater(result["relative_difference_coarse_to_medium"], 0.7)

    def test_all_perforated_speed_scan_does_not_bracket_transition(self) -> None:
        rows = [
            row(f"v{speed}", float(speed), 0.75, float(residual), 0.5, True, True)
            for speed, residual in [
                (1200, 500),
                (1600, 800),
                (2000, 1300),
                (2300, 700),
                (2600, 1200),
            ]
        ]
        result = build_speed_assessment(rows)
        self.assertFalse(result["perforation_transition_bracketed"])
        self.assertFalse(result["survivor_speed_monotonic_with_impact_speed"])
        self.assertEqual(result["velocity_response_fit_gate"], "fail")

    def test_math_helpers(self) -> None:
        self.assertAlmostEqual(relative_difference(75.0, 100.0), 0.25)
        self.assertIsNone(relative_difference(1.0, 0.0))
        self.assertTrue(is_monotonic([1.0, 1.0, 2.0]))
        self.assertFalse(is_monotonic([1.0, 3.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
