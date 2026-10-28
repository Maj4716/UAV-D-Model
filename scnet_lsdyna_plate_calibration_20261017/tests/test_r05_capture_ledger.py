from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from r05_capture_ledger import build_capture_ledger  # noqa: E402


def synthetic_states(
    first_z: list[float], second_z: list[float], third_z: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    element_count = len(first_z)
    node_count = 8 * element_count
    positions = np.zeros((3, node_count, 3), dtype=float)
    velocities = np.zeros_like(positions)
    connectivity = np.arange(node_count, dtype=np.int64).reshape(element_count, 8)
    for element in range(element_count):
        nodes = connectivity[element]
        positions[0, nodes, 2] = first_z[element]
        positions[1, nodes, 2] = second_z[element]
        positions[2, nodes, 2] = third_z[element]
        velocities[:, nodes, 2] = -800.0
    alive = np.ones((3, element_count), dtype=bool)
    masses = np.ones(element_count, dtype=float)
    return positions, velocities, connectivity, alive, masses


class R05CaptureLedgerTests(unittest.TestCase):
    def test_first_crossing_uses_initial_mass_not_terminal_survivor_average(self) -> None:
        positions, velocities, connectivity, alive, masses = synthetic_states(
            [-11.90, -11.00], [-12.05, -11.00], [-12.20, -11.00]
        )
        result = build_capture_ledger(
            times_ms=np.asarray([0.0, 0.001, 0.01]),
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity,
            element_alive=alive,
            element_reference_mass_kg=masses,
            expected_reference_mass_kg=float(masses.sum()),
            capture_plane_z_mm=-12.0,
            impact_speed_mps=1000.0,
            maximum_unambiguous_step_mm=0.25,
        )
        self.assertTrue(result["primary_qoi_available"])
        self.assertEqual(result["classification"], "confirmed_perforation")
        self.assertAlmostEqual(result["qM"], 0.5)
        self.assertAlmostEqual(result["qP"], 0.4)
        self.assertAlmostEqual(result["qE"], 0.32)
        self.assertAlmostEqual(result["qV_over_v0"], 0.8)
        self.assertAlmostEqual(
            result["mass_fraction_by_ledger_state"]["alive_upstream"], 0.5
        )

    def test_deleted_upstream_mass_makes_qoi_unavailable(self) -> None:
        positions, velocities, connectivity, alive, masses = synthetic_states(
            [-11.90, -11.00], [-12.05, -11.00], [-12.20, -11.00]
        )
        alive[1:, 1] = False
        result = build_capture_ledger(
            times_ms=np.asarray([0.0, 0.001, 0.01]),
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity,
            element_alive=alive,
            element_reference_mass_kg=masses,
            expected_reference_mass_kg=float(masses.sum()),
            capture_plane_z_mm=-12.0,
            impact_speed_mps=1000.0,
            maximum_unambiguous_step_mm=0.25,
        )
        self.assertFalse(result["primary_qoi_available"])
        self.assertIn("deleted_upstream_mass", result["unavailable_reasons"])
        self.assertIsNone(result["qM"])
        self.assertAlmostEqual(
            result["mass_fraction_by_ledger_state"]["deleted_upstream"], 0.5
        )

    def test_large_crossing_step_is_explicit_temporal_ambiguity(self) -> None:
        positions, velocities, connectivity, alive, masses = synthetic_states(
            [-11.50], [-12.50], [-12.70]
        )
        result = build_capture_ledger(
            times_ms=np.asarray([0.0, 0.001, 0.01]),
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity,
            element_alive=alive,
            element_reference_mass_kg=masses,
            expected_reference_mass_kg=float(masses.sum()),
            capture_plane_z_mm=-12.0,
            impact_speed_mps=1000.0,
            maximum_unambiguous_step_mm=0.25,
        )
        self.assertFalse(result["primary_qoi_available"])
        self.assertEqual(result["classification"], "unavailable")
        self.assertEqual(
            result["mass_fraction_by_ledger_state"]["temporal_ambiguity"], 1.0
        )

    def test_zero_crossing_is_confirmed_numerical_nonperforation(self) -> None:
        positions, velocities, connectivity, alive, masses = synthetic_states(
            [-11.0], [-11.1], [-11.2]
        )
        result = build_capture_ledger(
            times_ms=np.asarray([0.0, 0.001, 0.01]),
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity,
            element_alive=alive,
            element_reference_mass_kg=masses,
            expected_reference_mass_kg=float(masses.sum()),
            capture_plane_z_mm=-12.0,
            impact_speed_mps=1000.0,
            maximum_unambiguous_step_mm=0.25,
        )
        self.assertTrue(result["primary_qoi_available"])
        self.assertEqual(result["classification"], "confirmed_nonperforation")
        self.assertEqual(result["qM"], 0.0)
        self.assertIsNone(result["qV_over_v0"])

    def test_nonfinite_input_is_rejected_fail_closed(self) -> None:
        positions, velocities, connectivity, alive, masses = synthetic_states(
            [-11.9], [-12.05], [-12.2]
        )
        velocities[1, 0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "node_velocities.*finite"):
            build_capture_ledger(
                times_ms=np.asarray([0.0, 0.001, 0.01]),
                node_positions_mm=positions,
                node_velocities_mm_per_ms=velocities,
                element_connectivity=connectivity,
                element_alive=alive,
                element_reference_mass_kg=masses,
                expected_reference_mass_kg=1.0,
                capture_plane_z_mm=-12.0,
                impact_speed_mps=1000.0,
                maximum_unambiguous_step_mm=0.25,
            )

    def test_nonfinite_derived_output_is_rejected_fail_closed(self) -> None:
        positions, velocities, connectivity, alive, masses = synthetic_states(
            [-11.9], [-12.05], [-12.2]
        )
        velocities[:, :, 2] = -1e200
        with self.assertRaisesRegex(ValueError, "derived outputs.*finite"):
            build_capture_ledger(
                times_ms=np.asarray([0.0, 0.001, 0.01]),
                node_positions_mm=positions,
                node_velocities_mm_per_ms=velocities,
                element_connectivity=connectivity,
                element_alive=alive,
                element_reference_mass_kg=masses,
                expected_reference_mass_kg=1.0,
                capture_plane_z_mm=-12.0,
                impact_speed_mps=1000.0,
                maximum_unambiguous_step_mm=0.25,
            )

    def test_missing_reference_mass_makes_qoi_unavailable(self) -> None:
        positions, velocities, connectivity, alive, masses = synthetic_states(
            [-11.0], [-11.1], [-11.2]
        )
        result = build_capture_ledger(
            times_ms=np.asarray([0.0, 0.001, 0.01]),
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity,
            element_alive=alive,
            element_reference_mass_kg=masses,
            expected_reference_mass_kg=2.0,
            capture_plane_z_mm=-12.0,
            impact_speed_mps=1000.0,
            maximum_unambiguous_step_mm=0.25,
        )
        self.assertFalse(result["primary_qoi_available"])
        self.assertIn(
            "initial_discrete_mass_relative_error", result["unavailable_reasons"]
        )
        self.assertAlmostEqual(result["initial_discrete_mass_relative_error"], 0.5)
        self.assertIsNone(result["qM"])

    def test_expected_mass_is_the_qoi_denominator_and_missing_mass_is_unresolved(self) -> None:
        positions, velocities, connectivity, alive, masses = synthetic_states(
            [-11.90, -11.00], [-12.05, -11.00], [-12.20, -11.00]
        )
        masses[:] = [0.498, 0.498]
        result = build_capture_ledger(
            times_ms=np.asarray([0.0, 0.001, 0.01]),
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity,
            element_alive=alive,
            element_reference_mass_kg=masses,
            expected_reference_mass_kg=1.0,
            capture_plane_z_mm=-12.0,
            impact_speed_mps=1000.0,
            maximum_unambiguous_step_mm=0.25,
        )
        self.assertTrue(result["primary_qoi_available"])
        self.assertAlmostEqual(result["qM"], 0.498)
        self.assertAlmostEqual(result["qP"], 0.3984)
        self.assertAlmostEqual(result["missing_reference_mass_fraction"], 0.004)
        self.assertAlmostEqual(result["unresolved_crossing_mass_fraction"], 0.004)

    def test_alive_to_deleted_possible_crossing_is_temporal_ambiguity(self) -> None:
        positions, velocities, connectivity, alive, masses = synthetic_states(
            [-11.9], [-12.05], [-12.2]
        )
        alive[1:, 0] = False
        result = build_capture_ledger(
            times_ms=np.asarray([0.0, 0.001, 0.01]),
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity,
            element_alive=alive,
            element_reference_mass_kg=masses,
            expected_reference_mass_kg=1.0,
            capture_plane_z_mm=-12.0,
            impact_speed_mps=1000.0,
            maximum_unambiguous_step_mm=0.25,
        )
        self.assertFalse(result["primary_qoi_available"])
        self.assertEqual(result["classification"], "unavailable")
        self.assertEqual(
            result["mass_fraction_by_ledger_state"]["temporal_ambiguity"], 1.0
        )
        self.assertEqual(
            result["mass_fraction_by_ledger_state"]["deleted_upstream"], 0.0
        )

    def test_result_records_frozen_call_parameters_and_thresholds(self) -> None:
        positions, velocities, connectivity, alive, masses = synthetic_states(
            [-11.0], [-11.1], [-11.2]
        )
        result = build_capture_ledger(
            times_ms=np.asarray([0.0, 0.001, 0.01]),
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity,
            element_alive=alive,
            element_reference_mass_kg=masses,
            expected_reference_mass_kg=1.0,
            capture_plane_z_mm=-12.0,
            impact_speed_mps=1000.0,
            maximum_unambiguous_step_mm=0.25,
        )
        provenance = result["provenance"]
        self.assertEqual(result["schema_version"], "hf-p0-capture-ledger-result@1.1.0")
        self.assertEqual(provenance["finite_input_validation"], "pass")
        self.assertEqual(provenance["state_count"], 3)
        self.assertEqual(provenance["node_count"], 8)
        self.assertEqual(provenance["element_count"], 1)
        self.assertEqual(provenance["expected_reference_mass_kg"], 1.0)
        self.assertEqual(provenance["capture_plane_z_mm"], -12.0)
        self.assertEqual(provenance["impact_speed_mps"], 1000.0)
        self.assertEqual(provenance["maximum_unambiguous_step_mm"], 0.25)
        self.assertEqual(provenance["tail_window_fraction"], 0.10)
        self.assertEqual(
            provenance["thresholds"]["initial_discrete_mass_relative_error_max"],
            0.005,
        )


if __name__ == "__main__":
    unittest.main()
