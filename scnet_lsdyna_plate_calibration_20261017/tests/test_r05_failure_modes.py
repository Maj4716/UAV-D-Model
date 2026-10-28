from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_r05_failure_modes import (  # noqa: E402
    assess_contact_output_availability,
    associate_failure_events_with_d3part,
    bind_events_to_cases,
    classify_fragment_elements,
    extract_focus_solid_history,
    load_run_integrity_binding,
    parse_mes_failure_events,
    summarize_cases,
    validate_output_location,
)


def cube_nodes(center_z: float) -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, center_z - 0.5],
            [1.0, 0.0, center_z - 0.5],
            [1.0, 1.0, center_z - 0.5],
            [0.0, 1.0, center_z - 0.5],
            [0.0, 0.0, center_z + 0.5],
            [1.0, 0.0, center_z + 0.5],
            [1.0, 1.0, center_z + 0.5],
            [0.0, 1.0, center_z + 0.5],
        ],
        dtype=float,
    )


class R05FailureModeTests(unittest.TestCase):
    def test_run_integrity_binding_rechecks_manifest_and_d3part_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            manifest = output / "run_artifact_manifest.json"
            manifest.write_text('{"schema_version":"test"}\n', encoding="utf-8")
            manifest_sha = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
            evidence = {
                "schema_version": "hf-p0-r05-run-integrity-evidence@1.0.0",
                "evidence_id": "test-bound-run",
                "status": "pass",
                "run_id": "72252362",
                "solver_build": "solver-test",
                "d3part_family_sha256": "sha256:d3part-test",
                "returned_artifact_sha256": manifest_sha,
            }
            (output / "run_integrity_evidence.json").write_text(
                json.dumps(evidence), encoding="utf-8"
            )

            binding = load_run_integrity_binding(
                output,
                expected_d3part_family_sha256="sha256:d3part-test",
            )
            self.assertEqual(binding["status"], "pass")
            self.assertEqual(binding["returned_artifact_sha256"], manifest_sha)
            self.assertTrue(binding["evidence_sha256"].startswith("sha256:"))

            manifest.write_text('{"schema_version":"tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "returned artifact manifest hash"):
                load_run_integrity_binding(
                    output,
                    expected_d3part_family_sha256="sha256:d3part-test",
                )

    def test_focus_solid_history_uses_alive_elements_and_stops_first_failure_at_deletion(
        self,
    ) -> None:
        times = np.asarray([0.0, 0.001, 0.002])
        element_ids = np.asarray([11, 12, 21, 22], dtype=np.int64)
        part_ids = np.asarray([1002, 2002], dtype=np.int64)
        part_indexes = np.asarray([0, 0, 1, 1], dtype=np.int64)
        alive = np.asarray(
            [
                [1, 1, 1, 1],
                [1, 0, 1, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )
        effective_plastic_strain = np.asarray(
            [
                [[0.0], [0.0], [0.0], [0.0]],
                [[0.2], [9.0], [0.4], [0.6]],
                [[9.0], [9.0], [0.8], [9.0]],
            ],
            dtype=float,
        )
        stress = np.zeros((3, 4, 1, 6), dtype=float)
        # p = -(sxx + syy + szz) / 3, so these give p=1,2,3,4 GPa.
        stress[1, :, 0, 0:3] = np.asarray(
            [[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0],
             [-3.0, -3.0, -3.0], [-4.0, -4.0, -4.0]]
        )
        stress[2, :, 0, 0:3] = stress[1, :, 0, 0:3]
        focus_cases = [
            {
                "case_id": "p0_v0800_h0p75",
                "plate_part_id": 1002,
                "fragment_part_id": 2002,
                "first_failure_element_id": 11,
                "first_failure_time_ms": 0.0015,
                "first_failure_type": "negative_volume",
                "first_failure_part_kind": "plate",
            }
        ]

        result = extract_focus_solid_history(
            times_ms=times,
            element_stress_gpa=stress,
            element_effective_plastic_strain=effective_plastic_strain,
            element_alive=alive,
            element_ids=element_ids,
            element_part_indexes=part_indexes,
            part_ids=part_ids,
            focus_cases=focus_cases,
        )

        rows = result["part_history_rows"]
        self.assertEqual(len(rows), 6)
        plate_at_1 = next(
            row
            for row in rows
            if row["part_kind"] == "plate" and row["d3plot_state_index"] == 1
        )
        self.assertEqual(plate_at_1["alive_element_count"], 1)
        self.assertAlmostEqual(plate_at_1["effective_plastic_strain_mean"], 0.2)
        self.assertAlmostEqual(
            plate_at_1["derived_solid_hydrostatic_pressure_gpa_mean"], 1.0
        )
        fragment_at_1 = next(
            row
            for row in rows
            if row["part_kind"] == "fragment" and row["d3plot_state_index"] == 1
        )
        self.assertEqual(fragment_at_1["alive_element_count"], 2)
        self.assertAlmostEqual(fragment_at_1["effective_plastic_strain_p50"], 0.5)
        self.assertAlmostEqual(
            fragment_at_1["derived_solid_hydrostatic_pressure_gpa_mean"], 3.5
        )

        first_history = result["first_failure_element_history_rows"]
        self.assertEqual([row["d3plot_time_ms"] for row in first_history], [0.0, 0.001])
        self.assertAlmostEqual(first_history[-1]["effective_plastic_strain"], 0.2)
        self.assertAlmostEqual(
            first_history[-1]["derived_solid_hydrostatic_pressure_gpa"], 1.0
        )
        first_summary = result["cases"][0]["first_failure_element"]
        self.assertAlmostEqual(first_summary["last_alive_d3plot_time_ms"], 0.001)
        self.assertAlmostEqual(first_summary["first_deleted_d3plot_time_ms"], 0.002)

    def test_contact_output_assessment_distinguishes_solid_pressure_from_contact_pressure(
        self,
    ) -> None:
        header = SimpleNamespace(
            has_node_max_contact_penetration_absolute=False,
            has_node_max_contact_penetration_relative=False,
            has_node_contact_energy_density=False,
        )
        assessment = assess_contact_output_availability(
            header=header,
            d3plot_array_keys={
                "timesteps",
                "element_solid_stress",
                "element_solid_effective_plastic_strain",
            },
            binout_inventory={
                "glstat": ["time", "time_step"],
                "sleout": ["time", "total_energy", "friction_energy"],
            },
        )

        self.assertEqual(
            assessment["contact_penetration"]["status"], "unavailable_not_written"
        )
        self.assertEqual(
            assessment["contact_pressure"]["status"], "unavailable_not_written"
        )
        self.assertTrue(
            assessment["derived_solid_hydrostatic_pressure"]["available"]
        )
        self.assertEqual(
            assessment["derived_solid_hydrostatic_pressure"]["formula"],
            "p=-(sxx+syy+szz)/3",
        )
        self.assertFalse(
            assessment["derived_solid_hydrostatic_pressure"]["is_contact_pressure"]
        )

    def test_output_location_preserves_unrelated_stage_a_files_and_rejects_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            returned = root / "05"
            output = root / "analysis"
            returned.mkdir()
            output.mkdir()
            (output / "run_integrity_audit.json").write_text("{}", encoding="utf-8")

            validate_output_location(returned, output)

            (output / "failure_mode_analysis.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "failure_mode_analysis.json"):
                validate_output_location(returned, output)
            validate_output_location(returned, output, allow_replace_owned=True)

    def test_mes_parser_deduplicates_rank_observations_and_keeps_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "mes0000").write_text(
                "negative volume in solid element # 101 cycle 3 time 1.2500E-03\n"
                "solid element 202 failed at time 2.5000E-03\n",
                encoding="utf-8",
            )
            (root / "mes0001").write_text(
                "negative volume in solid element # 101 cycle 3 time 1.2500E-03\n"
                "solid element 202 failed at time 2.5000E-03\n"
                "solid element 202 failed at time 2.7500E-03\n",
                encoding="utf-8",
            )

            parsed = parse_mes_failure_events(root)

        self.assertEqual(parsed["rank_file_count"], 2)
        self.assertEqual(parsed["raw_observation_count"], 5)
        self.assertEqual(len(parsed["unique_events"]), 2)
        by_type = {row["failure_type"]: row for row in parsed["unique_events"]}
        self.assertEqual(by_type["negative_volume"]["observation_count"], 2)
        self.assertEqual(by_type["other_solid_failure"]["observation_count"], 3)
        self.assertAlmostEqual(by_type["other_solid_failure"]["time_ms"], 0.0025)
        self.assertEqual(
            by_type["other_solid_failure"]["source_refs"],
            "mes0000:2;mes0001:2;mes0001:3",
        )

    def test_event_binding_uses_deck_part_and_frozen_case_map(self) -> None:
        events = [
            {"failure_type": "negative_volume", "element_id": 101, "time_ms": 0.001},
            {"failure_type": "other_solid_failure", "element_id": 202, "time_ms": 0.002},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            deck = root / "model.k"
            deck.write_text(
                "*KEYWORD\n*NODE\n"
                "1,0,0,-0.5\n2,1,0,-0.5\n3,1,1,-0.5\n4,0,1,-0.5\n"
                "5,0,0,0.5\n6,1,0,0.5\n7,1,1,0.5\n8,0,1,0.5\n"
                "9,0,0,-11.5\n10,1,0,-11.5\n11,1,1,-11.5\n12,0,1,-11.5\n"
                "13,0,0,-10.5\n14,1,0,-10.5\n15,1,1,-10.5\n16,0,1,-10.5\n"
                "*ELEMENT_SOLID\n101,1001,1,2,3,4,5,6,7,8\n"
                "202,2001,9,10,11,12,13,14,15,16\n*END\n",
                encoding="utf-8",
            )
            case_map = root / "cases.csv"
            with case_map.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id",
                        "role",
                        "plate_part_id",
                        "fragment_part_id",
                        "impact_speed_mps",
                        "mesh_mm",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "case-a",
                        "role": "hf_response_fit",
                        "plate_part_id": 1001,
                        "fragment_part_id": 2001,
                        "impact_speed_mps": 800,
                        "mesh_mm": 0.75,
                    }
                )

            bound = bind_events_to_cases(events, deck, case_map)

        self.assertEqual(bound[0]["case_id"], "case-a")
        self.assertEqual(bound[0]["part_kind"], "plate")
        self.assertAlmostEqual(bound[0]["initial_deck_centroid_z_mm"], 0.0)
        self.assertEqual(bound[1]["part_kind"], "fragment")
        self.assertAlmostEqual(bound[1]["initial_deck_centroid_z_mm"], -11.0)

    def test_fragment_classifier_matches_capture_ledger_states_and_deletion_brackets(self) -> None:
        times = np.asarray([0.0, 0.001, 0.002])
        reference = np.concatenate([cube_nodes(-11.0) for _ in range(4)], axis=0)
        positions = np.stack([reference.copy() for _ in range(3)])
        connectivity = np.arange(32, dtype=np.int64).reshape(4, 8)

        # E1 crosses alive, then is deleted; E2 deletes upstream; E3 survives
        # upstream; E4 disappears in the interval where it might cross.
        positions[1:, 0:8, 2] -= 2.0
        positions[2, 0:8, 2] -= 0.2
        positions[1:, 8:16, 2] -= 0.2
        positions[1:, 16:24, 2] -= 0.2
        positions[1:, 24:32, 2] -= 2.0
        velocities = np.zeros_like(positions)
        velocities[:, :, 2] = -1000.0
        alive = np.asarray(
            [
                [1, 1, 1, 1],
                [1, 0, 1, 0],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )

        result = classify_fragment_elements(
            times_ms=times,
            node_positions_mm=positions,
            node_velocities_mm_per_ms=velocities,
            element_connectivity=connectivity,
            element_alive=alive,
            element_ids=np.asarray([1, 2, 3, 4]),
            element_reference_mass_kg=np.asarray([0.25, 0.25, 0.25, 0.25]),
            capture_plane_z_mm=-12.0,
            maximum_unambiguous_step_mm=3.0,
        )
        by_id = {row["element_id"]: row for row in result}

        self.assertEqual(by_id[1]["ledger_state"], "deleted_after_crossing")
        self.assertAlmostEqual(by_id[1]["crossing_time_ms"], 0.0005)
        self.assertEqual(by_id[1]["d3part_deletion_interval_start_ms"], 0.001)
        self.assertEqual(by_id[1]["d3part_deletion_interval_end_ms"], 0.002)
        self.assertEqual(by_id[2]["ledger_state"], "deleted_upstream")
        self.assertEqual(by_id[3]["ledger_state"], "alive_upstream")
        self.assertEqual(by_id[4]["ledger_state"], "temporal_ambiguity")

    def test_association_and_summary_keep_plate_initial_geometry_but_not_deformed_position(
        self,
    ) -> None:
        events = [
            {
                "failure_type": "negative_volume",
                "element_id": 101,
                "time_ms": 0.001,
                "part_id": 1001,
                "case_id": "case-a",
                "case_role": "hf_response_fit",
                "part_kind": "plate",
                "initial_deck_centroid_x_mm": 0.5,
                "initial_deck_centroid_y_mm": 0.5,
                "initial_deck_centroid_z_mm": 0.0,
            },
            {
                "failure_type": "other_solid_failure",
                "element_id": 202,
                "time_ms": 0.002,
                "part_id": 2001,
                "case_id": "case-a",
                "case_role": "hf_response_fit",
                "part_kind": "fragment",
                "initial_deck_centroid_x_mm": 0.5,
                "initial_deck_centroid_y_mm": 0.5,
                "initial_deck_centroid_z_mm": -11.0,
            },
        ]
        ledger_rows = [
            {
                "element_id": 202,
                "ledger_state": "deleted_upstream",
                "initial_centroid_x_mm": 0.5,
                "initial_centroid_y_mm": 0.5,
                "initial_centroid_z_mm": -11.0,
                "d3part_deletion_interval_start_ms": 0.001,
                "d3part_deletion_interval_end_ms": 0.002,
                "last_alive_centroid_z_mm": -11.2,
                "first_dead_centroid_z_mm": -11.4,
            }
        ]
        associated = associate_failure_events_with_d3part(events, ledger_rows)

        self.assertEqual(associated[0]["d3part_association"], "not_in_fragment_only_d3part")
        self.assertIsNone(associated[0]["ledger_state"])
        self.assertEqual(associated[1]["d3part_association"], "matched_fragment_element")
        self.assertEqual(associated[1]["ledger_state"], "deleted_upstream")
        self.assertEqual(associated[1]["fragment_initial_reference_crosscheck_status"], "match")
        self.assertAlmostEqual(
            associated[1]["fragment_initial_reference_centroid_max_abs_delta_mm"], 0.0
        )

        cases = [
            {
                "case_id": "case-a",
                "role": "hf_response_fit",
                "plate_part_id": "1001",
                "fragment_part_id": "2001",
                "impact_speed_mps": "800",
                "mesh_mm": "0.75",
            }
        ]
        ledger = {
            "case-a": {
                "primary_qoi_available": False,
                "mass_fraction_by_ledger_state": {"deleted_upstream": 0.25},
                "unavailable_reasons": ["deleted_upstream_mass"],
            }
        }
        summary = summarize_cases(associated, cases, ledger)
        self.assertEqual(summary[0]["first_failure_type"], "negative_volume")
        self.assertEqual(summary[0]["plate_negative_volume_count"], 1)
        self.assertEqual(summary[0]["fragment_other_failure_count"], 1)
        self.assertEqual(summary[0]["fragment_mes_time_in_d3part_deletion_interval_count"], 1)
        self.assertEqual(summary[0]["fragment_mes_time_outside_d3part_deletion_interval_count"], 0)
        self.assertEqual(summary[0]["deleted_upstream_mass_fraction"], 0.25)
        self.assertNotIn("capture_before_deletion_mass_fraction", summary[0])
        self.assertEqual(summary[0]["primary_qoi_available"], False)
        self.assertTrue(summary[0]["qualification_use_forbidden"])
        self.assertAlmostEqual(summary[0]["plate_failure_initial_deck_z_min_mm"], 0.0)
        self.assertAlmostEqual(summary[0]["plate_failure_initial_deck_z_max_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
