from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from extract_r05_d3part import ArrayType, D3plotFiletype  # noqa: E402
from build_p0_r06_diagnostic_bundles import build_bundles  # noqa: E402
from create_p0_r06f_measurement_repair_design import (  # noqa: E402
    build_contract,
    build_design,
)
from r06_single_job_adapter import (  # noqa: E402
    assess_r06_eroding_contact_outputs,
    extract_mat015_hvar6_from_arrays,
    normalize_r06_single_job_capture_inputs,
    summarize_ragged_ncforc_states,
    strict_json_dumps,
)
from stage_r06f_wave0_canary import stage_wave0  # noqa: E402
from validate_p0_r06_diagnostics import grouped_blocks, validate_batch  # noqa: E402


DESIGN = PROJECT / "design" / "p0_r06f_measurement_repair_design.json"
CONTRACT = PROJECT / "design" / "p0_r06f_real_output_contract.json"
BATCH = PROJECT / "manual_upload_20261027_r06f_measurement_contract"
JOB = BATCH / "job_01_v0800_control"
RELEASE = PROJECT / "artifacts" / "r06f_measurement_contract_20261027"


def damage_fixture() -> tuple[dict, SimpleNamespace, dict]:
    history = np.zeros((3, 4, 1, 7), dtype=float)
    history[..., 3] = -1.0e30  # Known irrelevant channel must not contaminate HVAR6.
    history[:, :, 0, 5] = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.2, 0.4, 0.6, 0.8],
            [0.3, 0.5, 0.7, 0.0],  # deleted solid's reset-to-zero value is ignored
        ]
    )
    history[:, :, 0, 6] = 99.0  # Channel 7 deliberately differs from HVAR6.
    arrays = {
        ArrayType.global_timesteps: np.asarray([0.0, 0.001, 0.002]),
        ArrayType.element_solid_history_variables: history,
        ArrayType.element_solid_is_alive: np.asarray(
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=float
        ),
        ArrayType.element_solid_ids: np.asarray([10, 11, 20, 21]),
        ArrayType.element_solid_part_indexes: np.asarray([0, 0, 1, 1]),
        ArrayType.part_ids: np.asarray([1002, 2002]),
    }
    header = SimpleNamespace(filetype=D3plotFiletype.D3PLOT)
    case = {
        "plate_part_id": 1002,
        "fragment_part_id": 2002,
        "plate_element_count": 2,
        "fragment_element_count": 2,
    }
    return arrays, header, case


class R06fMeasurementRepairTests(unittest.TestCase):
    def test_design_and_generated_batch_are_single_job_and_static_pass(self) -> None:
        design = json.loads(DESIGN.read_text(encoding="utf-8"))
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(design["package_revision"], "R06_measurement_contract_repair_20261027")
        self.assertEqual(design["submission_sequence"]["wave_0"], ["r06f_0800_ctrl"])
        self.assertEqual(
            [job["job_id"] for exp in design["paired_experiments"] for job in exp["jobs"]],
            ["r06f_0800_ctrl"],
        )
        self.assertFalse(
            design["shared_measurement_upgrade"]["direct_contact_output_policy"][
                "request_binary_intfor"
            ]
        )
        self.assertTrue(
            design["shared_measurement_upgrade"]["direct_contact_output_policy"][
                "request_ascii_ncforc"
            ]
        )
        self.assertEqual(contract["d3part_keyword"]["extent_cards"][0][0], 6)
        result = validate_batch(BATCH, write_reports=False)
        self.assertEqual(result["overall_status"], "pass")
        self.assertEqual(result["job_count"], 1)

    def test_frozen_design_and_contract_match_the_reproducible_builder(self) -> None:
        parent = json.loads(
            (PROJECT / "design" / "p0_r06_keyword_parser_repair_design.json").read_text(
                encoding="utf-8"
            )
        )
        r05 = json.loads(
            (PROJECT / "design" / "p0_cases_r05.json").read_text(encoding="utf-8")
        )
        contract = build_contract(r05)
        design = build_design(parent, contract)
        self.assertEqual(
            contract,
            json.loads(CONTRACT.read_text(encoding="utf-8")),
        )
        self.assertEqual(
            design,
            json.loads(DESIGN.read_text(encoding="utf-8")),
        )

    def test_deck_removes_only_unsupported_intfor_and_adds_d3part_extent(self) -> None:
        deck = (JOB / "r06f_0800_ctrl.k").read_text(encoding="ascii")
        grouped = grouped_blocks(deck)
        self.assertNotIn("*DATABASE_BINARY_INTFOR_FILE", grouped)
        self.assertNotIn("*DATABASE_EXTENT_INTFOR", grouped)
        self.assertIn("*DATABASE_NCFORC", grouped)
        self.assertEqual(
            grouped["*DATABASE_EXTENT_D3PART"][0],
            ["6,0,3,0,1,1,1,1", "0,0,0,0,0,0,0,0"],
        )
        report = json.loads((JOB / "static_validation.json").read_text(encoding="utf-8"))
        self.assertEqual(report["checks"]["immutable_r06e_physics_parity"]["status"], "pass")

    def test_single_job_contract_normalization_is_self_contained(self) -> None:
        manifest = json.loads((JOB / "manifest.json").read_text(encoding="utf-8"))
        design = json.loads((JOB / manifest["design"]["path"]).read_text(encoding="utf-8"))
        contract = json.loads(
            (JOB / manifest["successor_contract"]["path"]).read_text(encoding="utf-8")
        )
        job_contract = json.loads(
            (JOB / manifest["job_contract"]["path"]).read_text(encoding="utf-8")
        )
        with (JOB / manifest["case_map"]["path"]).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            case_row = next(csv.DictReader(handle))
        mapped, rows = normalize_r06_single_job_capture_inputs(
            job_contract=job_contract,
            design=design,
            measurement_contract=contract,
            case_row=case_row,
        )
        self.assertEqual(mapped["common"]["termination_time_ms"], 0.04)
        self.assertEqual(mapped["cases"][0]["impact_speed_mps"], 800.0)
        self.assertEqual(rows[0]["fragment_part_id"], "2002")

    def test_hvar6_uses_index_five_and_masks_post_deletion_zero(self) -> None:
        arrays, header, case = damage_fixture()
        result = extract_mat015_hvar6_from_arrays(
            arrays=arrays, header=header, case=case
        )
        self.assertEqual(result["status"], "available")
        fragment = next(part for part in result["parts"] if part["part"] == "fragment")
        self.assertAlmostEqual(fragment["maximum"], 0.8)
        element = next(row for row in result["element_last_alive"] if row["element_id"] == 21)
        self.assertAlmostEqual(element["last_alive_hvar6"], 0.8)
        self.assertEqual(element["first_deleted_state_index"], 2)
        self.assertNotEqual(fragment["maximum"], 99.0)

    def test_hvar6_invalid_alive_value_is_null_but_dead_sentinel_is_ignored(self) -> None:
        arrays, header, case = damage_fixture()
        arrays[ArrayType.element_solid_history_variables][2, 3, 0, 5] = -1.0e30
        result = extract_mat015_hvar6_from_arrays(
            arrays=arrays, header=header, case=case
        )
        self.assertEqual(result["status"], "available")
        arrays[ArrayType.element_solid_history_variables][1, 2, 0, 5] = -1.0e30
        result = extract_mat015_hvar6_from_arrays(
            arrays=arrays, header=header, case=case
        )
        self.assertEqual(result["status"], "unavailable")
        fragment = next(part for part in result["parts"] if part["part"] == "fragment")
        self.assertIsNone(fragment["maximum"])
        self.assertNotEqual(fragment["maximum"], 0)

    def test_hvar6_rejects_short_history_and_element_revival(self) -> None:
        arrays, header, case = damage_fixture()
        short = dict(arrays)
        short[ArrayType.element_solid_history_variables] = arrays[
            ArrayType.element_solid_history_variables
        ][..., :5]
        with self.assertRaisesRegex(ValueError, "nvar >= 6"):
            extract_mat015_hvar6_from_arrays(arrays=short, header=header, case=case)
        revival = dict(arrays)
        revival[ArrayType.element_solid_is_alive] = np.asarray(
            [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=float
        )
        with self.assertRaisesRegex(ValueError, "revived"):
            extract_mat015_hvar6_from_arrays(arrays=revival, header=header, case=case)

    def test_contact_warning_and_missing_are_not_zero(self) -> None:
        warning = (
            "Warning 20340 (STR+340): binary interface force data is unavailable "
            "for eroding contact types - interface # 1"
        )
        result = assess_r06_eroding_contact_outputs(
            message_sources={"std.out": warning + "\n" + warning},
            returned_names=["binout0000", "sleout"],
            binout_inventory={"sleout": ["time", "energy"]},
        )
        self.assertEqual(result["intfor"]["status"], "unavailable_by_solver_capability")
        self.assertIsNone(result["intfor"]["value"])
        self.assertEqual(result["warning_20340"]["occurrence_count"], 2)
        self.assertEqual(result["ncforc"]["status"], "unavailable_not_returned_best_effort")
        self.assertFalse(result["sleout"]["direct_contact_substitute"])
        unexplained = assess_r06_eroding_contact_outputs(
            message_sources={}, returned_names=[], binout_inventory={}
        )
        self.assertEqual(unexplained["intfor"]["status"], "missing_unexplained")
        with self.assertRaises(ValueError):
            strict_json_dumps({"invalid": float("nan")})

    def test_contract_not_requested_intfor_and_binout_contact_inventory(self) -> None:
        result = assess_r06_eroding_contact_outputs(
            message_sources={},
            returned_names=["binout0000", "binout0006"],
            binout_inventory={
                "JOBINFO": ["solver"],
                "ncforc": ["surfa_1"],
                "sleout": ["time", "energy"],
            },
            intfor_requested=False,
        )
        self.assertEqual(result["intfor"]["status"], "not_requested_under_contract")
        self.assertIsNone(result["intfor"]["value"])
        self.assertEqual(result["ncforc"]["status"], "present_pending_reader_validation")
        self.assertEqual(result["sleout"]["status"], "available_energy_only")
        self.assertFalse(result["missing_is_zero"])

    def test_ragged_ncforc_summary_preserves_per_state_node_ids(self) -> None:
        states = [
            {
                "state_name": "d000001",
                "time_ms": 0.0,
                "ids": np.asarray([10, 20]),
                "x": np.asarray([0.0, 1.0]),
                "y": np.asarray([0.0, 0.0]),
                "z": np.asarray([0.0, 0.0]),
                "pressure": np.asarray([1.0, 2.0]),
                "x_force": np.asarray([0.1, 0.2]),
                "y_force": np.asarray([0.0, 0.0]),
                "z_force": np.asarray([-0.1, -0.2]),
            },
            {
                "state_name": "d000002",
                "time_ms": 0.1,
                "ids": np.asarray([20, 30, 40]),
                "x": np.asarray([1.0, 2.0, 3.0]),
                "y": np.asarray([0.0, 0.0, 0.0]),
                "z": np.asarray([0.0, 0.0, 0.0]),
                "pressure": np.asarray([3.0, 4.0, 5.0]),
                "x_force": np.asarray([0.3, 0.4, 0.5]),
                "y_force": np.asarray([0.0, 0.0, 0.0]),
                "z_force": np.asarray([-0.3, -0.4, -0.5]),
            },
        ]
        summary = summarize_ragged_ncforc_states(states)
        self.assertEqual(summary["status"], "available_ragged_pending_semantic_validation")
        self.assertEqual(summary["state_count"], 2)
        self.assertEqual(summary["minimum_nodes_per_state"], 2)
        self.assertEqual(summary["maximum_nodes_per_state"], 3)
        self.assertEqual(summary["union_node_count"], 4)
        self.assertEqual(summary["all_state_intersection_node_count"], 1)
        self.assertEqual(summary["states"][0]["node_ids"], [10, 20])
        self.assertEqual(summary["states"][1]["node_ids"], [20, 30, 40])
        self.assertEqual(summary["raw_model_unit_ranges"]["pressure"], [1.0, 5.0])

        invalid = [dict(states[0])]
        invalid[0]["pressure"] = np.asarray([1.0, float("nan")])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            summarize_ragged_ncforc_states(invalid)

    def test_wave0_stage_has_exactly_four_files_and_no_directory(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06f-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            manifest = stage_wave0(RELEASE, output)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "uav_hf_r06f_0800_ctrl.zip",
                    "uav_hf_r06f_0800_ctrl.bundle_manifest.json",
                    "SUBMIT_ONLY_THIS_JOB.md",
                    "WAVE0_MANIFEST.json",
                },
            )
            self.assertTrue(all(path.is_file() for path in output.iterdir()))
            self.assertEqual(
                manifest["status"],
                "ready_for_user_submission_not_submitted_not_solver_validated",
            )

    def test_two_independent_bundle_builds_are_byte_identical(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06f-determinism-", dir=temp_root) as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            build_bundles(BATCH, first)
            build_bundles(BATCH, second)
            self.assertEqual(
                (first / "uav_hf_r06f_0800_ctrl.zip").read_bytes(),
                (second / "uav_hf_r06f_0800_ctrl.zip").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
