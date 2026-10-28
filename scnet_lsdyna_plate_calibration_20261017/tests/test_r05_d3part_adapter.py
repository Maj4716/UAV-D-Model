from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import extract_r05_d3part as d3part_adapter  # noqa: E402
from extract_r05_d3part import (  # noqa: E402
    ArrayType,
    D3plotFiletype,
    artifact_identity_from_paths,
    assemble_output_payload,
    bridge_extraction_to_assessor_payload,
    canonical_sha256,
    d3part_family_identity,
    extract_case_ledgers_from_arrays,
    parser_source_sha256_from_extraction,
    sha256_file,
    strict_json,
)
from assess_r05_response import (  # noqa: E402
    EXPECTED_CASE_MAP_SHA256,
    EXPECTED_DECK_SHA256,
    EXPECTED_DESIGN_CANONICAL_SHA256,
    EXPECTED_DESIGN_SHA256,
    EXPECTED_PACKAGE_MANIFEST_SHA256,
    assess_r05_response,
)


def fixture_design() -> dict:
    return {
        "package_revision": "R05_capture_ledger_preregistered",
        "measurement_contract": {
            "fragment_d3part_interval_ms": 0.001,
            "capture_plane": {"coordinate_mm": -12.0},
        },
        "common": {"termination_time_ms": 0.002},
        "qualification_gates": {
            "capture_ledger": {
                "initial_discrete_mass_relative_error_max": 0.005,
                "mass_accounting_relative_error_max": 1e-6,
                "temporal_ambiguity_mass_fraction_max": 0.01,
                "deleted_upstream_mass_fraction_max": 0.05,
                "tail_window_fraction": 0.10,
                "tail_increment_qM_max": 0.01,
                "tail_increment_qP_max": 0.01,
                "tail_increment_qE_max": 0.01,
                "tail_qV_relative_range_max": 0.05,
            },
            "numerical_perforation_classification": {
                "confirmed_perforation_qM_min": 0.02,
                "confirmed_perforation_qP_min": 0.01,
                "confirmed_nonperforation_qM_max": 0.005,
                "confirmed_nonperforation_qP_max": 0.005,
                "confirmed_nonperforation_worst_case_crossing_mass_max": 0.005,
            },
        },
        "cases": [
            {
                "case_id": "case-a",
                "role": "hf_response_fit",
                "impact_speed_mps": 1000.0,
                "mesh_mm": 4.0,
            }
        ],
    }


def fixture_case_rows() -> list[dict]:
    return [
        {
            "case_id": "case-a",
            "role": "hf_response_fit",
            "fragment_part_id": "2001",
            "impact_speed_mps": "1000.0",
            "mesh_mm": "4.0",
            "fragment_elements": "1",
            "capture_plane_z_mm": "-12.0",
            "fragment_initial_reference_mass_kg": "1.0",
        }
    ]


def fixture_arrays(*, displacement_mode: bool = False) -> dict[str, np.ndarray]:
    reference = np.asarray(
        [
            [0.0, 0.0, -11.5],
            [1.0, 0.0, -11.5],
            [1.0, 1.0, -11.5],
            [0.0, 1.0, -11.5],
            [0.0, 0.0, -10.5],
            [1.0, 0.0, -10.5],
            [1.0, 1.0, -10.5],
            [0.0, 1.0, -10.5],
        ]
    )
    positions = np.stack((reference, reference.copy(), reference.copy()))
    positions[1, :, 2] -= 1.1
    positions[2, :, 2] -= 1.2
    state_coordinates = positions - reference if displacement_mode else positions
    velocities = np.zeros_like(positions)
    velocities[:, :, 2] = -1000.0
    return {
        ArrayType.global_timesteps: np.asarray([0.0, 0.001, 0.002]),
        ArrayType.node_coordinates: reference,
        ArrayType.node_displacement: state_coordinates,
        ArrayType.node_velocity: velocities,
        ArrayType.element_solid_node_indexes: np.arange(8, dtype=np.int64).reshape(1, 8),
        ArrayType.element_solid_part_indexes: np.asarray([0], dtype=np.int64),
        ArrayType.element_solid_is_alive: np.ones((3, 1), dtype=float),
        ArrayType.part_ids: np.asarray([2001], dtype=np.int64),
    }


def fixture_header() -> SimpleNamespace:
    return SimpleNamespace(
        filetype=D3plotFiletype.D3PART,
        has_node_displacement=True,
        has_node_velocity=True,
        has_element_deletion_data=True,
    )


def fixture_solver_timeline() -> dict[str, np.ndarray]:
    return {
        "solver_state_times_ms": np.asarray([0.0, 0.001, 0.002]),
        "solver_time_steps_ms": np.asarray([0.0001, 0.0001, 0.0001]),
    }


def passing_assessor_cases(design: dict) -> list[dict]:
    classifications = {
        600.0: "confirmed_nonperforation",
        800.0: "confirmed_nonperforation",
        1000.0: "confirmed_nonperforation",
        1200.0: "confirmed_perforation",
        2000.0: "confirmed_perforation",
    }
    records = []
    for case in design["cases"]:
        perforated = classifications[float(case["impact_speed_mps"])] == "confirmed_perforation"
        records.append(
            {
                "ledger_schema_version": "hf-p0-capture-ledger-result@1.1.0",
                "case_id": case["case_id"],
                "impact_speed_mps": case["impact_speed_mps"],
                "mesh_mm": case["mesh_mm"],
                "primary_qoi_available": True,
                "unavailable_reasons": [],
                "run_integrity_gate": "not_assessed_by_d3part_extractor",
                "classification": classifications[float(case["impact_speed_mps"])],
                "qM": 0.20 if perforated else 0.0,
                "qP": 0.15 if perforated else 0.0,
                "qE": 0.12 if perforated else 0.0,
                "qV_over_v0": 0.75 if perforated else None,
                "initial_discrete_mass_relative_error": 0.0,
                "mass_accounting_relative_error": 0.0,
                "mass_fraction_by_ledger_state": {
                    "alive_crossed": 0.20 if perforated else 0.0,
                    "alive_upstream": 0.80 if perforated else 1.0,
                    "deleted_upstream": 0.0,
                    "deleted_after_crossing": 0.0,
                    "temporal_ambiguity": 0.0,
                },
                "missing_reference_mass_fraction": 0.0,
                "unresolved_crossing_mass_fraction": 0.0,
                "worst_case_crossing_mass_fraction": 0.20 if perforated else 0.0,
                "tail_window": {
                    "increment_qM": 0.0,
                    "increment_qP": 0.0,
                    "increment_qE": 0.0,
                    "qV_relative_range": 0.0,
                },
            }
        )
    return records


class R05D3partAdapterTests(unittest.TestCase):
    def test_reader_loader_appends_vendor_path_in_fresh_process_without_h5py_preload(self) -> None:
        code = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(SCRIPTS)!r})
import extract_r05_d3part as adapter
assert 'h5py' not in sys.modules
adapter._load_lasso_d3plot()
vendor = adapter.VENDOR_DIR.resolve()
assert Path(sys.modules['lasso'].__file__).resolve().is_relative_to(vendor)
assert Path(sys.path[-1]).resolve() == vendor
print(Path(sys.modules['h5py'].__file__).resolve())
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    @unittest.skipUnless(
        (PROJECT / "05" / "d3part").is_file()
        and (PROJECT / "05" / "binout0000").is_file(),
        "real R05 d3part/binout family is not present",
    )
    def test_real_r05_d3part_family_passes_adapter_integration_seam(self) -> None:
        result_dir = PROJECT / "05"
        design = json.loads((result_dir / "p0_cases_r05.json").read_text(encoding="utf-8"))
        material = json.loads((result_dir / "jc4340_reference.json").read_text(encoding="utf-8"))
        with (result_dir / "p0_case_part_map.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            case_rows = list(csv.DictReader(handle))

        reader_class = d3part_adapter._load_lasso_d3plot()
        d3part = reader_class(
            str(result_dir / "d3part"),
            state_array_filter=[
                ArrayType.global_timesteps,
                ArrayType.node_displacement,
                ArrayType.node_velocity,
                ArrayType.element_solid_is_alive,
            ],
            buffered_reading=True,
        )
        solver_timeline = d3part_adapter.load_solver_timeline_from_binout(
            result_dir / "binout*"
        )
        result = extract_case_ledgers_from_arrays(
            arrays=d3part.arrays,
            header=d3part.header,
            design=design,
            case_rows=case_rows,
            material=material,
            **solver_timeline,
        )

        summary = result["d3part_summary"]
        self.assertEqual(summary["state_count"], 802)
        self.assertEqual(summary["solid_count"], 5680)
        self.assertEqual(summary["part_ids"], [2001, 2002, 2003, 2004, 2005, 2006, 2007])
        self.assertEqual(
            summary["zero_solid_catalog_part_ids"],
            [1001, 1002, 1003, 1004, 1005, 1006, 1007],
        )
        self.assertGreater(
            summary["maximum_state_interval_ms"], summary["expected_state_interval_ms"]
        )
        self.assertEqual(summary["temporal_gate"]["status"], "pass")
        self.assertTrue(
            all(
                item["overlimit_alive_crossing_count"] == 0
                for item in summary["temporal_gate"]["case_crossing_resolution"]
            )
        )
        self.assertEqual(len(result["cases"]), 7)

    def test_accepts_nominal_output_delay_only_when_local_explicit_step_explains_it(self) -> None:
        arrays = fixture_arrays()
        arrays[ArrayType.global_timesteps] = np.asarray([0.0, 0.0012, 0.002])

        result = extract_case_ledgers_from_arrays(
            arrays=arrays,
            header=fixture_header(),
            design=fixture_design(),
            case_rows=fixture_case_rows(),
            material={"strength_model": {"density_kg_mm3": 1.0}},
            solver_state_times_ms=np.asarray([0.0, 0.0012, 0.002]),
            solver_time_steps_ms=np.asarray([0.00025, 0.00025, 0.0001]),
        )

        gate = result["d3part_summary"]["temporal_gate"]
        self.assertEqual(gate["schema_version"], "hf-p0-d3part-temporal-gate@1.0.0")
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["nominal_output_interval_ms"], 0.001)
        self.assertEqual(gate["maximum_actual_state_interval_ms"], 0.0012)
        self.assertEqual(gate["crossing_spatial_resolution_mesh_fraction"], 0.5)

    def test_rejects_nominal_output_delay_not_explained_by_explicit_solver_step(self) -> None:
        arrays = fixture_arrays()
        arrays[ArrayType.global_timesteps] = np.asarray([0.0, 0.0012, 0.002])

        with self.assertRaisesRegex(ValueError, "not realizable.*local explicit solver step"):
            extract_case_ledgers_from_arrays(
                arrays=arrays,
                header=fixture_header(),
                design=fixture_design(),
                case_rows=fixture_case_rows(),
                material={"strength_model": {"density_kg_mm3": 1.0}},
                solver_state_times_ms=np.asarray([0.0, 0.0012, 0.002]),
                solver_time_steps_ms=np.asarray([0.0001, 0.0001, 0.0001]),
            )

    def test_records_half_mesh_crossing_resolution_and_routes_overlimit_mass_to_ambiguity(self) -> None:
        design = fixture_design()
        design["cases"][0]["mesh_mm"] = 2.0
        rows = fixture_case_rows()
        rows[0]["mesh_mm"] = "2.0"

        result = extract_case_ledgers_from_arrays(
            arrays=fixture_arrays(),
            header=fixture_header(),
            design=design,
            case_rows=rows,
            material={"strength_model": {"density_kg_mm3": 1.0}},
            **fixture_solver_timeline(),
        )

        resolution = result["d3part_summary"]["temporal_gate"][
            "case_crossing_resolution"
        ][0]
        self.assertEqual(resolution["case_id"], "case-a")
        self.assertEqual(resolution["spatial_resolution_limit_mm"], 1.0)
        self.assertAlmostEqual(resolution["maximum_alive_crossing_step_mm"], 1.1)
        self.assertEqual(resolution["overlimit_alive_crossing_count"], 1)
        self.assertFalse(result["cases"][0]["primary_qoi_available"])
        self.assertIn("temporal_ambiguity_mass", result["cases"][0]["unavailable_reasons"])

    def test_allows_zero_solid_catalog_parts_but_rejects_wrong_used_solid_pid(self) -> None:
        arrays = fixture_arrays()
        arrays[ArrayType.part_ids] = np.asarray([1001, 2001], dtype=np.int64)
        arrays[ArrayType.element_solid_part_indexes] = np.asarray([1], dtype=np.int64)

        result = extract_case_ledgers_from_arrays(
            arrays=arrays,
            header=fixture_header(),
            design=fixture_design(),
            case_rows=fixture_case_rows(),
            material={"strength_model": {"density_kg_mm3": 1.0}},
            **fixture_solver_timeline(),
        )

        self.assertEqual(result["d3part_summary"]["part_ids"], [2001])
        self.assertEqual(result["d3part_summary"]["catalog_part_ids"], [1001, 2001])
        self.assertEqual(result["d3part_summary"]["zero_solid_catalog_part_ids"], [1001])

        arrays[ArrayType.element_solid_part_indexes] = np.asarray([0], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "used solid PID set"):
            extract_case_ledgers_from_arrays(
                arrays=arrays,
                header=fixture_header(),
                design=fixture_design(),
                case_rows=fixture_case_rows(),
                material={"strength_model": {"density_kg_mm3": 1.0}},
                **fixture_solver_timeline(),
            )

    def test_extracts_mass_bound_capture_ledger_from_absolute_state_coordinates(self) -> None:
        result = extract_case_ledgers_from_arrays(
            arrays=fixture_arrays(),
            header=fixture_header(),
            design=fixture_design(),
            case_rows=fixture_case_rows(),
            material={"strength_model": {"density_kg_mm3": 1.0}},
            **fixture_solver_timeline(),
        )

        self.assertEqual(result["d3part_summary"]["coordinate_mode"], "absolute_coordinates")
        self.assertEqual(result["d3part_summary"]["part_ids"], [2001])
        self.assertEqual(len(result["cases"]), 1)
        case = result["cases"][0]
        self.assertEqual(case["case_id"], "case-a")
        self.assertEqual(case["ledger_schema_version"], "hf-p0-capture-ledger-result@1.1.0")
        self.assertEqual(case["provenance"]["maximum_unambiguous_step_mm"], 2.0)
        self.assertAlmostEqual(case["reference_mass_kg"], 1.0)
        self.assertTrue(case["primary_qoi_available"])
        self.assertEqual(case["classification"], "confirmed_perforation")
        self.assertAlmostEqual(case["qM"], 1.0)
        self.assertAlmostEqual(case["qP"], 1.0)

    def test_accepts_displacement_from_reference_state_coordinates(self) -> None:
        result = extract_case_ledgers_from_arrays(
            arrays=fixture_arrays(displacement_mode=True),
            header=fixture_header(),
            design=fixture_design(),
            case_rows=fixture_case_rows(),
            material={"strength_model": {"density_kg_mm3": 1.0}},
            **fixture_solver_timeline(),
        )

        self.assertEqual(
            result["d3part_summary"]["coordinate_mode"],
            "displacement_from_reference",
        )
        self.assertAlmostEqual(result["cases"][0]["qM"], 1.0)

    def test_rejects_non_d3part_and_missing_required_header_flags(self) -> None:
        cases = (
            ("filetype", {"filetype": D3plotFiletype.D3PLOT}, "not D3PART"),
            (
                "displacement",
                {"has_node_displacement": False},
                "has_node_displacement",
            ),
            ("velocity", {"has_node_velocity": False}, "has_node_velocity"),
            (
                "deletion",
                {"has_element_deletion_data": False},
                "has_element_deletion_data",
            ),
        )
        for name, changes, expected in cases:
            with self.subTest(name=name):
                header = fixture_header()
                for key, value in changes.items():
                    setattr(header, key, value)
                with self.assertRaisesRegex(ValueError, expected):
                    extract_case_ledgers_from_arrays(
                        arrays=fixture_arrays(),
                        header=header,
                        design=fixture_design(),
                        case_rows=fixture_case_rows(),
                        material={"strength_model": {"density_kg_mm3": 1.0}},
                        **fixture_solver_timeline(),
                    )

    def test_rejects_each_missing_required_array(self) -> None:
        for missing_key in (
            ArrayType.global_timesteps,
            ArrayType.node_coordinates,
            ArrayType.node_displacement,
            ArrayType.node_velocity,
            ArrayType.element_solid_node_indexes,
            ArrayType.element_solid_part_indexes,
            ArrayType.element_solid_is_alive,
            ArrayType.part_ids,
        ):
            with self.subTest(missing_key=missing_key):
                arrays = fixture_arrays()
                arrays.pop(missing_key)
                with self.assertRaisesRegex(ValueError, "required arrays are missing"):
                    extract_case_ledgers_from_arrays(
                        arrays=arrays,
                        header=fixture_header(),
                        design=fixture_design(),
                        case_rows=fixture_case_rows(),
                        material={"strength_model": {"density_kg_mm3": 1.0}},
                        **fixture_solver_timeline(),
                    )

    def test_rejects_invalid_state_shapes_nonfinite_values_and_time_order(self) -> None:
        mutations = (
            (
                "time_order",
                lambda arrays: arrays.__setitem__(
                    ArrayType.global_timesteps, np.asarray([0.0, 0.001, 0.001])
                ),
                "strictly increasing",
            ),
            (
                "nonfinite_velocity",
                lambda arrays: arrays[ArrayType.node_velocity].__setitem__((1, 0, 2), np.nan),
                "node_velocity must be finite",
            ),
            (
                "alive_shape",
                lambda arrays: arrays.__setitem__(
                    ArrayType.element_solid_is_alive, np.ones((2, 1))
                ),
                r"shape \[state,solid\]",
            ),
            (
                "connectivity_shape",
                lambda arrays: arrays.__setitem__(
                    ArrayType.element_solid_node_indexes, np.arange(7).reshape(1, 7)
                ),
                r"shape \[solid,8\]",
            ),
            (
                "ambiguous_coordinates",
                lambda arrays: arrays[ArrayType.node_displacement].__setitem__(
                    (0, slice(None), slice(None)), 123.0
                ),
                "neither absolute coordinates nor displacement",
            ),
        )
        for name, mutate, expected in mutations:
            with self.subTest(name=name):
                arrays = fixture_arrays()
                mutate(arrays)
                with self.assertRaisesRegex(ValueError, expected):
                    extract_case_ledgers_from_arrays(
                        arrays=arrays,
                        header=fixture_header(),
                        design=fixture_design(),
                        case_rows=fixture_case_rows(),
                        material={"strength_model": {"density_kg_mm3": 1.0}},
                        **fixture_solver_timeline(),
                    )

    def test_rejects_nonzero_start_missing_state_volume_and_truncated_tail(self) -> None:
        scenarios = []

        nonzero_start = fixture_arrays()
        nonzero_start[ArrayType.global_timesteps] = np.asarray([0.0001, 0.001, 0.002])
        scenarios.append(
            ("nonzero_start", nonzero_start, fixture_design(), "start near zero", fixture_solver_timeline())
        )

        missing_volume = fixture_arrays()
        missing_volume[ArrayType.global_timesteps] = np.asarray([0.0, 0.001, 0.0025])
        scenarios.append(
            (
                "missing_volume",
                missing_volume,
                fixture_design(),
                "not realizable.*local explicit solver step",
                {
                    "solver_state_times_ms": np.asarray([0.0, 0.001, 0.0025]),
                    "solver_time_steps_ms": np.asarray([0.0001, 0.0001, 0.0001]),
                },
            )
        )

        truncated_design = fixture_design()
        truncated_design["common"]["termination_time_ms"] = 0.004
        scenarios.append(
            (
                "truncated_tail",
                fixture_arrays(),
                truncated_design,
                "does not cover termination",
                fixture_solver_timeline(),
            )
        )

        for name, arrays, design, expected, solver_timeline in scenarios:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, expected):
                extract_case_ledgers_from_arrays(
                    arrays=arrays,
                    header=fixture_header(),
                    design=design,
                    case_rows=fixture_case_rows(),
                    material={"strength_model": {"density_kg_mm3": 1.0}},
                    **solver_timeline,
                )

    def test_rejects_pid_part_index_initial_alive_and_element_count_mismatches(self) -> None:
        cases = []
        wrong_pid = fixture_arrays()
        wrong_pid[ArrayType.part_ids] = np.asarray([2002])
        cases.append(("pid", wrong_pid, fixture_case_rows(), "exact fragment PID set"))

        bad_part_index = fixture_arrays()
        bad_part_index[ArrayType.element_solid_part_indexes][0] = 1
        cases.append(("part_index", bad_part_index, fixture_case_rows(), "out-of-range part index"))

        dead_initial = fixture_arrays()
        dead_initial[ArrayType.element_solid_is_alive][0, 0] = 0.0
        cases.append(("initial_alive", dead_initial, fixture_case_rows(), "alive in the initial"))

        wrong_count_rows = copy.deepcopy(fixture_case_rows())
        wrong_count_rows[0]["fragment_elements"] = "2"
        cases.append(("element_count", fixture_arrays(), wrong_count_rows, "element count mismatch"))

        for name, arrays, rows, expected in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, expected):
                    extract_case_ledgers_from_arrays(
                        arrays=arrays,
                        header=fixture_header(),
                        design=fixture_design(),
                        case_rows=rows,
                        material={"strength_model": {"density_kg_mm3": 1.0}},
                        **fixture_solver_timeline(),
                    )

    def test_output_binds_artifacts_one_run_and_pending_independent_integration_status(self) -> None:
        extraction = extract_case_ledgers_from_arrays(
            arrays=fixture_arrays(),
            header=fixture_header(),
            design=fixture_design(),
            case_rows=fixture_case_rows(),
            material={"strength_model": {"density_kg_mm3": 1.0}},
            **fixture_solver_timeline(),
        )
        artifact_identity = {
            "package_revision": "R05_capture_ledger_preregistered",
            "design_sha256": "sha256:" + "1" * 64,
            "deck_sha256": "sha256:" + "2" * 64,
            "case_map_sha256": "sha256:" + "3" * 64,
            "material_sha256": "sha256:" + "4" * 64,
        }
        run_identity = {
            "run_id": "job-123",
            "solver_build": "LS-DYNA R15 fixture",
            "d3part_family_sha256": "sha256:" + "5" * 64,
            "d3part_family": [{"path": "d3part", "sha256": "sha256:" + "6" * 64}],
        }
        payload = assemble_output_payload(
            package_revision="R05_capture_ledger_preregistered",
            extraction=extraction,
            artifact_identity=artifact_identity,
            run_identity=run_identity,
            reader_identity={"name": "lasso-python", "version": "2.0.4"},
            extractor_identity={"name": "extract_r05_d3part", "version": "1.0.0"},
        )

        self.assertEqual(
            payload["integration_status"],
            "real_d3part_exercised_pending_independent_integration_acceptance",
        )
        self.assertIn("remains pending independent integration acceptance", payload["claim_boundary"])
        self.assertEqual(payload["artifact_identity"], artifact_identity)
        self.assertEqual(payload["run_identity"], run_identity)
        self.assertNotIn("run_identity", payload["cases"][0])
        self.assertEqual(json.loads(strict_json(payload))["run_identity"]["run_id"], "job-123")
        with self.assertRaises(ValueError):
            strict_json({"not_json": float("nan")})

    def test_artifact_and_d3part_family_hashes_are_bound_to_file_bytes(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r05-d3part-identity-", dir=temp_root) as temp:
            root = Path(temp)
            paths = {}
            for name in ("design.json", "deck.k", "case-map.csv", "material.json"):
                path = root / name
                path.write_bytes(b"abc")
                paths[name] = path
            artifact = artifact_identity_from_paths(
                package_revision="R05_capture_ledger_preregistered",
                design_path=paths["design.json"],
                deck_path=paths["deck.k"],
                case_map_path=paths["case-map.csv"],
                material_path=paths["material.json"],
            )
            expected_abc = "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            self.assertEqual(artifact["design_sha256"], expected_abc)
            self.assertEqual(artifact["deck_sha256"], expected_abc)
            self.assertEqual(artifact["case_map_sha256"], expected_abc)
            self.assertEqual(artifact["material_sha256"], expected_abc)

            (root / "d3part").write_bytes(b"base")
            (root / "d3part2").write_bytes(b"two")
            (root / "d3part10").write_bytes(b"ten")
            (root / "d3part-not-a-family-member").write_bytes(b"ignore")
            run = d3part_family_identity(
                root / "d3part", run_id="job-123", solver_build="LS-DYNA R15 fixture"
            )
            self.assertEqual(
                [entry["path"] for entry in run["d3part_family"]],
                ["d3part", "d3part2", "d3part10"],
            )
            self.assertRegex(run["d3part_family_sha256"], r"^sha256:[0-9a-f]{64}$")

    def test_bridge_is_assessor_compatible_only_with_bound_passing_evidence(self) -> None:
        design_path = PROJECT / "design" / "p0_cases_r05.json"
        manifest_path = PROJECT / "manual_upload_20261021_r05_candidate" / "manifest.json"
        design = json.loads(design_path.read_text(encoding="utf-8"))
        artifact_identity = {
            "package_revision": "R05_capture_ledger_preregistered",
            "design_sha256": EXPECTED_DESIGN_SHA256,
            "deck_sha256": EXPECTED_DECK_SHA256,
            "case_map_sha256": EXPECTED_CASE_MAP_SHA256,
            "material_sha256": "sha256:" + "4" * 64,
        }
        run_identity = {
            "run_id": "job-real-r05",
            "solver_build": "LS-DYNA R15 integration fixture",
            "d3part_family_sha256": "sha256:" + "5" * 64,
            "d3part_family": [
                {"path": "d3part", "size_bytes": 123, "sha256": "sha256:" + "6" * 64}
            ],
        }
        extraction_payload = assemble_output_payload(
            package_revision=artifact_identity["package_revision"],
            extraction={"d3part_summary": {"state_count": 801}, "cases": passing_assessor_cases(design)},
            artifact_identity=artifact_identity,
            run_identity=run_identity,
            reader_identity={"name": "lasso-python", "version": "2.0.4"},
            extractor_identity={
                "name": "extract_r05_d3part",
                "version": "1.0.0",
                "source_sha256": "sha256:" + "7" * 64,
            },
        )
        self.assertEqual(canonical_sha256(design), EXPECTED_DESIGN_CANONICAL_SHA256)
        self.assertEqual(sha256_file(manifest_path), EXPECTED_PACKAGE_MANIFEST_SHA256)

        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r05-assessor-bridge-", dir=temp_root) as temp:
            returned_artifact = Path(temp) / "returned-r05.zip"
            returned_artifact.write_bytes(b"real returned run artifact fixture")
            returned_hash = sha256_file(returned_artifact)
            run_evidence = {
                "schema_version": "hf-p0-r05-run-integrity-evidence@1.0.0",
                "evidence_id": "run-integrity-review-fixture",
                "status": "pass",
                "run_id": run_identity["run_id"],
                "solver_build": run_identity["solver_build"],
                "d3part_family_sha256": run_identity["d3part_family_sha256"],
                "returned_artifact_sha256": returned_hash,
            }
            integration_evidence = {
                "schema_version": "hf-p0-r05-d3part-integration-acceptance@1.0.0",
                "evidence_id": "real-d3part-adapter-review-fixture",
                "status": "pass",
                "accepted_extraction_sha256": canonical_sha256(extraction_payload),
                "accepted_parser_source_sha256": parser_source_sha256_from_extraction(
                    extraction_payload
                ),
            }
            assessor_payload = bridge_extraction_to_assessor_payload(
                extraction_payload=extraction_payload,
                design_path=design_path,
                package_manifest_path=manifest_path,
                returned_artifact_path=returned_artifact,
                run_integrity_evidence=run_evidence,
                integration_acceptance=integration_evidence,
            )

            self.assertEqual(assessor_payload["schema_version"], "hf-p0-r05-case-results@1.0.0")
            self.assertEqual(
                assessor_payload["artifact_identity"]["design_canonical_sha256"],
                EXPECTED_DESIGN_CANONICAL_SHA256,
            )
            self.assertEqual(
                assessor_payload["artifact_identity"]["package_manifest_sha256"],
                EXPECTED_PACKAGE_MANIFEST_SHA256,
            )
            self.assertEqual(
                assessor_payload["run_identity"]["returned_artifact_sha256"], returned_hash
            )
            self.assertTrue(
                all(case["run_integrity_gate"] == "pass" for case in assessor_payload["cases"])
            )
            assessment = assess_r05_response(design, assessor_payload)
            self.assertEqual(assessment["overall_status"], "pass")
            self.assertEqual(assessment["p0_response_calibration_admission"], "allowed")

            rejection_scenarios = (
                ("missing_run_integrity", None, integration_evidence, "run-integrity evidence"),
                (
                    "failed_run_integrity",
                    {**run_evidence, "status": "fail"},
                    integration_evidence,
                    "run-integrity evidence status must be pass",
                ),
                ("missing_integration", run_evidence, None, "integration-acceptance evidence"),
                (
                    "failed_integration",
                    run_evidence,
                    {**integration_evidence, "status": "fail"},
                    "integration-acceptance evidence status must be pass",
                ),
            )
            for name, run_proof, integration_proof, expected in rejection_scenarios:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, expected):
                    bridge_extraction_to_assessor_payload(
                        extraction_payload=extraction_payload,
                        design_path=design_path,
                        package_manifest_path=manifest_path,
                        returned_artifact_path=returned_artifact,
                        run_integrity_evidence=run_proof,
                        integration_acceptance=integration_proof,
                    )


if __name__ == "__main__":
    unittest.main()
