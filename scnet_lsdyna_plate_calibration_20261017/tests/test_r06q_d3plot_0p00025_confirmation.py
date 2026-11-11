from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_r06p_terminal_response import build_terminal_response  # noqa: E402
from build_p0_r06q_d3plot_0p00025_confirmation import (  # noqa: E402
    EXPECTED_DIFFERING_LINES,
    HEAD_DECK,
    JOB_DIR,
    build_batch,
    transform_deck,
    validate_transformed_deck,
)
from build_r06g_tssfac02_return_audit import sha256  # noqa: E402
from build_r06p_terminal_horizon_return_audit import build_audit  # noqa: E402
from create_p0_r06q_d3plot_0p00025_confirmation_design import (  # noqa: E402
    JOB_ID,
    PARENT_DECK_SHA256,
    REVISION,
    build_contract,
    build_design,
)
from stage_r06q_d3plot_0p00025_confirmation import stage_submission  # noqa: E402
from validate_r06m_adaptive_sph_outputs import build_validation  # noqa: E402


R06P_RETURN = PROJECT.parent / "uav_hf_r06p_0800_terminal_horizon_0p50"
ANALYSIS = PROJECT / "analysis" / "72915268_r06p_0800_terminal_horizon_0p50_20261102"
VALIDATION = ANALYSIS / "adaptive_sph_validation.json"
LEDGER = ANALYSIS / "terminal_response_ledger.json"
AUDIT = ANALYSIS / "r06p_terminal_horizon_return_audit.json"
PARENT_DECK = ANALYSIS / "immutable_parent_r06p_0800_terminal_horizon_0p50.k"
DESIGN = PROJECT / "design" / "p0_r06q_d3plot_0p00025_confirmation_design.json"
CONTRACT = PROJECT / "design" / "p0_r06q_d3plot_0p00025_confirmation_output_contract.json"
MATERIAL = PROJECT / "materials" / "jc4340_reference.json"
CASE_MAP = PROJECT / "manual_upload_20261102_r06p_terminal_horizon_0p50" / "job_01_v0800_terminal_horizon_0p50" / "p0_case_part_map.csv"
BATCH = PROJECT / "manual_upload_20261102_r06q_d3plot_0p00025_confirmation"
JOB = BATCH / JOB_DIR


class R06qD3plotMeasurementConfirmationTests(unittest.TestCase):
    def test_r06p_adaptive_validation_is_reproducible_at_half_ms(self) -> None:
        rebuilt = build_validation(
            R06P_RETURN,
            schema_version="hf-p0-r06p-adaptive-sph-validation@1.0.0",
            strict_gates=False,
            required_problem_time_ms=0.5,
        )
        stored = json.loads(VALIDATION.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        self.assertEqual(stored["adaptive_sph"]["conversion"]["plate_deleted_element_count"], 0)
        self.assertEqual(stored["adaptive_sph"]["conversion"]["final_converted_solid_count"], 45)
        self.assertEqual(stored["adaptive_sph"]["conversion"]["final_active_sph_count"], 360)

    def test_r06p_terminal_response_ledger_is_reproducible_and_fails_only_resolution(self) -> None:
        rebuilt = build_terminal_response(R06P_RETURN)
        stored = json.loads(LEDGER.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        response = stored["primary_response"]
        capture = stored["capture_observations"]
        self.assertFalse(response["available"])
        self.assertEqual(response["unavailable_reasons"], ["temporal_resolution"])
        self.assertEqual(capture["solid_crossing_event_count"], 0)
        self.assertEqual(capture["sph_crossing_event_count"], 0)
        self.assertGreater(capture["maximum_observed_consecutive_downward_carrier_z_step_mm"], 0.25)
        self.assertGreater(capture["minimum_observed_capture_plane_clearance_mm"], 6.0)
        self.assertEqual(stored["terminal_tail"]["gate"], "pass")

    def test_r06p_audit_is_reproducible_and_releases_only_measurement_confirmation(self) -> None:
        rebuilt = build_audit(return_dir=R06P_RETURN, analysis_dir=ANALYSIS)
        stored = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        self.assertEqual(stored["engineering_gates"]["overall_engineering"], "pass")
        self.assertEqual(stored["response_assessment"]["temporal_resolution"]["status"], "fail")
        self.assertTrue(stored["decision"]["release_r06q_d3plot_0p00025_measurement_confirmation"])
        self.assertFalse(stored["decision"]["release_speed_scan"])
        self.assertEqual(stored["decision"]["engine_fitting"], "denied")
        self.assertEqual(stored["decision"]["damage_tree_fitting"], "denied")

    def test_frozen_design_and_contract_match_reproducible_builder(self) -> None:
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        contract = build_contract(audit)
        design = build_design(contract)
        self.assertEqual(contract, json.loads(CONTRACT.read_text(encoding="utf-8")))
        self.assertEqual(design, json.loads(DESIGN.read_text(encoding="utf-8")))
        self.assertEqual(design["package_revision"], REVISION)
        self.assertEqual(contract["single_change"]["layer"], "measurement_only")
        self.assertLess(contract["prospective_resolution"]["projected_maximum_downward_step_mm"], 0.25)

    def test_generated_deck_changes_only_title_and_d3plot_interval(self) -> None:
        self.assertEqual(sha256(PARENT_DECK), PARENT_DECK_SHA256)
        parent = PARENT_DECK.read_text(encoding="ascii")
        candidate = (JOB / HEAD_DECK).read_text(encoding="ascii")
        self.assertEqual(candidate, transform_deck(parent))
        report = validate_transformed_deck(parent, candidate)
        stored = json.loads((JOB / "static_validation.json").read_text(encoding="utf-8"))
        expected = dict(report)
        expected["candidate_deck_sha256"] = sha256(JOB / HEAD_DECK)
        self.assertEqual(expected, stored)
        self.assertEqual(stored["differing_line_numbers"], EXPECTED_DIFFERING_LINES)
        self.assertTrue(stored["all_other_deck_lines_byte_identical"])
        self.assertTrue(stored["physical_model_semantics_unchanged"])

    def test_case_map_has_current_identity(self) -> None:
        lines = (JOB / "p0_case_part_map.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith(f"{JOB_ID},r06q_terminal_response_measurement_confirmation_800,"))

    def test_batch_is_reproducible(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06q-build-", dir=temp_root) as temp:
            regenerated = Path(temp) / "batch"
            build_batch(parent_deck=PARENT_DECK, design_path=DESIGN, contract_path=CONTRACT, material_path=MATERIAL, case_map_path=CASE_MAP, output_dir=regenerated)
            expected = {path.relative_to(BATCH).as_posix(): path for path in BATCH.rglob("*") if path.is_file()}
            actual = {path.relative_to(regenerated).as_posix(): path for path in regenerated.rglob("*") if path.is_file()}
            self.assertEqual(set(actual), set(expected))
            for relative, path in actual.items():
                self.assertEqual(path.read_bytes(), expected[relative].read_bytes(), relative)

    def test_staged_submission_is_direct_folder_and_exact_resource_contract(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06q-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            delivery = stage_submission(BATCH, output)
            self.assertEqual(delivery["job_id"], JOB_ID)
            self.assertEqual(delivery["resource_contract"]["nodes"], 1)
            self.assertEqual(delivery["resource_contract"]["ranks"], 32)
            self.assertEqual(delivery["output_intervals_ms"]["d3plot"], 0.00025)
            self.assertTrue(delivery["physical_model_semantics_unchanged"])
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in output.rglob("*")))
            upload = output / delivery["upload_folder"]
            self.assertEqual(sha256(upload / delivery["head_deck"]), delivery["head_deck_sha256"])


if __name__ == "__main__":
    unittest.main()
