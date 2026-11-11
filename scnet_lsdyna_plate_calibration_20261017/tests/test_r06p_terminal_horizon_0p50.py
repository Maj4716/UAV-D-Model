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

from build_p0_r06p_terminal_horizon_0p50 import (  # noqa: E402
    EXPECTED_DIFFERING_LINES,
    HEAD_DECK,
    JOB_DIR,
    build_batch,
    transform_deck,
    validate_transformed_deck,
)
from build_r06g_tssfac02_return_audit import sha256  # noqa: E402
from build_r06o_plate_erod1_return_audit import build_audit  # noqa: E402
from compare_r06n_r06o_plate_erod_and_horizon import build_comparison  # noqa: E402
from create_p0_r06p_terminal_horizon_0p50_design import (  # noqa: E402
    CANDIDATE_HORIZON_MS,
    JOB_ID,
    MAXIMUM_CYCLE_COUNT,
    PARENT_DECK_SHA256,
    REVISION,
    build_contract,
    build_design,
)
from stage_r06p_terminal_horizon_0p50 import stage_submission  # noqa: E402


R06O_RETURN = PROJECT.parent / "uav_hf_r06o_0800_plate_erod1_isolation"
ANALYSIS = PROJECT / "analysis" / "72894870_r06o_0800_plate_erod1_isolation_20261102"
R06N_SUMMARY = PROJECT / "analysis" / "72864757_r06n_0800_adaptive_sph_32rank_confirmation_20261102" / "r06m_r06n_resource_sensitivity.json"
R06N_ASCII = PROJECT / "analysis" / "72864757_r06n_0800_adaptive_sph_32rank_confirmation_20261102" / "ascii_metrics.json"
R06O_ASCII = ANALYSIS / "ascii_metrics.json"
COMPARISON = ANALYSIS / "r06n_r06o_plate_erod_horizon_comparison.json"
AUDIT = ANALYSIS / "r06o_plate_erod1_return_audit.json"
PARENT_DECK = ANALYSIS / "immutable_parent_r06o_0800_plate_erod1_isolation.k"
DESIGN = PROJECT / "design" / "p0_r06p_terminal_horizon_0p50_design.json"
CONTRACT = PROJECT / "design" / "p0_r06p_terminal_horizon_0p50_output_contract.json"
MATERIAL = PROJECT / "materials" / "jc4340_reference.json"
CASE_MAP = PROJECT / "manual_upload_20261102_r06o_plate_erod1_isolation" / "job_01_v0800_plate_erod1_isolation" / "p0_case_part_map.csv"
BATCH = PROJECT / "manual_upload_20261102_r06p_terminal_horizon_0p50"
JOB = BATCH / JOB_DIR


class R06pTerminalHorizonTests(unittest.TestCase):
    def test_r06n_r06o_comparison_is_reproducible_and_selects_half_ms(self) -> None:
        rebuilt = build_comparison(
            r06n_summary=R06N_SUMMARY,
            r06o_return=R06O_RETURN,
            r06n_ascii=R06N_ASCII,
            r06o_ascii=R06O_ASCII,
        )
        stored = json.loads(COMPARISON.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        paired = stored["paired_diagnostics"]
        self.assertEqual(paired["recommended_next_horizon_ms"], 0.5)
        self.assertGreater(paired["earliest_linear_projected_capture_crossing_time_ms"], 0.2)
        self.assertTrue(paired["target_plate_element_terminal_alive_r06o"])

    def test_r06o_audit_is_reproducible_and_releases_only_horizon(self) -> None:
        rebuilt = build_audit(return_dir=R06O_RETURN, analysis_dir=ANALYSIS)
        stored = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        self.assertEqual(stored["decision"]["numerical_engineering_confirmation"], "pass_plate_erod1_isolation")
        self.assertTrue(stored["decision"]["release_r06p_terminal_horizon_0p50"])
        self.assertFalse(stored["response_availability"]["primary_qoi_available"])
        self.assertEqual(stored["decision"]["engine_fitting"], "denied")
        self.assertEqual(stored["decision"]["damage_tree_fitting"], "denied")

    def test_frozen_design_and_contract_match_reproducible_builder(self) -> None:
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        contract = build_contract(audit)
        design = build_design(contract)
        self.assertEqual(contract, json.loads(CONTRACT.read_text(encoding="utf-8")))
        self.assertEqual(design, json.loads(DESIGN.read_text(encoding="utf-8")))
        self.assertEqual(design["package_revision"], REVISION)
        self.assertEqual(contract["single_factor"]["candidate_level"], CANDIDATE_HORIZON_MS)
        self.assertEqual(contract["horizon_selection"]["selected_ms"], 0.5)
        self.assertEqual(contract["run_planning"]["linear_cycle_projection"], 735113)
        self.assertLess(contract["output_resolution_gate"]["maximum_projected_interstate_displacement_mm"], 0.025)
        self.assertEqual(contract["output_resolution_gate"]["status"], "pass")

    def test_generated_deck_has_only_registered_horizon_and_output_differences(self) -> None:
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
        self.assertTrue(stored["model_semantics_unchanged"])
        self.assertEqual(stored["maximum_cycle_count"], MAXIMUM_CYCLE_COUNT)
        self.assertEqual(stored["plate_erod"], 1)
        self.assertEqual(stored["fragment_erod"], 1)

    def test_case_map_has_current_identity_and_horizon(self) -> None:
        lines = (JOB / "p0_case_part_map.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith(f"{JOB_ID},r06p_terminal_response_horizon_0p50_800,"))
        header = lines[0].split(",")
        values = lines[1].split(",")
        row = dict(zip(header, values))
        self.assertEqual(row["termination_time_ms"], "0.5")

    def test_batch_is_reproducible(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06p-build-", dir=temp_root) as temp:
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
        with tempfile.TemporaryDirectory(prefix="r06p-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            delivery = stage_submission(BATCH, output)
            self.assertEqual(delivery["job_id"], JOB_ID)
            self.assertEqual(delivery["resource_contract"]["nodes"], 1)
            self.assertEqual(delivery["resource_contract"]["ranks"], 32)
            self.assertEqual(delivery["solver_controls"]["problem_time_ms"], 0.5)
            self.assertEqual(delivery["single_factor"], "terminal_response_horizon_0p04_to_0p50_ms")
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in output.rglob("*")))
            upload = output / delivery["upload_folder"]
            self.assertEqual(sha256(upload / delivery["head_deck"]), delivery["head_deck_sha256"])


if __name__ == "__main__":
    unittest.main()
