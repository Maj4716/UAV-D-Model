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

from build_p0_r06t_fragment_only_psfail import (  # noqa: E402
    EXPECTED_DIFFERING_LINES,
    HEAD_DECK,
    JOB_DIR,
    build_batch,
    transform_deck,
    validate_transformed_deck,
)
from build_r06g_tssfac02_return_audit import sha256  # noqa: E402
from build_r06s_stage1_return_audit import build_audit  # noqa: E402
from create_p0_r06t_fragment_only_psfail_design import (  # noqa: E402
    JOB_ID,
    MINIMUM_TIME_STEP_FLOOR_MS,
    PARENT_DECK_SHA256,
    REVISION,
    build_contract,
    build_design,
)
from stage_r06t_fragment_only_psfail import stage_submission  # noqa: E402


R06S_RETURN = PROJECT.parent / "uav_hf_r06s_1200_speed_stability_diagnostic"
ANALYSIS = PROJECT / "analysis" / "73059442_r06s_1200_speed_stability_stage1_20261103"
AUDIT = ANALYSIS / "r06s_stage1_return_audit.json"
PARENT_DECK = R06S_RETURN / "r06s_1200_speed_stability_diagnostic.k"
DESIGN = PROJECT / "design" / "p0_r06t_fragment_only_psfail_design.json"
CONTRACT = PROJECT / "design" / "p0_r06t_fragment_only_psfail_output_contract.json"
MATERIAL = PROJECT / "materials" / "jc4340_reference.json"
CASE_MAP = PROJECT / "manual_upload_20261103_r06s_1200_speed_stability_diagnostic" / "job_01_v1200_speed_stability_diagnostic" / "p0_case_part_map.csv"
BATCH = PROJECT / "manual_upload_20261103_r06t_fragment_only_psfail_diagnostic"
JOB = BATCH / JOB_DIR


class R06tFragmentOnlyPsfailTests(unittest.TestCase):
    def test_r06s_stage1_audit_is_reproducible_and_releases_r06t(self) -> None:
        rebuilt = build_audit(R06S_RETURN)
        stored = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        self.assertEqual(stored["execution"]["solution_resource_identity"], "pass")
        self.assertTrue(stored["execution"]["normal_termination_at_required_time"])
        self.assertEqual(stored["negative_volume_attribution"]["plate_unique_count"], 70)
        self.assertEqual(stored["negative_volume_attribution"]["fragment_unique_count"], 276)
        self.assertFalse(stored["negative_volume_attribution"]["target_plate_negative_volume_event"])
        self.assertEqual(stored["global_engineering"]["sliding_energy_gate"], "pass")
        self.assertTrue(stored["stage1_decision"]["stop_heavy_download"])
        self.assertTrue(stored["stage1_decision"]["release_r06t_fragment_only_psfail_model_form_diagnostic"])

    def test_frozen_design_and_contract_match_reproducible_builder(self) -> None:
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        contract = build_contract(audit)
        design = build_design(contract)
        self.assertEqual(contract, json.loads(CONTRACT.read_text(encoding="utf-8")))
        self.assertEqual(design, json.loads(DESIGN.read_text(encoding="utf-8")))
        self.assertEqual(design["package_revision"], REVISION)
        self.assertAlmostEqual(contract["preregistered_stage1_gates"]["minimum_time_step_ms"]["minimum"], MINIMUM_TIME_STEP_FLOOR_MS)
        self.assertEqual(contract["admission"]["engine_fitting"], "denied")

    def test_generated_deck_changes_only_two_registered_lines(self) -> None:
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
        self.assertEqual(stored["psfail_set_part_ids"], [2002])
        self.assertTrue(stored["plate_part_1002_excluded_from_psfail_set"])

    def test_case_map_has_current_identity_and_frozen_speed(self) -> None:
        lines = (JOB / "p0_case_part_map.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith(f"{JOB_ID},r06t_plate_negative_volume_psfail_scope_1200,"))
        self.assertIn(",1200.0,0.75,", lines[1])

    def test_batch_is_reproducible(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06t-build-", dir=temp_root) as temp:
            regenerated = Path(temp) / "batch"
            build_batch(parent_deck=PARENT_DECK, design_path=DESIGN, contract_path=CONTRACT, material_path=MATERIAL, case_map_path=CASE_MAP, output_dir=regenerated)
            expected = {path.relative_to(BATCH).as_posix(): path for path in BATCH.rglob("*") if path.is_file()}
            actual = {path.relative_to(regenerated).as_posix(): path for path in regenerated.rglob("*") if path.is_file()}
            self.assertEqual(set(actual), set(expected))
            for relative, path in actual.items():
                self.assertEqual(path.read_bytes(), expected[relative].read_bytes(), relative)

    def test_staged_submission_is_direct_folder_and_staged_download_contract(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06t-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            delivery = stage_submission(BATCH, output)
            self.assertEqual(delivery["job_id"], JOB_ID)
            self.assertEqual(delivery["resource_contract"]["nodes"], 1)
            self.assertEqual(delivery["resource_contract"]["ranks"], 32)
            self.assertEqual(delivery["output_intervals_ms"]["d3plot"], 0.00015)
            self.assertIn("d3plot*", delivery["staged_download_policy"]["do_not_download_initially"])
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in output.rglob("*")))
            upload = output / delivery["upload_folder"]
            self.assertEqual(sha256(upload / delivery["head_deck"]), delivery["head_deck_sha256"])


if __name__ == "__main__":
    unittest.main()
