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

from build_p0_r06u_plate_adaptive_sph_canary import (  # noqa: E402
    HEAD_DECK,
    JOB_DIR,
    build_batch,
    transform_deck,
    validate_transformed_deck,
)
from build_r06g_tssfac02_return_audit import sha256  # noqa: E402
from build_r06t_failure_audit import build_audit  # noqa: E402
from create_p0_r06u_plate_adaptive_sph_canary_design import (  # noqa: E402
    BRANCH_PARENT_DECK_SHA256,
    JOB_ID,
    REVISION,
    build_contract,
    build_design,
)
from stage_r06u_plate_adaptive_sph_canary import stage_submission  # noqa: E402


R06T_RETURN = PROJECT.parent / "uav_hf_r06t_1200_fragment_only_psfail_diagnostic"
R06T_ANALYSIS = PROJECT / "analysis" / "73087868_r06t_fragment_only_psfail_failure_20261104"
R06T_AUDIT = R06T_ANALYSIS / "r06t_failure_audit.json"
R06S_AUDIT = PROJECT / "analysis" / "73059442_r06s_1200_speed_stability_stage1_20261103" / "r06s_stage1_return_audit.json"
PARENT_DECK = PROJECT.parent / "uav_hf_r06s_1200_speed_stability_diagnostic" / "r06s_1200_speed_stability_diagnostic.k"
DESIGN = PROJECT / "design" / "p0_r06u_plate_adaptive_sph_canary_design.json"
CONTRACT = PROJECT / "design" / "p0_r06u_plate_adaptive_sph_canary_output_contract.json"
MATERIAL = PROJECT / "materials" / "jc4340_reference.json"
CASE_MAP = PROJECT / "manual_upload_20261103_r06s_1200_speed_stability_diagnostic" / "job_01_v1200_speed_stability_diagnostic" / "p0_case_part_map.csv"
BATCH = PROJECT / "manual_upload_20261104_r06u_plate_adaptive_sph_canary"
JOB = BATCH / JOB_DIR


class R06uPlateAdaptiveSphCanaryTests(unittest.TestCase):
    def test_r06t_failure_audit_is_reproducible_and_releases_r06u(self) -> None:
        rebuilt = build_audit(R06T_RETURN)
        stored = json.loads(R06T_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        self.assertEqual(stored["execution"]["solution_resource_identity"], "pass")
        self.assertTrue(stored["execution"]["error_termination"])
        self.assertEqual(stored["terminal_failure"]["element_id"], 23830)
        self.assertEqual(stored["terminal_failure"]["part_id"], 1002)
        self.assertEqual(stored["terminal_failure"]["time_step_ms"], 1.99e-27)
        self.assertEqual(stored["negative_volume_attribution_before_failure"]["plate_unique_count"], 0)
        self.assertEqual(stored["negative_volume_attribution_before_failure"]["fragment_unique_count"], 415)
        self.assertTrue(stored["decision"]["release_r06u_plate_adaptive_sph_capability_canary"])
        self.assertFalse(stored["decision"]["download_heavy_binary_families"])

    def test_frozen_design_and_contract_match_reproducible_builder(self) -> None:
        r06s = json.loads(R06S_AUDIT.read_text(encoding="utf-8"))
        r06t = json.loads(R06T_AUDIT.read_text(encoding="utf-8"))
        contract = build_contract(r06s, r06t)
        design = build_design(contract)
        self.assertEqual(contract, json.loads(CONTRACT.read_text(encoding="utf-8")))
        self.assertEqual(design, json.loads(DESIGN.read_text(encoding="utf-8")))
        self.assertEqual(design["package_revision"], REVISION)
        self.assertEqual(contract["admission"]["engine_fitting"], "denied")
        self.assertEqual(contract["frozen_controls"]["termination_time_ms"], 0.15)

    def test_generated_deck_adds_only_registered_plate_adaptive_bundle(self) -> None:
        self.assertEqual(sha256(PARENT_DECK), BRANCH_PARENT_DECK_SHA256)
        parent = PARENT_DECK.read_text(encoding="ascii")
        candidate = (JOB / HEAD_DECK).read_text(encoding="ascii")
        self.assertEqual(candidate, transform_deck(parent))
        report = validate_transformed_deck(parent, candidate)
        stored = json.loads((JOB / "static_validation.json").read_text(encoding="utf-8"))
        expected = dict(report)
        expected["candidate_deck_sha256"] = sha256(JOB / HEAD_DECK)
        self.assertEqual(expected, stored)
        self.assertEqual(stored["control_solid_psfail"], 0)
        self.assertEqual(stored["plate_adaptive_sph"]["sph_part_id"], 1003)
        self.assertEqual(stored["termination_time_ms"], 0.15)
        self.assertTrue(stored["all_other_keyword_blocks_identical"])

    def test_case_map_has_current_identity_speed_and_horizon(self) -> None:
        lines = (JOB / "p0_case_part_map.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith(f"{JOB_ID},r06u_plate_adaptive_sph_capability_1200,"))
        self.assertIn(",1200.0,0.75,", lines[1])
        self.assertIn(",0.15,", lines[1])

    def test_batch_is_reproducible(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06u-build-", dir=temp_root) as temp:
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
        with tempfile.TemporaryDirectory(prefix="r06u-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            delivery = stage_submission(BATCH, output)
            self.assertEqual(delivery["job_id"], JOB_ID)
            self.assertEqual(delivery["resource_contract"]["nodes"], 1)
            self.assertEqual(delivery["resource_contract"]["ranks"], 32)
            self.assertEqual(delivery["solver_controls"]["problem_time_ms"], 0.15)
            self.assertIn("d3plot*", delivery["staged_download_policy"]["do_not_download_initially"])
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in output.rglob("*")))
            upload = output / delivery["upload_folder"]
            self.assertEqual(sha256(upload / delivery["head_deck"]), delivery["head_deck_sha256"])


if __name__ == "__main__":
    unittest.main()
