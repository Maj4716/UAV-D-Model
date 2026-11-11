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

from build_r06k_fragment_psfail_return_audit import build_audit  # noqa: E402
from create_p0_r06l_contact_soft1_isolation_design import build_contract, build_design  # noqa: E402
from generate_p0_r06_diagnostics import generate_batch  # noqa: E402
from stage_r06l_contact_soft1_isolation import stage_contact_soft1_diagnostic  # noqa: E402
from validate_p0_r06_diagnostics import comparable_blocks, grouped_blocks, split_fields, validate_batch  # noqa: E402


DESIGN = PROJECT / "design" / "p0_r06l_contact_soft1_isolation_design.json"
CONTRACT = PROJECT / "design" / "p0_r06l_contact_soft1_output_contract.json"
PARENT_DESIGN = PROJECT / "design" / "p0_r06k_fragment_negative_volume_isolation_design.json"
PARENT_CONTRACT = PROJECT / "design" / "p0_r06k_fragment_negative_volume_output_contract.json"
ANALYSIS = PROJECT / "analysis" / "72764387_r06k_0800_fragment_psfail_plate_20261031"
AUDIT = ANALYSIS / "r06k_fragment_psfail_return_audit.json"
BASELINE_AUDIT = (
    PROJECT
    / "analysis"
    / "72751725_r06j_0800_fragment_erod1_20261030"
    / "r06j_fragment_erod1_return_audit.json"
)
RETURN = PROJECT.parent / "LS_DYNA_1030_233727"
BATCH = PROJECT / "manual_upload_20261031_r06l_contact_soft1_isolation"
JOB = BATCH / "job_01_v0800_contact_soft1"
BASELINE_DECK = ANALYSIS / "immutable_parent_r06k_0800_fragment_psfail_plate.k"


class R06lContactSoft1IsolationTests(unittest.TestCase):
    def test_r06k_return_audit_is_reproducible_and_releases_only_soft1(self) -> None:
        rebuilt = build_audit(return_dir=RETURN, analysis_dir=ANALYSIS, baseline_audit_path=BASELINE_AUDIT)
        stored = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        comparison = stored["formal_fragment_negative_volume_comparison_to_job_72751725"]
        self.assertEqual(comparison["status"], "supported_fragment_deletion_removed_but_numerical_instability_exposed")
        self.assertEqual(comparison["deletion_change"]["control_deleted_element_count"], 35)
        self.assertEqual(comparison["deletion_change"]["treatment_deleted_element_count"], 0)
        self.assertEqual(comparison["deletion_change"]["treatment_negative_volume_fragment_element_count"], 0)
        self.assertEqual(comparison["performance_and_numerical_stability"]["status"], "fail")
        self.assertGreater(comparison["performance_and_numerical_stability"]["cycle_count_ratio"], 60.0)
        self.assertTrue(stored["decision"]["release_r06l_contact_soft1_isolation"])
        self.assertFalse(stored["decision"]["release_2000_control"])
        self.assertEqual(stored["decision"]["engine_fitting"], "denied")
        self.assertEqual(stored["decision"]["damage_tree_fitting"], "denied")

    def test_frozen_design_and_contract_match_reproducible_builder(self) -> None:
        parent_design = json.loads(PARENT_DESIGN.read_text(encoding="utf-8"))
        parent_contract = json.loads(PARENT_CONTRACT.read_text(encoding="utf-8"))
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        contract = build_contract(parent_contract, audit)
        design = build_design(parent_design, contract, audit)
        self.assertEqual(contract, json.loads(CONTRACT.read_text(encoding="utf-8")))
        self.assertEqual(design, json.loads(DESIGN.read_text(encoding="utf-8")))
        self.assertEqual(contract["contact_soft_isolation"]["control_level"], 2)
        self.assertEqual(contract["contact_soft_isolation"]["treatment_level"], 1)

    def test_generated_job_changes_only_contact_soft_from_returned_parent(self) -> None:
        result = validate_batch(BATCH, write_reports=False)
        self.assertEqual(result["overall_status"], "pass")
        candidate = (JOB / "r06l_0800_contact_soft1.k").read_text(encoding="ascii")
        baseline = BASELINE_DECK.read_text(encoding="ascii")
        ignored = {"*TITLE", "*CONTACT_ERODING_SINGLE_SURFACE"}
        self.assertEqual(comparable_blocks(candidate, ignored), comparable_blocks(baseline, ignored))
        candidate_grouped = grouped_blocks(candidate)
        baseline_grouped = grouped_blocks(baseline)
        candidate_contact = candidate_grouped["*CONTACT_ERODING_SINGLE_SURFACE"][0]
        baseline_contact = baseline_grouped["*CONTACT_ERODING_SINGLE_SURFACE"][0]
        self.assertEqual(candidate_contact[:4], baseline_contact[:4])
        candidate_a = split_fields(candidate_contact[4])
        baseline_a = split_fields(baseline_contact[4])
        self.assertEqual(int(baseline_a[0]), 2)
        self.assertEqual(int(candidate_a[0]), 1)
        self.assertEqual(candidate_a[1:], baseline_a[1:])
        self.assertEqual(candidate_grouped["*CONTROL_SOLID"], baseline_grouped["*CONTROL_SOLID"])
        self.assertEqual(candidate_grouped["*CONTROL_TIMESTEP"], baseline_grouped["*CONTROL_TIMESTEP"])
        self.assertEqual(candidate_grouped["*MAT_JOHNSON_COOK"], baseline_grouped["*MAT_JOHNSON_COOK"])
        stored = json.loads((JOB / "static_validation.json").read_text(encoding="utf-8"))
        isolation = stored["checks"]["immutable_r06k_contact_soft_factor_isolation"]
        self.assertEqual(isolation["status"], "pass")
        self.assertEqual(isolation["control_level"], 2)
        self.assertEqual(isolation["treatment_level"], 1)

    def test_batch_is_reproducible(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06l-soft1-", dir=temp_root) as temp:
            regenerated = Path(temp) / "batch"
            generate_batch(
                design_path=DESIGN,
                r05_design_path=PROJECT / "design" / "p0_cases_r05.json",
                material_path=PROJECT / "materials" / "jc4340_reference.json",
                successor_contract_path=CONTRACT,
                output_dir=regenerated,
            )
            validate_batch(regenerated, write_reports=True)
            for relative in (
                "batch_manifest.json",
                "R06_SUBMISSION_READINESS.json",
                "submission_matrix.csv",
                "job_01_v0800_contact_soft1/r06l_0800_contact_soft1.k",
                "job_01_v0800_contact_soft1/manifest.json",
                "job_01_v0800_contact_soft1/static_validation.json",
            ):
                self.assertEqual((regenerated / relative).read_bytes(), (BATCH / relative).read_bytes())

    def test_staged_submission_is_direct_folder_without_archive(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06l-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            manifest = stage_contact_soft1_diagnostic(BATCH, output, AUDIT)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"uav_hf_r06l_0800_contact_soft1", "DELIVERY_MANIFEST.json", "SUBMIT_ONLY_THIS_FOLDER.md"},
            )
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in output.rglob("*")))
            upload = output / manifest["upload_folder"]
            self.assertTrue((upload / manifest["head_deck"]).is_file())
            self.assertEqual(manifest["single_factor"]["control"], 2)
            self.assertEqual(manifest["single_factor"]["treatment"], 1)
            self.assertEqual(manifest["status"], "ready_for_user_submission_not_submitted_not_solver_validated")


if __name__ == "__main__":
    unittest.main()
