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

from build_p0_r06m_adaptive_sph_canary import (  # noqa: E402
    HEAD_DECK,
    JOB_DIR,
    build_batch,
    transform_deck,
    validate_transformed_deck,
)
from build_r06l_contact_soft1_return_audit import build_audit  # noqa: E402
from create_p0_r06m_adaptive_sph_canary_design import (  # noqa: E402
    JOB_ID,
    MAXIMUM_CYCLE_COUNT,
    REVISION,
    build_contract,
    build_design,
)
from stage_r06m_adaptive_sph_canary import stage_adaptive_sph_canary  # noqa: E402
from validate_p0_r06_diagnostics import grouped_blocks, sha256, split_fields  # noqa: E402


DESIGN = PROJECT / "design" / "p0_r06m_adaptive_sph_canary_design.json"
CONTRACT = PROJECT / "design" / "p0_r06m_adaptive_sph_output_contract.json"
R06J_ANALYSIS = PROJECT / "analysis" / "72751725_r06j_0800_fragment_erod1_20261030"
R06J_AUDIT = R06J_ANALYSIS / "r06j_fragment_erod1_return_audit.json"
R06J_DECK = R06J_ANALYSIS / "immutable_parent_r06j_0800_fragment_erod1.k"
R06K_AUDIT = (
    PROJECT
    / "analysis"
    / "72764387_r06k_0800_fragment_psfail_plate_20261031"
    / "r06k_fragment_psfail_return_audit.json"
)
R06L_ANALYSIS = PROJECT / "analysis" / "72782972_r06l_0800_contact_soft1_20261101"
R06L_AUDIT = R06L_ANALYSIS / "r06l_contact_soft1_return_audit.json"
R06L_RETURN = PROJECT.parent / "uav_hf_r06l_0800_contact_soft1"
MATERIAL = PROJECT / "materials" / "jc4340_reference.json"
BATCH = PROJECT / "manual_upload_20261101_r06m_adaptive_sph_canary"
JOB = BATCH / JOB_DIR


class R06mAdaptiveSphCanaryTests(unittest.TestCase):
    def test_r06l_cancelled_return_audit_is_reproducible_and_releases_only_model_form_canary(self) -> None:
        rebuilt = build_audit(
            return_dir=R06L_RETURN,
            analysis_dir=R06L_ANALYSIS,
            baseline_audit_path=R06K_AUDIT,
        )
        stored = json.loads(R06L_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        self.assertEqual(
            stored["decision"]["classification"],
            "valid_negative_result_soft1_rejected_model_form_pivot_required",
        )
        self.assertTrue(stored["decision"]["release_r06m_adaptive_solid_to_sph_canary"])
        self.assertFalse(stored["decision"]["release_additional_contact_parameter_tuning"])
        self.assertEqual(stored["decision"]["engine_fitting"], "denied")
        self.assertEqual(stored["decision"]["damage_tree_fitting"], "denied")

    def test_frozen_design_and_contract_match_reproducible_builder(self) -> None:
        r06j = json.loads(R06J_AUDIT.read_text(encoding="utf-8"))
        r06l = json.loads(R06L_AUDIT.read_text(encoding="utf-8"))
        contract = build_contract(r06j, r06l)
        design = build_design(contract, r06j, r06l)
        self.assertEqual(contract, json.loads(CONTRACT.read_text(encoding="utf-8")))
        self.assertEqual(design, json.loads(DESIGN.read_text(encoding="utf-8")))
        self.assertEqual(design["package_revision"], REVISION)
        bundle = contract["adaptive_solid_to_sph_canary"]["model_form_bundle"]
        self.assertEqual(bundle["define_adaptive_solid_to_sph"]["nq"], 2)
        self.assertEqual(bundle["define_adaptive_solid_to_sph"]["particles_per_failed_hexahedron"], 8)
        self.assertEqual(bundle["runaway_guard"]["control_termination_endcyc"], MAXIMUM_CYCLE_COUNT)

    def test_generated_deck_has_exact_bundle_and_immutable_parent_content(self) -> None:
        baseline = R06J_DECK.read_text(encoding="ascii")
        candidate = (JOB / HEAD_DECK).read_text(encoding="ascii")
        self.assertEqual(candidate, transform_deck(baseline))
        report = validate_transformed_deck(baseline, candidate)
        self.assertEqual(report, json.loads((JOB / "static_validation.json").read_text(encoding="utf-8")))
        self.assertEqual(report["overall_status"], "pass")

        grouped = grouped_blocks(candidate)
        adaptive = split_fields(grouped["*DEFINE_ADAPTIVE_SOLID_TO_SPH"][0][0])
        self.assertEqual([int(value) for value in adaptive[:7]], [2002, 0, 2, 2003, 2, 0, 0])
        sph_contact = grouped["*CONTACT_AUTOMATIC_NODES_TO_SURFACE"][0]
        self.assertEqual([int(value) for value in split_fields(sph_contact[0])[:4]], [2003, 920001, 3, 2])
        self.assertEqual(int(split_fields(sph_contact[3])[0]), 1)
        solid_contact = grouped["*CONTACT_ERODING_SINGLE_SURFACE"][0]
        self.assertEqual(int(split_fields(solid_contact[4])[0]), 2)
        termination = split_fields(grouped["*CONTROL_TERMINATION"][0][0])
        self.assertEqual(int(termination[1]), MAXIMUM_CYCLE_COUNT)

    def test_batch_is_reproducible(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06m-build-", dir=temp_root) as temp:
            regenerated = Path(temp) / "batch"
            build_batch(
                baseline_deck=R06J_DECK,
                design_path=DESIGN,
                contract_path=CONTRACT,
                material_path=MATERIAL,
                output_dir=regenerated,
            )
            expected_files = {
                path.relative_to(BATCH).as_posix(): path
                for path in BATCH.rglob("*")
                if path.is_file()
            }
            actual_files = {
                path.relative_to(regenerated).as_posix(): path
                for path in regenerated.rglob("*")
                if path.is_file()
            }
            self.assertEqual(set(actual_files), set(expected_files))
            for relative, actual in actual_files.items():
                self.assertEqual(actual.read_bytes(), expected_files[relative].read_bytes(), relative)

    def test_staged_submission_is_direct_folder_without_archive(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06m-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            delivery = stage_adaptive_sph_canary(BATCH, output)
            self.assertEqual(delivery["job_id"], JOB_ID)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"uav_hf_r06m_0800_adaptive_sph_canary", "DELIVERY_MANIFEST.json", "SUBMIT_ONLY_THIS_FOLDER.md"},
            )
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in output.rglob("*")))
            upload = output / delivery["upload_folder"]
            self.assertTrue((upload / delivery["head_deck"]).is_file())
            self.assertEqual(sha256(upload / delivery["head_deck"]), delivery["head_deck_sha256"])
            self.assertEqual(delivery["capability_bundle"]["maximum_cycle_count"], MAXIMUM_CYCLE_COUNT)
            self.assertEqual(delivery["status"], "ready_for_user_submission_not_submitted_not_solver_validated")


if __name__ == "__main__":
    unittest.main()
