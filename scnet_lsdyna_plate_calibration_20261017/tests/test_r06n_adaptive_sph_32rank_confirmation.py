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

from build_p0_r06n_adaptive_sph_32rank_confirmation import (  # noqa: E402
    HEAD_DECK,
    JOB_DIR,
    build_batch,
    transform_deck,
    validate_transformed_deck,
)
from build_r06m_adaptive_sph_return_audit import build_audit  # noqa: E402
from build_r06g_tssfac02_return_audit import sha256  # noqa: E402
from create_p0_r06n_adaptive_sph_32rank_confirmation_design import (  # noqa: E402
    JOB_ID,
    PARENT_DECK_SHA256,
    REVISION,
    build_contract,
    build_design,
)
from stage_r06n_adaptive_sph_32rank_confirmation import stage_confirmation  # noqa: E402
from validate_r06m_adaptive_sph_outputs import build_validation  # noqa: E402


ANALYSIS = PROJECT / "analysis" / "72861050_r06m_0800_adaptive_sph_canary_20261101"
RETURN = PROJECT.parent / "uav_hf_r06m_0800_adaptive_sph_canary"
AUDIT = ANALYSIS / "r06m_adaptive_sph_return_audit.json"
VALIDATION = ANALYSIS / "adaptive_sph_validation.json"
PARENT_DECK = ANALYSIS / "immutable_parent_r06m_0800_adaptive_sph_canary.k"
DESIGN = PROJECT / "design" / "p0_r06n_adaptive_sph_32rank_confirmation_design.json"
CONTRACT = PROJECT / "design" / "p0_r06n_adaptive_sph_32rank_output_contract.json"
MATERIAL = PROJECT / "materials" / "jc4340_reference.json"
CASE_MAP = (
    PROJECT
    / "manual_upload_20261101_r06m_adaptive_sph_canary"
    / "job_01_v0800_adaptive_sph_canary"
    / "p0_case_part_map.csv"
)
BATCH = PROJECT / "manual_upload_20261101_r06n_adaptive_sph_32rank_confirmation"
JOB = BATCH / JOB_DIR
DELIVERY = PROJECT / "artifacts" / "r06n_adaptive_sph_32rank_confirmation_submission_20261101"


class R06nAdaptiveSph32RankConfirmationTests(unittest.TestCase):
    def test_completed_r06m_validation_is_reproducible_and_matsum_sum_is_denied(self) -> None:
        rebuilt = build_validation(RETURN)
        stored = json.loads(VALIDATION.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        matsum = stored["adaptive_sph"]["matsum"]
        self.assertTrue(matsum["source_part_2002_and_sph_part_2003_duplicate_at_initial_state"])
        self.assertFalse(matsum["naive_sum_across_source_part_2002_and_sph_part_2003_is_admissible"])

    def test_r06m_return_audit_is_reproducible_and_releases_only_exact_confirmation(self) -> None:
        rebuilt = build_audit(return_dir=RETURN, analysis_dir=ANALYSIS)
        stored = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        self.assertEqual(stored["execution"]["resource_contract"]["ranks"], 32)
        self.assertEqual(stored["execution"]["observed_resources"]["ranks"], 64)
        self.assertTrue(stored["decision"]["release_r06n_exact_32rank_confirmation"])
        self.assertFalse(stored["decision"]["release_terminal_horizon_extension"])
        self.assertEqual(stored["decision"]["engine_fitting"], "denied")
        self.assertEqual(stored["decision"]["damage_tree_fitting"], "denied")

    def test_frozen_design_and_contract_match_reproducible_builder(self) -> None:
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        contract = build_contract(audit)
        design = build_design(contract)
        self.assertEqual(contract, json.loads(CONTRACT.read_text(encoding="utf-8")))
        self.assertEqual(design, json.loads(DESIGN.read_text(encoding="utf-8")))
        self.assertEqual(design["package_revision"], REVISION)
        self.assertEqual(contract["single_factor"]["candidate_required_level"], 32)
        self.assertEqual(contract["single_factor"]["permitted_deck_difference"], "TITLE_identifier_line_only")

    def test_generated_deck_changes_title_only(self) -> None:
        self.assertEqual(sha256(PARENT_DECK), PARENT_DECK_SHA256)
        parent = PARENT_DECK.read_text(encoding="ascii")
        candidate = (JOB / HEAD_DECK).read_text(encoding="ascii")
        self.assertEqual(candidate, transform_deck(parent))
        report = validate_transformed_deck(parent, candidate)
        stored = json.loads((JOB / "static_validation.json").read_text(encoding="utf-8"))
        expected = dict(report)
        expected["candidate_deck_sha256"] = sha256(JOB / HEAD_DECK)
        self.assertEqual(expected, stored)
        self.assertEqual(stored["differing_line_numbers"], [3])
        self.assertTrue(stored["solver_model_output_cards_byte_identical"])

    def test_batch_is_reproducible(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06n-build-", dir=temp_root) as temp:
            regenerated = Path(temp) / "batch"
            build_batch(
                parent_deck=PARENT_DECK,
                design_path=DESIGN,
                contract_path=CONTRACT,
                material_path=MATERIAL,
                case_map_path=CASE_MAP,
                output_dir=regenerated,
            )
            expected = {path.relative_to(BATCH).as_posix(): path for path in BATCH.rglob("*") if path.is_file()}
            actual = {path.relative_to(regenerated).as_posix(): path for path in regenerated.rglob("*") if path.is_file()}
            self.assertEqual(set(actual), set(expected))
            for relative, path in actual.items():
                self.assertEqual(path.read_bytes(), expected[relative].read_bytes(), relative)

    def test_staged_submission_is_direct_folder_and_exact_32rank_contract(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06n-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            delivery = stage_confirmation(BATCH, output)
            self.assertEqual(delivery["job_id"], JOB_ID)
            self.assertEqual(delivery["resource_contract"]["nodes"], 1)
            self.assertEqual(delivery["resource_contract"]["ranks"], 32)
            self.assertTrue(delivery["resource_contract"]["hard_gate"])
            self.assertEqual(delivery["deck_change"], "TITLE_identifier_line_only")
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in output.rglob("*")))
            upload = output / delivery["upload_folder"]
            self.assertEqual(sha256(upload / delivery["head_deck"]), delivery["head_deck_sha256"])


if __name__ == "__main__":
    unittest.main()
