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

from build_p0_r06o_plate_erod1_isolation import (  # noqa: E402
    HEAD_DECK,
    JOB_DIR,
    build_batch,
    transform_deck,
    validate_transformed_deck,
)
from build_r06g_tssfac02_return_audit import sha256  # noqa: E402
from build_r06n_adaptive_sph_32rank_return_audit import build_audit  # noqa: E402
from compare_r06m_r06n_adaptive_sph_resources import build_comparison  # noqa: E402
from create_p0_r06o_plate_erod1_isolation_design import (  # noqa: E402
    JOB_ID,
    PARENT_DECK_SHA256,
    REVISION,
    build_contract,
    build_design,
)
from stage_r06o_plate_erod1_isolation import stage_isolation  # noqa: E402
from validate_r06m_adaptive_sph_outputs import build_validation  # noqa: E402


R06M_RETURN = PROJECT.parent / "uav_hf_r06m_0800_adaptive_sph_canary"
R06M_ASCII = PROJECT / "analysis" / "72861050_r06m_0800_adaptive_sph_canary_20261101" / "ascii_metrics.json"
R06N_RETURN = PROJECT.parent / "uav_hf_r06n_0800_adaptive_sph_32rank_confirmation"
ANALYSIS = PROJECT / "analysis" / "72864757_r06n_0800_adaptive_sph_32rank_confirmation_20261102"
R06N_ASCII = ANALYSIS / "ascii_metrics.json"
VALIDATION = ANALYSIS / "adaptive_sph_validation.json"
COMPARISON = ANALYSIS / "r06m_r06n_resource_sensitivity.json"
AUDIT = ANALYSIS / "r06n_adaptive_sph_32rank_return_audit.json"
PARENT_DECK = ANALYSIS / "immutable_parent_r06n_0800_adaptive_sph_32rank_confirmation.k"
DESIGN = PROJECT / "design" / "p0_r06o_plate_erod1_isolation_design.json"
CONTRACT = PROJECT / "design" / "p0_r06o_plate_erod1_output_contract.json"
MATERIAL = PROJECT / "materials" / "jc4340_reference.json"
CASE_MAP = (
    PROJECT
    / "manual_upload_20261101_r06n_adaptive_sph_32rank_confirmation"
    / "job_01_v0800_adaptive_sph_32rank_confirmation"
    / "p0_case_part_map.csv"
)
BATCH = PROJECT / "manual_upload_20261102_r06o_plate_erod1_isolation"
JOB = BATCH / JOB_DIR


class R06oPlateErod1IsolationTests(unittest.TestCase):
    def test_r06n_diagnostic_validation_is_reproducible_and_reports_plate_element(self) -> None:
        rebuilt = build_validation(
            R06N_RETURN,
            schema_version="hf-p0-r06n-adaptive-sph-validation@1.0.0",
            strict_gates=False,
        )
        stored = json.loads(VALIDATION.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        conversion = stored["adaptive_sph"]["conversion"]
        self.assertEqual(conversion["plate_deleted_element_count"], 1)
        self.assertEqual(conversion["final_deleted_plate_element_ids"], [39948])
        self.assertTrue(conversion["statewise_exact_activation_correspondence"])
        self.assertEqual(conversion["final_active_sph_count"], 8 * conversion["final_converted_solid_count"])

    def test_resource_sensitivity_comparison_is_reproducible(self) -> None:
        rebuilt = build_comparison(
            r06m_return=R06M_RETURN,
            r06n_return=R06N_RETURN,
            r06m_ascii=R06M_ASCII,
            r06n_ascii=R06N_ASCII,
        )
        stored = json.loads(COMPARISON.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        paired = stored["paired_diagnostics"]
        self.assertLess(paired["cycle_count_ratio_r06n_over_r06m"], 0.8)
        self.assertEqual(paired["converted_fragment_count_difference"], 3)
        self.assertTrue(paired["r06n_plate_failure_is_not_in_negative_volume_set"])

    def test_r06n_audit_is_reproducible_and_releases_only_plate_erod_isolation(self) -> None:
        rebuilt = build_audit(return_dir=R06N_RETURN, analysis_dir=ANALYSIS)
        stored = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        self.assertEqual(stored["decision"]["resource_identity"], "pass")
        self.assertEqual(stored["decision"]["numerical_engineering_confirmation"], "fail_plate_deletion_and_cycle_repeatability")
        self.assertTrue(stored["decision"]["release_r06o_plate_mat_erod1_isolation"])
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
        self.assertEqual(contract["single_factor"]["parent_level"], 0)
        self.assertEqual(contract["single_factor"]["candidate_level"], 1)

    def test_generated_deck_changes_title_and_plate_erod_only(self) -> None:
        self.assertEqual(sha256(PARENT_DECK), PARENT_DECK_SHA256)
        parent = PARENT_DECK.read_text(encoding="ascii")
        candidate = (JOB / HEAD_DECK).read_text(encoding="ascii")
        self.assertEqual(candidate, transform_deck(parent))
        report = validate_transformed_deck(parent, candidate)
        stored = json.loads((JOB / "static_validation.json").read_text(encoding="utf-8"))
        expected = dict(report)
        expected["candidate_deck_sha256"] = sha256(JOB / HEAD_DECK)
        self.assertEqual(expected, stored)
        self.assertEqual(stored["differing_line_numbers"], [3, 59])
        self.assertTrue(stored["all_other_deck_lines_byte_identical"])
        case_map_lines = (JOB / "p0_case_part_map.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(case_map_lines), 2)
        self.assertTrue(case_map_lines[1].startswith(f"{JOB_ID},r06o_plate_material_failure_erosion_isolation_800,"))

    def test_batch_is_reproducible(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06o-build-", dir=temp_root) as temp:
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

    def test_staged_submission_is_direct_folder_and_exact_resource_contract(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06o-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            delivery = stage_isolation(BATCH, output)
            self.assertEqual(delivery["job_id"], JOB_ID)
            self.assertEqual(delivery["resource_contract"]["nodes"], 1)
            self.assertEqual(delivery["resource_contract"]["ranks"], 32)
            self.assertEqual(delivery["single_factor"], "plate_MAT_JOHNSON_COOK_EROD_0_to_1")
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in output.rglob("*")))
            upload = output / delivery["upload_folder"]
            self.assertEqual(sha256(upload / delivery["head_deck"]), delivery["head_deck_sha256"])


if __name__ == "__main__":
    unittest.main()
