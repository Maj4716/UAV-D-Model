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

from build_r06j_fragment_erod1_return_audit import build_audit  # noqa: E402
from create_p0_r06k_fragment_negative_volume_isolation_design import (  # noqa: E402
    build_contract,
    build_design,
)
from generate_p0_r06_diagnostics import generate_batch  # noqa: E402
from stage_r06k_fragment_negative_volume_isolation import (  # noqa: E402
    stage_fragment_negative_volume_diagnostic,
)
from validate_p0_r06_diagnostics import (  # noqa: E402
    comparable_blocks,
    grouped_blocks,
    split_fields,
    validate_batch,
)


DESIGN = PROJECT / "design" / "p0_r06k_fragment_negative_volume_isolation_design.json"
CONTRACT = PROJECT / "design" / "p0_r06k_fragment_negative_volume_output_contract.json"
PARENT_DESIGN = PROJECT / "design" / "p0_r06j_fragment_erosion_isolation_design.json"
PARENT_CONTRACT = PROJECT / "design" / "p0_r06j_fragment_erosion_output_contract.json"
ANALYSIS = PROJECT / "analysis" / "72751725_r06j_0800_fragment_erod1_20261030"
AUDIT = ANALYSIS / "r06j_fragment_erod1_return_audit.json"
BASELINE_AUDIT = (
    PROJECT
    / "analysis"
    / "72740022_r06i_0800_tssfac005_20261030"
    / "r06i_tssfac005_return_audit.json"
)
RETURN = PROJECT.parent / "uav_hf_r06j_0800_fragment_erod1"
BATCH = PROJECT / "manual_upload_20261030_r06k_fragment_negative_volume_isolation"
JOB = BATCH / "job_01_v0800_fragment_psfail_plate"
BASELINE_DECK = ANALYSIS / "immutable_parent_r06j_0800_fragment_erod1.k"


class R06kFragmentNegativeVolumeIsolationTests(unittest.TestCase):
    def test_r06j_return_audit_is_reproducible_and_releases_only_psfail_isolation(self) -> None:
        rebuilt = build_audit(
            return_dir=RETURN,
            analysis_dir=ANALYSIS,
            baseline_audit_path=BASELINE_AUDIT,
        )
        stored = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        comparison = stored["formal_fragment_erosion_mechanism_comparison_to_job_72740022"]
        self.assertEqual(comparison["status"], "partial_support_remaining_negative_volume_path")
        self.assertEqual(comparison["deletion_change"]["control_deleted_element_count"], 47)
        self.assertEqual(comparison["deletion_change"]["candidate_deleted_element_count"], 35)
        self.assertEqual(comparison["deletion_change"]["deleted_element_count_reduction"], 12)
        self.assertGreater(comparison["deletion_change"]["absolute_fraction_reduction"], 0.0)
        self.assertEqual(
            comparison["remaining_deletion_attribution"]["status"],
            "pass_exact_count_and_fragment_topology",
        )
        self.assertEqual(comparison["response_availability_gate"]["status"], "fail")
        self.assertGreater(comparison["response_availability_gate"]["observed"], 0.05)
        self.assertTrue(stored["decision"]["release_r06k_fragment_negative_volume_psfail_isolation"])
        self.assertFalse(stored["decision"]["release_2000_control"])
        self.assertFalse(stored["decision"]["release_treatments"])
        topology = stored["execution"]["negative_volume_topology"]
        self.assertEqual(topology["unique_element_count"], 35)
        self.assertEqual(topology["plate_unique_element_count"], 0)
        self.assertEqual(topology["fragment_unique_element_count"], 35)

    def test_frozen_design_and_contract_match_reproducible_builder(self) -> None:
        parent_design = json.loads(PARENT_DESIGN.read_text(encoding="utf-8"))
        parent_contract = json.loads(PARENT_CONTRACT.read_text(encoding="utf-8"))
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        contract = build_contract(parent_contract, audit)
        design = build_design(parent_design, contract, audit)
        self.assertEqual(contract, json.loads(CONTRACT.read_text(encoding="utf-8")))
        self.assertEqual(design, json.loads(DESIGN.read_text(encoding="utf-8")))
        self.assertEqual(
            contract["fragment_negative_volume_isolation"]["interpretation_rules"]["primary_qoi_policy"],
            "never promote PSFAIL-isolation response values to physical labels, calibration targets or reference data",
        )

    def test_generated_job_changes_only_control_solid_psfail_from_returned_parent(self) -> None:
        result = validate_batch(BATCH, write_reports=False)
        self.assertEqual(result["overall_status"], "pass")
        self.assertEqual(result["job_count"], 1)
        candidate = (JOB / "r06k_0800_fragment_psfail_plate.k").read_text(encoding="ascii")
        baseline = BASELINE_DECK.read_text(encoding="ascii")
        self.assertEqual(
            comparable_blocks(candidate, {"*TITLE", "*CONTROL_SOLID"}),
            comparable_blocks(baseline, {"*TITLE", "*CONTROL_SOLID"}),
        )
        candidate_grouped = grouped_blocks(candidate)
        baseline_grouped = grouped_blocks(baseline)
        control_solid = split_fields(baseline_grouped["*CONTROL_SOLID"][0][0])
        treatment_solid = split_fields(candidate_grouped["*CONTROL_SOLID"][0][0])
        self.assertEqual(int(control_solid[4]), 0)
        self.assertEqual(int(treatment_solid[4]), 910001)
        self.assertEqual(control_solid[:4] + control_solid[5:], treatment_solid[:4] + treatment_solid[5:])
        self.assertEqual(candidate_grouped["*CONTROL_TIMESTEP"], baseline_grouped["*CONTROL_TIMESTEP"])
        self.assertEqual(candidate_grouped["*MAT_JOHNSON_COOK"], baseline_grouped["*MAT_JOHNSON_COOK"])
        fragment_card4 = split_fields(candidate_grouped["*MAT_JOHNSON_COOK"][1][3])
        self.assertEqual(int(fragment_card4[2]), 1)
        stored = json.loads((JOB / "static_validation.json").read_text(encoding="utf-8"))
        isolation = stored["checks"]["immutable_r06j_fragment_psfail_factor_isolation"]
        self.assertEqual(isolation["status"], "pass")
        self.assertEqual(isolation["only_changed_field"], "CONTROL_SOLID Card 1 field 5 PSFAIL")

    def test_batch_is_reproducible(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06k-psfail-", dir=temp_root) as temp:
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
                "job_01_v0800_fragment_psfail_plate/r06k_0800_fragment_psfail_plate.k",
                "job_01_v0800_fragment_psfail_plate/manifest.json",
                "job_01_v0800_fragment_psfail_plate/static_validation.json",
            ):
                self.assertEqual((regenerated / relative).read_bytes(), (BATCH / relative).read_bytes())

    def test_staged_submission_is_direct_folder_without_archive(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06k-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            manifest = stage_fragment_negative_volume_diagnostic(BATCH, output, AUDIT)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "uav_hf_r06k_0800_fragment_psfail_plate",
                    "DELIVERY_MANIFEST.json",
                    "SUBMIT_ONLY_THIS_FOLDER.md",
                },
            )
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in output.rglob("*")))
            upload = output / manifest["upload_folder"]
            self.assertTrue((upload / manifest["head_deck"]).is_file())
            self.assertEqual(manifest["single_factor"]["control"], 0)
            self.assertEqual(manifest["single_factor"]["treatment"], 910001)
            self.assertEqual(
                manifest["status"],
                "ready_for_user_submission_not_submitted_not_solver_validated",
            )


if __name__ == "__main__":
    unittest.main()
