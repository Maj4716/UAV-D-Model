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

from build_p0_r06_diagnostic_bundles import build_bundles  # noqa: E402
from build_r06f_tssfac04_return_audit import build_audit  # noqa: E402
from create_p0_r06g_timestep_refinement_design import (  # noqa: E402
    build_contract,
    build_design,
)
from generate_p0_r06_diagnostics import generate_batch  # noqa: E402
from stage_r06g_timestep_refinement import stage_refinement  # noqa: E402
from validate_p0_r06_diagnostics import (  # noqa: E402
    comparable_blocks,
    grouped_blocks,
    split_fields,
    validate_batch,
)
from verify_p0_r06_release import verify_zip  # noqa: E402


DESIGN = PROJECT / "design" / "p0_r06g_timestep_refinement_design.json"
CONTRACT = PROJECT / "design" / "p0_r06g_timestep_output_contract.json"
PARENT_DESIGN = PROJECT / "design" / "p0_r06f_timestep_sensitivity_design.json"
PARENT_CONTRACT = PROJECT / "design" / "p0_r06f_timestep_output_contract.json"
ANALYSIS = PROJECT / "analysis" / "72630855_r06f_0800_tssfac04_20261028"
AUDIT = ANALYSIS / "r06f_tssfac04_return_audit.json"
BASELINE_AUDIT = (
    PROJECT
    / "analysis"
    / "72617857_r06f_0800_ctrl_20261028"
    / "r06f_return_audit.json"
)
RETURN = PROJECT.parent / "uav_hf_r06f_0800_tssfac04"
BATCH = PROJECT / "manual_upload_20261028_r06g_timestep_refinement_v2"
JOB = BATCH / "job_01_v0800_tssfac02"
RELEASE = PROJECT / "artifacts" / "r06g_timestep_refinement_20261028_v2"
BASELINE_DECK = (
    PROJECT
    / "manual_upload_20261028_r06f_timestep_sensitivity"
    / "job_01_v0800_tssfac04"
    / "r06f_0800_tssfac04.k"
)


class R06gTimestepRefinementTests(unittest.TestCase):
    def test_return_audit_is_reproducible_and_authorizes_only_refinement(self) -> None:
        rebuilt = build_audit(
            return_dir=RETURN,
            analysis_dir=ANALYSIS,
            baseline_audit_path=BASELINE_AUDIT,
        )
        stored = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        self.assertEqual(stored["formal_timestep_comparison_to_job_72617857"]["status"], "fail")
        self.assertTrue(stored["decision"]["release_r06g_0800_tssfac02"])
        self.assertFalse(stored["decision"]["release_r06f_2000_ctrl"])

    def test_frozen_design_and_contract_match_reproducible_builder(self) -> None:
        parent_design = json.loads(PARENT_DESIGN.read_text(encoding="utf-8"))
        parent_contract = json.loads(PARENT_CONTRACT.read_text(encoding="utf-8"))
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        contract = build_contract(parent_contract, audit)
        design = build_design(parent_design, contract, audit)
        self.assertEqual(contract, json.loads(CONTRACT.read_text(encoding="utf-8")))
        self.assertEqual(design, json.loads(DESIGN.read_text(encoding="utf-8")))

    def test_generated_job_changes_only_tssfac_from_tssfac04_parent(self) -> None:
        result = validate_batch(BATCH, write_reports=False)
        self.assertEqual(result["overall_status"], "pass")
        self.assertEqual(result["job_count"], 1)
        candidate = (JOB / "r06g_0800_tssfac02.k").read_text(encoding="ascii")
        baseline = BASELINE_DECK.read_text(encoding="ascii")
        candidate_timestep = split_fields(grouped_blocks(candidate)["*CONTROL_TIMESTEP"][0][0])
        baseline_timestep = split_fields(grouped_blocks(baseline)["*CONTROL_TIMESTEP"][0][0])
        self.assertEqual(baseline_timestep[1], "4.000000000e-01")
        self.assertEqual(candidate_timestep[1], "2.000000000e-01")
        self.assertEqual(
            candidate_timestep[:1] + candidate_timestep[2:],
            baseline_timestep[:1] + baseline_timestep[2:],
        )
        self.assertEqual(candidate_timestep[4], "0.000000000e+00")
        self.assertEqual(
            comparable_blocks(candidate, {"*TITLE", "*CONTROL_TIMESTEP"}),
            comparable_blocks(baseline, {"*TITLE", "*CONTROL_TIMESTEP"}),
        )
        stored = json.loads((JOB / "static_validation.json").read_text(encoding="utf-8"))
        isolation = stored["checks"]["immutable_tssfac04_parent_factor_isolation"]
        self.assertEqual(isolation["status"], "pass")
        self.assertEqual(isolation["only_changed_field"], "CONTROL_TIMESTEP.TSSFAC")

    def test_batch_is_reproducible_and_zip_is_deterministic(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06g-ts-", dir=temp_root) as temp:
            temp = Path(temp)
            regenerated = temp / "batch"
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
                "job_01_v0800_tssfac02/r06g_0800_tssfac02.k",
                "job_01_v0800_tssfac02/manifest.json",
                "job_01_v0800_tssfac02/static_validation.json",
            ):
                self.assertEqual((regenerated / relative).read_bytes(), (BATCH / relative).read_bytes())
            first = temp / "release1"
            second = temp / "release2"
            build_bundles(BATCH, first)
            build_bundles(BATCH, second)
            self.assertEqual(
                (first / "uav_hf_r06g_0800_tssfac02.zip").read_bytes(),
                (second / "uav_hf_r06g_0800_tssfac02.zip").read_bytes(),
            )

    def test_staged_submission_contains_one_verified_job(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06g-ts-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            manifest = stage_refinement(RELEASE, output, AUDIT)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "uav_hf_r06g_0800_tssfac02.zip",
                    "uav_hf_r06g_0800_tssfac02.bundle_manifest.json",
                    "SUBMIT_ONLY_THIS_JOB.md",
                    "WAVE0_MANIFEST.json",
                },
            )
            self.assertEqual(
                manifest["status"],
                "ready_for_user_submission_not_submitted_not_solver_validated",
            )
            verify_zip(
                output / "uav_hf_r06g_0800_tssfac02.zip",
                output / "uav_hf_r06g_0800_tssfac02.bundle_manifest.json",
            )


if __name__ == "__main__":
    unittest.main()
