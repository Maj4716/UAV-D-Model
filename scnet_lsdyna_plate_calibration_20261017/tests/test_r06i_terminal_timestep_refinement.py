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

from build_r06h_tssfac01_return_audit import build_audit  # noqa: E402
from create_p0_r06i_terminal_timestep_refinement_design import (  # noqa: E402
    build_contract,
    build_design,
)
from generate_p0_r06_diagnostics import generate_batch  # noqa: E402
from stage_r06i_terminal_timestep_refinement import stage_terminal_refinement  # noqa: E402
from validate_p0_r06_diagnostics import (  # noqa: E402
    comparable_blocks,
    grouped_blocks,
    split_fields,
    validate_batch,
)


DESIGN = PROJECT / "design" / "p0_r06i_terminal_timestep_refinement_design.json"
CONTRACT = PROJECT / "design" / "p0_r06i_terminal_timestep_output_contract.json"
PARENT_DESIGN = PROJECT / "design" / "p0_r06h_timestep_confirmation_design.json"
PARENT_CONTRACT = PROJECT / "design" / "p0_r06h_timestep_output_contract.json"
ANALYSIS = PROJECT / "analysis" / "72701744_r06h_0800_tssfac01_20261030"
AUDIT = ANALYSIS / "r06h_tssfac01_return_audit.json"
BASELINE_AUDIT = (
    PROJECT
    / "analysis"
    / "72639935_r06g_0800_tssfac02_20261029"
    / "r06g_tssfac02_return_audit.json"
)
RETURN = PROJECT.parent / "uav_hf_r06h_0800_tssfac01"
BATCH = PROJECT / "manual_upload_20261030_r06i_terminal_timestep_refinement"
JOB = BATCH / "job_01_v0800_tssfac005"
BASELINE_DECK = ANALYSIS / "immutable_parent_r06h_0800_tssfac01.k"


class R06iTerminalTimestepRefinementTests(unittest.TestCase):
    def test_r06h_return_audit_is_reproducible_and_authorizes_only_terminal_refinement(self) -> None:
        rebuilt = build_audit(
            return_dir=RETURN,
            analysis_dir=ANALYSIS,
            baseline_audit_path=BASELINE_AUDIT,
        )
        stored = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt, stored)
        comparison = stored["formal_timestep_comparison_to_job_72639935"]
        self.assertEqual(comparison["status"], "fail")
        self.assertEqual(comparison["gates"]["qM"]["status"], "pass")
        self.assertEqual(comparison["gates"]["qP"]["status"], "pass")
        self.assertEqual(comparison["gates"]["qE"]["status"], "pass")
        self.assertEqual(
            comparison["gates"]["qV"]["status"],
            "pass_not_applicable_zero_confirmed_crossing_mass",
        )
        self.assertEqual(comparison["gates"]["deleted_upstream_mass_fraction"]["status"], "fail")
        self.assertGreater(
            comparison["gates"]["deleted_upstream_mass_fraction"]["absolute_difference"],
            0.01,
        )
        self.assertEqual(comparison["response_availability_gate"]["status"], "fail")
        self.assertTrue(stored["decision"]["release_r06i_0800_tssfac005"])
        self.assertFalse(stored["decision"]["release_any_further_timestep_halving_after_r06i"])
        self.assertFalse(stored["decision"]["release_2000_control"])
        self.assertFalse(stored["decision"]["release_treatments"])
        topology = stored["execution"]["negative_volume_topology"]
        self.assertEqual(topology["unique_element_count"], 5)
        self.assertEqual(topology["plate_unique_element_count"], 0)
        self.assertEqual(topology["fragment_unique_element_count"], 5)

    def test_frozen_design_and_contract_match_reproducible_builder(self) -> None:
        parent_design = json.loads(PARENT_DESIGN.read_text(encoding="utf-8"))
        parent_contract = json.loads(PARENT_CONTRACT.read_text(encoding="utf-8"))
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        contract = build_contract(parent_contract, audit)
        design = build_design(parent_design, contract, audit)
        self.assertEqual(contract, json.loads(CONTRACT.read_text(encoding="utf-8")))
        self.assertEqual(design, json.loads(DESIGN.read_text(encoding="utf-8")))

    def test_generated_job_changes_only_tssfac_from_returned_parent(self) -> None:
        result = validate_batch(BATCH, write_reports=False)
        self.assertEqual(result["overall_status"], "pass")
        self.assertEqual(result["job_count"], 1)
        candidate = (JOB / "r06i_0800_tssfac005.k").read_text(encoding="ascii")
        baseline = BASELINE_DECK.read_text(encoding="ascii")
        candidate_timestep = split_fields(grouped_blocks(candidate)["*CONTROL_TIMESTEP"][0][0])
        baseline_timestep = split_fields(grouped_blocks(baseline)["*CONTROL_TIMESTEP"][0][0])
        self.assertEqual(baseline_timestep[1], "1.000000000e-01")
        self.assertEqual(candidate_timestep[1], "5.000000000e-02")
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
        isolation = stored["checks"]["immutable_tssfac01_parent_factor_isolation"]
        self.assertEqual(isolation["status"], "pass")
        self.assertEqual(isolation["only_changed_field"], "CONTROL_TIMESTEP.TSSFAC")

    def test_batch_is_reproducible(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06i-ts-", dir=temp_root) as temp:
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
                "job_01_v0800_tssfac005/r06i_0800_tssfac005.k",
                "job_01_v0800_tssfac005/manifest.json",
                "job_01_v0800_tssfac005/static_validation.json",
            ):
                self.assertEqual((regenerated / relative).read_bytes(), (BATCH / relative).read_bytes())

    def test_staged_submission_is_direct_folder_without_archive(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06i-stage-", dir=temp_root) as temp:
            output = Path(temp) / "stage"
            manifest = stage_terminal_refinement(BATCH, output, AUDIT)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "uav_hf_r06i_0800_tssfac005",
                    "DELIVERY_MANIFEST.json",
                    "SUBMIT_ONLY_THIS_FOLDER.md",
                },
            )
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in output.rglob("*")))
            upload = output / manifest["upload_folder"]
            self.assertTrue((upload / manifest["head_deck"]).is_file())
            self.assertEqual(manifest["terminal_stop_rule"]["no_automatic_tssfac_below"], 0.05)
            self.assertEqual(
                manifest["status"],
                "ready_for_user_submission_not_submitted_not_solver_validated",
            )


if __name__ == "__main__":
    unittest.main()
