from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_r05_run_integrity import (  # noqa: E402
    build_artifact_manifest,
    evaluate_run_integrity,
    parse_added_mass_parts,
    required_return_equivalences,
    sha256_file,
)


class R05RunIntegrityTests(unittest.TestCase):
    def temporary_root(self, prefix: str) -> tempfile.TemporaryDirectory[str]:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix=prefix, dir=temp_root)

    def test_added_mass_summary_must_be_explicit(self) -> None:
        text = "number of parts with added mass................       0\n"
        self.assertEqual(parse_added_mass_parts(text), 0)
        with self.assertRaisesRegex(ValueError, "added-mass part count"):
            parse_added_mass_parts("mass scaling configured off\n")

    def test_binout_and_solver_output_are_explicit_return_equivalents(self) -> None:
        inventory = {
            "solver_output": ["job_example.out"],
            "rank_messages": [f"mes{index:04d}" for index in range(32)],
            "glstat": ["glstat"],
            "binout": ["binout0000"],
            "d3plot": ["d3plot"],
            "d3part": ["d3part"],
            "solver_stdout": ["std.out.123"],
            "solver_stderr": ["std.err.123"],
            "slurm_script": ["job.slurm"],
            "resource_record": ["job.ma"],
        }
        result = required_return_equivalences(
            inventory=inventory,
            binout_directories={"JOBINFO", "glstat", "matsum", "nodout", "sleout"},
            expected_rank_count=32,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["items"]["d3hsp"]["mode"], "solver_o_output")
        self.assertEqual(result["items"]["matsum"]["mode"], "binout_logical_database")

        missing = required_return_equivalences(
            inventory=inventory,
            binout_directories={"JOBINFO", "glstat", "matsum", "nodout"},
            expected_rank_count=32,
        )
        self.assertEqual(missing["status"], "fail")
        self.assertFalse(missing["items"]["sleout"]["satisfied"])

    def test_frozen_integrity_gates_pass_only_with_complete_finite_evidence(self) -> None:
        observed = {
            "rank_log_count": 32,
            "normal_termination_rank_count": 32,
            "normal_termination": True,
            "error_termination": False,
            "fatal_or_error_count": 0,
            "nonfinite_token_count": 0,
            "total_energy_ratio_minimum": 0.968,
            "total_energy_ratio_maximum": 1.002,
            "hourglass_to_peak_internal_energy": 0.0,
            "added_mass_parts": 0,
            "dt2ms": 0.0,
            "return_contract_status": "pass",
        }
        gates = {
            "all_ranks_normal_termination": True,
            "fatal_or_error_count_max": 0,
            "total_energy_ratio_min": 0.95,
            "total_energy_ratio_max": 1.05,
            "absolute_hourglass_to_peak_internal_energy_max": 0.05,
            "added_mass_fraction_max": 0.0,
            "missing_or_nonfinite_policy": "unavailable and fail-closed",
        }
        passed = evaluate_run_integrity(observed, gates, expected_rank_count=32)
        self.assertEqual(passed["status"], "pass")
        self.assertTrue(all(item["status"] == "pass" for item in passed["gates"].values()))

        failed = evaluate_run_integrity(
            {**observed, "total_energy_ratio_maximum": None},
            gates,
            expected_rank_count=32,
        )
        self.assertEqual(failed["status"], "fail")
        self.assertEqual(failed["gates"]["total_energy_ratio"]["status"], "fail")

    def test_artifact_manifest_is_sorted_and_hash_bound(self) -> None:
        with self.temporary_root("r05-run-manifest-") as temp:
            root = Path(temp)
            (root / "b").write_bytes(b"second")
            (root / "a").write_bytes(b"first")
            manifest = build_artifact_manifest(
                result_dir=root,
                paths=[root / "b", root / "a"],
                run_id="job-123",
                solver_build="LS-DYNA R15 fixture",
            )
            self.assertEqual([item["path"] for item in manifest["files"]], ["a", "b"])
            self.assertTrue(
                all(item["sha256"].startswith("sha256:") for item in manifest["files"])
            )
            json.dumps(manifest, allow_nan=False)

    @unittest.skipUnless(
        (PROJECT / "analysis" / "72252362_r05_stage_ab_20261023" / "run_integrity_audit.json").is_file(),
        "Job 72252362 Stage-A evidence is not present",
    )
    def test_real_job_72252362_evidence_remains_hash_bound_and_fail_closed(self) -> None:
        output = PROJECT / "analysis" / "72252362_r05_stage_ab_20261023"
        audit = json.loads((output / "run_integrity_audit.json").read_text(encoding="utf-8"))
        evidence = json.loads(
            (output / "run_integrity_evidence.json").read_text(encoding="utf-8")
        )
        self.assertEqual(audit["assessment"]["status"], "pass")
        self.assertEqual(audit["returned_evidence_contract"]["status"], "pass")
        self.assertEqual(
            evidence["returned_artifact_sha256"],
            sha256_file(output / "run_artifact_manifest.json"),
        )
        self.assertEqual(
            evidence["d3part_family_sha256"],
            "sha256:8f24005080f09cbfd1b81dd2f0f97d4abb3158e7133207f44beccb9852346af7",
        )
        self.assertIn("remain separate fail-closed gates", audit["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
