from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_p0_r06_diagnostics import generate_batch  # noqa: E402
from generate_p0_r05 import guard_frozen_cli_revision  # noqa: E402
from validate_p0_r06_diagnostics import REVISION, validate_batch  # noqa: E402


DESIGN = PROJECT / "design" / "p0_r06_diagnostic_design.json"
R05_DESIGN = PROJECT / "design" / "p0_cases_r05.json"
MATERIAL = PROJECT / "materials" / "jc4340_reference.json"
SUCCESSOR = PROJECT / "design" / "p0_r06_real_output_contract.json"
ACTUAL = PROJECT / "manual_upload_20261023_r06_diagnostics"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R06DiagnosticPackageTests(unittest.TestCase):
    def test_preregistered_design_has_two_pairs_and_four_jobs(self) -> None:
        design = json.loads(DESIGN.read_text(encoding="utf-8"))
        self.assertEqual(design["package_revision"], REVISION)
        self.assertEqual(len(design["paired_experiments"]), 2)
        jobs = [job for experiment in design["paired_experiments"] for job in experiment["jobs"]]
        self.assertEqual(len(jobs), 4)
        self.assertEqual(sum(job["role"] == "paired_control" for job in jobs), 2)
        self.assertEqual(sum(job["role"] == "single_factor_treatment" for job in jobs), 2)
        self.assertEqual(
            [experiment["factor"] for experiment in design["paired_experiments"]],
            ["termination_time_ms", "control_solid_psfail_part_set"],
        )

    def test_historical_batch_is_withdrawn_after_target_parser_failure(self) -> None:
        with self.assertRaisesRegex(AssertionError, "known target-R15 parser defect"):
            validate_batch(ACTUAL, write_reports=False)
        historical = json.loads(
            (ACTUAL / "R06_SUBMISSION_READINESS.json").read_text(encoding="utf-8")
        )
        self.assertEqual(historical["overall_status"], "pass")
        self.assertEqual(
            historical["lifecycle_status"],
            "prepared_not_submitted_not_solver_validated",
        )

    def test_generation_is_deterministic_and_refuses_overwrite(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06-generation-", dir=temp_root) as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            kwargs = {
                "design_path": DESIGN,
                "r05_design_path": R05_DESIGN,
                "material_path": MATERIAL,
                "successor_contract_path": SUCCESSOR,
            }
            generate_batch(output_dir=first, **kwargs)
            generate_batch(output_dir=second, **kwargs)
            first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
            second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
            self.assertEqual(first_files, second_files)
            for relative in first_files:
                self.assertEqual(digest(first / relative), digest(second / relative), str(relative))
            with self.assertRaises(FileExistsError):
                generate_batch(output_dir=first, **kwargs)

    def test_successor_generator_rejects_r05_revision(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06-revision-", dir=temp_root) as temp:
            temp_path = Path(temp)
            design = json.loads(DESIGN.read_text(encoding="utf-8"))
            design["package_revision"] = "R05_capture_ledger_preregistered"
            bad_design = temp_path / "bad_design.json"
            bad_design.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refuses non-R06 revision"):
                generate_batch(
                    design_path=bad_design,
                    r05_design_path=R05_DESIGN,
                    material_path=MATERIAL,
                    successor_contract_path=SUCCESSOR,
                    output_dir=temp_path / "candidate",
                )

    def test_legacy_r05_cli_requires_explicit_fixture_override(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "colliding identity"):
            guard_frozen_cli_revision(R05_DESIGN, False)
        guard_frozen_cli_revision(R05_DESIGN, True)

    def test_psfail_tamper_is_rejected(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06-psfail-", dir=temp_root) as temp:
            batch = Path(temp) / "candidate"
            generate_batch(
                design_path=DESIGN,
                r05_design_path=R05_DESIGN,
                material_path=MATERIAL,
                successor_contract_path=SUCCESSOR,
                output_dir=batch,
            )
            job_dir = batch / "job_04_v2000_psfail_plate_only"
            manifest_path = job_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            deck = job_dir / manifest["deck"]["path"]
            text = deck.read_text(encoding="ascii")
            self.assertIn("0,0,0,0,910001,0.000000000e+00,0,0", text)
            deck.write_text(
                text.replace("0,0,0,0,910001,0.000000000e+00,0,0", "0,0,0,0,900001,0.000000000e+00,0,0", 1),
                encoding="ascii",
                newline="\n",
            )
            manifest["deck"]["sha256"] = "sha256:" + digest(deck)
            manifest["deck"]["size_bytes"] = deck.stat().st_size
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            batch_manifest_path = batch / "batch_manifest.json"
            batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
            for entry in batch_manifest["jobs"]:
                if entry["job_id"] == "r06d_2000_psfail_plate":
                    entry["deck_sha256"] = manifest["deck"]["sha256"]
                    entry["manifest_sha256"] = "sha256:" + digest(manifest_path)
            batch_manifest_path.write_text(json.dumps(batch_manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "CONTROL_SOLID"):
                validate_batch(batch, allow_known_failed_revision=True)

    def test_observability_tamper_is_rejected(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06-intfor-", dir=temp_root) as temp:
            batch = Path(temp) / "candidate"
            generate_batch(
                design_path=DESIGN,
                r05_design_path=R05_DESIGN,
                material_path=MATERIAL,
                successor_contract_path=SUCCESSOR,
                output_dir=batch,
            )
            job_dir = batch / "job_01_v0800_control"
            manifest_path = job_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            deck = job_dir / manifest["deck"]["path"]
            text = deck.read_text(encoding="ascii")
            deck.write_text(text.replace("1,1,1,1,1,1,0,0", "1,1,0,1,1,1,0,0", 1), encoding="ascii", newline="\n")
            manifest["deck"]["sha256"] = "sha256:" + digest(deck)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            batch_manifest_path = batch / "batch_manifest.json"
            batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
            for entry in batch_manifest["jobs"]:
                if entry["job_id"] == "r06d_0800_ctrl":
                    entry["manifest_sha256"] = "sha256:" + digest(manifest_path)
            batch_manifest_path.write_text(json.dumps(batch_manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "EXTENT_INTFOR"):
                validate_batch(batch, allow_known_failed_revision=True)

    def test_historical_r05_and_successor_contract_are_unchanged(self) -> None:
        expected = {
            PROJECT / "manual_upload_20261021_r05_candidate" / "p0_velocity_mesh_calibration.k": "6fbd753a113d0214e7ba4ff5136dc526d151e6f2c52c94e70399b29c4b9d3d0c",
            PROJECT / "manual_upload_20261021_r05_candidate" / "manifest.json": "e178b6c28832519aaa2befdc969dcd93d2141ea4073ad740b52b17968d92ba44",
            SUCCESSOR: "71371ee8726fe95899de33aef163d8a9725d9f43781d5775de10fa2afc150795",
        }
        for path, expected_hash in expected.items():
            self.assertEqual(digest(path), expected_hash, str(path))


if __name__ == "__main__":
    unittest.main()
