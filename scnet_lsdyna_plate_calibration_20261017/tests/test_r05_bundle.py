from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_p0_r05_candidate_bundle import build_bundle  # noqa: E402


CANDIDATE = PROJECT / "manual_upload_20261021_r05_candidate"


class R05BundleTests(unittest.TestCase):
    def temporary_root(self, prefix: str) -> tempfile.TemporaryDirectory[str]:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix=prefix, dir=temp_root)

    def test_rejects_stale_static_pass_after_package_tamper(self) -> None:
        with self.temporary_root("r05-bundle-stale-") as temp:
            root = Path(temp)
            package = root / "candidate"
            shutil.copytree(CANDIDATE, package)
            deck = package / "p0_velocity_mesh_calibration.k"
            text = deck.read_text(encoding="ascii")
            original = "5.000000000e-05,0,0,0,900001"
            tampered = "1.000000000e-04,0,0,0,900001"
            self.assertIn(original, text)
            deck.write_text(
                text.replace(original, tampered, 1),
                encoding="ascii",
                newline="\n",
            )

            def reject_current_package(path: Path) -> dict:
                self.assertEqual(path, package.resolve())
                self.assertIn(tampered, deck.read_text(encoding="ascii"))
                raise AssertionError("live validation rejected the current package")

            with patch(
                "build_p0_r05_candidate_bundle.validate_candidate",
                side_effect=reject_current_package,
            ) as live_validation:
                with self.assertRaisesRegex(AssertionError, "live validation rejected"):
                    build_bundle(
                        package,
                        root / "candidate.zip",
                        root / "candidate.manifest.json",
                    )

            live_validation.assert_called_once_with(package.resolve())

    def test_rejects_identical_bundle_and_manifest_paths_before_writing(self) -> None:
        with self.temporary_root("r05-bundle-alias-") as temp:
            shared_output = Path(temp) / "candidate.output"

            with self.assertRaisesRegex(ValueError, "must be different"):
                build_bundle(CANDIDATE, shared_output, shared_output)

            self.assertFalse(shared_output.exists())

    def test_rejects_output_paths_inside_package_before_writing(self) -> None:
        for inside_output in ("bundle", "manifest"):
            with self.subTest(inside_output=inside_output):
                with self.temporary_root(f"r05-bundle-inside-{inside_output}-") as temp:
                    root = Path(temp)
                    package = root / "candidate"
                    shutil.copytree(CANDIDATE, package)
                    bundle = (
                        package / "candidate.zip"
                        if inside_output == "bundle"
                        else root / "candidate.zip"
                    )
                    manifest = (
                        package / "candidate.manifest.json"
                        if inside_output == "manifest"
                        else root / "candidate.manifest.json"
                    )

                    with self.assertRaisesRegex(ValueError, "outside the package directory"):
                        build_bundle(package, bundle, manifest)

                    self.assertFalse(bundle.exists())
                    self.assertFalse(manifest.exists())

    def test_builds_valid_candidate_to_distinct_external_paths(self) -> None:
        with self.temporary_root("r05-bundle-valid-") as temp:
            root = Path(temp)
            bundle = root / "candidate.zip"
            manifest = root / "candidate.manifest.json"
            stored_validation = json.loads(
                (CANDIDATE / "static_validation.json").read_text(encoding="utf-8")
            )

            with patch(
                "build_p0_r05_candidate_bundle.validate_candidate",
                return_value=stored_validation,
            ) as live_validation:
                result = build_bundle(CANDIDATE, bundle, manifest)

            self.assertTrue(bundle.is_file())
            self.assertTrue(manifest.is_file())
            self.assertEqual(result["bundle"]["path"], bundle.name)
            live_validation.assert_called_once_with(CANDIDATE.resolve())


if __name__ == "__main__":
    unittest.main()
