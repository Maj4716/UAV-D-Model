from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_p0_r06_diagnostic_bundles import build_bundles  # noqa: E402


BATCH = PROJECT / "manual_upload_20261025_r06e_parser_repair"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R06DiagnosticBundleTests(unittest.TestCase):
    def test_bundles_are_deterministic_one_deck_explicit_whitelists(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06-bundles-", dir=temp_root) as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            release_first = build_bundles(BATCH, first)
            release_second = build_bundles(BATCH, second)
            self.assertEqual(release_first, release_second)
            self.assertEqual(release_first["bundle_count"], 4)
            for entry in release_first["bundles"]:
                zip_name = entry["zip"]["path"]
                self.assertEqual(digest(first / zip_name), digest(second / zip_name))
                self.assertEqual("sha256:" + digest(first / zip_name), entry["zip"]["sha256"])
                with zipfile.ZipFile(first / zip_name) as archive:
                    names = archive.namelist()
                    decks = [name for name in names if name.endswith(".k")]
                    self.assertEqual(len(decks), 1)
                    self.assertIn("candidate_bundle_manifest.json", names)
                    self.assertFalse(any(name.startswith(("05/", "analysis/", ".vendor/")) for name in names))
                    embedded = json.loads(archive.read("candidate_bundle_manifest.json"))
                    self.assertEqual(embedded["head_deck"], decks[0])
                    self.assertEqual(embedded["file_policy"], "explicit_whitelist_one_head_deck")
                    self.assertEqual(embedded["status"], "prepared_not_submitted_not_solver_validated")

    def test_builder_refuses_overwrite(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06-bundle-overwrite-", dir=temp_root) as temp:
            output = Path(temp) / "release"
            build_bundles(BATCH, output)
            with self.assertRaises(FileExistsError):
                build_bundles(BATCH, output)


if __name__ == "__main__":
    unittest.main()
