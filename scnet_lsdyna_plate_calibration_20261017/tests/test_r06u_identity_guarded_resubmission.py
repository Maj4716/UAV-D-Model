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

from build_r06g_tssfac02_return_audit import sha256  # noqa: E402
from build_r06u_identity_mismatch_audit import build_audit  # noqa: E402
from stage_r06u_identity_guarded_resubmission import (  # noqa: E402
    EXPECTED_FILES,
    HEAD_DECK,
    HEAD_DECK_SHA256,
    UPLOAD_DIR,
    stage_resubmission,
)


RETURN = PROJECT.parent / "uav_hf_r06u_1200_plate_adaptive_sph_canary"
SOURCE = PROJECT / "artifacts" / "r06u_plate_adaptive_sph_canary_submission_20261104" / "uav_hf_r06u_1200_plate_adaptive_sph_canary"


class R06uIdentityGuardedResubmissionTests(unittest.TestCase):
    def test_identity_audit_rejects_run_as_r06u_evidence(self) -> None:
        audit = build_audit(RETURN, SOURCE)
        self.assertEqual(audit["identity_gate"]["status"], "fail")
        self.assertEqual(audit["identity_gate"]["executed_input_revision"], "R06t")
        self.assertFalse(audit["identity_gate"]["intended_deck_present_in_return"])
        self.assertFalse(audit["identity_gate"]["deck_hashes_equal"])
        self.assertFalse(audit["decision"]["r06u_was_executed"])
        self.assertFalse(audit["decision"]["change_r06u_model"])
        self.assertTrue(audit["decision"]["release_exact_r06u_identity_guarded_resubmission"])
        self.assertEqual(audit["repeated_r06t_failure_signature"]["problem_time_ms"], 0.10872)
        self.assertEqual(audit["repeated_r06t_failure_signature"]["cycle"], 603398)

    def test_resubmission_is_byte_identical_direct_folder(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06u-retry-", dir=temp_root) as temp:
            output = Path(temp) / "delivery"
            delivery = stage_resubmission(SOURCE, output)
            upload = output / UPLOAD_DIR
            self.assertEqual(delivery["model_change_from_original_r06u"], "none_exact_byte_identical_resubmission")
            self.assertEqual({path.name for path in upload.iterdir()}, EXPECTED_FILES)
            for name in EXPECTED_FILES:
                self.assertEqual((upload / name).read_bytes(), (SOURCE / name).read_bytes(), name)
            self.assertEqual(sha256(upload / HEAD_DECK), HEAD_DECK_SHA256)
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in output.rglob("*")))

    def test_delivery_contains_human_and_machine_identity_guards(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06u-guard-", dir=temp_root) as temp:
            output = Path(temp) / "delivery"
            stage_resubmission(SOURCE, output)
            manifest = json.loads((output / "RESUBMISSION_MANIFEST.json").read_text(encoding="utf-8"))
            instructions = (output / "IDENTITY_CHECK_BEFORE_SUBMIT.md").read_text(encoding="utf-8")
            self.assertEqual(manifest["head_deck"], HEAD_DECK)
            self.assertEqual(manifest["head_deck_sha256"], HEAD_DECK_SHA256)
            self.assertIn("r06t", instructions)
            self.assertIn("全新工作目录", instructions)
            self.assertIn(HEAD_DECK, instructions)


if __name__ == "__main__":
    unittest.main()
