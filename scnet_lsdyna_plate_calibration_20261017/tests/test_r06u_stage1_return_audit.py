from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_r06u_stage1_return_audit import (  # noqa: E402
    EXPECTED_DECK_SHA256,
    build_audit,
)


RETURN = PROJECT.parent / "uav_hf_r06u_1200_plate_adaptive_sph_canary_retry"


class R06uStage1ReturnAuditTests(unittest.TestCase):
    def test_identity_resources_and_completion_pass(self) -> None:
        audit = build_audit(RETURN)
        self.assertEqual(audit["source_evidence"]["deck_sha256"], EXPECTED_DECK_SHA256)
        self.assertEqual(audit["execution"]["input_identity"], "pass")
        self.assertEqual(audit["execution"]["resource_identity"], "pass")
        self.assertTrue(audit["execution"]["normal_termination_at_required_time"])
        self.assertEqual(audit["execution"]["normal_termination_rank_count"], 32)
        self.assertEqual(audit["execution"]["final_problem_time_ms"], 0.15)
        self.assertEqual(audit["execution"]["cycle_count"], 244053)
        self.assertFalse(audit["execution"]["complex_sound_speed_present"])

    def test_target_element_is_traversed_without_releasing_physics(self) -> None:
        audit = build_audit(RETURN)
        adaptive = audit["adaptive_sph_runtime_evidence"]
        self.assertEqual(adaptive["r06t_target_element"]["element_id"], 23830)
        self.assertEqual(adaptive["r06t_target_element"]["part_id"], 1002)
        self.assertEqual(adaptive["r06t_target_element"]["first_time_ms"], 0.042562)
        self.assertGreater(adaptive["continued_from_target_failure_to_terminal_time_ms"], 0.1)
        self.assertEqual(audit["decision"]["engine_fitting"], "denied")
        self.assertEqual(audit["decision"]["damage_tree_fitting"], "denied")

    def test_only_missing_small_databases_are_requested(self) -> None:
        audit = build_audit(RETURN)
        self.assertEqual(
            audit["missing_required_small_databases"],
            ["glstat", "matsum", "sleout", "nodout", "ncforc"],
        )
        self.assertFalse(audit["decision"]["stage1_complete"])
        self.assertFalse(audit["decision"]["release_heavy_binary_download"])
        self.assertFalse(audit["decision"]["release_next_compute_submission"])
        self.assertFalse(audit["decision"]["change_model"])
        self.assertEqual(audit["next_single_action"]["do_not_download_yet"], ["d3plot*", "d3part*", "binout*"])


if __name__ == "__main__":
    unittest.main()
