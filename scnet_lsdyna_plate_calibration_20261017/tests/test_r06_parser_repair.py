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

from analyze_r06_wave1_keyword_failure import DEFAULT_CASES, audit_case  # noqa: E402
from create_p0_r06_parser_repair_design import build_design  # noqa: E402
from stage_r06e_wave0_canary import stage_wave0  # noqa: E402
from validate_p0_r06_diagnostics import (  # noqa: E402
    grouped_blocks,
    validate_batch,
    validate_schema,
)
from validate_package import keyword_blocks  # noqa: E402


PARENT_DESIGN = PROJECT / "design" / "p0_r06_diagnostic_design.json"
REPAIR_DESIGN = PROJECT / "design" / "p0_r06_keyword_parser_repair_design.json"
LEGACY_BATCH = PROJECT / "manual_upload_20261023_r06_diagnostics"
REPAIR_BATCH = PROJECT / "manual_upload_20261025_r06e_parser_repair"
REPAIR_RELEASE = PROJECT / "artifacts" / "r06e_parser_repair_20261025"
WAVE0_STAGE = PROJECT / "artifacts" / "r06e_wave0_canary_20261025"


class R06ParserRepairTests(unittest.TestCase):
    def test_repair_design_is_deterministic_and_staged(self) -> None:
        design = json.loads(REPAIR_DESIGN.read_text(encoding="utf-8"))
        self.assertEqual(design["package_revision"], "R06_keyword_parser_repair_20261025")
        self.assertEqual(design["submission_sequence"]["wave_0"], ["r06e_0800_ctrl"])
        self.assertEqual(
            design["shared_measurement_upgrade"]["r15_parser_contract"],
            {
                "target_solver": "LS-DYNA R15.0.2",
                "database_extent_binary_cards": [
                    [6, 0, 3, 0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 2, 1],
                ],
                "intfor_file_timing_card_tail": [0, 0, 0],
                "intfor_file_ioopt_card": [0],
            },
        )
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06-repair-design-", dir=temp_root) as temp:
            first = Path(temp) / "first.json"
            second = Path(temp) / "second.json"
            build_design(PARENT_DESIGN, first)
            build_design(PARENT_DESIGN, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes(), REPAIR_DESIGN.read_bytes())

    def test_actual_repair_batch_has_exact_r15_cards(self) -> None:
        result = validate_batch(REPAIR_BATCH, write_reports=False)
        self.assertEqual(result["overall_status"], "pass")
        self.assertEqual(
            result["admission"]["r06_diagnostic_submission"],
            "ready_for_user_submission_in_staged_canary",
        )
        deck = (
            REPAIR_BATCH
            / "job_01_v0800_control"
            / "r06e_0800_ctrl.k"
        ).read_text(encoding="ascii")
        grouped = grouped_blocks(deck)
        self.assertEqual(
            grouped["*DATABASE_EXTENT_BINARY"][0],
            ["6,0,3,0,1,1,1,1", "0,0,0,1,1,1,2,1"],
        )
        self.assertEqual(
            grouped["*DATABASE_BINARY_INTFOR_FILE"][0],
            ["intfor", "5.000000000e-05,0,0,0", "0"],
        )
        self.assertEqual(
            grouped["*DATABASE_BINARY_D3PART"][0],
            ["5.000000000e-05,0,0,0,900001"],
        )

    def test_repair_changes_only_title_and_shared_output_serialization(self) -> None:
        pairs = (
            (
                LEGACY_BATCH / "job_01_v0800_control" / "r06d_0800_ctrl.k",
                REPAIR_BATCH / "job_01_v0800_control" / "r06e_0800_ctrl.k",
            ),
            (
                LEGACY_BATCH / "job_03_v2000_control" / "r06d_2000_ctrl.k",
                REPAIR_BATCH / "job_03_v2000_control" / "r06e_2000_ctrl.k",
            ),
        )
        ignored = {
            "*TITLE",
            "*DATABASE_EXTENT_BINARY",
            "*DATABASE_BINARY_INTFOR_FILE",
        }
        for legacy_path, repair_path in pairs:
            legacy = [
                block
                for block in keyword_blocks(legacy_path.read_text(encoding="ascii"))
                if block[0] not in ignored
            ]
            repair = [
                block
                for block in keyword_blocks(repair_path.read_text(encoding="ascii"))
                if block[0] not in ignored
            ]
            self.assertEqual(legacy, repair)

    def test_validator_rejects_both_historical_card_defects_in_repair_mode(self) -> None:
        design = json.loads(REPAIR_DESIGN.read_text(encoding="utf-8"))
        deck = (
            REPAIR_BATCH
            / "job_01_v0800_control"
            / "r06e_0800_ctrl.k"
        ).read_text(encoding="ascii")
        missing_card2 = deck.replace("0,0,0,1,1,1,2,1\n", "", 1)
        with self.assertRaisesRegex(AssertionError, "requires exactly 2 cards"):
            validate_schema(missing_card2, design)
        five_field_intfor = deck.replace(
            "5.000000000e-05,0,0,0\n0\n*DATABASE_EXTENT_INTFOR",
            "5.000000000e-05,0,0,0,0\n0\n*DATABASE_EXTENT_INTFOR",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "requires 4 fields"):
            validate_schema(five_field_intfor, design)

    def test_downloaded_returns_reproduce_common_keyword_failure(self) -> None:
        reports = [audit_case(case) for case in DEFAULT_CASES]
        self.assertEqual([report["job_id"] for report in reports], [
            "r06d_0800_ctrl",
            "r06d_2000_ctrl",
        ])
        for report in reports:
            self.assertEqual(report["zip"]["crc"], "pass")
            self.assertTrue(report["deck"]["manifest_match"])
            self.assertEqual(report["deck"]["extent_binary_card_count"], 1)
            self.assertEqual(report["deck"]["intfor_timing_field_count"], 5)
            self.assertFalse(report["execution"]["time_integration_started"])
            self.assertTrue(report["diagnostics"]["error_10246_lines"])
            self.assertTrue(report["diagnostics"]["error_10450_lines"])
            self.assertTrue(report["diagnostics"]["warning_10457_lines"])

    def test_wave0_stage_contains_only_the_canary_and_evidence(self) -> None:
        self.assertEqual(
            {path.name for path in WAVE0_STAGE.iterdir() if path.is_file()},
            {
                "uav_hf_r06e_0800_ctrl.zip",
                "uav_hf_r06e_0800_ctrl.bundle_manifest.json",
                "SUBMIT_ONLY_THIS_JOB.md",
                "WAVE0_MANIFEST.json",
            },
        )
        manifest = json.loads(
            (WAVE0_STAGE / "WAVE0_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["upload_only"], "uav_hf_r06e_0800_ctrl.zip")
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r06-wave0-stage-", dir=temp_root) as temp:
            staged = Path(temp) / "stage"
            report = stage_wave0(REPAIR_RELEASE, staged)
            self.assertEqual(report["zip"]["sha256"], manifest["zip"]["sha256"])
            self.assertEqual(
                (staged / "uav_hf_r06e_0800_ctrl.zip").read_bytes(),
                (WAVE0_STAGE / "uav_hf_r06e_0800_ctrl.zip").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
