from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from parse_lsdyna_ascii import (  # noqa: E402
    assess,
    bind_negative_volume_events,
    parse_control_timestep,
    parse_glstat,
    parse_glstat_series,
    parse_rank_diagnostics,
    parse_termination,
)


class PostprocessorTests(unittest.TestCase):
    def test_real_r15_glstat_dot_fill_format(self) -> None:
        text = """
 time...........................   0.00000E+00
 time step......................   6.46272E-05
 kinetic energy.................   2.80500E+04
 internal energy................   6.40000E-19
 hourglass energy ..............   0.00000E+00
 sliding interface energy.......   0.00000E+00
 eroded kinetic energy..........   0.00000E+00
 eroded internal energy.........   0.00000E+00
 eroded hourglass energy........   0.00000E+00
 total energy...................   2.80500E+04
 total energy / initial energy..   1.00000E+00
 energy ratio w/o eroded energy.   1.00000E+00

 time...........................   1.99984E-02
 time step......................   7.56953E-06
 kinetic energy.................   1.66200E+04
 internal energy................   5.55313E+03
 hourglass energy ..............   0.00000E+00
 sliding interface energy.......   5.24882E+03
 eroded kinetic energy..........   7.90039E+03
 eroded internal energy.........   1.86553E+03
 eroded hourglass energy........   0.00000E+00
 total energy...................   2.74219E+04
 total energy / initial energy..   9.77608E-01
 energy ratio w/o eroded energy.   6.29447E-01
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glstat"
            path.write_text(text, encoding="utf-8")
            records = parse_glstat_series(path)
            summary = parse_glstat(path)
        self.assertEqual(len(records), 2)
        self.assertAlmostEqual(summary["final_total_energy_ratio"], 0.977608)
        self.assertAlmostEqual(summary["final_energy_ratio_without_eroded"], 0.629447)
        self.assertEqual(summary["maximum_absolute_hourglass_over_internal"], 0.0)
        self.assertAlmostEqual(summary["final_eroded_energy_sum"], 9765.92)

    def test_all_rank_logs_and_case_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mes0000").write_text(
                "*** Warning 30364 (INI+364)\n"
                "*** Warning 40509 (SOL+509)\n"
                "negative volume in solid element # 10 cycle 28 time 1.2192E-03\n"
                "N o r m a l    t e r m i n a t i o n\n",
                encoding="utf-8",
            )
            (root / "mes0001").write_text(
                "*** Warning 30364 (INI+364)\n"
                "*** Warning 40509 (SOL+509)\n"
                "negative volume in solid element # 20 cycle 30 time 1.3000E-03\n"
                "N o r m a l    t e r m i n a t i o n\n",
                encoding="utf-8",
            )
            (root / "std.out.1").write_text(
                "Mass Scaling ....... 1.0E-03 0.01\nN o r m a l termination\n",
                encoding="utf-8",
            )
            deck = root / "model.k"
            deck.write_text(
                "*KEYWORD LONG=Y\n"
                "*CONTROL_TIMESTEP\n"
                "0,0.8,0,0,0,0,0,0\n"
                "*ELEMENT_SOLID\n"
                "10,1001,1,2,3,4,5,6,7,8\n"
                "*ELEMENT_SOLID\n"
                "20,2001,11,12,13,14,15,16,17,18\n"
                "*END\n",
                encoding="utf-8",
            )
            case_map = root / "cases.csv"
            with case_map.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id",
                        "role",
                        "plate_part_id",
                        "fragment_part_id",
                        "impact_speed_mps",
                        "mesh_mm",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "case-a",
                        "role": "hf_response_fit",
                        "plate_part_id": 1001,
                        "fragment_part_id": 2001,
                        "impact_speed_mps": 1200,
                        "mesh_mm": 0.75,
                    }
                )
            diagnostics = parse_rank_diagnostics(root)
            termination = parse_termination(root)
            bound, summary = bind_negative_volume_events(
                diagnostics["negative_volume_events"], deck, case_map
            )
            timestep = parse_control_timestep(deck)

        self.assertEqual(diagnostics["rank_log_count"], 2)
        self.assertEqual(diagnostics["normal_termination_rank_count"], 2)
        self.assertEqual(diagnostics["negative_volume_unique_element_count"], 2)
        self.assertEqual(termination["mass_scaling_mentions"], 0)
        self.assertEqual({row["part_kind"] for row in bound}, {"plate", "fragment"})
        self.assertEqual(summary[0]["total_negative_volume_events"], 2)
        self.assertFalse(timestep["configured_mass_scaling"])

    def test_assessment_requires_negative_volume_review(self) -> None:
        termination = {
            "normal_termination": True,
            "error_termination": False,
            "negative_volume_mentions": 1,
        }
        glstat = {
            "final_total_energy_ratio": 1.0,
            "maximum_absolute_hourglass_over_internal": 0.0,
        }
        status, reasons = assess(termination, glstat)
        self.assertEqual(status, "review_or_fail")
        self.assertEqual(reasons, ["negative-volume events require case-level review"])


if __name__ == "__main__":
    unittest.main()
