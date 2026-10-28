from __future__ import annotations

import tempfile
from pathlib import Path

from parse_lsdyna_ascii import assess, parse_glstat, parse_termination


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="hf_postprocess_selftest_") as directory:
        root = Path(directory)
        (root / "d3hsp").write_text(
            "LS-DYNA synthetic parser fixture\nN o r m a l   t e r m i n a t i o n\n",
            encoding="utf-8",
        )
        # Include the compact wording used by multiple LS-DYNA cohorts and a
        # Fortran D exponent to exercise the parser's normalization.
        (root / "glstat").write_text(
            "energy ratio = 1.002D+00\n"
            "kinetic energy = 2.500D+00\n"
            "internal energy = 1.000D+01\n"
            "hourglass energy = 2.000D-01\n"
            "sliding interface energy = 5.000D-02\n"
            "added mass = 0.000D+00\n",
            encoding="utf-8",
        )
        termination = parse_termination(root)
        glstat = parse_glstat(root / "glstat")
        status, reasons = assess(termination, glstat)
        assert termination["normal_termination"]
        assert not termination["error_termination"]
        assert glstat["final_total_energy_ratio"] == 1.002
        assert glstat["final_hourglass_energy"] == 0.2
        assert status == "pass", reasons

        (root / "messag").write_text("fatal error: negative volume\n", encoding="utf-8")
        termination = parse_termination(root)
        status, reasons = assess(termination, glstat)
        assert status == "review_or_fail"
        # Generic wording is not counted as an element-level SOL+509 event.
        assert termination["negative_volume_mentions"] == 0
        assert termination["error_termination"]
        assert reasons
    print("postprocessor self-test: PASS")


if __name__ == "__main__":
    main()
