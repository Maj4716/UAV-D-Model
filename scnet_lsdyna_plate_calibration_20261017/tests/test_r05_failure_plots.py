from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
MODULE_PATH = SCRIPTS / "plot_r05_failure_modes.py"

spec = importlib.util.spec_from_file_location("plot_r05_failure_modes", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
plots = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plots
spec.loader.exec_module(plots)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fixture_inputs(root: Path) -> tuple[Path, Path, Path]:
    events_path = root / "failure_events_unique.csv"
    summary_path = root / "failure_summary_by_case.csv"
    design_path = root / "p0_cases_r05.json"

    event_fields = [
        "failure_type",
        "element_id",
        "time_ms",
        "case_id",
        "part_kind",
        "initial_deck_centroid_z_mm",
        "initial_position_semantics",
    ]
    _write_csv(
        events_path,
        event_fields,
        [
            {
                "failure_type": "negative_volume",
                "element_id": 1,
                "time_ms": 0.003,
                "case_id": "p0_v0800_h0p75",
                "part_kind": "plate",
                "initial_deck_centroid_z_mm": -0.5,
                "initial_position_semantics": "deck_reference_geometry_not_failure_time_position",
            },
            {
                "failure_type": "other_solid_failure",
                "element_id": 2,
                "time_ms": 0.004,
                "case_id": "p0_v0800_h0p75",
                "part_kind": "plate",
                "initial_deck_centroid_z_mm": -0.7,
                "initial_position_semantics": "deck_reference_geometry_not_failure_time_position",
            },
            {
                "failure_type": "negative_volume",
                "element_id": 3,
                "time_ms": 0.005,
                "case_id": "p0_v0800_h0p75",
                "part_kind": "fragment",
                "initial_deck_centroid_z_mm": 2.0,
                "initial_position_semantics": "deck_reference_geometry_not_failure_time_position",
            },
            {
                "failure_type": "other_solid_failure",
                "element_id": 4,
                "time_ms": 0.002,
                "case_id": "p0_v2000_h0p75",
                "part_kind": "plate",
                "initial_deck_centroid_z_mm": -1.0,
                "initial_position_semantics": "deck_reference_geometry_not_failure_time_position",
            },
            {
                "failure_type": "negative_volume",
                "element_id": 5,
                "time_ms": 0.006,
                "case_id": "p0_v2000_h0p75",
                "part_kind": "fragment",
                "initial_deck_centroid_z_mm": 3.0,
                "initial_position_semantics": "deck_reference_geometry_not_failure_time_position",
            },
            {
                "failure_type": "other_solid_failure",
                "element_id": 6,
                "time_ms": 0.007,
                "case_id": "p0_v2000_h0p75",
                "part_kind": "fragment",
                "initial_deck_centroid_z_mm": 4.0,
                "initial_position_semantics": "deck_reference_geometry_not_failure_time_position",
            },
        ],
    )

    summary_fields = [
        "case_id",
        "impact_speed_mps",
        "mesh_mm",
        "failure_event_count",
        "plate_negative_volume_count",
        "plate_other_failure_count",
        "fragment_negative_volume_count",
        "fragment_other_failure_count",
        "deleted_upstream_mass_fraction",
    ]
    _write_csv(
        summary_path,
        summary_fields,
        [
            {
                "case_id": "p0_v0800_h0p75",
                "impact_speed_mps": 800,
                "mesh_mm": 0.75,
                "failure_event_count": 3,
                "plate_negative_volume_count": 1,
                "plate_other_failure_count": 1,
                "fragment_negative_volume_count": 1,
                "fragment_other_failure_count": 0,
                "deleted_upstream_mass_fraction": 0.04,
            },
            {
                "case_id": "p0_v2000_h0p75",
                "impact_speed_mps": 2000,
                "mesh_mm": 0.75,
                "failure_event_count": 3,
                "plate_negative_volume_count": 0,
                "plate_other_failure_count": 1,
                "fragment_negative_volume_count": 1,
                "fragment_other_failure_count": 1,
                "deleted_upstream_mass_fraction": 0.20,
            },
        ],
    )
    design_path.write_text(
        json.dumps(
            {
                "qualification_gates": {
                    "capture_ledger": {"deleted_upstream_mass_fraction_max": 0.05}
                }
            }
        ),
        encoding="utf-8",
    )
    return events_path, summary_path, design_path


class R05FailurePlotTests(unittest.TestCase):
    def temporary_root(self, prefix: str) -> tempfile.TemporaryDirectory[str]:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix=prefix, dir=temp_root)

    def test_source_tables_preserve_structural_zeros_and_raw_focus_events(self) -> None:
        with self.temporary_root("r05-viz-source-") as temp:
            events_path, summary_path, design_path = _fixture_inputs(Path(temp))
            inputs = plots.load_inputs(events_path, summary_path, design_path)
            tables = plots.build_source_tables(inputs)

        panel_a = tables["panel_a"]
        self.assertEqual(len(panel_a), 2 * 4)
        zero_rows = [row for row in panel_a if row["count"] == 0]
        self.assertEqual(
            {(row["case_id"], row["part_kind"], row["failure_type"]) for row in zero_rows},
            {
                ("p0_v0800_h0p75", "fragment", "other_solid_failure"),
                ("p0_v2000_h0p75", "plate", "negative_volume"),
            },
        )
        self.assertEqual(len(tables["panel_b"]), 6)
        self.assertEqual(
            [row["element_id"] for row in tables["panel_b"]], [1, 2, 3, 4, 5, 6]
        )
        self.assertEqual(
            [row["gate_status"] for row in tables["panel_c"]],
            ["at_or_below_frozen_gate", "above_frozen_gate"],
        )

    def test_rejects_event_counts_that_disagree_with_case_summary(self) -> None:
        with self.temporary_root("r05-viz-count-mismatch-") as temp:
            root = Path(temp)
            events_path, summary_path, design_path = _fixture_inputs(root)
            with summary_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["plate_negative_volume_count"] = "2"
            _write_csv(summary_path, list(rows[0]), rows)
            with self.assertRaisesRegex(
                ValueError,
                "category counts do not sum|event counts differ from case summary",
            ):
                plots.load_inputs(events_path, summary_path, design_path)

    def test_generation_exports_opaque_png_svg_source_data_manifest_and_alt_text(self) -> None:
        with self.temporary_root("r05-viz-generate-") as temp:
            root = Path(temp)
            inputs_dir = root / "inputs"
            outputs_dir = root / "outputs"
            inputs_dir.mkdir()
            events_path, summary_path, design_path = _fixture_inputs(inputs_dir)
            manifest = plots.generate_visualizations(
                events_path=events_path,
                summary_path=summary_path,
                design_path=design_path,
                output_dir=outputs_dir,
                replace=False,
            )

            self.assertEqual(manifest["schema_version"], plots.MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest["frozen_gate"]["value"], 0.05)
            self.assertFalse(manifest["uncertainty"]["displayed"])
            self.assertEqual(manifest["source_data"]["panel_a"]["row_count"], 8)
            self.assertEqual(manifest["source_data"]["panel_b"]["row_count"], 6)
            self.assertEqual(manifest["source_data"]["panel_c"]["row_count"], 2)
            self.assertIn("not physical causation", manifest["claim_boundary"])

            for name in plots.PNG_OUTPUTS:
                path = outputs_dir / name
                self.assertTrue(path.is_file(), name)
                with Image.open(path) as image:
                    self.assertEqual(image.mode, "RGB")
                    expected = plots.FIGURE_SPECS[name.removesuffix(".png")]
                    self.assertEqual(
                        image.size,
                        (
                            round(expected["width_in"] * plots.PNG_DPI),
                            round(expected["height_in"] * plots.PNG_DPI),
                        ),
                    )
                metadata = manifest["artifacts"][name]["metadata"]
                self.assertFalse(metadata["has_alpha_channel"])
                self.assertEqual(metadata["background"], "opaque_white")

            for name in plots.SVG_OUTPUTS:
                text = (outputs_dir / name).read_text(encoding="utf-8")
                self.assertIn("<svg", text)
                self.assertIn("#ffffff", text.lower())
                metadata = manifest["artifacts"][name]["metadata"]
                self.assertEqual(metadata["background"], "opaque_white")
                self.assertGreater(metadata["text_element_count"], 0)
                self.assertEqual(metadata["embedded_raster_count"], 0)

            notes = (outputs_dir / plots.NOTES_NAME).read_text(encoding="utf-8")
            self.assertIn("初始参考位置", notes)
            self.assertIn("不代表失效瞬时位置", notes)
            self.assertIn("未绘制误差条", notes)

    @unittest.skipUnless(sys.platform == "win32", "Windows publication regression")
    def test_generation_does_not_require_windows_hardlink_privilege(self) -> None:
        with self.temporary_root("r05-viz-no-hardlink-") as temp:
            root = Path(temp)
            inputs_dir = root / "inputs"
            outputs_dir = root / "outputs"
            inputs_dir.mkdir()
            events_path, summary_path, design_path = _fixture_inputs(inputs_dir)

            with mock.patch.object(
                plots.os,
                "link",
                side_effect=PermissionError(5, "hard links are not permitted"),
            ):
                manifest = plots.generate_visualizations(
                    events_path=events_path,
                    summary_path=summary_path,
                    design_path=design_path,
                    output_dir=outputs_dir,
                    replace=False,
                )

            self.assertEqual(manifest["schema_version"], plots.MANIFEST_SCHEMA_VERSION)
            self.assertTrue(all((outputs_dir / name).is_file() for name in plots.OWNED_OUTPUTS))

    @unittest.skipUnless(sys.platform == "win32", "Windows publication regression")
    def test_mid_publish_failure_removes_partial_new_generation(self) -> None:
        with self.temporary_root("r05-viz-publish-rollback-") as temp:
            root = Path(temp)
            inputs_dir = root / "inputs"
            outputs_dir = root / "outputs"
            inputs_dir.mkdir()
            events_path, summary_path, design_path = _fixture_inputs(inputs_dir)
            real_rename = plots.os.rename
            rename_count = 0

            def fail_third_rename(source: Path, destination: Path) -> None:
                nonlocal rename_count
                rename_count += 1
                if rename_count == 3:
                    raise OSError(5, "injected publish failure")
                real_rename(source, destination)

            with mock.patch.object(plots.os, "rename", side_effect=fail_third_rename):
                with self.assertRaisesRegex(OSError, "injected publish failure"):
                    plots.generate_visualizations(
                        events_path=events_path,
                        summary_path=summary_path,
                        design_path=design_path,
                        output_dir=outputs_dir,
                        replace=False,
                    )

            self.assertFalse(
                [name for name in plots.OWNED_OUTPUTS if (outputs_dir / name).exists()]
            )
            self.assertFalse(list(outputs_dir.glob(".r05-viz-stage-*")))

    def test_replace_failure_restores_the_complete_previous_generation(self) -> None:
        with self.temporary_root("r05-viz-replace-rollback-") as temp:
            root = Path(temp)
            inputs_dir = root / "inputs"
            outputs_dir = root / "outputs"
            inputs_dir.mkdir()
            outputs_dir.mkdir()
            events_path, summary_path, design_path = _fixture_inputs(inputs_dir)
            previous = {
                name: f"previous::{name}".encode("utf-8") for name in plots.OWNED_OUTPUTS
            }
            for name, payload in previous.items():
                (outputs_dir / name).write_bytes(payload)
            real_replace = plots.os.replace
            replace_count = 0

            def fail_third_replace(source: Path, destination: Path) -> None:
                nonlocal replace_count
                replace_count += 1
                if replace_count == 3:
                    raise OSError(5, "injected replace failure")
                real_replace(source, destination)

            with mock.patch.object(plots.os, "replace", side_effect=fail_third_replace):
                with self.assertRaisesRegex(OSError, "injected replace failure"):
                    plots.generate_visualizations(
                        events_path=events_path,
                        summary_path=summary_path,
                        design_path=design_path,
                        output_dir=outputs_dir,
                        replace=True,
                    )

            self.assertEqual(
                {name: (outputs_dir / name).read_bytes() for name in plots.OWNED_OUTPUTS},
                previous,
            )
            self.assertFalse(list(outputs_dir.glob(".r05-viz-stage-*")))

    def test_refuses_implicit_overwrite_before_writing_any_output(self) -> None:
        with self.temporary_root("r05-viz-overwrite-") as temp:
            root = Path(temp)
            inputs_dir = root / "inputs"
            outputs_dir = root / "outputs"
            inputs_dir.mkdir()
            outputs_dir.mkdir()
            events_path, summary_path, design_path = _fixture_inputs(inputs_dir)
            sentinel = outputs_dir / plots.PNG_OUTPUTS[0]
            sentinel.write_bytes(b"sentinel")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                plots.generate_visualizations(
                    events_path=events_path,
                    summary_path=summary_path,
                    design_path=design_path,
                    output_dir=outputs_dir,
                    replace=False,
                )
            self.assertEqual(sentinel.read_bytes(), b"sentinel")
            self.assertEqual(
                sorted(path.name for path in outputs_dir.iterdir()), [sentinel.name]
            )

    def test_real_stage_b_tables_have_expected_bound_counts(self) -> None:
        analysis = PROJECT / "analysis" / "72252362_r05_stage_ab_20261023"
        inputs = plots.load_inputs(
            analysis / "failure_events_unique.csv",
            analysis / "failure_summary_by_case.csv",
            PROJECT / "05" / "p0_cases_r05.json",
        )
        tables = plots.build_source_tables(inputs)
        self.assertEqual(len(tables["panel_a"]), 28)
        self.assertEqual(sum(row["count"] for row in tables["panel_a"]), 2248)
        self.assertEqual(len(tables["panel_b"]), 847)
        self.assertEqual(len(tables["panel_c"]), 7)
        self.assertEqual(
            sum(row["gate_status"] == "above_frozen_gate" for row in tables["panel_c"]),
            6,
        )


if __name__ == "__main__":
    unittest.main()
