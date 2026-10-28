from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_p0_r05 import generate_candidate  # noqa: E402
from validate_p0_r05 import validate_candidate, validate_d3part_contract  # noqa: E402


DESIGN = PROJECT / "design" / "p0_cases_r05.json"
MATERIAL = PROJECT / "materials" / "jc4340_reference.json"
P1 = PROJECT / "design" / "p1_preregistered_cases.csv"
CANDIDATE = PROJECT / "manual_upload_20261021_r05_candidate"
R06_REAL_OUTPUT_CONTRACT = PROJECT / "design" / "p0_r06_real_output_contract.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R05PackageTests(unittest.TestCase):
    @staticmethod
    def _write_design_and_refresh_manifest(package: Path, design: dict) -> None:
        design_path = package / "p0_cases_r05.json"
        design_path.write_text(
            json.dumps(design, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_path = package / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        design_hash = "sha256:" + digest(design_path)
        manifest["design"]["sha256"] = design_hash
        manifest["design_hash"] = design_hash
        manifest["measurement_contract"] = design["measurement_contract"]
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_initial_mass_gate_tamper_is_rejected(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r05-gate-tamper-", dir=temp_root) as temp:
            temp_dir = Path(temp)
            design = json.loads(DESIGN.read_text(encoding="utf-8"))
            design["qualification_gates"]["capture_ledger"][
                "initial_discrete_mass_relative_error_max"
            ] = 0.006
            design_path = temp_dir / "p0_cases_r05.json"
            design_path.write_text(
                json.dumps(design, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            package = temp_dir / "candidate"
            generate_candidate(
                design_path=design_path,
                material_path=MATERIAL,
                p1_design_path=P1,
                output_dir=package,
            )
            with self.assertRaisesRegex(
                AssertionError, "initial_discrete_mass_relative_error_max"
            ):
                validate_candidate(package)

    def test_each_frozen_gate_family_rejects_semantic_tamper(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r05-gate-families-", dir=temp_root) as temp:
            package = Path(temp) / "candidate"
            generate_candidate(
                design_path=DESIGN,
                material_path=MATERIAL,
                p1_design_path=P1,
                output_dir=package,
            )
            frozen = json.loads(DESIGN.read_text(encoding="utf-8"))
            mutations = (
                (
                    "run_integrity",
                    ("qualification_gates", "run_integrity", "total_energy_ratio_min"),
                    0.90,
                ),
                (
                    "tail_stability",
                    ("qualification_gates", "capture_ledger", "tail_increment_qE_max"),
                    0.02,
                ),
                (
                    "speed_bracket",
                    ("qualification_gates", "speed_scan", "maximum_nearest_bracket_width_mps"),
                    400.0,
                ),
                (
                    "speed_monotonicity",
                    ("qualification_gates", "speed_scan", "forbid_perforation_to_nonperforation_reversal"),
                    False,
                ),
                (
                    "speed_downward_jump",
                    ("qualification_gates", "speed_scan", "confirmed_perforation_qV_downward_jump_mps_max"),
                    200.0,
                ),
                (
                    "admission",
                    ("qualification_gates", "admission", "p1_submission"),
                    "allowed",
                ),
            )
            for name, path, replacement in mutations:
                with self.subTest(gate=name):
                    design = deepcopy(frozen)
                    target = design
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = replacement
                    self._write_design_and_refresh_manifest(package, design)
                    with self.assertRaisesRegex(AssertionError, "frozen qualification gate"):
                        validate_candidate(package)
            self._write_design_and_refresh_manifest(package, frozen)

    def test_expected_mass_measurement_contract_rejects_semantic_tamper(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r05-mass-contract-", dir=temp_root) as temp:
            package = Path(temp) / "candidate"
            generate_candidate(
                design_path=DESIGN,
                material_path=MATERIAL,
                p1_design_path=P1,
                output_dir=package,
            )
            frozen = json.loads(DESIGN.read_text(encoding="utf-8"))
            mutations = (
                (
                    ("measurement_contract", "expected_reference_mass_source"),
                    "D3PART observed mass",
                ),
                (
                    ("measurement_contract", "runtime_completeness_policy"),
                    "missing elements are ignored",
                ),
                (
                    ("measurement_contract", "primary_qois", 0, "definition"),
                    "first-crossing mass divided by observed D3PART mass",
                ),
                (
                    ("measurement_contract", "primary_qois", 1, "definition"),
                    "signed momentum divided by observed mass",
                ),
                (
                    (
                        "qualification_gates",
                        "numerical_perforation_classification",
                        "confirmed_nonperforation_worst_case_crossing_mass_definition",
                    ),
                    "qM + temporal_ambiguity_mass_fraction + deleted_upstream_mass_fraction",
                ),
            )
            for path, replacement in mutations:
                with self.subTest(path=path):
                    design = deepcopy(frozen)
                    target = design
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = replacement
                    self._write_design_and_refresh_manifest(package, design)
                    with self.assertRaisesRegex(
                        AssertionError, "frozen (measurement contract|qualification gate)"
                    ):
                        validate_candidate(package)
            self._write_design_and_refresh_manifest(package, frozen)

    def test_self_consistent_material_tuning_is_rejected_by_r04_baseline(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r05-material-baseline-", dir=temp_root) as temp:
            temp_dir = Path(temp)
            material = json.loads(MATERIAL.read_text(encoding="utf-8"))
            material["strength_model"]["A_gpa"] = 0.800
            material_path = temp_dir / "jc4340_reference.json"
            material_path.write_text(
                json.dumps(material, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            package = temp_dir / "candidate"
            generate_candidate(
                design_path=DESIGN,
                material_path=material_path,
                p1_design_path=P1,
                output_dir=package,
            )
            with self.assertRaisesRegex(AssertionError, "frozen R04 material baseline"):
                validate_candidate(package)

    def test_timestep_card_tuning_is_rejected_even_with_refreshed_hash(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r05-card-baseline-", dir=temp_root) as temp:
            package = Path(temp) / "candidate"
            generate_candidate(
                design_path=DESIGN,
                material_path=MATERIAL,
                p1_design_path=P1,
                output_dir=package,
            )
            deck = package / "p0_velocity_mesh_calibration.k"
            text = deck.read_text(encoding="ascii")
            self.assertIn("0.000000000e+00,8.000000000e-01,0", text)
            deck.write_text(
                text.replace(
                    "0.000000000e+00,8.000000000e-01,0",
                    "0.000000000e+00,9.000000000e-01,0",
                    1,
                ),
                encoding="ascii",
                newline="\n",
            )
            manifest_path = package / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["deck"]["sha256"] = "sha256:" + digest(deck)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AssertionError, "frozen R04 physics deck"):
                validate_candidate(package)

    def test_frozen_design_and_generated_candidate_validate(self) -> None:
        design = json.loads(DESIGN.read_text(encoding="utf-8"))
        self.assertEqual(design["package_revision"], "R05_capture_ledger_preregistered")
        self.assertEqual(
            [case["impact_speed_mps"] for case in design["cases"][:5]],
            [600.0, 800.0, 1000.0, 1200.0, 2000.0],
        )
        self.assertEqual(
            design["measurement_contract"]["fragment_d3part_interval_ms"], 0.00005
        )
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r15-candidate-", dir=temp_root) as temp:
            package = Path(temp) / "candidate"
            generate_candidate(
                design_path=DESIGN,
                material_path=MATERIAL,
                p1_design_path=P1,
                output_dir=package,
            )
            report = validate_candidate(package)
            self.assertEqual(report["overall_status"], "pass")
            self.assertEqual(
                report["checks"]["measurement_contract"]["primary_qois"],
                ["qM", "qP", "qE", "qV_over_v0"],
            )
            self.assertEqual(report["checks"]["d3part_contract"]["card_1_field_count"], 5)

    def test_generated_d3part_card_uses_r15_five_field_contract(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r15-d3part-card-", dir=temp_root) as temp:
            package = Path(temp) / "candidate"
            generate_candidate(
                design_path=DESIGN,
                material_path=MATERIAL,
                p1_design_path=P1,
                output_dir=package,
            )
            lines = (package / "p0_velocity_mesh_calibration.k").read_text(
                encoding="ascii"
            ).splitlines()
            keyword_index = lines.index("*DATABASE_BINARY_D3PART")
            self.assertEqual(
                lines[keyword_index + 1],
                "5.000000000e-05,0,0,0,900001",
            )

    def test_generator_manifest_wording_uses_supplied_design_revision(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="successor-wording-", dir=temp_root) as temp:
            root = Path(temp)
            design = json.loads(DESIGN.read_text(encoding="utf-8"))
            design["package_revision"] = "R06_successor_fixture"
            design_path = root / "p0_cases_r06_fixture.json"
            design_path.write_text(
                json.dumps(design, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest = generate_candidate(
                design_path=design_path,
                material_path=MATERIAL,
                p1_design_path=P1,
                output_dir=root / "candidate",
            )
            limitations = "\n".join(manifest["limitations"])
            self.assertEqual(manifest["package_revision"], "R06_successor_fixture")
            self.assertIn("R06_successor_fixture", limitations)
            self.assertNotIn("The R05 keyword deck", limitations)
            self.assertNotIn("R05 gates", limitations)

    def test_validator_report_marks_live_contract_as_successor_only(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="successor-boundary-", dir=temp_root) as temp:
            package = Path(temp) / "candidate"
            generate_candidate(
                design_path=DESIGN,
                material_path=MATERIAL,
                p1_design_path=P1,
                output_dir=package,
            )
            report = validate_candidate(package)
            self.assertEqual(report["schema_version"], "hf-static-validation@1.3.0")
            boundary = report["version_boundary"]
            self.assertEqual(
                boundary["live_contract_scope"],
                "future_r06_successor_candidate_only",
            )
            self.assertEqual(
                boundary["successor_contract"]["path"],
                "design/p0_r06_real_output_contract.json",
            )
            self.assertEqual(
                boundary["successor_contract"]["sha256"],
                "sha256:" + digest(R06_REAL_OUTPUT_CONTRACT),
            )
            self.assertEqual(
                boundary["historical_r05"]["job_id"], "72252362"
            )
            self.assertEqual(
                boundary["historical_r05"]["evidence_status"],
                "read_only_immutable",
            )
            self.assertIs(
                boundary["historical_r05"]["retroactive_admission"], False
            )

    def test_r04_historical_inputs_are_unchanged(self) -> None:
        expected = {
            "p0_velocity_mesh_calibration.k": "312f2ce1c42dd070097dad1036d1d670790704c55c5b534a3b845a0709709e08",
            "p0_case_part_map.csv": "7ef72a22eec38691a8662e0efff4d0e8ca5aeaa5757720ac8958ce2dfd1e453a",
            "manifest.json": "8e767187759cc47b4dd837dd3b9b6908abc85bc7a986bd6fcaaddee30e937220",
            "static_validation.json": "11d0facf716df33373cbd39e0cd3d4072704e3bc2f6c0b9e0cd952aeb4feacc1",
        }
        r04 = PROJECT / "manual_upload_20261021_r04"
        for name, expected_hash in expected.items():
            self.assertEqual(digest(r04 / name), expected_hash, name)
        self.assertEqual(
            digest(PROJECT / "design" / "p0_cases.json"),
            "805916aa97986bb7fbe29d6a2a43d039ac18f6a587cd56d26e37bceb0bc04348",
        )

    def test_r05_historical_candidate_is_unchanged(self) -> None:
        expected = {
            "p0_velocity_mesh_calibration.k": "6fbd753a113d0214e7ba4ff5136dc526d151e6f2c52c94e70399b29c4b9d3d0c",
            "p0_cases_r05.json": "b1d2027d57fc6d9cd1ee3f932ea044fd16a112344d8f2b6f508071f45191a06f",
            "p0_case_part_map.csv": "b05b59249e49c3e9d15904f475c2f9478bdff1638fb797d1b552298490aa5295",
            "manifest.json": "e178b6c28832519aaa2befdc969dcd93d2141ea4073ad740b52b17968d92ba44",
            "static_validation.json": "491753050734ba46ded5460221e35a70e37bbccd179b6b172d753bfc18c8b17e",
            "UPLOAD_README.txt": "2e873f731308fa2027b06184c658a31fe31dabe3d084ac0abe7c0159083deb8d",
            "jc4340_reference.json": "9b462b920459e57641d5f89d10d36c0aa0498c86f2478fef8e4c3efa96888311",
            "p1_preregistered_cases.csv": "22262e09c1d1bddd4da19eaecc9439a2f9ce81af4245495aeb61e854436b322e",
        }
        for name, expected_hash in expected.items():
            self.assertEqual(digest(CANDIDATE / name), expected_hash, name)

    def test_generation_is_deterministic_and_refuses_overwrite(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r05-determinism-", dir=temp_root) as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            kwargs = {
                "design_path": DESIGN,
                "material_path": MATERIAL,
                "p1_design_path": P1,
            }
            generate_candidate(output_dir=first, **kwargs)
            generate_candidate(output_dir=second, **kwargs)
            first_files = sorted(path.name for path in first.iterdir())
            second_files = sorted(path.name for path in second.iterdir())
            self.assertEqual(first_files, second_files)
            for name in first_files:
                self.assertEqual(digest(first / name), digest(second / name), name)
            with self.assertRaises(FileExistsError):
                generate_candidate(output_dir=first, **kwargs)

    def test_case_map_semantic_tamper_is_rejected(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r05-tamper-", dir=temp_root) as temp:
            tamper_cases = (
                ("role", "hf_response_fit", "hf_response_holdout", "ID/role"),
                ("speed", "600.0", "601.0", "speed differs"),
                ("mesh", ",0.75,", ",0.76,", "mesh differs"),
            )
            for name, old, new, expected_error in tamper_cases:
                with self.subTest(field=name):
                    package = Path(temp) / name
                    generate_candidate(
                        design_path=DESIGN,
                        material_path=MATERIAL,
                        p1_design_path=P1,
                        output_dir=package,
                    )
                    case_map = package / "p0_case_part_map.csv"
                    text = case_map.read_text(encoding="utf-8")
                    self.assertIn(old, text)
                    case_map.write_text(text.replace(old, new, 1), encoding="utf-8")
                    with self.assertRaisesRegex(AssertionError, expected_error):
                        validate_candidate(package)

    def test_d3part_interval_tamper_is_rejected(self) -> None:
        temp_root = PROJECT / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="r05-d3part-", dir=temp_root) as temp:
            package = Path(temp) / "candidate"
            generate_candidate(
                design_path=DESIGN,
                material_path=MATERIAL,
                p1_design_path=P1,
                output_dir=package,
            )
            deck = package / "p0_velocity_mesh_calibration.k"
            text = deck.read_text(encoding="ascii")
            deck.write_text(
                text.replace("5.000000000e-05,0,0,0,900001", "1.000000000e-04,0,0,0,900001", 1),
                encoding="ascii",
                newline="\n",
            )
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            manifest["deck"]["sha256"] = "sha256:" + digest(deck)
            (package / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(AssertionError, "D3PART interval"):
                validate_candidate(package)

    def test_d3part_card_requires_exact_r15_five_field_contract(self) -> None:
        design = json.loads(DESIGN.read_text(encoding="utf-8"))
        fragment_ids = set(range(2001, 2008))
        compatible = """*KEYWORD LONG=Y
*DATABASE_BINARY_D3PART
5.000000000e-05,0,0,0,900001
*SET_PART_LIST_TITLE
fragments
900001,0.0,0.0,0.0,0.0,MECH,,
2001,2002,2003,2004,2005,2006,2007
*END
"""
        report = validate_d3part_contract(compatible, fragment_ids, design)
        self.assertEqual(report["card_1_field_count"], 5)
        self.assertEqual(
            report["card_1_fields"], ["DT", "CYCL", "IOOPT", "LCDT", "PSETID"]
        )

        legacy_tail = compatible.replace(
            "5.000000000e-05,0,0,0,900001",
            "5.000000000e-05,0,0,0,900001,0,1,",
        )
        with self.assertRaisesRegex(AssertionError, "requires exactly 5 fields"):
            validate_d3part_contract(legacy_tail, fragment_ids, design)

    def test_d3part_part_set_tamper_is_rejected(self) -> None:
        design = json.loads(DESIGN.read_text(encoding="utf-8"))
        fragment_ids = set(range(2001, 2008))
        template = """*KEYWORD LONG=Y
*DATABASE_BINARY_D3PART
5.000000000e-05,0,0,0,900001
*SET_PART_LIST_TITLE
fragments
900001,0.0,0.0,0.0,0.0,MECH,,
{part_ids}
*END
"""
        cases = (
            (
                "wrong_psetid",
                template.replace(",900001", ",900002", 1).format(
                    part_ids="2001,2002,2003,2004,2005,2006,2007"
                ),
                "frozen PSETID",
            ),
            (
                "plate_mixed_in",
                template.format(part_ids="2001,2002,2003,2004,2005,2006,1001"),
                "only and all fragment PIDs",
            ),
            (
                "duplicate_fragment",
                template.format(part_ids="2001,2002,2003,2004,2005,2006,2006"),
                "duplicate PID",
            ),
        )
        for name, deck_text, expected_error in cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(AssertionError, expected_error):
                    validate_d3part_contract(deck_text, fragment_ids, design)


if __name__ == "__main__":
    unittest.main()
