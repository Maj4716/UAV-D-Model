from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from assess_r05_response import assess_r05_response, strict_json_loads  # noqa: E402


FROZEN_DESIGN_CANONICAL_SHA256 = (
    "sha256:85b35f1a23ac0b822d0ce65fd9232327c9e566e5cfdddb080814e01414ebdf97"
)
FROZEN_DESIGN_SHA256 = (
    "sha256:b1d2027d57fc6d9cd1ee3f932ea044fd16a112344d8f2b6f508071f45191a06f"
)
ZERO_SHA256 = "sha256:" + "0" * 64
ONE_SHA256 = "sha256:" + "1" * 64
FROZEN_DECK_SHA256 = (
    "sha256:6fbd753a113d0214e7ba4ff5136dc526d151e6f2c52c94e70399b29c4b9d3d0c"
)
FROZEN_CASE_MAP_SHA256 = (
    "sha256:b05b59249e49c3e9d15904f475c2f9478bdff1638fb797d1b552298490aa5295"
)
FROZEN_PACKAGE_MANIFEST_SHA256 = (
    "sha256:e178b6c28832519aaa2befdc969dcd93d2141ea4073ad740b52b17968d92ba44"
)


def passing_records(design: dict) -> list[dict]:
    classification = {
        600.0: "confirmed_nonperforation",
        800.0: "confirmed_nonperforation",
        1000.0: "confirmed_nonperforation",
        1200.0: "confirmed_perforation",
        2000.0: "confirmed_perforation",
    }
    records = []
    for case in design["cases"]:
        is_perforated = classification[float(case["impact_speed_mps"])] == "confirmed_perforation"
        records.append(
            {
                "ledger_schema_version": "hf-p0-capture-ledger-result@1.1.0",
                "case_id": case["case_id"],
                "impact_speed_mps": case["impact_speed_mps"],
                "mesh_mm": case["mesh_mm"],
                "primary_qoi_available": True,
                "unavailable_reasons": [],
                "run_integrity_gate": "pass",
                "classification": classification[float(case["impact_speed_mps"])],
                "qM": 0.20 if is_perforated else 0.0,
                "qP": 0.15 if is_perforated else 0.0,
                "qE": 0.12 if is_perforated else 0.0,
                "qV_over_v0": 0.75 if is_perforated else None,
                "initial_discrete_mass_relative_error": 0.0,
                "mass_accounting_relative_error": 0.0,
                "mass_fraction_by_ledger_state": {
                    "alive_crossed": 0.20 if is_perforated else 0.0,
                    "alive_upstream": 0.80 if is_perforated else 1.0,
                    "deleted_upstream": 0.0,
                    "deleted_after_crossing": 0.0,
                    "temporal_ambiguity": 0.0,
                },
                "missing_reference_mass_fraction": 0.0,
                "unresolved_crossing_mass_fraction": 0.0,
                "worst_case_crossing_mass_fraction": 0.20 if is_perforated else 0.0,
                "tail_window": {
                    "increment_qM": 0.0,
                    "increment_qP": 0.0,
                    "increment_qE": 0.0,
                    "qV_relative_range": 0.0,
                },
            }
        )
    return records


def passing_payload(design: dict) -> dict:
    return {
        "schema_version": "hf-p0-r05-case-results@1.0.0",
        "artifact_identity": {
            "package_revision": "R05_capture_ledger_preregistered",
            "design_sha256": FROZEN_DESIGN_SHA256,
            "design_canonical_sha256": FROZEN_DESIGN_CANONICAL_SHA256,
            "deck_sha256": FROZEN_DECK_SHA256,
            "case_map_sha256": FROZEN_CASE_MAP_SHA256,
            "package_manifest_sha256": FROZEN_PACKAGE_MANIFEST_SHA256,
        },
        "run_identity": {
            "run_id": "test-run-r05",
            "solver_build": "LS-DYNA R15 test fixture",
            "returned_artifact_sha256": ZERO_SHA256,
        },
        "parser_identity": {
            "name": "r05-d3part-capture-ledger",
            "version": "1.1.0-test",
            "source_sha256": ONE_SHA256,
            "acceptance_status": "pass",
        },
        "cases": passing_records(design),
    }


class R05ResponseGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = json.loads((ROOT / "design" / "p0_cases_r05.json").read_text(encoding="utf-8"))

    def test_preregistered_passing_shape_is_allowed(self) -> None:
        result = assess_r05_response(self.design, passing_payload(self.design))
        self.assertEqual(result["overall_status"], "pass")
        self.assertEqual(result["p0_response_calibration_admission"], "allowed")
        self.assertEqual(result["gates"]["speed_scan"]["nearest_bracket_width_mps"], 200.0)
        self.assertEqual(result["bound_identity"]["run_identity"]["run_id"], "test-run-r05")

    def test_reported_classification_is_recomputed_and_mismatch_fails_closed(self) -> None:
        payload = passing_payload(self.design)
        payload["cases"][0]["classification"] = "confirmed_perforation"
        result = assess_r05_response(self.design, payload)
        self.assertEqual(result["overall_status"], "fail_closed")
        self.assertEqual(result["gates"]["per_case_contract"]["status"], "fail")
        self.assertIn(
            "reported classification does not match recomputed classification",
            " ".join(result["gates"]["per_case_contract"]["reasons"]),
        )

    def test_missing_case_fails_closed(self) -> None:
        payload = passing_payload(self.design)
        payload["cases"] = payload["cases"][:-1]
        result = assess_r05_response(self.design, payload)
        self.assertEqual(result["overall_status"], "fail_closed")
        self.assertEqual(result["gates"]["case_identity"]["status"], "fail")
        self.assertEqual(result["gates"]["mesh_scan_2000_mps"]["status"], "fail")

    def test_transition_is_not_silently_bridged_for_adjacent_qv(self) -> None:
        payload = passing_payload(self.design)
        by_id = {record["case_id"]: record for record in payload["cases"]}
        by_id["p0_v1200_h0p75"]["classification"] = "transition_or_indeterminate"
        by_id["p0_v1200_h0p75"]["qM"] = 0.01
        by_id["p0_v1200_h0p75"]["qP"] = 0.008
        by_id["p0_v1200_h0p75"]["qE"] = 0.01
        by_id["p0_v1200_h0p75"]["qV_over_v0"] = None
        by_id["p0_v1200_h0p75"]["mass_fraction_by_ledger_state"]["alive_crossed"] = 0.01
        by_id["p0_v1200_h0p75"]["mass_fraction_by_ledger_state"]["alive_upstream"] = 0.99
        by_id["p0_v1200_h0p75"]["worst_case_crossing_mass_fraction"] = 0.01
        result = assess_r05_response(self.design, payload)
        checks = result["gates"]["speed_scan"]["adjacent_confirmed_perforation_qV_checks"]
        self.assertFalse(any(check["lower_case_id"] == "p0_v1200_h0p75" for check in checks))
        self.assertEqual(result["gates"]["speed_scan"]["status"], "fail")
        self.assertIsNone(result["gates"]["speed_scan"]["nearest_bracket_width_mps"])

    def test_adjacent_perforation_qv_downward_jump_fails(self) -> None:
        payload = passing_payload(self.design)
        by_id = {record["case_id"]: record for record in payload["cases"]}
        by_id["p0_v1200_h0p75"]["qV_over_v0"] = 0.90
        by_id["p0_v1200_h0p75"]["qP"] = 0.18
        by_id["p0_v2000_h0p75"]["qV_over_v0"] = 0.40
        by_id["p0_v2000_h0p75"]["qP"] = 0.08
        result = assess_r05_response(self.design, payload)
        self.assertEqual(result["gates"]["speed_scan"]["status"], "fail")
        checks = result["gates"]["speed_scan"]["adjacent_confirmed_perforation_qV_checks"]
        self.assertEqual(checks[-1]["downward_jump_mps"], 280.0)

    def test_mesh_metric_drift_fails(self) -> None:
        payload = passing_payload(self.design)
        changed = copy.deepcopy(payload)
        by_id = {record["case_id"]: record for record in changed["cases"]}
        by_id["p0_v2000_h1p00"]["qM"] = 0.50
        by_id["p0_v2000_h1p00"]["qV_over_v0"] = 0.30
        by_id["p0_v2000_h1p00"]["mass_fraction_by_ledger_state"]["alive_crossed"] = 0.50
        by_id["p0_v2000_h1p00"]["mass_fraction_by_ledger_state"]["alive_upstream"] = 0.50
        by_id["p0_v2000_h1p00"]["worst_case_crossing_mass_fraction"] = 0.50
        result = assess_r05_response(self.design, changed)
        self.assertEqual(result["gates"]["mesh_scan_2000_mps"]["status"], "fail")
        self.assertEqual(
            result["gates"]["mesh_scan_2000_mps"]["metric_checks"]["qM"]["status"],
            "fail",
        )

    def test_missing_run_integrity_fails_closed(self) -> None:
        payload = passing_payload(self.design)
        payload["cases"][0].pop("run_integrity_gate")
        result = assess_r05_response(self.design, payload)
        self.assertEqual(result["gates"]["all_case_run_integrity"]["status"], "fail")
        self.assertEqual(result["p0_response_calibration_admission"], "denied")

    def test_missing_nan_and_non_object_case_records_fail_closed_without_exception(self) -> None:
        mutations = []

        missing = passing_payload(self.design)
        missing["cases"][0].pop("qM")
        mutations.append(missing)

        nonfinite = passing_payload(self.design)
        nonfinite["cases"][0]["qM"] = float("nan")
        mutations.append(nonfinite)

        non_object = passing_payload(self.design)
        non_object["cases"][0] = "not-an-object"
        mutations.append(non_object)

        for payload in mutations:
            with self.subTest(payload=payload):
                result = assess_r05_response(self.design, payload)
                self.assertEqual(result["overall_status"], "fail_closed")
                self.assertEqual(result["p0_response_calibration_admission"], "denied")

    def test_strict_json_rejects_duplicate_keys_and_every_nonfinite_spelling(self) -> None:
        invalid_documents = (
            '{"case_id":"first","case_id":"second"}',
            '{"qM":NaN}',
            '{"qM":Infinity}',
            '{"qM":-Infinity}',
            '{"qM":1e999}',
        )
        for document in invalid_documents:
            with self.subTest(document=document), self.assertRaises(ValueError):
                strict_json_loads(document)

    def test_mutated_design_is_denied_even_if_payload_uses_looser_threshold(self) -> None:
        changed_design = copy.deepcopy(self.design)
        changed_design["qualification_gates"]["speed_scan"][
            "maximum_nearest_bracket_width_mps"
        ] = 1000.0
        result = assess_r05_response(changed_design, passing_payload(self.design))
        self.assertEqual(result["gates"]["frozen_design_contract"]["status"], "fail")
        self.assertEqual(result["overall_status"], "fail_closed")

    def test_top_level_identity_mismatch_and_case_local_identity_are_denied(self) -> None:
        payload = passing_payload(self.design)
        payload["artifact_identity"]["deck_sha256"] = ZERO_SHA256
        payload["cases"][0]["run_identity"] = {"run_id": "different-run"}
        result = assess_r05_response(self.design, payload)
        self.assertEqual(result["gates"]["top_level_identity_binding"]["status"], "fail")
        self.assertEqual(result["gates"]["per_case_contract"]["status"], "fail")
        self.assertEqual(result["overall_status"], "fail_closed")

    def test_duplicate_case_record_is_denied(self) -> None:
        payload = passing_payload(self.design)
        payload["cases"].append(copy.deepcopy(payload["cases"][0]))
        result = assess_r05_response(self.design, payload)
        self.assertEqual(result["gates"]["case_identity"]["status"], "fail")
        self.assertEqual(result["overall_status"], "fail_closed")

    def test_claimed_availability_is_recomputed_from_frozen_diagnostics(self) -> None:
        payload = passing_payload(self.design)
        payload["cases"][0]["tail_window"]["increment_qM"] = 0.02
        result = assess_r05_response(self.design, payload)
        self.assertEqual(result["gates"]["per_case_contract"]["status"], "fail")
        self.assertEqual(
            result["gates"]["all_case_capture_ledger_availability"]["status"],
            "fail",
        )

    def test_mesh_unavailable_classification_is_explicitly_forbidden(self) -> None:
        payload = passing_payload(self.design)
        mesh_ids = {
            "p0_v2000_h1p00",
            "p0_v2000_h0p75",
            "p0_v2000_h0p50",
        }
        for record in payload["cases"]:
            if record["case_id"] not in mesh_ids:
                continue
            record["primary_qoi_available"] = False
            record["unavailable_reasons"] = ["temporal_ambiguity_mass"]
            record["classification"] = "unavailable"
            record["qM"] = None
            record["qP"] = None
            record["qE"] = None
            record["qV_over_v0"] = None
            record["mass_fraction_by_ledger_state"].update(
                {
                    "alive_crossed": 0.0,
                    "alive_upstream": 0.98,
                    "temporal_ambiguity": 0.02,
                }
            )
            record["unresolved_crossing_mass_fraction"] = 0.02
            record["worst_case_crossing_mass_fraction"] = 0.02
        result = assess_r05_response(self.design, payload)
        mesh_gate = result["gates"]["mesh_scan_2000_mps"]
        self.assertEqual(mesh_gate["status"], "fail")
        self.assertIn("mesh-scan classification unavailable is forbidden", mesh_gate["reasons"])

    def test_matching_transition_mesh_is_allowed_when_every_qoi_is_available(self) -> None:
        payload = passing_payload(self.design)
        mesh_ids = {
            "p0_v2000_h1p00",
            "p0_v2000_h0p75",
            "p0_v2000_h0p50",
        }
        for record in payload["cases"]:
            if record["case_id"] not in mesh_ids:
                continue
            record["classification"] = "transition_or_indeterminate"
            record["qM"] = 0.03
            record["qP"] = 0.009
            record["qE"] = 0.02
            record["qV_over_v0"] = 0.30
            record["mass_fraction_by_ledger_state"]["alive_crossed"] = 0.03
            record["mass_fraction_by_ledger_state"]["alive_upstream"] = 0.97
            record["worst_case_crossing_mass_fraction"] = 0.03
        result = assess_r05_response(self.design, payload)
        self.assertEqual(result["gates"]["mesh_scan_2000_mps"]["status"], "pass")
        self.assertEqual(result["overall_status"], "pass")


if __name__ == "__main__":
    unittest.main()
