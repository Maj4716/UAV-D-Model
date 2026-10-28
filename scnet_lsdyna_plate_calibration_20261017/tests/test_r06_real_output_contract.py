from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT / "design" / "p0_r06_real_output_contract.json"


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class R06RealOutputContractTests(unittest.TestCase):
    def test_future_only_contract_freezes_real_output_seams(self) -> None:
        contract = json.loads(
            CONTRACT.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
        self.assertEqual(
            set(contract),
            {
                "schema_version",
                "contract_id",
                "status",
                "version_boundary",
                "d3part_keyword",
                "d3part_population",
                "timeline",
                "crossing_resolution",
                "return_evidence",
                "admission",
                "claim_boundary",
            },
        )
        self.assertEqual(
            contract["schema_version"], "hf-p0-r06-real-output-contract@1.0.0"
        )

        boundary = contract["version_boundary"]
        self.assertEqual(boundary["applies_to_revision_family"], "R06")
        self.assertEqual(boundary["effective_scope"], "future_successor_candidates_only")
        self.assertEqual(boundary["r06_deck_status"], "not_generated")
        self.assertEqual(boundary["r06_submission_status"], "not_submitted")
        self.assertEqual(boundary["historical_r05"]["job_id"], "72252362")
        self.assertEqual(
            boundary["historical_r05"]["evidence_policy"], "read_only_immutable"
        )
        self.assertIs(boundary["historical_r05"]["retroactive_admission"], False)

        keyword = contract["d3part_keyword"]
        self.assertEqual(keyword["keyword"], "*DATABASE_BINARY_D3PART")
        self.assertEqual(keyword["field_count"], 5)
        self.assertEqual(
            [field["name"] for field in keyword["fields"]],
            ["DT", "CYCL", "IOOPT", "LCDT", "PSETID"],
        )
        self.assertEqual(
            [
                field.get("frozen_value_ms", field.get("frozen_value"))
                for field in keyword["fields"]
            ],
            [0.00005, 0, 0, 0, 900001],
        )
        self.assertEqual(keyword["extra_field_policy"], "reject")
        self.assertEqual(
            keyword["serialization"],
            "exactly_five_comma_separated_nonblank_fields_without_trailing_comma",
        )

        population = contract["d3part_population"]
        self.assertEqual(population["admission_identity"], "used_solid_part_ids")
        self.assertEqual(
            population["required_relation"],
            "used_solid_part_ids == expected_fragment_part_ids",
        )
        self.assertEqual(
            population["zero_solid_catalog_part_ids"], "allowed_but_must_be_reported"
        )
        self.assertEqual(population["unexpected_used_solid_part_ids"], "fail_closed")
        self.assertEqual(population["missing_expected_used_solid_part_ids"], "fail_closed")

        timeline = contract["timeline"]
        self.assertEqual(timeline["nominal_output_interval_ms"], 0.00005)
        self.assertEqual(timeline["solver_timeline_source"]["artifact_family"], "binout*")
        self.assertEqual(timeline["solver_timeline_source"]["database"], "glstat")
        self.assertEqual(timeline["solver_timeline_source"]["time_field"], "time")
        self.assertEqual(
            timeline["solver_timeline_source"]["time_step_field"], "time_step"
        )
        self.assertEqual(timeline["interval_rule"]["operator"], "<=")
        self.assertEqual(timeline["interval_rule"]["time_word_tolerance_ulps"], 4)
        self.assertEqual(
            timeline["interval_rule"]["right_hand_side"],
            "nominal output interval plus the maximum positive GLSTAT explicit solver time_step whose solver state lies in the same D3PART interval, plus time-word tolerance",
        )
        self.assertIn("first D3PART time must be zero", timeline["start_rule"])
        self.assertIn("cover every D3PART interval", timeline["solver_evidence_coverage_rule"])
        self.assertIn("reach termination_time_ms", timeline["termination_coverage_rule"])
        self.assertEqual(
            timeline["missing_nonfinite_nonmonotone_policy"], "fail_closed"
        )

        crossing = contract["crossing_resolution"]
        self.assertEqual(
            crossing["maximum_directional_cross_step_travel_mesh_fraction"], 0.5
        )
        self.assertEqual(
            crossing["overlimit_policy"], "route_reference_mass_to_temporal_ambiguity"
        )
        self.assertEqual(
            crossing["alive_to_deleted_possible_crossing_policy"],
            "route_reference_mass_to_temporal_ambiguity",
        )

        returned = contract["return_evidence"]
        self.assertEqual(returned["d3hsp"]["accepted_equivalent"], "solver_o_output")
        self.assertEqual(returned["d3hsp"]["cardinality"], "exactly_one")
        self.assertIn("exact LS-DYNA o= output path", returned["d3hsp"]["command_binding"])
        self.assertIn("hash-bound", returned["d3hsp"]["identity_policy"])
        self.assertEqual(
            returned["binout_logical_database_equivalences"]["logical_databases"],
            ["matsum", "nodout", "sleout"],
        )
        self.assertIs(
            returned["binout_logical_database_equivalences"]["all_required"], True
        )
        self.assertIn(
            "hash-bound",
            returned["binout_logical_database_equivalences"]["identity_policy"],
        )
        self.assertEqual(returned["missing_equivalent_policy"], "fail_closed")

        admission = contract["admission"]
        self.assertEqual(
            admission["p0_response_calibration"],
            "denied_until_all_future_r06_applicable_gates_pass",
        )
        self.assertEqual(
            admission["p1_submission"],
            "denied_until_p0_response_calibration_is_allowed",
        )
        self.assertEqual(
            admission["local_engine_fitting"],
            "denied_until_p0_response_calibration_is_allowed",
        )
        self.assertEqual(
            admission["dataset_generation"],
            "denied_until_p0_response_calibration_is_allowed",
        )
        self.assertEqual(
            admission["historical_r05_job_72252362"], "not_retroactively_admitted"
        )


if __name__ == "__main__":
    unittest.main()
