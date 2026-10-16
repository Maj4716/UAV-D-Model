from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.fragment_audit import (
    diagnose_front_axis_semantics,
    summarize_component_reachability,
)
from damage_engine.registry import PhysicsRegistry


ROOT = Path(__file__).resolve().parents[1]


def _trace(
    fragment_id: int,
    candidate_ids: list[str],
    processed: list[dict] | None = None,
    *,
    blocker: str | None = None,
) -> dict:
    return {
        "record_type": "fragment_trace",
        "fragment_id": fragment_id,
        "geometry_candidate_path": [
            {"object_id": object_id, "role": "component"}
            for object_id in candidate_ids
        ],
        "processed_path": processed or [],
        "blocking_object_id": blocker,
        "blocking_reason": "test_block" if blocker else None,
    }


def _intersection(
    fragment_id: int,
    *,
    penetrated: bool,
    response_probability: float,
) -> dict:
    return {
        "record_type": "intersection",
        "fragment_id": fragment_id,
        "object_id": "critical-component",
        "role": "component",
        "penetrated": penetrated,
        "damage_state_response_probabilities": {
            "failed": response_probability,
        },
    }


def test_component_reachability_summary_exhausts_five_categories():
    audit = [
        _trace(0, []),
        _trace(
            1,
            ["upstream-armor", "critical-component"],
            [
                {
                    "object_id": "upstream-armor",
                    "role": "armor",
                    "disposition": "not_penetrated_block",
                }
            ],
            blocker="upstream-armor",
        ),
        _trace(
            2,
            ["critical-component"],
            [{"object_id": "critical-component", "role": "component"}],
        ),
        _intersection(2, penetrated=False, response_probability=0.0),
        _trace(
            3,
            ["critical-component"],
            [{"object_id": "critical-component", "role": "component"}],
        ),
        _intersection(3, penetrated=True, response_probability=0.0),
        _trace(
            4,
            ["critical-component"],
            [{"object_id": "critical-component", "role": "component"}],
        ),
        _intersection(4, penetrated=True, response_probability=0.25),
    ]

    report = summarize_component_reachability(audit, ["critical-component"])

    component = report["components"]["critical-component"]
    assert report["fragment_trace_count"] == 5
    assert report["replicate_fragment_counts"] == {"0": 5}
    assert component["counts"] == {
        "no_geometric_intersection": 1,
        "blocked_before_component": 1,
        "hit_no_penetration": 1,
        "penetrated_no_damage": 1,
        "response_possible": 1,
    }
    assert sum(component["fractions"].values()) == pytest.approx(1.0)
    json.dumps(report, allow_nan=False)

    wrapped_report = summarize_component_reachability(
        [{"replicate": 7, "events": audit}],
        ["critical-component"],
    )
    assert wrapped_report["replicate_fragment_counts"] == {"7": 5}
    assert wrapped_report["components"]["critical-component"]["counts"] == component[
        "counts"
    ]


def test_component_reachability_rejects_legacy_hit_only_audit():
    with pytest.raises(InputValidationError, match="no fragment_trace"):
        summarize_component_reachability(
            [_intersection(0, penetrated=True, response_probability=0.25)],
            ["critical-component"],
        )


def test_current_front_axis_semantics_are_explicit_and_observational():
    case = json.loads(
        (ROOT / "raw_data" / "case.loitering_impact.fragment_only.v1.json").read_text(
            encoding="utf-8"
        )
    )
    registry = PhysicsRegistry.from_file(ROOT / "examples" / "physics.loitering_impact.json")
    source_model = registry.fragment_source(
        case["warhead"]["fragment_source"]["model_id"]
    )
    original_case = deepcopy(case)
    original_model = deepcopy(source_model)

    report = diagnose_front_axis_semantics(case, source_model)

    assert report["detonation_point"] == "front"
    assert report["body_forward_dot_velocity"] == pytest.approx(1.0)
    assert report["source_axis_dot_velocity"] == pytest.approx(-1.0)
    assert report["source_axis_dot_body_forward"] == pytest.approx(-1.0)
    assert report["reference_front_backward_rule_satisfied"] is True
    assert report["primary_fragment_axis_opposes_projectile_velocity"] is True
    assert report["diagnosis"] == "front_backward_axis_opposes_velocity"
    assert case == original_case
    assert source_model == original_model
