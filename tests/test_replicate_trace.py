import copy

import pytest

from damage_engine.engine import build_replicate_trace
from damage_engine.errors import InputValidationError


def record(index):
    return {
        "parameter_realization_hash": "sha256:" + str(index) * 64,
        "mechanism_states": {"combined": {"component": {"degraded": 0.2 + index * 0.1}}},
        "fragment_metrics": {"component": {"hits": float(index)}},
        "blast_metrics": {"component": {"distance_m": 2.0}},
        "tree": {
            "outputs": {"mission_loss": 0.1 + index * 0.2},
            "node_probabilities": {"component": 0.2 + index * 0.1},
        },
        "summary": {"total_component_hits": float(index)},
    }


def test_trace_preserves_joint_replicates_before_aggregation():
    source = [record(0), record(1)]
    frozen = copy.deepcopy(source)
    trace = build_replicate_trace(source)
    assert trace["replicate_count"] == 2
    assert [row["system_outputs"]["mission_loss"] for row in trace["records"]] == pytest.approx([0.1, 0.3])
    assert [row["fragment_metrics"]["component"]["hits"] for row in trace["records"]] == [0.0, 1.0]
    assert trace["semantics"] == "joint_per_replicate_outputs_before_aggregation"
    assert source == frozen


def test_trace_fails_closed_on_incomplete_replica():
    value = record(0)
    value.pop("parameter_realization_hash")
    with pytest.raises(InputValidationError, match="incomplete"):
        build_replicate_trace([value])
