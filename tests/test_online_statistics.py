from __future__ import annotations

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.online_statistics import OnlineStatisticsStore
from damage_engine.utils import statistics_dict


def test_online_statistics_match_batch_contract():
    values = [0.0, 0.2, 0.9, 1.0, 0.4]
    store = OnlineStatisticsStore()
    for value in values:
        store.observe_replicate({"nested": {"series": value}})

    observed = store.statistics(("nested", "series"), probability=True)
    expected = statistics_dict(values, probability=True)
    assert observed["sample_count"] == expected["sample_count"]
    assert observed["mean"] == pytest.approx(expected["mean"])
    assert observed["stddev"] == pytest.approx(expected["stddev"])
    assert observed["ci95_lower"] == pytest.approx(expected["ci95_lower"])
    assert observed["ci95_upper"] == pytest.approx(expected["ci95_upper"])


def test_online_statistics_reject_changed_replicate_shape():
    store = OnlineStatisticsStore()
    store.observe_replicate({"a": 1.0})
    with pytest.raises(InputValidationError, match="paths changed"):
        store.observe_replicate({"b": 2.0})


def test_online_statistics_exposes_deterministic_child_keys():
    store = OnlineStatisticsStore()
    store.observe_replicate({"metrics": {"z": 1.0, "a": 2.0}})
    assert store.child_keys(("metrics",)) == ("a", "z")
