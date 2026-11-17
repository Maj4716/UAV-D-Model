from __future__ import annotations

import pytest

from tools.benchmark_online_statistics import (
    _distribution_summary,
    _source_identity_drift,
    execution_order,
)


def test_benchmark_summary_reports_median_and_iqr():
    summary = _distribution_summary([9.0, 1.0, 5.0, 3.0, 7.0])

    assert summary["median"] == pytest.approx(5.0)
    assert summary["q1"] == pytest.approx(3.0)
    assert summary["q3"] == pytest.approx(7.0)
    assert summary["iqr"] == pytest.approx(4.0)


def test_benchmark_alternates_execution_order():
    assert execution_order(0) == ("online", "raw_trace")
    assert execution_order(1) == ("raw_trace", "online")
    assert execution_order(2) == ("online", "raw_trace")


def test_benchmark_source_identity_reports_tool_or_input_drift():
    start = {
        "engine_code_hash": "engine-a",
        "benchmark_tool_hash": "tool-a",
        "target_hash": "target-a",
    }
    end = {**start, "benchmark_tool_hash": "tool-b", "target_hash": "target-b"}

    assert _source_identity_drift(start, start) == []
    assert _source_identity_drift(start, end) == [
        "benchmark_tool_hash",
        "target_hash",
    ]
