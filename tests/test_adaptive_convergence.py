from __future__ import annotations

from copy import deepcopy
import json

import pytest

from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError


def _case_for(engine, case_data):
    case = deepcopy(case_data)
    case["target_ref"]["content_hash"] = engine.target_hash
    case["analysis"]["replicates"] = 64
    case["analysis"]["adaptive_ci_half_width"] = None
    return case


def test_physical_ci_can_drive_adaptive_stopping_without_probability_target(
    target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    case = _case_for(engine, case_data)
    case["analysis"]["adaptive_physical_ci"] = {
        "component.blast.effective_peak_pressure_pa": {
            "absolute_half_width": 1.0e12
        }
    }

    result = engine.run(case).to_dict()

    assert result["replicate_count"] == 32
    convergence = result["convergence"]
    assert convergence["mode"] == "adaptive"
    assert convergence["physical_metrics_in_scope"] is True
    assert convergence["probability_target_met"] is None
    assert convergence["physical_target_met"] is True
    assert convergence["target_met"] is True
    metric = convergence["physical_metric_results"][
        "component.blast.effective_peak_pressure_pa"
    ]
    assert metric["unit"] == "Pa"
    assert metric["series_count"] == 1
    json.dumps(result, allow_nan=False)


def test_physical_ci_report_identifies_unmet_series(target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    component_id = engine.compiled_target.components[0].object_id
    replicates = [
        {
            "fragment_metrics": {
                component_id: {"deposited_energy_j": float(index % 2) * 10.0}
            }
        }
        for index in range(32)
    ]
    report = engine._physical_convergence_report(
        replicates,
        {
            "component.fragment.deposited_energy_j": {
                "absolute_half_width": 1.0e-9,
                "relative_half_width": 1.0e-6,
                "relative_floor": 1.0,
            }
        },
    )
    assert report["target_met"] is False
    assert report["unmet_series_count"] == 1
    assert report["worst_series"]["series_id"] == component_id
    assert report["worst_series"]["ci95_half_width"] > report["worst_series"][
        "allowed_half_width"
    ]


def test_physical_ci_rejects_unknown_or_disabled_metrics(
    target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    unknown = _case_for(engine, case_data)
    unknown["analysis"]["adaptive_physical_ci"] = {
        "component.fragment.not_a_metric": {"absolute_half_width": 1.0}
    }
    with pytest.raises(InputValidationError, match="unsupported metric selectors"):
        engine.run(unknown)

    disabled = _case_for(engine, case_data)
    disabled["analysis"]["mechanisms"] = ["blast"]
    disabled["analysis"]["adaptive_physical_ci"] = {
        "component.fragment.deposited_energy_j": {"absolute_half_width": 1.0}
    }
    with pytest.raises(InputValidationError, match="disabled mechanism"):
        engine.run(disabled)


def test_physical_ci_schema_requires_at_least_32_replicates(
    target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    case = _case_for(engine, case_data)
    case["analysis"]["replicates"] = 31
    case["analysis"]["adaptive_physical_ci"] = {
        "summary.total_component_hits": {"absolute_half_width": 1.0}
    }
    with pytest.raises(InputValidationError, match="simulation_case.schema.json"):
        engine.run(case)

