import numpy as np
import pytest

from damage_engine.sensitivity import (
    ParameterInterval,
    analyze_morris,
    analyze_sobol,
    epistemic_corner_set,
    morris_design,
    sobol_saltelli_design,
)


def _intervals():
    return (
        ParameterInterval("x1", 0.0, 1.0),
        ParameterInterval("x2", 1.0, 10.0, "log"),
    )


def test_morris_recovers_linear_elementary_effects():
    design = morris_design(_intervals(), trajectories=8, levels=8, seed=17)
    x = design["samples"]
    result = analyze_morris(design, 3.0 * x[:, 0] - 2.0 * x[:, 1])
    rows = {row["parameter_id"]: row for row in result["parameters"]}
    assert rows["x1"]["mu"][0] == pytest.approx(3.0)
    assert rows["x2"]["mu"][0] == pytest.approx(-2.0)


def test_sobol_ranks_dominant_additive_parameter():
    design = sobol_saltelli_design(_intervals(), base_size=256, seed=19)
    x = design["samples"]
    result = analyze_sobol(design, 10.0 * x[:, 0] + 0.01 * x[:, 1])
    rows = {row["parameter_id"]: row for row in result["parameters"]}
    assert rows["x1"]["total_order"][0] > rows["x2"]["total_order"][0]
    assert rows["x1"]["total_order"][0] > 0.9


def test_corner_set_is_small_and_declares_scenario_semantics():
    result = epistemic_corner_set(
        _intervals(), active_parameter_ids=["x1", "x2"], maximum_corners=4
    )
    assert len(result["corners"]) == 4
    assert result["semantics"] == "scenario_envelope_not_probability_distribution"
