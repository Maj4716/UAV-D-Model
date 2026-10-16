import pytest

from damage_engine.errors import InputValidationError
from damage_engine.evaluation_planning import plan_reference_soft_event_mass


def test_soft_event_mass_plan_reproduces_f1_engineering_starting_points():
    report = plan_reference_soft_event_mass(0.004, target_soft_event_masses=(50, 100))
    rows = report["soft_event_mass_plans"]
    assert rows[0]["iid_case_count"] == 12500
    assert rows[1]["iid_case_count"] == 25000
    assert "pilot uncertainty" in report["evidence_boundary"]


def test_soft_event_mass_plan_inflates_for_predeclared_effective_fraction():
    report = plan_reference_soft_event_mass(
        0.005, target_soft_event_masses=(50,), effective_sample_fraction=0.5
    )
    row = report["soft_event_mass_plans"][0]
    assert row["iid_case_count"] == 10000
    assert row["design_adjusted_case_count"] == 20000


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan")])
def test_soft_event_mass_plan_rejects_invalid_mean(value):
    with pytest.raises(InputValidationError):
        plan_reference_soft_event_mass(value)
