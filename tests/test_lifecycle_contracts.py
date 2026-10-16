"""Compatibility and experimental symbol lifecycle contracts."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from damage_engine.adaptive_sampling import tempered_prevalence_target
from damage_engine.errors import DamageEngineError, InputValidationError, NumericalError


PACKAGE_ROOT = Path(__file__).parents[1] / "damage_engine"


def test_tempered_prevalence_target_is_bounded_and_not_on_active_package_paths() -> None:
    observed = tempered_prevalence_target([0.0, 0.25, 1.0], gamma=0.5)
    assert observed.dtype == np.dtype("float64")
    assert np.all(observed > 0.0)
    assert np.sum(observed) == pytest.approx(1.0)

    callers: list[str] = []
    for path in PACKAGE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "tempered_prevalence_target"
            ):
                callers.append(path.name)
    assert callers == []


@pytest.mark.parametrize(
    "values, kwargs",
    [([], {}), ([float("nan")], {}), ([-0.1], {}), ([0.1], {"gamma": 0.0})],
)
def test_tempered_prevalence_target_remains_fail_closed(values, kwargs) -> None:
    with pytest.raises(InputValidationError):
        tempered_prevalence_target(values, **kwargs)


def test_numerical_error_keeps_its_reserved_serialized_contract() -> None:
    error = NumericalError("reserved", context={"source": "compatibility-test"})
    assert isinstance(error, DamageEngineError)
    assert error.to_dict() == {
        "code": "REF_NUMERICAL",
        "message": "reserved",
        "context": {"source": "compatibility-test"},
    }
