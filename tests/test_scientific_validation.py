import copy
import json
from pathlib import Path

import pytest

import damage_engine.scientific_validation as scientific_validation
from damage_engine.errors import InputValidationError
from damage_engine.scientific_validation import validate_scientific_validation_record


PROJECT_ROOT = Path(__file__).parents[1]


def _passing_scientific_validation():
    return {
        "schema_version": "1.0.0",
        "independent_data_sources": [{
            "source_id": "withheld", "role": "validation",
            "content_hash": "sha256:" + "1" * 64, "used_for_calibration": False,
        }],
        "dimensional_error_metrics": [{
            "metric_id": "velocity", "mechanism": "fragment", "unit": "m/s",
            "observed_error": 1.0, "acceptance_limit": 2.0, "passed": True,
        }],
        "interval_coverage": [{
            "metric_id": "coverage", "mechanism": "blast", "nominal": 0.95,
            "observed": 0.95, "sample_count": 20, "passed": True,
        }],
        "mechanism_stratified_errors": [{
            "stratum_id": "near", "metric_id": "velocity", "value": 1.0,
            "acceptance_limit": 2.0, "passed": True,
        }],
        "extrapolation_failure_rules": [{
            "rule_id": "outside", "condition": "outside frozen domain", "action": "refuse",
        }],
        "uncertainty_ledger": {
            "parameter_epistemic": "assessed", "model_form": "assessed",
            "numerical_error": "assessed",
        },
        "approval": {
            "approver_id": "reviewer", "approved_at": "2026-10-13T00:00:00Z",
            "protocol_version": "1",
        },
    }


def test_complete_scientific_validation_structure_passes_shape_gate():
    validate_scientific_validation_record(_passing_scientific_validation())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["independent_data_sources"].clear(),
        lambda value: value["dimensional_error_metrics"][0].update(passed=False),
        lambda value: value["interval_coverage"][0].update(sample_count=0),
        lambda value: value["mechanism_stratified_errors"][0].update(value=3.0),
        lambda value: value["extrapolation_failure_rules"][0].update(action="warn"),
        lambda value: value["uncertainty_ledger"].update(model_form="missing"),
    ],
)
def test_incomplete_scientific_validation_fails_closed(mutation):
    record = copy.deepcopy(_passing_scientific_validation())
    mutation(record)
    with pytest.raises(InputValidationError):
        validate_scientific_validation_record(record)


def test_checked_in_nonapproving_template_is_the_only_template_source():
    """The retired code helper must not drift from a second template copy."""

    assert not hasattr(
        scientific_validation, "example_scientific_validation_record"
    )
    template_path = (
        PROJECT_ROOT / "examples" / "scientific_validation_record.template.json"
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert template["_instructions"].startswith(
        "Template only and intentionally non-admissible."
    )
    record = {
        key: value for key, value in template.items() if key != "_instructions"
    }
    with pytest.raises(InputValidationError):
        validate_scientific_validation_record(record)
