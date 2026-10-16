from __future__ import annotations

from copy import deepcopy

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.sampling_spec import SamplingSpec, latent_dimensions


def _sampling_spec_data() -> dict:
    return {
        "schema_version": "1.0.0",
        "sampling_spec_id": "p0-demo",
        "revision": "1",
        "status": "provisional",
        "reference_distribution": {
            "distribution_id": "p0-reference",
            "interpretation": "design_distribution",
            "variables": [
                {
                    "name": "temperature",
                    "path": "/environment/temperature_k",
                    "distribution": "uniform",
                    "low": 280.0,
                    "high": 300.0,
                },
                {
                    "name": "velocity_direction",
                    "path": "/encounter/velocity_target_mps",
                    "distribution": "unit_sphere_direction",
                },
                {
                    "name": "body_attitude",
                    "path": "/encounter/body_to_target_orientation_xyzw",
                    "distribution": "haar_quaternion",
                },
            ],
            "constraints": [],
        },
        "budgets": {
            "train_pilot": 4,
            "validation": 2,
            "test": 2,
            "rare_stress_test": 0,
        },
        "seeds": {
            "design": 101,
            "selection": 202,
            "split": 303,
            "screening": 404,
            "formal": 505,
        },
        "formal_policy": {
            "replicates": 32,
            "adaptive_ci_half_width": None,
            "epistemic_policy": "unique_per_lineage",
        },
        "split_policy": {
            "registry_id": "split-p0-demo",
            "freeze_before_outcomes": True,
            "adaptive_train_only": True,
        },
        "quality_gates": {
            "positivity_min": 0.1,
            "min_ess_fraction": 0.5,
            "max_weight_cv": 2.0,
        },
    }


def test_sampling_spec_validates_schema_and_exposes_dimensioned_contract() -> None:
    spec = SamplingSpec(_sampling_spec_data())

    assert spec.spec_id == "p0-demo"
    assert spec.distribution_id == "p0-reference"
    assert spec.latent_dimension == 6
    assert spec.budgets == {
        "train_pilot": 4,
        "validation": 2,
        "test": 2,
        "rare_stress_test": 0,
    }
    assert spec.formal_replicates == 32
    assert latent_dimensions("unit_sphere_direction") == 2
    assert latent_dimensions("haar_quaternion") == 3


def test_sampling_spec_schema_fails_closed_on_unknown_fields() -> None:
    data = _sampling_spec_data()
    data["unreviewed_extension"] = True

    with pytest.raises(InputValidationError, match="failed schema validation") as error:
        SamplingSpec(data)
    assert any("Additional properties" in item for item in error.value.context["errors"])


def test_sampling_spec_hash_is_canonical_and_semantic_changes_are_bound() -> None:
    data = _sampling_spec_data()
    reordered = dict(reversed(list(data.items())))

    baseline = SamplingSpec(data)
    assert SamplingSpec(reordered).hash == baseline.hash

    changed = deepcopy(data)
    changed["reference_distribution"]["variables"][0]["high"] = 301.0
    assert SamplingSpec(changed).hash != baseline.hash


def test_sampling_spec_requires_distinct_rng_namespaces() -> None:
    data = _sampling_spec_data()
    data["seeds"]["formal"] = data["seeds"]["screening"]

    with pytest.raises(InputValidationError, match="seeds must be distinct"):
        SamplingSpec(data)
def test_integer_uniform_rejects_fractional_bounds_instead_of_truncating() -> None:
    data = _sampling_spec_data()
    data["reference_distribution"]["variables"][0] = {
        "name": "replicate_class",
        "path": "/analysis/replicates",
        "distribution": "integer_uniform",
        "low": 2.5,
        "high": 8,
    }

    with pytest.raises(InputValidationError, match="bounds must be integers"):
        SamplingSpec(data)
