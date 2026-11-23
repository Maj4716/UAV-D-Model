import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.target_encoding import (
    fit_target_encoder,
    inverse_target,
    project_encoded_prediction,
    transform_target,
)


@pytest.mark.parametrize("method", ["identity", "log1p", "box_cox"])
def test_reversible_target_encoders_round_trip_original_scale(method):
    values = np.asarray([0.0, 0.25, 1.0, 4.0, 12.0])
    contract = fit_target_encoder(values, method=method, task_name="energy")
    restored = inverse_target(transform_target(values, contract), contract)
    assert np.allclose(restored, values, rtol=1.0e-10, atol=1.0e-10)
    assert contract["fit_scope"] == "train_only"
    assert contract["label_mutation"] is False


def test_quantile_encoder_is_monotone_and_inverts_training_knots():
    values = np.asarray([0.0, 0.5, 1.0, 2.0, 8.0, 20.0])
    contract = fit_target_encoder(
        values,
        method="empirical_quantile_normal",
        task_name="positive_energy",
        quantile_knots=values.size,
    )
    transformed = transform_target(values, contract)
    restored = inverse_target(transformed, contract)
    assert np.all(np.diff(transformed) > 0.0)
    assert np.allclose(restored, values, rtol=0.0, atol=1.0e-8)


def test_encoder_refuses_validation_fit_and_hash_drift():
    with pytest.raises(InputValidationError, match="Train only"):
        fit_target_encoder([1.0, 2.0], method="identity", task_name="x", source_split="validation")
    contract = fit_target_encoder([1.0, 2.0], method="identity", task_name="x")
    contract["training_observation_count"] = 3
    with pytest.raises(InputValidationError, match="hash"):
        transform_target([1.0], contract)


def test_box_cox_prediction_projection_is_train_only_and_invertible():
    values = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0])
    contract = fit_target_encoder(values, method="box_cox", task_name="count")
    projected = project_encoded_prediction(
        np.asarray([-1.0e9, 0.0, 1.0e9]), contract
    )
    parameters = contract["parameters"]

    assert np.all(projected >= parameters["transformed_training_min"])
    assert np.all(projected <= parameters["transformed_training_max"])
    restored = inverse_target(projected, contract)
    assert np.isfinite(restored).all()
    assert np.all(restored >= parameters["training_value_min"] - 1.0e-12)
    assert np.all(restored <= parameters["training_value_max"] + 1.0e-12)


def test_box_cox_constant_train_target_has_valid_one_point_projection():
    contract = fit_target_encoder(
        np.asarray([2.5]), method="box_cox", task_name="constant-count"
    )
    projected = project_encoded_prediction(
        np.asarray([-1.0e6, 0.0, 1.0e6]), contract
    )

    assert np.all(projected == projected[0])
    restored = inverse_target(projected, contract)
    np.testing.assert_allclose(restored, np.full(3, 2.5), atol=1.0e-12)
