import pytest

torch = pytest.importorskip("torch")

from damage_engine.errors import InputValidationError
from damage_engine.hurdle_learning import (
    masked_binomial_nll_from_logits,
    masked_heteroscedastic_gaussian_nll,
    masked_interval_squared_loss,
    monte_carlo_precision_weight,
    ordered_state_probabilities,
)


def test_interval_loss_is_zero_inside_and_squared_distance_outside():
    prediction = torch.tensor([[0.2, 0.8, 1.4]])
    lower = torch.tensor([[0.1, 0.6, 0.9]])
    upper = torch.tensor([[0.3, 1.0, 1.0]])
    mask = torch.ones_like(prediction, dtype=torch.bool)
    loss = masked_interval_squared_loss(prediction, lower, upper, mask)
    assert float(loss) == pytest.approx(0.16 / 3.0)


def test_ordered_state_parameterization_cannot_invert_states():
    degraded, failed = ordered_state_probabilities(
        torch.tensor([[-2.0, 2.0]]), torch.tensor([[8.0, -8.0]])
    )
    assert torch.all(failed <= degraded)
    assert torch.all((failed >= 0.0) & (degraded <= 1.0))


def test_binomial_loss_uses_counts_and_gaussian_loss_accepts_mc_variance():
    logits = torch.zeros((2, 1))
    successes = torch.tensor([[2.0], [8.0]])
    trials = torch.tensor([[4.0], [16.0]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    loss = masked_binomial_nll_from_logits(logits, successes, trials, mask)
    assert float(loss) == pytest.approx(10.0 * 0.6931471805599453)

    variance = torch.tensor([[1.0], [4.0]])
    count = torch.tensor([[4.0], [16.0]])
    precision = monte_carlo_precision_weight(
        variance, count, mask, variance_floor=0.01
    )
    assert precision[0, 0] == pytest.approx(precision[1, 0])
    gaussian = masked_heteroscedastic_gaussian_nll(
        torch.zeros_like(logits),
        torch.zeros_like(logits),
        torch.tensor([[0.0], [1.0]]),
        mask,
        observation_variance=variance / count,
        precision_weight=precision,
    )
    assert torch.isfinite(gaussian)


def test_losses_keep_design_and_precision_weight_shapes_distinct():
    values = torch.zeros((2, 3))
    mask = torch.ones_like(values, dtype=torch.bool)
    with pytest.raises(InputValidationError, match="case axes"):
        masked_interval_squared_loss(
            values, values, values, mask, design_weight=torch.ones(2, 1)
        )
