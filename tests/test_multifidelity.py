import numpy as np

from damage_engine.multifidelity import (
    control_variate_mean,
    fit_control_variate,
    paired_efficiency_report,
)


def test_control_variate_recovers_linear_teacher_and_reduces_variance():
    rng = np.random.default_rng(7)
    proxy = rng.normal(size=200)
    teacher = 3.0 + 2.0 * proxy + rng.normal(scale=0.1, size=200)
    fit = fit_control_variate(proxy, teacher)
    estimate = control_variate_mean(
        proxy,
        teacher,
        proxy_reference_mean=np.asarray([0.0]),
        fit=fit,
    )
    report = paired_efficiency_report(proxy, teacher)
    assert abs(estimate["estimate"][0] - 3.0) < 0.03
    assert report["observed_variance_reduction_fraction"][0] > 0.99
