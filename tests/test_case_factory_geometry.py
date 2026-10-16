from __future__ import annotations

import numpy as np

from damage_engine.case_factory import _haar_quaternion, _unit_direction, randomized_lhs


def test_randomized_lhs_is_replayable_and_uses_each_stratum_once() -> None:
    sample_count = 64
    dimension = 6

    first = randomized_lhs(sample_count, dimension, seed=20261010)
    replay = randomized_lhs(sample_count, dimension, seed=20261010)
    different_seed = randomized_lhs(sample_count, dimension, seed=20261011)

    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, different_seed)
    assert np.all((0.0 < first) & (first < 1.0))
    for axis in range(dimension):
        occupied = np.sort(np.floor(first[:, axis] * sample_count).astype(int))
        np.testing.assert_array_equal(occupied, np.arange(sample_count))


def test_unit_sphere_transform_has_unit_norm_and_is_directionally_balanced() -> None:
    latent = randomized_lhs(4096, 2, seed=6174)
    directions = np.asarray([_unit_direction(u, v) for u, v in latent])

    np.testing.assert_allclose(
        np.linalg.norm(directions, axis=1),
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(directions.mean(axis=0), 0.0, rtol=0.0, atol=0.025)
    np.testing.assert_allclose(
        np.mean(directions * directions, axis=0),
        [1.0 / 3.0] * 3,
        rtol=0.0,
        atol=0.025,
    )
def test_haar_quaternion_transform_is_unit_norm_with_canonical_sign() -> None:
    latent = randomized_lhs(4096, 3, seed=2718)
    quaternions = np.asarray([_haar_quaternion(u, v, w) for u, v, w in latent])

    np.testing.assert_allclose(
        np.linalg.norm(quaternions, axis=1),
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.all(quaternions[:, 3] >= 0.0)
    np.testing.assert_allclose(
        np.mean(quaternions * quaternions, axis=0),
        [0.25] * 4,
        rtol=0.0,
        atol=0.025,
    )
