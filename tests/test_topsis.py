"""Tests for AHP weight derivation and TOPSIS, incl. hand-computed cases."""

import numpy as np
import pytest

from src import config
from src.topsis import (ahp_from_importance, ahp_weights,
                        criterion_contributions, pairwise_matrix, topsis)


def test_ahp_consistent_matrix_hand_computed():
    # Perfectly consistent matrix: a_ij = w_i / w_j for w = (4, 2, 1).
    matrix = np.array([[1.0, 2.0, 4.0],
                       [0.5, 1.0, 2.0],
                       [0.25, 0.5, 1.0]])
    weights, lambda_max, ci, cr = ahp_weights(matrix)
    assert weights == pytest.approx([4 / 7, 2 / 7, 1 / 7], abs=1e-6)
    assert lambda_max == pytest.approx(3.0, abs=1e-6)
    assert ci == pytest.approx(0.0, abs=1e-6)
    assert cr == pytest.approx(0.0, abs=1e-6)


def test_ahp_equal_importance_gives_uniform_weights():
    importance = {c: 3 for c in config.CRITERIA}
    weights, cr = ahp_from_importance(importance)
    assert list(weights.values()) == pytest.approx([0.2] * 5, abs=1e-9)
    assert cr == pytest.approx(0.0, abs=1e-9)


def test_ahp_one_dominant_criterion_hand_computed():
    # Ratings (5,1,1,1,1): Saaty diff 4 -> intensity 9 against each other
    # criterion, which are mutually equal -> w = (9,1,1,1,1)/13, CR = 0.
    importance = {"budget": 5, "distance": 1, "safety": 1, "wifi": 1,
                  "food": 1}
    weights, cr = ahp_from_importance(importance)
    assert weights["budget"] == pytest.approx(9 / 13, abs=1e-6)
    for criterion in ("distance", "safety", "wifi", "food"):
        assert weights[criterion] == pytest.approx(1 / 13, abs=1e-6)
    assert cr == pytest.approx(0.0, abs=1e-6)


def test_ahp_default_profile_is_consistent_enough():
    weights, cr = ahp_from_importance(config.DEFAULT_PROFILE["weights"])
    assert sum(weights.values()) == pytest.approx(1.0)
    assert cr < config.CR_THRESHOLD
    # Budget and safety share the top importance (5) in the default profile.
    assert weights["budget"] == pytest.approx(weights["safety"], abs=1e-9)
    assert weights["budget"] > weights["distance"] > weights["food"]


def test_pairwise_matrix_is_reciprocal():
    matrix = pairwise_matrix(config.DEFAULT_PROFILE["weights"])
    n = matrix.shape[0]
    for i in range(n):
        assert matrix[i, i] == 1.0
        for j in range(n):
            assert matrix[i, j] == pytest.approx(1.0 / matrix[j, i])


def test_topsis_dominance_hand_computed():
    # Alternative 1 is the ideal on both criteria, alternative 2 the
    # anti-ideal -> closeness exactly 1 and 0.
    closeness = topsis(np.array([[1.0, 1.0], [0.0, 0.0]]), [0.5, 0.5])
    assert closeness == pytest.approx([1.0, 0.0], abs=1e-9)


def test_topsis_symmetric_hand_computed():
    # Mirror-image alternatives must tie at 0.5.
    closeness = topsis(np.array([[1.0, 0.0], [0.0, 1.0]]), [0.5, 0.5])
    assert closeness == pytest.approx([0.5, 0.5], abs=1e-9)


def test_topsis_identical_rows_get_half():
    closeness = topsis(np.array([[2.0, 3.0], [2.0, 3.0]]), [0.6, 0.4])
    assert closeness == pytest.approx([0.5, 0.5], abs=1e-9)


def test_topsis_bounds_on_random_matrix():
    rng = np.random.default_rng(0)
    matrix = rng.random((25, 5))
    closeness = topsis(matrix, [5, 4, 5, 3, 2])
    assert closeness.min() >= 0.0 and closeness.max() <= 1.0


def test_contributions_sum_to_one():
    rng = np.random.default_rng(1)
    matrix = rng.random((10, 5))
    shares = criterion_contributions(matrix, [0.3, 0.2, 0.3, 0.1, 0.1])
    assert shares.sum(axis=1) == pytest.approx(np.ones(10), abs=1e-9)
