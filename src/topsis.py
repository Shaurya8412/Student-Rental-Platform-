"""AHP weight derivation + TOPSIS ranking, pure numpy.

AHP: the user's 1-5 importance ratings are turned into a pairwise
comparison matrix with a Saaty-style difference mapping (see
config.SAATY_DIFF_MAP). Priority weights come from the principal
eigenvector; the consistency ratio (CR) is reported alongside and
should stay below config.CR_THRESHOLD (0.10).

TOPSIS: vector-normalise the decision matrix, apply the AHP weights,
measure each listing's distance to the ideal and anti-ideal points, and
return the closeness coefficient in [0, 1]. All five criteria
(budget, distance, safety, wifi, food) are benefit-type scores where
higher is better, because the specialist agents already convert cost
and distance into "higher is better" scores.
"""

from __future__ import annotations

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# AHP
# ---------------------------------------------------------------------------
def pairwise_matrix(importance: dict, criteria: list | None = None) -> np.ndarray:
    """Build the pairwise comparison matrix from 1-5 importance ratings."""
    criteria = criteria or config.CRITERIA
    values = [float(importance[c]) for c in criteria]
    n = len(values)
    matrix = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            diff = int(round(values[i] - values[j]))
            intensity = config.SAATY_DIFF_MAP[abs(diff)]
            matrix[i, j] = intensity if diff >= 0 else 1.0 / intensity
    return matrix


def ahp_weights(matrix: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Priority vector + consistency diagnostics for a comparison matrix.

    Returns (weights, lambda_max, consistency_index, consistency_ratio).
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    if n == 1:
        return np.array([1.0]), 1.0, 0.0, 0.0
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    idx = int(np.argmax(eigenvalues.real))
    lambda_max = float(eigenvalues[idx].real)
    weights = np.abs(eigenvectors[:, idx].real)
    weights = weights / weights.sum()
    ci = (lambda_max - n) / (n - 1)
    ri = config.RI_TABLE.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return weights, lambda_max, float(ci), float(cr)


def ahp_from_importance(importance: dict,
                        criteria: list | None = None) -> tuple[dict, float]:
    """Convenience wrapper: importance dict -> (weights dict, CR)."""
    criteria = criteria or config.CRITERIA
    matrix = pairwise_matrix(importance, criteria)
    weights, _, _, cr = ahp_weights(matrix)
    return {c: float(w) for c, w in zip(criteria, weights)}, cr


# ---------------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------------
def topsis(decision_matrix: np.ndarray, weights) -> np.ndarray:
    """Closeness coefficients for a benefit-only decision matrix.

    ``decision_matrix`` is (n_listings, n_criteria); ``weights`` is a
    sequence aligned with the columns. Returns an array in [0, 1];
    a listing identical to every other gets 0.5 by convention.
    """
    X = np.asarray(decision_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("decision_matrix must be 2-dimensional")
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    norms = np.sqrt((X ** 2).sum(axis=0))
    norms = np.where(norms == 0, 1.0, norms)
    weighted = (X / norms) * w

    ideal = weighted.max(axis=0)
    anti_ideal = weighted.min(axis=0)
    d_plus = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    closeness = np.divide(d_minus, denom,
                          out=np.full_like(denom, 0.5), where=denom > 0)
    return np.clip(closeness, 0.0, 1.0)


def criterion_contributions(decision_matrix: np.ndarray,
                            weights) -> np.ndarray:
    """Each criterion's share of a listing's weighted normalised score.

    Returns an (n_listings, n_criteria) array of shares summing to 1 per
    row; used to state *which* criterion contributed most to a score.
    If a row is all zeros, shares are uniform (nothing distinguishes it).
    """
    X = np.asarray(decision_matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    norms = np.sqrt((X ** 2).sum(axis=0))
    norms = np.where(norms == 0, 1.0, norms)
    contrib = (X / norms) * w
    totals = contrib.sum(axis=1, keepdims=True)
    uniform = np.full_like(contrib, 1.0 / contrib.shape[1])
    return np.divide(contrib, totals, out=uniform, where=totals > 0)
