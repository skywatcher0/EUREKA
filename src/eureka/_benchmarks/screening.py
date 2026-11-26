"""DGReg dependence-gap screen using source covariates and environments."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression


def marginal_environment_shift(
    X: np.ndarray,
    env: np.ndarray,
) -> np.ndarray:
    """Measure feature-wise distribution shift across source environments.

    The score is the largest pairwise Kolmogorov-Smirnov distance.  It uses
    source inputs and source environment identifiers only.
    """

    X = np.asarray(X, dtype=np.float64)
    env = np.asarray(env).reshape(-1)
    if X.ndim != 2 or len(X) != len(env):
        raise ValueError("X must be 2-D and aligned with env.")
    environments = sorted(np.unique(env).tolist())
    if len(environments) < 2:
        return np.zeros(X.shape[1], dtype=np.float64)
    scores = np.zeros(X.shape[1], dtype=np.float64)
    for column in range(X.shape[1]):
        distances = [
            float(
                stats.ks_2samp(
                    X[env == left, column],
                    X[env == right, column],
                    method="asymp",
                ).statistic
            )
            for left, right in combinations(environments, 2)
        ]
        scores[column] = max(distances, default=0.0)
    return scores



def select_exogenous_independent_features(
    X: np.ndarray,
    env: np.ndarray,
    *,
    count: int | None,
    max_samples: int = 6000,
    random_state: int = 0,
    marginal_rank_weight: float = 0.01,
    global_dependence_weight: float = 0.5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select a low-dependence, low-shift source feature subset.

    The intended use is to suppress environment-dependent proxy descendants:
    mutually independent source variables receive lower scores than nonlinear
    mixtures of those variables.  The selector uses source inputs only.
    """

    X = np.asarray(X, dtype=np.float64)
    env = np.asarray(env).reshape(-1)
    if X.ndim != 2 or len(X) != len(env):
        raise ValueError("X must be 2-D and aligned with env.")
    rng = np.random.default_rng(random_state)
    if len(X) > int(max_samples):
        chosen_rows = np.sort(
            rng.choice(len(X), size=int(max_samples), replace=False)
        )
        sample = X[chosen_rows]
    else:
        sample = X
    dimensions = sample.shape[1]
    matrix = np.zeros((dimensions, dimensions), dtype=np.float64)
    for target in range(dimensions):
        predictors = [index for index in range(dimensions) if index != target]
        if not predictors:
            continue
        values = mutual_info_regression(
            sample[:, predictors],
            sample[:, target],
            n_neighbors=5,
            random_state=random_state,
        )
        matrix[target, predictors] = values
    matrix = 0.5 * (matrix + matrix.T)
    dependence_scores = np.mean(matrix, axis=1)
    if count is None:
        ordered_dependence = np.sort(dependence_scores, kind="stable")
        candidate_splits = np.arange(2, dimensions, dtype=int)
        if len(candidate_splits) == 0:
            selected_count = dimensions
            count_decision = "all_features_no_valid_gap"
            largest_gap = 0.0
            gap_ratio = 0.0
        else:
            gaps = (
                ordered_dependence[candidate_splits]
                - ordered_dependence[candidate_splits - 1]
            )
            winner = int(np.argmax(gaps))
            largest_gap = float(gaps[winner])
            positive_gaps = gaps[gaps > 1e-12]
            baseline_gap = float(
                np.median(positive_gaps)
                if len(positive_gaps)
                else 0.0
            )
            gap_ratio = largest_gap / max(baseline_gap, 1e-12)
            if largest_gap >= 0.02 and gap_ratio >= 3.0:
                selected_count = int(candidate_splits[winner])
                count_decision = "dependence_gap"
            else:
                selected_count = dimensions
                count_decision = "all_features_no_clear_gap"
    else:
        selected_count = max(1, min(int(count), dimensions))
        count_decision = "fixed_count"
        largest_gap = None
        gap_ratio = None
    marginal = marginal_environment_shift(X, env)
    marginal_rank = np.argsort(np.argsort(marginal, kind="stable"), kind="stable")
    marginal_rank = marginal_rank.astype(float) / max(dimensions - 1, 1)
    selected: list[int] = []
    remaining = set(range(dimensions))
    while remaining and len(selected) < selected_count:
        candidate_scores = []
        for candidate in sorted(remaining):
            if selected:
                selected_dependence = float(
                    np.mean(matrix[candidate, selected])
                )
                global_weight = float(
                    np.clip(global_dependence_weight, 0.0, 1.0)
                )
                dependence = (
                    (1.0 - global_weight) * selected_dependence
                    + global_weight * float(dependence_scores[candidate])
                )
            else:
                dependence = float(dependence_scores[candidate])
            candidate_scores.append(
                (
                    dependence
                    + float(marginal_rank_weight) * marginal_rank[candidate],
                    candidate,
                )
            )
        _, winner = min(candidate_scores)
        selected.append(int(winner))
        remaining.remove(winner)
    indices = np.asarray(sorted(selected), dtype=np.int64)
    audit = {
        "method": "source_exogenous_independence",
        "selected_indices": indices.astype(int).tolist(),
        "selection_order": selected,
        "selected_count": int(len(indices)),
        "pairwise_mutual_information": matrix.tolist(),
        "mean_pairwise_mutual_information": dependence_scores.tolist(),
        "marginal_shift": marginal.tolist(),
        "marginal_rank_weight": float(marginal_rank_weight),
        "global_dependence_weight": float(global_dependence_weight),
        "count_decision": count_decision,
        "largest_dependence_gap": largest_gap,
        "dependence_gap_ratio": gap_ratio,
        "uses_source_response": False,
        "uses_target_inputs": False,
        "uses_target_response": False,
    }
    return indices, audit
