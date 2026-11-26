"""Ridge coefficient fitting and structural cost for symbolic candidates."""

import numpy as np

from eureka._symbolic.core.ast import Program


def _fit_linear_coeffs(Phi: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, float]:
    Phi64 = np.asarray(Phi, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    if Phi64.shape[1] == 0:
        return np.zeros(0, dtype=np.float32), float(np.mean(y64))

    center = np.mean(Phi64, axis=0)
    scale = np.std(Phi64, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (Phi64 - center) / scale
    target_mean = float(np.mean(y64))
    target = y64 - target_mean
    A = standardized.T @ standardized
    A += max(0.0, float(ridge)) * np.eye(A.shape[0], dtype=np.float64)
    b = standardized.T @ target
    try:
        standardized_coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        standardized_coeffs = np.linalg.lstsq(
            standardized,
            target,
            rcond=None,
        )[0]
    coeffs64 = standardized_coeffs / scale
    intercept = float(target_mean - center @ coeffs64)
    coeffs = coeffs64.reshape(-1).astype(np.float32)
    return coeffs, intercept


def mdl_loss(program: Program, input_dim: int) -> float:
    node_cost = float(program.count_nodes())
    feat_cost = float(np.log2(max(2, input_dim)) * len(program.used_features()))
    if_cost = float(15.0 * program.count_ifelse())
    # Additional affine slots in conditional output heads have an explicit
    # structural-proxy cost. Legacy term heads retain exactly their old score.
    extra_coefficients = max(0, program.coefficient_count() - len(program.terms))
    return node_cost + feat_cost + if_cost + extra_coefficients
