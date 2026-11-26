"""Compress multiple single-coordinate terms using a fitted exponential rate."""
import numpy as np
from scipy.optimize import minimize_scalar

from eureka._method.eureka import constant, expr, variable
from eureka._symbolic.core.ast import Program


def groups(program):
    if program.coefficient_mode != "term" or not 3 <= len(program.terms) <= 12:
        return {}
    # Avoid expanding nested projections just to test this sparse rewrite.
    if program.count_unique_nodes() > 40:
        return {}
    result = {}
    for i, term in enumerate(program.terms):
        indices = term.collect_feature_indices()
        if len(indices) == 1:
            result.setdefault(next(iter(indices)), []).append(i)
    return {j: idx for j, idx in result.items() if len(idx) >= 3}


def fit_effect(program, j, indices, X, y):
    other = [t for i, t in enumerate(program.terms) if i not in indices]
    fixed = np.column_stack([np.ones(len(X)), *(t.evaluate(X) for t in other)])
    y = np.ravel(y).astype(float)
    x = X[:, j].astype(float)
    center, span = float(x.mean()), float(np.max(abs(x - x.mean())))
    if span < 1e-6 or len(np.unique(x)) < 8:
        return None
    z = (x-center)/span
    # Project the fixed columns out once for one-dimensional variable projection.
    U, singular, _ = np.linalg.svd(fixed, full_matrices=False)
    U = U[:, singular > 1e-7 * singular[0]]
    residual = y - U @ (U.T @ y)

    def objective(rate):
        wave = np.expm1(rate * z) / rate if rate else z
        wave -= U @ (U.T @ wave)
        norm = wave @ wave
        return -(wave @ residual)**2 / max(norm, 1e-16)

    grid = np.linspace(-6, 6, 25)
    i = min(range(len(grid)), key=lambda k: objective(grid[k]))
    result = minimize_scalar(objective, bounds=(grid[max(i-1, 0)], grid[min(i+1, len(grid)-1)]),
                             method="bounded")
    rate = float(result.x)
    if abs(rate) < .05:
        return None
    arg = expr("*", constant(rate/span), expr("-", variable(j), constant(center)))
    wave = expr("exp", arg)
    design = np.column_stack([fixed, wave.evaluate(X)])
    beta = np.linalg.lstsq(design, y, rcond=1e-7)[0]
    return Program([*other, wave], beta[1:], float(beta[0]))


def select(program, X, y, env, V, v, val_env):
    candidates = groups(program)
    if not candidates:
        return program, dict(accepted=False, reason="structure")
    v, val_env = np.ravel(v), np.ravel(val_env)
    labels = np.unique(val_env)
    def losses(model):
        sq = (model.predict(V)[0].astype(float) - v)**2
        return np.asarray([sq[val_env == label].mean() for label in labels])
    initial = losses(program)
    best, score, axis = program, np.inf, None
    for j, indices in candidates.items():
        candidate = fit_effect(program, j, indices, X, y)
        if candidate is None:
            continue
        loss = losses(candidate)
        # Fewer estimated parameters with equivalent source-validation error.
        if np.all(loss <= 1.02**2 * initial) and loss.mean() <= 1.005**2 * initial.mean() and loss.mean() < score:
            best, score, axis = candidate, float(loss.mean()), j
    return best, dict(accepted=axis is not None, coordinate=axis,
                      before=float(np.sqrt(initial.mean())), after=float(np.sqrt(score)) if axis is not None else None)
