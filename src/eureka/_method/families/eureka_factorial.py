"""Regularized response surfaces for sources measured on repeated factor levels."""
from itertools import combinations

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def supported(X):
    counts = [len(np.unique(column)) for column in np.asarray(X).T]
    varying = [n for n in counts if n > 1]
    return (len(X) >= 80 and 2 <= len(varying) <= 12 and max(varying) <= 32
            and min(varying) <= 8 and len(X) >= 3 * max(varying))


def surface(X, y, penalty):
    y = np.asarray(y, dtype=float).ravel()
    main, linear = [], []
    for j, values in enumerate(X.T):
        levels = np.unique(values)
        if len(levels) < 2:
            continue
        node = variable(j)
        linear.append(node)
        main.append(node)
        # A threshold basis models unequal increments at successive levels.
        for a, b in zip(levels[:-1], levels[1:]):
            main.append(expr("gt", node, constant((float(a) + float(b)) / 2)))
    terms = main + [expr("*", a, b) for a, b in combinations(linear, 2)]
    matrix = np.column_stack([term.evaluate(X) for term in terms]).astype(float)
    mean = matrix.mean(0)
    scale = matrix.std(0)
    scale[scale < 1e-8] = 1.
    centered = (matrix - mean) / scale
    gram = centered.T @ centered
    gram.flat[::len(gram)+1] += penalty
    coeffs = np.linalg.solve(gram, centered.T @ (y - y.mean())) / scale
    return Program(terms, coeffs, float(y.mean() - mean @ coeffs))


def mse(program, X, y):
    return float(np.mean((program.predict(X)[0].astype(float) - np.ravel(y))**2))


def select(incumbent, X, y, env, V, v, val_env):
    env, val_env = np.ravel(env), np.ravel(val_env)
    groups = np.unique(env)
    if not supported(X) or len(groups) < 2:
        return incumbent, dict(accepted=False, reason="insufficient_factorial_support")
    penalties = (1., 10., 50.)
    fold_mse = []
    for held in np.array_split(groups, min(5, len(groups))):
        train = ~np.isin(env, held)
        candidates = [refit(incumbent, X[train], y[train])]
        candidates.extend(surface(X[train], y[train], p) for p in penalties)
        for g in held:
            valid = val_env == g
            if valid.any():
                fold_mse.append([mse(model, V[valid], v[valid]) for model in candidates])
    if not fold_mse:
        return incumbent, dict(accepted=False, reason="no_matched_validation_environment")
    blocked = np.sqrt(np.mean(fold_mse, axis=0))
    choice = int(np.argmin(blocked[1:])) + 1
    candidate = surface(X, y, penalties[choice-1])
    validation = np.sqrt(np.mean([[mse(model, V[val_env == g], v[val_env == g])
                                    for model in (incumbent, candidate)] for g in np.unique(val_env)], axis=0))
    accepted = bool(blocked[choice] < .95 * blocked[0]
                    and validation[1] <= validation[0] + .03 * np.std(y))
    return (candidate if accepted else incumbent), dict(
        accepted=accepted, reason="source_comparison", penalty=penalties[choice-1],
        blocked=blocked.tolist(), validation=validation.tolist(), fold_mse=fold_mse)
