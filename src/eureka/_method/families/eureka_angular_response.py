"""Source-selected angular harmonics for two periodic measurement coordinates."""
import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families import eureka_relative_phase as phase
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def supported(X):
    return (X.shape[1] == 2 and len(X) >= 100 and
            all(8 <= len(np.unique(col)) and .5 <= np.ptp(col) <= 2*np.pi+.01 and
                np.abs(col).max() <= 2*np.pi+.01 for col in X.T))


def terms(spec, order):
    i, j, sign, frequency, _ = spec
    directions = [(1, sign), (1, 0), (0, 1), (1, -sign)]
    if order >= 2:
        directions += [(2, sign), (1, 2*sign), (2, 2*sign)]
    if order >= 3:
        directions += [(2, 0), (0, 2), (2, -sign), (1, -2*sign)]
    basis = []
    for a, b in directions:
        components = [expr("*", constant(frequency * k), variable(axis))
                      for axis, k in ((i, a), (j, b)) if k]
        angle = components[0] if len(components) == 1 else expr("+", *components)
        basis.extend([expr("cos", angle), expr("sin", angle)])
    return basis


def estimate(basis, X, y, env, precision):
    A = np.column_stack([term.evaluate(X) for term in basis]).astype(float)
    _, index, counts = np.unique(env, return_counts=True, return_inverse=True)
    weight = 1 / (len(counts) * counts[index])
    def solve(w):
        mean, center = w @ A, w @ y
        centered = A - mean
        gram = centered.T @ (w[:, None] * centered)
        gram.flat[::len(gram)+1] += .00002
        coefficients = np.linalg.solve(gram, centered.T @ (w * (y-center)))
        return Program(basis, coefficients, float(center-mean @ coefficients))
    first = solve(weight)
    if not precision:
        return first
    fitted = first.predict(X)[0].astype(float)
    residual = y-fitted
    floor = max(np.mean(residual**2) * .05, 1e-10)
    groups = np.array_split(np.argsort(fitted, kind="stable"), min(8, len(X)//20))
    centers = [fitted[group].mean() for group in groups]
    variance = [max(floor, np.mean(residual[group]**2)) for group in groups]
    adjusted = weight / np.interp(fitted, centers, variance)
    return solve(adjusted / adjusted.sum())


def select(old, X, y, env, V, v, val_env, seed=0):
    y, env, v, val_env = (np.asarray(a).ravel() for a in (y, env, v, val_env))
    groups = np.unique(env)
    if not supported(X) or len(groups) < 2 or not np.array_equal(groups, np.unique(val_env)):
        return old, dict(accepted=False, reason="support")
    spec = phase.search(X.astype(float), y, seed)
    if spec is None:
        return old, dict(accepted=False, reason="periodicity")
    configs = [(order, precision) for order in (1, 2, 3) for precision in (False, True)]
    bases = [(terms(spec, order), precision) for order, precision in configs]
    models = [estimate(basis, X, y, env, precision) for basis, precision in bases]
    def scores(models, held):
        return [np.mean((p.predict(V[held])[0].astype(float)-v[held])**2) for p in models]
    validation = np.sqrt(np.mean([scores([old, *models], val_env == g) for g in groups], axis=0))
    if validation[1:].min() >= .995 * validation[0]:
        return old, dict(accepted=False, reason="validation", validation=validation.tolist())
    folds = []
    for group in groups:
        keep = env != group
        fold_spec = phase.search(X[keep].astype(float), y[keep], seed)
        if fold_spec != spec:
            return old, dict(accepted=False, reason="fold_periodicity")
        competitors = [refit(old, X[keep], y[keep])]
        competitors.extend(estimate(basis, X[keep], y[keep], env[keep], precision)
                           for basis, precision in bases)
        folds.append(scores(competitors, val_env == group))
    losses = np.array(folds)
    blocked = np.sqrt(losses.mean(0))
    wins = (losses[:, 1:] < losses[:, [0]]).sum(0)
    viable = np.flatnonzero((blocked[1:] < .99 * blocked[0]) &
                            (validation[1:] < .995 * validation[0]) & (wins >= len(groups)/2))
    report = dict(accepted=bool(len(viable)), blocked=blocked.tolist(),
                  validation=validation.tolist(), wins=wins.tolist())
    if not len(viable):
        return old, report
    index = int(viable[np.argmin(blocked[1:][viable])])
    selected = models[index]
    report.update(order=configs[index][0], precision=configs[index][1], nodes=selected.count_unique_nodes())
    return selected, report
