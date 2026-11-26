"""Fit bounded residual corrections to EUREKA's source predictions."""

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.eureka_design_neighbors import model_node, sum_nodes
from eureka._method.families.eureka_size_contrasts import clip as bounded
from eureka._method.families.eureka_source_surfaces import shrink_fit
from eureka._method.families.blocked_selection import _fit_linear_coeffs
from eureka._symbolic.core.ast import Program

LABELS = ("incumbent", "response_bounds", "output_calibration", "local_interactions", "stable_interactions")


def bounds(y):
    levels = np.unique(y)
    if 2 <= len(levels) <= 8 and np.allclose(np.diff(levels), np.diff(levels).mean(), rtol=1e-4):
        return float(levels[0]), float(levels[-1])
    lo, hi = y.min(), y.max()
    margin = .15 * (hi - lo)
    return float(lo - margin), float(hi + margin)


def environment_weights(env):
    _, ix, n = np.unique(env, return_inverse=True, return_counts=True)
    return .5 + .5 * len(env) / (len(n) * n[ix])


def coordinates(X, p, residual, env, V, q):
    scale_p = max(float(p.std()), .05)
    mu = float(p.mean())
    z = (p - mu) / scale_p
    w = (q - mu) / scale_p
    limits = tuple(np.quantile(z, [.01, .99]))
    z = np.clip(z, *limits)
    w = np.clip(w, *limits)
    knots = np.unique(np.quantile(z, [.2, .5, .8]))
    core = np.column_stack([z, *[np.maximum(z - knot, 0.) for knot in knots]])
    other_core = np.column_stack([w, *[np.maximum(w - knot, 0.) for knot in knots]])
    center = np.median(X, axis=0)
    scale = np.maximum(X.std(axis=0).astype(float), .02)
    U = np.clip((X - center) / scale, -3., 3.)
    Z = np.clip((V - center) / scale, -3., 3.)
    within = []
    for g in np.unique(env):
        grouprows = env == g
        a, b = U[grouprows], residual[grouprows]
        if len(a) > 4:
            within.append(((a - a.mean(0)).T @ (b - b.mean())) / len(b))
    association = np.abs(np.mean(within, axis=0)) if within else np.zeros(X.shape[1])
    axes = np.argsort(-association, kind="stable")[:4]
    A = np.column_stack((core, U[:, axes], z[:, None] * U[:, axes]))
    B = np.column_stack((other_core, Z[:, axes], w[:, None] * Z[:, axes]))
    return A, B, (mu, scale_p, limits, knots, center[axes], scale[axes], axes, core.shape[1])


def fits(X, p, V, q, y, env):
    y = np.asarray(y, float).ravel()
    residual = y - p
    A, B, basis = coordinates(X, p, residual, env, V, q)
    size = basis[-1]
    weights = environment_weights(env)
    safe = bounds(y)
    cap = max(float(.65 * y.std()), .05)
    full, preds = [], [q, np.clip(q, *safe)]
    for shape, penalty, fixed_effects in ((size, .02, False), (A.shape[1], .05, False),
                                          (A.shape[1], .03, True)):
        Z = A[:, :shape]
        rt = residual
        if fixed_effects:
            Z, rt = Z.copy(), residual.copy()
            for group in np.unique(env):
                held = env == group
                Z[held] -= Z[held].mean(0)
                rt[held] -= rt[held].mean()
        beta, intercept = shrink_fit(Z, rt, weights, np.ones(shape), penalty)
        if fixed_effects:
            intercept = float(np.average(residual - A[:, :shape] @ beta, weights=weights))
        estimates = q + np.clip(B[:, :shape] @ beta + intercept, -cap, cap)
        preds.append(np.clip(estimates, *safe))
        full.append((beta, intercept))
    return (basis, safe, cap, full), np.column_stack(preds)


def expression(anchor, fitted, index):
    basis, safe, cap, coefficients = fitted
    p = model_node(anchor)
    if index == 1:
        return Program([bounded(p, *safe)], np.ones(1), 0.)
    mu, scale_p, limits, knots, center, scale, axes, nmain = basis
    z = bounded(expr("*", constant(1. / scale_p), expr("-", p, constant(mu))), *limits)
    terms = [z]
    for k in knots:
        terms.append(expr("*", expr("gt", z, constant(k)), expr("-", z, constant(k))))
    for axis, loc, sd in zip(axes, center, scale, strict=True):
        terms.append(bounded(expr("*", constant(1. / sd), expr("-", variable(axis), constant(loc))), -3., 3.))
    terms.extend(expr("*", z, node) for node in terms[nmain:])
    beta, intercept = coefficients[index - 2]
    drift = sum_nodes([constant(intercept), *[expr("*", constant(c), t) for c, t in zip(beta, terms)]])
    output = bounded(expr("+", p, bounded(drift, -cap, cap)), *safe)
    return Program([output], np.ones(1), 0.)


def select(anchor, X, y, env, V, v, val_env):
    if anchor.coeffs is None or X.shape[1] not in range(4, 22):
        return anchor, dict(accepted=False, reason="source_support")
    levels = np.unique(env)
    if len(levels) < 2 or not np.array_equal(levels, np.unique(val_env)):
        return anchor, dict(accepted=False, reason="source_groups")
    y, v = np.asarray(y, float).ravel(), np.asarray(v, float).ravel()
    train_phi = anchor.design_matrix(X)[0]
    val_phi = anchor.design_matrix(V)[0]
    p, q = (anchor._combine_terms(phi).astype(float) for phi in (train_phi, val_phi))
    fit, estimates = fits(X, p, V, q, y, env)
    val = np.sqrt(np.mean([np.mean((estimates[val_env == g] - v[val_env == g, None]) ** 2, axis=0)
                          for g in levels], axis=0))
    report = dict(accepted=False, validation=val.tolist())
    if min(val[1:]) >= .998 * val[0]:
        return anchor, dict(report, reason="source_validation")
    blocked = []
    for g in levels:
        training, valid = env != g, val_env == g
        if training.sum() < 30:
            return anchor, dict(report, reason="source_fold_support")
        beta, bias = _fit_linear_coeffs(train_phi[training].astype(float), y[training])
        p_train = (train_phi[training] @ beta.astype(np.float32) + bias).astype(float)
        p_valid = (val_phi[valid] @ beta.astype(np.float32) + bias).astype(float)
        _, P = fits(X[training], p_train, V[valid], p_valid, y[training], env[training])
        blocked.append(np.mean((P - v[valid, None]) ** 2, axis=0))
    losses = np.asarray(blocked)
    blocked = np.sqrt(losses.mean(0))
    wins = (losses[:, 1:] < losses[:, [0]]).sum(0)
    possible = np.flatnonzero((val[1:] < .998 * val[0]) & (blocked[1:] < .998 * blocked[0])
                              & (wins * 2 >= len(levels)))
    report.update(blocked=blocked.tolist(), source_wins=wins.tolist())
    if not len(possible):
        return anchor, dict(report, reason="blocked_validation")
    i = int(possible[np.argmin(val[1:][possible] / val[0] + blocked[1:][possible] / blocked[0])]) + 1
    result = expression(anchor, fit, i)
    np.testing.assert_allclose(result.predict(V)[0], estimates[:, i], rtol=8e-4, atol=2e-4)
    return result, dict(report, accepted=True, variant=LABELS[i], nodes=result.count_unique_nodes())
