"""Bounded piecewise-linear main effects and sparse pairwise products."""
import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._method.method import select_formula
from eureka._symbolic.core.ast import Program


def coordinate_basis(x, j):
    levels = np.unique(x)
    leaf = variable(j)
    if len(levels) < 2:
        return []
    if len(levels) <= 8:
        return [expr("eq", leaf, constant(value)) for value in levels]
    knots = np.unique(np.quantile(x, np.linspace(0, 1, 9)).astype(np.float32))
    if len(knots) < 2:
        return []
    ramps = []
    for low, high in zip(knots, knots[1:]):
        slope = expr("*", constant(1. / float(high-low)), expr("-", leaf, constant(low)))
        ramps.append(expr("ifelse", expr("gt", leaf, constant(high)), constant(1.),
                          expr("ifelse", expr("gt", leaf, constant(low)), slope, constant(0.))))
    return [expr("-", constant(1.), ramps[0]),
            *(expr("-", a, b) for a, b in zip(ramps, ramps[1:])), ramps[-1]]


def fit_interpolant(X, y):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float).reshape(-1)
    ym = y.mean()
    blocks = []
    for j in range(X.shape[1]):
        terms = coordinate_basis(X[:, j], j)
        if not terms:
            continue
        columns = np.column_stack([node.evaluate(X) for node in terms]).astype(float)
        centered = columns - columns.mean(0)
        association = centered.T @ (y-ym)
        strength = np.sum(association**2 / (np.sum(centered**2, axis=0) + 2.))
        blocks.append((strength, terms, columns))
    if not blocks:
        return Program([], np.empty(0), float(ym))
    blocks = sorted(blocks, key=lambda b: -b[0])[:8]
    terms = [term for _, group, _ in blocks for term in group]
    design = np.column_stack([b[2] for b in blocks])

    def ridge(matrix):
        mean = matrix.mean(0)
        A = matrix - mean
        gram = A.T @ A
        gram.flat[::len(gram)+1] += 2.
        coef = np.linalg.solve(gram, A.T @ (y-ym))
        return coef, float(ym - mean @ coef)

    coef, intercept = ridge(design)
    residual = y - (design @ coef + intercept)
    hits = []
    start = 0
    for _, group, _ in blocks:
        stop = start + len(group)
        for i in range(start, stop):
            products = design[:, i, None] * design[:, stop:]
            products -= products.mean(0)
            gain = (products.T @ residual)**2 / (np.sum(products**2, axis=0) + 2.)
            hits.extend((score, i, stop+j) for j, score in enumerate(gain) if score > 1e-12)
        start = stop
    pairs = sorted(hits, reverse=True)[:min(24, len(y)//15)]
    if pairs:
        extra = np.column_stack([design[:, i] * design[:, j] for _, i, j in pairs])
        terms += [expr("*", terms[i], terms[j]) for _, i, j in pairs]
        coef, intercept = ridge(np.column_stack([design, extra]))
    return Program(terms, coef, intercept)


def assess_interpolation(incumbent, X, y, env, V, v, val_env, seed=0):
    X, V = np.asarray(X), np.asarray(V)
    y, v = np.asarray(y).ravel(), np.asarray(v).ravel()
    env, val_env = np.asarray(env).ravel(), np.asarray(val_env).ravel()
    full = fit_interpolant(X, y)
    groups = np.unique(env)
    moments, fold_ids, boundary_moments = [], [], []
    for fold, held in enumerate(np.array_split(groups, min(5, len(groups)))):
        if len(groups) == 1:
            train, held_x, truth, labels = X, V, v, val_env
            base, candidate = incumbent, full
        else:
            hold, valid = np.isin(env, held), np.isin(val_env, held)
            train, response = X[~hold], y[~hold]
            held_x, truth = np.vstack([X[hold], V[valid]]), np.r_[y[hold], v[valid]]
            labels = np.r_[env[hold], val_env[valid]]
            base = refit(incumbent, train, response)
            candidate = fit_interpolant(train, response)
        a = base.predict(held_x)[0].astype(float)
        diff = candidate.predict(held_x)[0].astype(float) - a
        residual = truth - a
        loss = np.column_stack([residual**2, residual * diff, diff**2])
        outside = np.any((held_x < train.min(0)) | (held_x > train.max(0)), axis=1)
        if len(groups) == 1:
            labels = np.arange(len(labels)) % 2
        for group in np.unique(labels):
            mask = labels == group
            moments.append(loss[mask].mean(0))
            fold_ids.append(group if len(groups) == 1 else fold)
            boundary_moments.append((int((mask & outside).sum()), loss[mask & outside].sum(0)))
    moments, fold_ids = np.asarray(moments), np.asarray(fold_ids)
    def weight(data):
        return float(np.clip(data[:, 1].sum() / max(1e-12, data[:, 2].sum()), 0., 1.))
    weights = np.array([weight(moments[fold_ids != i]) for i in fold_ids])
    def blend_loss(m, w):
        return np.maximum(0., m[..., 0] - 2*w*m[..., 1] + w*w*m[..., 2])
    errors = np.column_stack([moments[:, 0], blend_loss(moments, weights)])
    boundary = [[n, float(m[0]), float(blend_loss(m, w))] for (n, m), w in zip(boundary_moments, weights) if n]
    w = weight(moments)
    if w == 0:
        blended = incumbent
    elif w == 1:
        blended = full
    else:
        terms = [expr("*", constant(1.), t) if incumbent.coefficient_mode == "conditional_affine"
                 and t.op == "ifelse" else t for t in full.terms]
        coeffs = incumbent.coeffs if incumbent.coeffs is not None else np.ones(incumbent.coefficient_count())
        blended = Program([*incumbent.terms, *terms], np.r_[(1.-w)*coeffs, w*full.coeffs],
                          (1.-w)*incumbent.intercept+w*full.intercept, incumbent.coefficient_mode)
    scores = np.sqrt(np.mean(errors, axis=0))
    chosen, decision = select_formula(incumbent, blended, *scores, V, v, val_env)
    decision.update(fold_mse=np.asarray(errors).tolist(), outside_count_and_sse=boundary,
                    candidate_nodes=blended.count_unique_nodes(), weight=w)
    return chosen, decision
