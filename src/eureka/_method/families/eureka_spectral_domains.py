"""Select a separate spectral correction for interpolation across operating cells."""

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families import eureka_supported_spectrum as transfer
from eureka._method.families import eureka_swept_response as sweep
from eureka._method.families.eureka_design_neighbors import model_node, sum_nodes
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program

WEIGHTS = (.35, .7, 1.)


def cell_support(X, query, geometry, velocity):
    known_geometry = set(map(tuple, X[:, geometry]))
    seen = np.array([tuple(row) in known_geometry for row in query[:, geometry]])
    return seen & np.isin(query[:, velocity], np.unique(X[:, velocity]))


def candidate_predictions(anchor, predictions, V):
    reference = anchor.predict(V)[0].astype(float)
    return np.column_stack([reference, *[reference + w * (p - reference)
                                        for p in predictions for w in WEIGHTS]])


def select(anchor, X, y, env, V, v, val_env):
    shape = transfer.support(X, env)
    if shape is None or anchor.coeffs is None:
        return anchor, {"accepted": False, "reason": "source_support"}
    geometry_axes, geometry = shape
    velocity = sweep.axes(X, env)[1]
    outer, outer_decision = transfer.select(anchor, X, y, env, V, v, val_env)
    report = {"accepted": outer_decision["accepted"], "velocity_transfer": outer_decision}
    y, v = np.asarray(y, float).ravel(), np.asarray(v, float).ravel()
    full = sweep.fit_block(X, y, V, env)
    if full is None:
        return outer, report
    models, predictions = full
    active = cell_support(X, V, geometry_axes, velocity)
    if not active.any():
        return outer, dict(report, cells="validation_support")
    full_predictions = candidate_predictions(anchor, predictions, V)
    validation = np.sqrt(np.mean((full_predictions[active] - v[active, None]) ** 2, axis=0))
    if min(validation[1:]) >= .995 * validation[0]:
        return outer, dict(report, cells="source_validation")
    key = [*geometry_axes, velocity]
    cells = np.unique(X[:, key], axis=0)
    rng = np.random.default_rng(6112389)
    losses = []
    for held in np.array_split(rng.permutation(cells), 5):
        omitted = set(map(tuple, held))
        tr = np.array([tuple(row) not in omitted for row in X[:, key]])
        va = np.array([tuple(row) in omitted for row in V[:, key]])
        va &= cell_support(X[tr], V, geometry_axes, velocity)
        if va.sum() < 15:
            continue
        fold = sweep.fit_block(X[tr], y[tr], V[va], env[tr])
        if fold is None:
            continue
        old = refit(anchor, X[tr], y[tr])
        pred = candidate_predictions(old, fold[1], V[va])
        losses.append(np.mean((pred - v[va, None]) ** 2, axis=0))
    if len(losses) < 4:
        return outer, dict(report, cells="blocked_support", cell_folds=len(losses))
    losses = np.asarray(losses)
    blocked = np.sqrt(losses.mean(0))
    wins = (losses[:, 1:] < losses[:, [0]]).sum(0)
    possible = np.flatnonzero((blocked[1:] < .99 * blocked[0]) & (validation[1:] < .995 * validation[0])
                             & (wins * 2 >= len(losses)))
    report.update(cells="blocked_validation", cell_folds=len(losses), blocked=blocked.tolist(),
                  validation=validation.tolist(), source_wins=wins.tolist())
    if not len(possible):
        return outer, report
    index = int(possible[np.argmin(blocked[1:][possible] / blocked[0] + validation[1:][possible] / validation[0])])
    trial, w = divmod(index, len(WEIGHTS))
    fitted = sweep.serialize(*models[trial])
    weight = WEIGHTS[w]
    a = model_node(anchor)
    candidate = expr("+", expr("*", constant(1. - weight), a),
                     expr("*", constant(weight), model_node(fitted)))
    j, k = geometry_axes
    known_shape = sum_nodes([expr("*", expr("eq", variable(j), constant(b)),
                                  expr("eq", variable(k), constant(c))) for b, c in geometry])
    known_velocity = sum_nodes([expr("eq", variable(velocity), constant(z)) for z in np.unique(X[:, velocity])])
    gate = expr("*", known_shape, known_velocity)
    result = Program([expr("ifelse", gate, candidate, model_node(outer))], np.ones(1), 0.)
    target = np.where(active, full_predictions[:, index + 1], outer.predict(V)[0])
    np.testing.assert_allclose(result.predict(V)[0], target, rtol=6e-4, atol=8e-5)
    return result, dict(report, accepted=True, cells="accepted", cell_weight=weight, nodes=result.count_unique_nodes())
