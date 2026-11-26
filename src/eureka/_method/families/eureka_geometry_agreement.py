"""Correct an unseen replicated design when source-block refits support the change."""
import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families import eureka_design_neighbors as neighbor
from eureka._method.method import _fit_measurement_expression
from eureka._symbolic.core.ast import Program


def peer_fits(X, y, env, V, v, val_env, key, seed):
    models = []
    levels = np.unique(X[:, key])
    for hold in np.array_split(levels, 3):
        tr, va = ~np.isin(X[:, key], hold), ~np.isin(V[:, key], hold)
        if len(np.unique(env[tr])) < 2 or not va.any():
            return []
        p, _ = _fit_measurement_expression(X[tr], y[tr, None], env[tr], V[va], v[va, None], val_env[va], seed=seed)
        if p.coeffs is None or p.count_nodes() > 400:
            return []
        models.append(p)
    return models


def candidates(current, anchor, peers, scale):
    direction = np.sign(anchor-current)
    votes = ((peers-current) * direction > scale).sum(axis=0)
    delta = anchor-current
    return [current+np.where(votes >= threshold, fraction*delta, 0.)
            for threshold in (1, 2, 3) for fraction in (.5, 1.)]


def select(base, X, y, env, V, v, val_env, seed=0):
    y, v = np.ravel(y).astype(float), np.ravel(v).astype(float)
    env, val_env = np.ravel(env), np.ravel(val_env)
    structure = neighbor.profile(X, env)
    if structure is None or base.coeffs is None or base.count_nodes()>400:
        return base, dict(accepted=False, reason="support")
    key = structure[0]
    models = peer_fits(X, y, env, V, v, val_env, key, seed)
    if len(models) != 3:
        return base, dict(accepted=False, reason="peer_support")
    losses = []
    scale = .05*max(float(np.std(y)), 1e-8)
    for held in np.array_split(np.unique(X[:, key]), 5):
        tr, va = ~np.isin(X[:, key], held), ~np.isin(V[:, key], held)
        if not (~va).any() or len(np.unique(env[tr])) < 2:
            continue
        args = X[tr], y[tr, None], env[tr], V[va], v[va, None], val_env[va]
        inner, _ = _fit_measurement_expression(*args, seed=seed)
        peers = peer_fits(X[tr], y[tr], env[tr], V[va], v[va], val_env[va], key, seed)
        if len(peers)!=3:
            continue
        test = V[~va]
        p = inner.predict(test)[0].astype(float)
        anchor = neighbor.predict(neighbor.fit(X[tr], y[tr], structure), test)
        predictions = np.array([peer.predict(test)[0] for peer in peers])
        losses.append([np.mean((pred-v[~va])**2) for pred in [p, *candidates(p, anchor, predictions, scale)]])
    if len(losses)<3:
        return base, dict(accepted=False, reason="fold_support")
    losses = np.array(losses)
    blocked = np.sqrt(losses.mean(0))
    viable = np.flatnonzero((blocked[1:] < .985*blocked[0]) &
                            ((losses[:, 1:] < losses[:, [0]]).sum(0) >= len(losses)/2))
    report = dict(accepted=bool(len(viable)), blocked=blocked.tolist(), folds=len(losses))
    if not len(viable):
        return base, report
    choice = int(viable[np.argmin(blocked[1:][viable])])
    threshold, fraction = 1+choice//2, (.5, 1.)[choice%2]
    full = neighbor.fit(X, y, structure)
    anchor, current = neighbor.anchor_node(full), neighbor.model_node(base)
    upwards = expr("gt", anchor, current)
    votes = [expr("ifelse", upwards, expr("gt", peer, expr("+", current, constant(scale))),
                  expr("lt", peer, expr("-", current, constant(scale)))) for peer in map(neighbor.model_node, models)]
    agree = expr("gt", neighbor.sum_nodes(votes), constant(threshold-.5))
    change = expr("*", constant(fraction), expr("-", anchor, current))
    known = neighbor.sum_nodes([expr("eq", variable(key), constant(k)) for k in np.unique(X[:, key])])
    correction = expr("ifelse", known, constant(0.), expr("ifelse", agree, change, constant(0.)))
    result = Program([*base.terms, correction], np.r_[base.coeffs, 1.], base.intercept, base.coefficient_mode)
    # A conditional-affine Program allocates one coefficient per terminal branch.
    if base.coefficient_mode == "conditional_affine":
        root = expr("*", constant(1.), correction)
        result = Program([*base.terms, root], np.r_[base.coeffs, 1.], base.intercept, base.coefficient_mode)
    for values in (X, V):
        old = base.predict(values)[0].astype(float)
        predictions = np.array([p.predict(values)[0] for p in models])
        alternate = candidates(old, neighbor.predict(full, values), predictions, scale)[choice]
        known_mask = np.min(abs(values[:, [key]]-np.unique(X[:, key])), axis=1) < 1e-4
        np.testing.assert_allclose(result.predict(values)[0], np.where(known_mask, old, alternate),
                                   rtol=2e-4, atol=2e-4)
    report.update(votes=threshold, fraction=fraction, nodes=result.count_unique_nodes())
    return result, report
