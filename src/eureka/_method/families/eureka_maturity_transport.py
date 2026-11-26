"""Transport source recipe effects by a monotone maturity ratio outside age support."""

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.eureka_design_neighbors import model_node
from eureka._method.families.eureka_maturity_response import identify_maturity, maturity_node, recipe_coordinates, select as interior
from eureka._method.families.eureka_source_surfaces import bounded, minus, mul, plus, scaled
from eureka._method.method import _fit_structured_expression
from eureka._symbolic.core.ast import Program

SHARES = np.array([.65, 1.])


def replace_age(node, axis, value):
    if not node.children:
        return value if node.op == "var" and node.feature_idx == axis else node
    return expr(node.op, *[replace_age(child, axis, value) for child in node.children])


def maturity_parameters(X, y_raw, age, fraction, axis):
    recipe = np.delete(X, axis, axis=1)
    _, groups = np.unique(recipe, axis=0, return_inverse=True)
    return identify_maturity(y_raw, age, fraction, groups)


def transport_predictions(model, train, y, query, side, meta):
    features, scaler, response = meta
    axis = features.index("age_days")
    raw, fraction, _ = recipe_coordinates(features, scaler)
    age = raw["age_days"]
    xage = np.rint(age.evaluate(train)).astype(float)
    y_raw = y * response["scale"] + response["mean"]
    params = maturity_parameters(train, y_raw, xage, fraction.evaluate(train), axis)
    if params is None:
        return None
    edge = xage.min() if side == "early" else xage.max()
    normalized_edge = (edge - scaler["mean"][axis]) / scaler["scale"][axis]
    edge_rows = query.copy()
    edge_rows[:, axis] = normalized_edge
    growth = maturity_node(age, fraction, params[0])
    growth_ratio = growth.evaluate(query).astype(float) / growth.evaluate(edge_rows).astype(float)
    floor = max(.1, .6 * float(y_raw.min()))
    ceiling = 1.25 * float(y_raw.max())
    model_edge = model.predict(edge_rows)[0].astype(float) * response["scale"] + response["mean"]
    expected = (np.clip(model_edge, floor, ceiling) * growth_ratio - response["mean"]) / response["scale"]
    return expected, normalized_edge, floor, ceiling, params


def source_tail(X, y, env, V, v, val_env, side, meta):
    features, scaler, _ = meta
    axis = features.index("age_days")
    ages = np.rint(np.r_[X[:, axis], V[:, axis]] * scaler["scale"][axis] + scaler["mean"][axis]).astype(int)
    data, target = np.vstack((X, V)), np.r_[y, v]
    n = len(X)
    values = np.unique(ages) if side == "early" else np.unique(ages)[::-1]
    folds, observed = [], []
    used = 0
    for edge in values[:min(6, len(values) - 2)]:
        held = ages <= edge if side == "early" else ages >= edge
        if held.sum() < 24 or held[:n].sum() == used or (~held[:n]).sum() < 190 or (~held[n:]).sum() < 20:
            continue
        keep = ~held[:n]
        native = ~held[n:]
        if len(np.unique(env[keep])) < 2 or not np.array_equal(np.unique(env[keep]), np.unique(val_env[native])):
            continue
        old, _ = _fit_structured_expression(X[keep], y[keep, None], env[keep], V[native], v[native, None], val_env[native], seed=0)
        new = transport_predictions(old, X[keep], y[keep], data[held], side, meta)
        if new is None:
            continue
        base = old.predict(data[held])[0].astype(float)
        prediction = base[:, None] + SHARES * (new[0] - base)[:, None]
        folds.append(np.mean((np.column_stack((base, prediction)) - target[held, None]) ** 2, axis=0))
        observed.append(int(edge))
        used = int(held[:n].sum())
        if len(folds) == 3:
            break
    if len(folds) < 2:
        return None, dict(source_edges=observed, reason="insufficient_source_tails")
    loss = np.array(folds)
    risk = np.sqrt(loss.mean(0))
    gains = np.sqrt(loss[:, [0]]) - np.sqrt(loss[:, 1:])
    viable = np.flatnonzero((risk[1:] <= .99 * risk[0]) & ((gains > .001).sum(0) * 2 >= len(folds))
                            & (gains.min(0) >= -.025))
    report = dict(source_edges=observed, source_risk=risk.tolist(), source_wins=(gains > .001).sum(0).tolist(),
                  source_max_loss=(-gains.min(0)).tolist())
    if not len(viable):
        return None, dict(report, reason="blocked_source_tail")
    return float(SHARES[viable[np.argmin(risk[1:][viable])]]), report


def select(anchor, X, y, env, V, v, val_env, *, feature_names, x_scaler, y_scaler):
    y, v = np.ravel(y).astype(float), np.ravel(v).astype(float)
    meta = tuple(feature_names), x_scaler, y_scaler
    feature, _, _ = meta
    axis = feature.index("age_days")
    inside, info = interior(anchor, X, y, env, V, v, val_env, feature_names=feature,
                            x_scaler=x_scaler, y_scaler=y_scaler)
    if inside.coeffs is None:
        return inside, dict(accepted=info["accepted"], interior=info)
    base = model_node(inside)
    observed = np.rint(X[:, axis] * x_scaler["scale"][axis] + x_scaler["mean"][axis])
    sides = []
    if observed.min() > 3:
        sides.append(("early", "lt"))
    if observed.max() < 365:
        sides.append(("late", "gt"))
    tree, reports = base, {}
    raw, fraction, _ = recipe_coordinates(feature, x_scaler)
    age = raw["age_days"]
    expected = inside.predict(V)[0].astype(float)
    for side, comparison in sides:
        share, report = source_tail(X, y, env, V, v, val_env, side, meta)
        reports[side] = report
        if share is None:
            continue
        predicted, edge, floor, upper, params = transport_predictions(inside, X, y, V, side, meta)
        t = constant(edge)
        at_edge = replace_age(base, axis, t)
        m = maturity_node(age, fraction, params[0])
        at_m = replace_age(m, axis, t)
        original_units = plus(constant(y_scaler["mean"]), scaled(at_edge, y_scaler["scale"]))
        forecast_units = mul(bounded(original_units, floor, upper), expr("/", m, at_m))
        forecast = scaled(minus(forecast_units, constant(y_scaler["mean"])), 1. / y_scaler["scale"])
        if share < 1.:
            forecast = plus(scaled(base, 1. - share), scaled(forecast, share))
        tree = expr("ifelse", expr(comparison, variable(axis), t), forecast, tree)
        mask = V[:, axis] < np.float32(edge) if side == "early" else V[:, axis] > np.float32(edge)
        expected[mask] += share * (predicted[mask] - expected[mask])
        reports[side] = dict(report, share=share, maturity=list(params[0]), clipped_edge=[floor, upper])
    if tree is base:
        return inside, dict(accepted=info["accepted"], interior=info, tails=reports)
    new = Program([tree], np.ones(1), 0.)
    np.testing.assert_allclose(new.predict(V)[0], expected, atol=8e-5, rtol=5e-4)
    return new, dict(accepted=True, variant="maturity_transport", interior=info, tails=reports,
                     nodes=new.count_unique_nodes())
