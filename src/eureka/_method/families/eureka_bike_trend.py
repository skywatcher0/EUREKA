"""Learn a weather-only rental response after accounting for source calendar drift."""

import numpy as np

from eureka._method.eureka import constant, expr
from eureka._method.families import eureka_group_response as group
from eureka._method.families import eureka_weather_stack as stack
from eureka._method.families.eureka_design_neighbors import model_node, sum_nodes
from eureka._method.families.eureka_source_surfaces import add_formula, bounded, hinge, mul, raw_coordinates, scaled
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program

FEATURES = ("temp", "atemp", "hum", "windspeed")
CONFIGS = ((.75, .006, 1.), (1., .012, 1.), (1., .008, .55), (.65, .025, .72))


def weather_columns(X, feature_names, scaler):
    raw = raw_coordinates(feature_names, scaler)
    _, general, cost = group.library("bike", X, feature_names, scaler)
    t = scaled(expr("+", raw["temp"], raw["atemp"]), .5)
    h, w = raw["hum"], raw["windspeed"]
    hot, muggy = hinge(t, .70), hinge(h, .68)
    specific = [t, h, w, hinge(t, .25), hinge(t, .44), hot, hinge(t, .79),
                hinge(h, .82), hinge(w, .20), mul(muggy, t), mul(muggy, hot), mul(w, hinge(t, .42))]
    return [*specific, *general], np.r_[np.ones(len(specific)), cost + .45]


def time_basis(t, knots):
    step = knots[1] - knots[0]
    return np.maximum(1. - np.abs((t[:, None] - knots) / step), 0.)


def calendar_columns(days, weights):
    knots = np.linspace(days.min(), days.max(), 9)
    long = time_basis(days, knots)
    cycle = 2 * np.pi * days / 365.25
    week = days.astype(int) % 7
    weekly = np.column_stack([week == i for i in range(7)])
    seasons = np.column_stack((np.cos(cycle), np.sin(cycle), np.cos(2 * cycle), np.sin(2 * cycle)))
    nuisance = np.column_stack((long, weekly, seasons))
    nuisance -= np.average(nuisance, axis=0, weights=weights)
    smooth = np.diff(np.eye(len(knots)), n=2, axis=0)
    penalty = np.diag(np.r_[np.full(len(knots), .009), np.full(7, .035), np.full(4, .12)])
    penalty[:len(knots), :len(knots)] += .08 * (smooth.T @ smooth)
    return nuisance, penalty


def bank(anchor, X, y, env, days, query, names, scale):
    nodes, penalty = weather_columns(X, names, scale)
    features = Program(nodes)
    A, Q = (features.design_matrix(z)[0].astype(float) for z in (X, query))
    weight = group.equal_weights(env)
    weight /= weight.sum()
    raw_mean = np.average(A, axis=0, weights=weight)
    sd = np.maximum(A.std(0), .0025)
    Z = (A - raw_mean) / sd
    C, cal_cost = calendar_columns(days, weight)
    train_old, test_old = (anchor.predict(z)[0].astype(float) for z in (X, query))
    gram_c = C.T @ (weight[:, None] * C) + cal_cost
    cross_c = C.T @ (weight[:, None] * Z)
    gram_z = Z.T @ (weight[:, None] * Z)
    fits, predictions = [], []
    for share, ridge, fraction in CONFIGS:
        base = anchor if fraction < .65 else None
        target = y - train_old if base is not None else y
        average = float(weight @ target)
        centered = target - average
        adjustment = np.linalg.solve(gram_c, np.column_stack((cross_c, C.T @ (weight * centered))))
        gram = gram_z - share * (cross_c.T @ adjustment[:, :-1])
        gram.flat[::len(gram) + 1] += ridge * penalty
        beta = np.linalg.solve(gram, Z.T @ (weight * centered) - share * (cross_c.T @ adjustment[:, -1])) / sd
        bias = average - float(raw_mean @ beta)
        prediction = (Q @ beta + bias)
        if base is not None:
            model = add_formula(base, nodes, fraction * beta, fraction * bias)
            prediction = test_old + fraction * prediction
        else:
            model = add_formula(None, nodes, beta, bias)
            if fraction < 1.:
                model = Program([sum_nodes([scaled(model_node(anchor), 1. - fraction),
                                            scaled(model_node(model), fraction)])], np.ones(1), 0.)
                prediction = (1. - fraction) * test_old + fraction * prediction
        fits.append(model)
        predictions.append(prediction)
    return fits, np.column_stack(predictions)


def source_panels(days, validation_days, env, val_env):
    for label in np.unique(env):
        yield env != label, val_env == label
    phase, vp = days % 365.25, validation_days % 365.25
    for a, b in ((16, 81), (95, 160), (180, 250), (267, 332)):
        yield (phase < a - 6) | (b + 6 <= phase), (a <= vp) & (vp < b)


def select(anchor, X, y, env, V, v, val_env, *, feature_names, x_scaler, y_scaler, days):
    if tuple(feature_names) != FEATURES or anchor.coeffs is None:
        return anchor, dict(accepted=False, reason="weather_schema")
    t, vt = (np.asarray(part, dtype=float) for part in days)
    if len(t) != len(X) or len(vt) != len(V) or len(np.unique(np.r_[t, vt])) != len(t) + len(vt):
        raise ValueError("bike source dates must be unique and match source arrays")
    fallback, previous = stack.select(anchor, X, y, env, V, v, val_env,
                                       feature_names=feature_names, x_scaler=x_scaler, y_scaler=y_scaler)
    y, v = (np.ravel(part).astype(float) for part in (y, v))
    env, val_env = np.ravel(env), np.ravel(val_env)
    loss = []
    for train, val in source_panels(t, vt, env, val_env):
        if train.sum() < 190 or val.sum() < 12:
            continue
        prior = refit(anchor, X[train], y[train])
        baseline = prior.predict(V[val])[0].astype(float)
        _, other = bank(prior, X[train], y[train], env[train], t[train], V[val], feature_names, x_scaler)
        before = baseline if fallback is anchor else refit(fallback, X[train], y[train]).predict(V[val])[0]
        loss.append(np.mean((np.column_stack((baseline, before, other)) - v[val, None]) ** 2, axis=0))
    status = dict(accepted=previous["accepted"], prior=previous, panels=len(loss), variant=previous.get("variant"))
    if len(loss) < 6:
        return fallback, dict(status, reason="calendar_panels")
    losses = np.asarray(loss)
    risk = np.sqrt(losses.mean(0))
    comparison = np.minimum(losses[:, 0], losses[:, 1])
    wins = (losses[:, 2:] + 1e-6 < comparison[:, None]).sum(0)
    worst = (np.sqrt(losses[:, 2:]) - np.sqrt(losses[:, [0]])).max(0)
    fits, pred = bank(anchor, X, y, env, t, V, feature_names, x_scaler)
    previous_score = np.sqrt(np.mean((v - fallback.predict(V)[0]) ** 2))
    check = np.sqrt(np.mean((v[:, None] - pred) ** 2, axis=0))
    viable = np.flatnonzero((risk[2:] <= .998 * min(risk[:2])) & (wins * 2 >= len(losses))
                            & (worst <= .065) & (check <= 1.012 * previous_score))
    status.update(source_risk=risk.tolist(), source_wins=wins.tolist(), validation=check.tolist())
    if not len(viable):
        return fallback, dict(status, reason="weather_transfer")
    pick = int(viable[np.argmin(risk[viable + 2])])
    zero = -y_scaler["mean"] / y_scaler["scale"]
    max_y = float(y.max() + .38 * np.ptp(y))
    selected = Program([bounded(model_node(fits[pick]), zero, max_y)], np.ones(1), 0.)
    np.testing.assert_allclose(selected.predict(V)[0], np.clip(pred[:, pick], zero, max_y), atol=2e-4, rtol=1e-3)
    return selected, dict(status, accepted=True, variant="continuous_source_calendar", config=CONFIGS[pick])
