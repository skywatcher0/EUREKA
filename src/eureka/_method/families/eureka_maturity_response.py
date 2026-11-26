"""Fit a monotone maturity response using repeated source recipe observations."""

import numpy as np

from eureka._method.eureka import constant, expr
from eureka._method.families.eureka_conditional_response import ratio
from eureka._method.families.eureka_design_neighbors import model_node, sum_nodes
from eureka._method.families.eureka_source_surfaces import (
    bounded, mul, normalized, plus, polynomial_basis, raw_coordinates, scaled, shrink_fit,
)
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


MATURITY = tuple((days, exponent, delay) for days in (3., 8., 18., 35.)
                 for exponent in (.5, 1.) for delay in (0., 2.5))
OPTIONS = ((.65, .007), (1., .007), (1., .026))


def recipe_coordinates(names, scaler):
    r = raw_coordinates(names, scaler)
    cement, slag, ash = r["cement"], r["blast_furnace_slag"], r["fly_ash"]
    binder = sum_nodes([cement, slag, ash])
    supplemental = ratio(plus(slag, ash), binder)
    return r, supplemental, {key: node for key, node in r.items() if key != "age_days"} | {
        "binder": binder, "water_binder": ratio(r["water"], binder), "supplemental": supplemental,
        "slag_fraction": ratio(slag, binder), "ash_fraction": ratio(ash, binder),
    }


def maturity_node(age, supplemental, item):
    days, exponent, delay = item
    effective = plus(constant(days), scaled(supplemental, days * delay))
    measure = ratio(age, plus(age, effective), .1)
    return expr("exp", scaled(expr("log", measure), .5)) if exponent == .5 else measure


def identify_maturity(y_raw, age, fraction, groups):
    counts = np.bincount(groups)
    lo, hi = np.full(len(counts), np.inf), np.full(len(counts), -np.inf)
    np.minimum.at(lo, groups, age)
    np.maximum.at(hi, groups, age)
    valid = (hi - lo >= 2.)[groups]
    if len(np.unique(groups[valid])) < 10:
        return None
    weights = valid / counts[groups]
    log_y = np.log(np.maximum(y_raw, .25))
    risks = []
    for days, exponent, delay in MATURITY:
        curve = (age / (age + days * (1 + delay * fraction))) ** exponent
        delta = log_y - np.log(curve)
        mean = np.bincount(groups, weights=delta, minlength=len(counts)) / counts
        risks.append(float(np.average((delta - mean[groups]) ** 2, weights=weights)))
    best = int(np.argmin(risks))
    return MATURITY[best], len(np.unique(groups[valid])), risks[best]


def candidate_fits(anchor, X, y, env, V, names, scaler, target):
    raw, fraction, values = recipe_coordinates(names, scaler)
    age = raw["age_days"]
    y_raw = y * target["scale"] + target["mean"]
    keys = [i for i, name in enumerate(names) if name != "age_days"]
    _, ids = np.unique(X[:, keys], axis=0, return_inverse=True)
    chosen = identify_maturity(y_raw, age.evaluate(X), fraction.evaluate(X), ids)
    if chosen is None:
        return None
    m = maturity_node(age, fraction, chosen[0])
    maturity, maturity_v = (m.evaluate(a).astype(float) for a in (X, V))
    features = {key: z for key, node in values.items() if (z := normalized(node, X)) is not None}
    nodes = [feature for z in features.values() for feature in polynomial_basis(z, X)]
    added = [("binder", "water_binder"), ("water_binder", "superplasticizer"), ("water_binder", "supplemental")]
    interactions = [mul(features[a], features[b]) for a, b in added if a in features and b in features]
    costs = np.r_[np.ones(len(nodes)), np.full(len(interactions), 3.)]
    nodes += interactions
    design = Program(nodes)
    M = design.design_matrix(X)[0].astype(float)
    A = design.design_matrix(V)[0].astype(float)
    _, g, count = np.unique(env, return_inverse=True, return_counts=True)
    weights = maturity ** 2 * (.55 + .45 * len(X) / (len(count) * count[g]))
    strength = y_raw / maturity
    lower, upper = np.quantile(strength, [.03, .97])
    lower, upper = max(2., .7 * lower), 1.15 * upper
    old = anchor.predict(V)[0].astype(float)
    ts, predictions = [], [old]
    for share, penalty in OPTIONS:
        beta, intercept = shrink_fit(M, strength, weights, costs, penalty)
        expected = (maturity_v * np.clip(A @ beta + intercept, lower, upper) - target["mean"]) / target["scale"]
        predictions.append(old + share * (expected - old))
        ts.append((nodes, beta, intercept, lower, upper, m, share))
    return np.column_stack(predictions), ts, chosen


def select(anchor, X, y, env, V, v, val_env, *, feature_names, x_scaler, y_scaler):
    names = tuple(feature_names)
    assert names[-1] == "age_days"
    y, v = np.ravel(y).astype(float), np.ravel(v).astype(float)
    env, val_env = np.ravel(env), np.ravel(val_env)
    meta = names, x_scaler, y_scaler
    full = candidate_fits(anchor, X, y, env, V, *meta)
    if full is None:
        return anchor, dict(accepted=False, reason="replicated_recipe_support")
    validation, formulas, choice = full
    validated = np.sqrt(np.mean((validation - v[:, None]) ** 2, axis=0))
    xage = np.rint(raw_coordinates(names, x_scaler)["age_days"].evaluate(X)).astype(int)
    vage = np.rint(raw_coordinates(names, x_scaler)["age_days"].evaluate(V)).astype(int)
    panel_losses, age_panels = [], []
    for current in np.unique(vage):
        tr, va = xage != current, vage == current
        if va.sum() < 9 or tr.sum() < 160:
            continue
        old = refit(anchor, X[tr], y[tr])
        fit = candidate_fits(old, X[tr], y[tr], env[tr], V[va], *meta)
        if fit is not None:
            panel_losses.append(np.mean((fit[0] - v[va, None]) ** 2, axis=0))
            age_panels.append(int(current))
    info = dict(accepted=False, maturity=list(choice[0]), repeated_recipes=choice[1],
                matched_source_loss=choice[2], source_validation=validated.tolist(), age_panels=age_panels)
    if len(panel_losses) < 3:
        return anchor, dict(info, reason="source_age_panels")
    panel_losses = np.array(panel_losses)
    risk = np.sqrt(panel_losses.mean(0))
    wins = (panel_losses[:, 1:] < panel_losses[:, [0]]).sum(0)
    worst = (np.sqrt(panel_losses[:, 1:]) - np.sqrt(panel_losses[:, [0]])).max(0)
    viable = np.flatnonzero((risk[1:] < .99 * risk[0]) & (validated[1:] <= 1.015 * validated[0])
                            & (wins * 2 >= len(panel_losses)) & (worst <= .075 + .12 * risk[0]))
    info.update(source_age_risk=risk.tolist(), source_age_wins=wins.tolist(), source_worst=worst.tolist())
    if not len(viable):
        return anchor, dict(info, reason="blocked_source")
    index = int(viable[np.argmin(risk[1:][viable])])
    nodes, beta, intercept, lo, hi, maturity, share = formulas[index]
    baseline = model_node(anchor)
    recipe = sum_nodes([constant(intercept), *[scaled(node, b) for node, b in zip(nodes, beta, strict=True)]])
    raw = mul(maturity, bounded(recipe, lo, hi))
    outcome = scaled(plus(raw, constant(-y_scaler["mean"])), 1. / y_scaler["scale"])
    top = outcome if share == 1. else plus(scaled(baseline, 1 - share), scaled(outcome, share))
    new = Program([top], np.ones(1), 0.)
    np.testing.assert_allclose(new.predict(V)[0], validation[:, index + 1], rtol=5e-4, atol=7e-5)
    return new, dict(info, accepted=True, variant="recipe_maturity", weight=share, nodes=new.count_unique_nodes())
