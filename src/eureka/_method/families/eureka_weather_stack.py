"""Estimate a bounded weather-response mixture on held source seasons."""

import numpy as np
from scipy.optimize import nnls

from eureka._method.families import eureka_group_response as pooled
from eureka._method.families import eureka_weather_effects as calendar
from eureka._method.families.eureka_design_neighbors import model_node, sum_nodes
from eureka._method.families.eureka_source_surfaces import scaled
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program

PENALTIES = (.002, .008)


def bank(anchor, X, y, env, V, features, scaler, target):
    root, pred = pooled.case_fits("bike", anchor, X, y, env, V, features, scaler, target)
    models = [pooled.expression(anchor, form, target) for form in root]
    by_calendar, q = calendar.fitted_weather(anchor, X, y, env, V, features, scaler)
    return models + by_calendar, np.column_stack((pred[:, 1:], q[:, 1:]))


def strengths(panels, ridge):
    columns = np.concatenate([a for a, _ in panels])
    error = np.concatenate([y for _, y in panels])
    weights = np.sqrt(np.concatenate([np.full(len(y), 1. / (len(panels) * len(y))) for _, y in panels]))
    variables = columns.shape[1]
    a = np.vstack((columns * weights[:, None], np.sqrt(ridge) * np.eye(variables)))
    b = np.r_[error * weights, np.zeros(variables)]
    fit, _ = nnls(a, b, maxiter=100)
    return fit / max(1., fit.sum())


def select(anchor, X, y, env, V, v, val_env, *, feature_names, x_scaler, y_scaler):
    fallback, outcome = calendar.select(anchor, X, y, env, V, v, val_env, feature_names=feature_names,
                                        x_scaler=x_scaler, y_scaler=y_scaler)
    y, v = np.ravel(y).astype(float), np.ravel(v).astype(float)
    env, val_env = np.ravel(env), np.ravel(val_env)
    groups = np.unique(env)
    report = dict(accepted=outcome["accepted"], previous=outcome, variant="year_season_weather" if outcome["accepted"] else "source_expression")
    if len(groups) < 5 or not np.array_equal(groups, np.unique(val_env)):
        return fallback, dict(report, reason="calendar_support")
    parameters = feature_names, x_scaler, y_scaler
    panels = []
    for group in groups:
        fit, hold = env != group, val_env == group
        reference = refit(anchor, X[fit], y[fit])
        previous = refit(fallback, X[fit], y[fit]) if fallback is not anchor else reference
        old = reference.predict(V[hold])[0].astype(float)
        q = previous.predict(V[hold])[0].astype(float)
        _, p = bank(reference, X[fit], y[fit], env[fit], V[hold], *parameters)
        panels.append((np.column_stack((q, p)) - old[:, None], v[hold] - old))
    chances = []
    for penalty in PENALTIES:
        predictions = [M @ strengths(panels[:i] + panels[i + 1:], penalty) for i, (M, _) in enumerate(panels)]
        chances.append([np.mean((r - p) ** 2) for (_, r), p in zip(panels, predictions, strict=True)])
    origin = np.array([np.mean(r ** 2) for _, r in panels])
    previous = np.array([np.mean((r - M[:, 0]) ** 2) for M, r in panels])
    folds = np.array(chances).T
    risk = np.sqrt(folds.mean(0))
    wins = (folds < origin[:, None] - 1e-6).sum(0)
    losses = (np.sqrt(folds) - np.sqrt(origin[:, None])).max(0)
    report.update(blocked_risks=risk.tolist(), previous_blocked=float(np.sqrt(previous.mean())),
                  source_wins=wins.tolist(), source_worst=losses.tolist())
    opts = np.flatnonzero((risk < .998 * min(np.sqrt(origin.mean()), np.sqrt(previous.mean())))
                          & (wins * 2 >= len(groups)) & (losses <= .035))
    if not len(opts):
        return fallback, dict(report, reason="blocked_weather_source")
    choose = int(opts[np.argmin(risk[opts])])
    strength = strengths(panels, PENALTIES[choose])
    models, pred = bank(anchor, X, y, env, V, *parameters)
    ref = anchor.predict(V)[0].astype(float)
    q = fallback.predict(V)[0].astype(float)
    direct = ref + (np.column_stack((q, pred)) - ref[:, None]) @ strength
    rates = np.sqrt(np.mean((np.column_stack((ref, q, direct)) - v[:, None]) ** 2, axis=0))
    if rates[-1] > 1.003 * min(rates[:2]):
        return fallback, dict(report, reason="validation_weather_source", validation=rates.tolist())
    expressions = [scaled(model_node(anchor), 1. - strength.sum())]
    expressions += [scaled(model_node(model), weight) for model, weight in zip([fallback, *models], strength, strict=True)
                    if weight > 0.]
    new = Program([sum_nodes(expressions)], np.ones(1), 0.)
    np.testing.assert_allclose(new.predict(V)[0], direct, atol=2e-4, rtol=1e-3)
    return new, dict(report, accepted=True, variant="source_weather_mixture", weights=strength.tolist(),
                     source_penalty=PENALTIES[choose], validation=rates.tolist())
