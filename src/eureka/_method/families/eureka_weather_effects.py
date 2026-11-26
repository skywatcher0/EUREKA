"""Separate year and season effects when fitting weather-only rental responses."""

import numpy as np
from scipy.linalg import solve

from eureka._method.families import eureka_group_response as previous
from eureka._method.families.eureka_source_surfaces import add_formula
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program

OPTIONS = ((.55, .010, False), (.85, .010, False), (.72, .014, True))


def fitted_weather(anchor, X, y, env, V, names, scaler):
    _, nodes, penalty = previous.library("bike", X, names, scaler)
    model = Program(nodes)
    A, W = (model.design_matrix(rows)[0].astype(float) for rows in (X, V))
    weight = previous.equal_weights(env)
    year, season = env // 4, env % 4
    categories = np.column_stack([group == value for group in (year, season) for value in np.unique(group)]).astype(float)
    C = categories - np.average(categories, axis=0, weights=weight)
    H = C.T @ (weight[:, None] * C)
    H.flat[::len(H) + 1] += .045 * weight.sum()
    mu = np.average(A, axis=0, weights=weight)
    sd = np.maximum(A.std(0), 1e-5)
    Z = (A - mu) / sd
    explained_z = C @ solve(H, C.T @ (weight[:, None] * Z), assume_a="pos")
    ref, ref_val = (anchor.predict(rows)[0].astype(float) for rows in (X, V))
    estimates, shapes = [ref_val], []
    for projection, shrink, residual in OPTIONS:
        g = y - ref if residual else y
        avg = float(np.average(g, weights=weight))
        centered = g - avg
        explained = C @ solve(H, C.T @ (weight * centered), assume_a="pos")
        Xfit, yfit = Z - projection * explained_z, centered - projection * explained
        gram = Xfit.T @ (weight[:, None] * Xfit)
        gram.flat[::len(gram) + 1] += shrink * weight.sum() * penalty
        beta = solve(gram, Xfit.T @ (weight * yfit), assume_a="pos") / sd
        bias = avg - float(mu @ beta)
        alpha = .63 if residual else 1.
        form = add_formula(anchor if residual else None, nodes, alpha * beta, alpha * bias)
        pred = alpha * (W @ beta + bias)
        if residual:
            pred += ref_val
        estimates.append(pred)
        shapes.append(form)
    return shapes, np.column_stack(estimates)


def select(anchor, X, y, env, V, v, val_env, *, feature_names, x_scaler, y_scaler):
    incumbent, pooled_decision = previous.select("bike", anchor, X, y, env, V, v, val_env,
                                      feature_names=feature_names, x_scaler=x_scaler, y_scaler=y_scaler)
    env, val_env = np.ravel(env).astype(int), np.ravel(val_env).astype(int)
    y, v = np.ravel(y).astype(float), np.ravel(v).astype(float)
    groups = np.unique(env)
    info = dict(accepted=pooled_decision["accepted"], base=pooled_decision,
                variant="group_centred_weather" if pooled_decision["accepted"] else "source_expression")
    if len(groups) < 4 or not np.array_equal(groups, np.unique(val_env)):
        return incumbent, dict(info, reason="calendar_group_support")
    full, val = fitted_weather(incumbent, X, y, env, V, feature_names, x_scaler)
    valid_loss = np.sqrt(np.mean((val - v[:, None]) ** 2, axis=0))
    panels = []
    for group in groups:
        tr, va = env != group, val_env == group
        reference = refit(incumbent, X[tr], y[tr])
        _, P = fitted_weather(reference, X[tr], y[tr], env[tr], V[va], feature_names, x_scaler)
        panels.append(np.mean((P - v[va, None]) ** 2, axis=0))
    block = np.array(panels)
    risk = np.sqrt(block.mean(0))
    diff = np.sqrt(block[:, [0]]) - np.sqrt(block[:, 1:])
    options = np.flatnonzero((risk[1:] < .997 * risk[0]) & (valid_loss[1:] <= 1.003 * valid_loss[0])
                             & ((diff > 1e-4).sum(0) * 2 >= len(groups)) & (diff.min(0) >= -.06))
    info.update(validation=valid_loss.tolist(), blocked=risk.tolist(), source_wins=(diff > 1e-4).sum(0).tolist())
    if not len(options):
        return incumbent, dict(info, reason="blocked_calendar_source")
    choice = int(options[np.argmin(risk[1:][options])])
    selected = full[choice]
    np.testing.assert_allclose(selected.predict(V)[0], val[:, choice + 1], rtol=1e-3, atol=2e-4)
    return selected, dict(info, accepted=True, variant="separate_year_season", option=list(OPTIONS[choice]))
