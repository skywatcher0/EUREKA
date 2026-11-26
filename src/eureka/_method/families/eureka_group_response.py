"""Partial pooling of source effects with task-specific symbolic response links."""

import numpy as np
from scipy.linalg import solve

from eureka._method.eureka import constant, expr
from eureka._method.families import eureka_conditional_response as previous
from eureka._method.families import eureka_source_surfaces as terms
from eureka._method.families.eureka_design_neighbors import model_node, sum_nodes
from eureka._method.families.eureka_source_surfaces import bounded, hinge, log_positive, minus, mul, plus, scaled
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program

AUDITED = ("beijing_o3", "bike", "tetouan", "appliances", "temp_forecast", "gas_turbine", "seoul_bike", "yacht", "energy")
POSITIVE = {"appliances", "seoul_bike", "gas_turbine", "yacht"}
VARIANTS = (("partial_pool", .55, .010, 1.), ("within_source", .88, .007, 1.),
            ("moderated_residual", .72, .013, .65))


def physical(name, r):
    if name == "beijing_o3":
        ventilation = log_positive(plus(r["wind_speed"], constant(1.)), 1.)
        nox = log_positive(plus(r["no2"], constant(1.)), 1.)
        co = log_positive(plus(r["co"], constant(1.)), 1.)
        return {"oxidation_balance": minus(nox, co), "ventilation": ventilation,
                "thermal_ventilation": mul(minus(r["temperature"], r["dew_point"]), ventilation)}
    if name == "bike":
        return {"humid_heat": mul(r["temp"], r["hum"]), "apparent_temperature_gap": minus(r["atemp"], r["temp"]),
                "drying_wind": mul(minus(constant(1.), r["hum"]), r["windspeed"])}
    if name == "tetouan":
        return {"daylight": log_positive(plus(r["general_diffuse_flow"], constant(1.)), 1.),
                "thermal_humidity": mul(r["temperature"], r["humidity"])}
    if name == "appliances":
        return {"light_on": expr("gt", r["lights"], constant(.25)),
                "humidity_gradient": minus(r["rh_in"], r["rh_out"]),
                "thermal_gradient": minus(r["t_in"], r["t_out"])}
    if name == "temp_forecast":
        return {"terrain_cloud": mul(r["DEM"], scaled(plus(r["LDAPS_CC3"], r["LDAPS_CC4"]), .5)),
                "cloud_tendency": minus(plus(r["LDAPS_CC3"], r["LDAPS_CC4"]), plus(r["LDAPS_CC1"], r["LDAPS_CC2"])),
                "max_bias_proxy": minus(r["Present_Tmax"], r["LDAPS_Tmax_lapse"])}
    if name == "gas_turbine":
        return {"inverse_fire_temperature": previous.ratio(constant(1000.), plus(r["TIT"], constant(273.15))),
                "expansion_ratio": previous.ratio(scaled(r["CDP"], 1000.), r["AP"]),
                "flow_power": previous.ratio(r["AFDP"], r["TEY"])}
    if name == "seoul_bike":
        return {"rain_active": expr("gt", r["rainfall"], constant(.05)),
                "warm_humidity": mul(r["temperature"], r["humidity"]),
                "dark_cloud": log_positive(plus(r["solar_radiation"], constant(1.)), 1.)}
    if name == "yacht":
        return {"inverse_slenderness": previous.ratio(r["beam_draught_ratio"], r["length_beam_ratio"]),
                "geometry_wave": mul(r["prismatic_coefficient"], mul(r["froude_number"], r["froude_number"]))}
    floor = mul(r["roof_area"], r["overall_height"])
    return {"walls_per_volume": previous.ratio(r["wall_area"], floor), "roof_per_volume": previous.ratio(r["roof_area"], floor),
            "glazing_per_volume": previous.ratio(mul(r["glazing_area"], r["wall_area"]), floor),
            "height_glazing": mul(r["overall_height"], r["glazing_area"])}


def clock_terms(raw):
    for phase in np.arange(12) * np.pi / 6:
        similarity = plus(scaled(raw["hour_sin"], np.sin(phase)), scaled(raw["hour_cos"], np.cos(phase)))
        yield expr("exp", scaled(minus(constant(1.), similarity), -7.))


def library(name, X, names, x_scale):
    raw = terms.raw_coordinates(names, x_scale)
    nodes, primary, cost = previous.library(name, X, names, x_scale)
    features = physical(name, raw)
    available = {key: z for key, feature in features.items() if (z := terms.normalized(feature, X)) is not None}
    new = [term for node in available.values() for term in terms.polynomial_basis(node, X)]
    expensive = []
    if "hour_sin" in raw:
        clocks = list(clock_terms(raw))
        new.extend(clocks)
        fields = {"tetouan": ("weekend", "temperature", "daylight"),
                  "appliances": ("weekend", "light_on", "thermal_gradient"),
                  "seoul_bike": ("holiday", "rain_active", "temperature")}[name]
        for field in fields:
            z = available.get(field)
            if z is None and field in raw:
                z = terms.normalized(raw[field], X)
            if z is not None:
                expensive.extend(mul(c, z) for c in clocks)
    tech = nodes[:primary] + new + nodes[primary:] + expensive
    penalty = np.r_[cost[:primary], np.full(len(new), 1.8), cost[primary:], np.full(len(expensive), 3.8)]
    return raw, tech, penalty


def equal_weights(env):
    _, labels, sizes = np.unique(env, return_inverse=True, return_counts=True)
    return .35 + .65 * len(env) / (len(sizes) * sizes[labels])


def pooled_coefficients(A, y, env, weight, cost, share, penalty):
    mean_x = np.average(A, axis=0, weights=weight)
    spread = np.maximum(A.std(0), 1e-5)
    mean_y = float(np.average(y, weights=weight))
    D = (A - mean_x) / spread
    centered = y - mean_y
    Y = centered.copy()
    original = D.copy()
    for label in np.unique(env):
        part = env == label
        D[part] -= share * np.average(original[part], axis=0, weights=weight[part])
        Y[part] -= share * float(np.average(centered[part], weights=weight[part]))
    gram = D.T @ (weight[:, None] * D)
    gram.flat[::len(gram) + 1] += weight.sum() * penalty * cost
    fitted = solve(gram, D.T @ (weight * Y), assume_a="pos") / spread
    return fitted, float(mean_y - mean_x @ fitted)


def hull_profile(node, X, y, weight):
    speed = np.round(node.evaluate(X).astype(float), 4)
    coordinates, labels = np.unique(speed, return_inverse=True)
    amount = np.bincount(labels, weights=weight)
    assert len(coordinates) >= 3
    mean = np.bincount(labels, weights=weight * y) / amount
    rate = np.diff(mean) / np.diff(coordinates)
    z = bounded(node, coordinates[0], coordinates[-1])
    return sum_nodes([constant(mean[0]), scaled(minus(z, constant(coordinates[0])), rate[0]),
                      *[scaled(hinge(z, p), s) for p, s in zip(coordinates[1:-1], np.diff(rate), strict=True)]])


def case_fits(name, anchor, X, y, env, V, names, xs, scale):
    raw, nodes, cost = library(name, X, names, xs)
    structure = Program(nodes)
    train = structure.design_matrix(X)[0].astype(float)
    valid = structure.design_matrix(V)[0].astype(float)
    old = anchor.predict(V)[0].astype(float)
    train_prediction = anchor.predict(X)[0].astype(float)
    weight = equal_weights(env)
    forms, estimates = [], [old]
    for kind, share, penalty, shrink in VARIANTS:
        target = y - train_prediction if kind == "moderated_residual" else y
        beta, bias = pooled_coefficients(train, target, env, weight, cost, share, penalty)
        candidate = valid @ beta + bias
        if kind == "moderated_residual":
            candidate = old + shrink * np.clip(candidate, -.65, .65)
        else:
            lo, hi = np.quantile(y, [.002, .998])
            lo, hi = float(lo - .50), float(hi + .50)
            candidate = np.clip(candidate, lo, hi)
        forms.append((kind, nodes, beta, bias, shrink, None if kind == "moderated_residual" else (lo, hi)))
        estimates.append(candidate)

    if name == "temp_forecast":
        forecast = plus(scaled(raw["LDAPS_Tmax_lapse"], .8), scaled(raw["Present_Tmax"], .2))
        basis = scaled(minus(forecast, constant(scale["mean"])), 1. / scale["scale"])
        delta = y - basis.evaluate(X)
        beta, bias = pooled_coefficients(train, delta, env, weight, cost, .62, .014)
        estimates.append(basis.evaluate(V) + np.clip(valid @ beta + bias, -1.6, 1.6))
        forms.append(("forecast_bias", nodes, beta, bias, basis, None))
    if name in POSITIVE:
        shift = max(.03, .025 * scale["scale"])
        magnitude = np.maximum(y * scale["scale"] + scale["mean"] + shift, shift)
        weighting = weight * np.clip(np.sqrt(magnitude / np.median(magnitude)), .3, 4.)
        logarithm = np.log(magnitude)
        if name == "yacht":
            profile = hull_profile(raw["froude_number"], X, logarithm, weighting)
            at_source, at_query = (profile.evaluate(a).astype(float) for a in (X, V))
        else:
            profile = constant(0.)
            at_source, at_query = 0., 0.
        bounds = float(logarithm.min() - .1), float(logarithm.max() + .16)
        beta, bias = pooled_coefficients(train, logarithm - at_source, env, weighting, cost, .58, .013)
        fitted = np.exp(np.clip(train @ beta + bias + at_source, *bounds))
        smearing = float(np.clip(np.average(magnitude, weights=weight) / np.average(fitted, weights=weight), .7, 1.35))
        expected = smearing * np.exp(np.clip(valid @ beta + bias + at_query, *bounds))
        estimates.append((expected - shift - scale["mean"]) / scale["scale"])
        forms.append(("positive_link", nodes, beta, bias, profile, (shift, *bounds, smearing)))
    return forms, np.column_stack(estimates)


def expression(anchor, fit, scale):
    mode, nodes, beta, bias, setting, auxiliary = fit
    response = sum_nodes([constant(bias), *[scaled(node, weight) for node, weight in zip(nodes, beta, strict=True)]])
    if mode == "moderated_residual":
        expression = plus(model_node(anchor), scaled(bounded(response, -.65, .65), setting))
    elif mode == "forecast_bias":
        expression = plus(setting, bounded(response, -1.6, 1.6))
    elif mode == "positive_link":
        shift, lo, hi, factor = auxiliary
        raw = scaled(expr("exp", bounded(plus(setting, response), lo, hi)), factor)
        expression = scaled(minus(raw, constant(shift + scale["mean"])), 1. / scale["scale"])
    else:
        expression = bounded(response, *auxiliary)
    return Program([expression], np.ones(1), 0.)


def select(name, anchor, X, y, env, V, v, val_env, *, feature_names, x_scaler, y_scaler):
    env, val_env = np.ravel(env), np.ravel(val_env)
    y, v = np.ravel(y).astype(float), np.ravel(v).astype(float)
    labels = np.unique(env)
    if len(labels) < 2 or anchor.coeffs is None or not np.array_equal(labels, np.unique(val_env)):
        return anchor, dict(accepted=False, reason="source_support")
    args = feature_names, x_scaler, y_scaler
    forms, predicted = case_fits(name, anchor, X, y, env, V, *args)
    val = np.sqrt(np.mean(terms.rmse_by_group(predicted, v, val_env, labels) ** 2, axis=0))
    info = dict(accepted=False, validation=val.tolist())
    if val[1:].min() > 1.010 * val[0]:
        return anchor, dict(info, reason="source_validation")
    cross = []
    for label in labels:
        train, held = env != label, val_env == label
        if train.sum() < 27:
            return anchor, dict(info, reason="source_panel_size")
        old = refit(anchor, X[train], y[train])
        _, pred = case_fits(name, old, X[train], y[train], env[train], V[held], *args)
        cross.append(np.mean((pred - v[held, None]) ** 2, axis=0))
    folds = np.array(cross)
    risk = np.sqrt(folds.mean(0))
    diff = np.sqrt(folds[:, [0]]) - np.sqrt(folds[:, 1:])
    choices = np.flatnonzero((risk[1:] < .997 * risk[0]) & (val[1:] <= 1.01 * val[0])
                            & ((diff > .0002).sum(0) * 2 >= len(labels))
                            & (diff.min(0) >= -.06 - .12 * risk[0]))
    info.update(blocked=risk.tolist(), source_wins=(diff > .0002).sum(0).tolist(), worst=(-diff.min(0)).tolist())
    if not len(choices):
        return anchor, dict(info, reason="blocked_source")
    loss = .7 * risk[1:] / max(risk[0], 1e-9) + .3 * val[1:] / max(val[0], 1e-9)
    best = int(choices[np.argmin(loss[choices])])
    selected = expression(anchor, forms[best], y_scaler)
    np.testing.assert_allclose(selected.predict(V)[0], predicted[:, best + 1], rtol=.001, atol=3e-4)
    return selected, dict(info, accepted=True, variant=forms[best][0], nodes=selected.count_unique_nodes())
