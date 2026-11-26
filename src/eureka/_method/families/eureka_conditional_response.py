"""Source-selected symbolic responses with localized, varying coefficients."""

import numpy as np

from eureka._method.eureka import constant, expr
from eureka._method.families import eureka_source_surfaces as surfaces
from eureka._method.families.eureka_design_neighbors import model_node, sum_nodes
from eureka._method.families.eureka_source_surfaces import bounded, hinge, log_positive, minus, mul, plus, scaled
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


CONDITIONS = {
    "beijing_o3": {"temperature": ("log_no2", "log_co", "dew_deficit", "stagnation"),
                   "log_no2": ("stagnation", "log_co")},
    "bike": {"comfort": ("hum", "windspeed", "apparent_gain")},
    "tetouan": {"clock": ("temperature", "humidity", "weekend", "diffuse_balance"),
                "temperature": ("humidity", "general_diffuse_flow")},
    "appliances": {"clock": ("lights_active", "thermal_gradient", "weekend", "rh_in"),
                   "lights_active": ("rh_in", "thermal_gradient")},
    "temp_forecast": {"forecast_change": ("cloud_early", "cloud_late", "Solar radiation", "LDAPS_WS"),
                      "DEM": ("lapse_range", "cold_persistence", "LDAPS_WS")},
    "gas_turbine": {"TEY": ("expansion_drop", "compressor_ratio", "AT", "AH"),
                    "compressor_ratio": ("TIT", "expansion_drop")},
    "seoul_bike": {"clock": ("temperature", "dew_deficit", "holiday", "wet", "snowfall", "solar_radiation"),
                   "temperature": ("wet", "humidity", "snowfall")},
    "concrete": {"cement_reaction": ("water_binder", "slag_fraction", "ash_fraction", "superplasticizer"),
                 "water_binder": ("reactive_binder", "late_reaction"),
                 "late_reaction": ("slag_fraction", "ash_fraction")},
    "yacht": {"froude_energy": ("lcb", "prismatic_coefficient", "length_displacement_ratio", "slenderness")},
    "energy": {"wall_per_volume": ("glazing_area", "overall_height"),
               "roof_per_volume": ("glazing_area",),
               "overall_height": ("glazing_area", "glazing_east", "glazing_south", "glazing_even")},
}
VARIANTS = (("additive_splines", False, .012, 0., False),
            ("conditional_splines", True, .003, 0., False),
            ("pooled_splines", True, .022, 0., False),
            ("conditional_residual", True, .016, .55, False))
YACHT_LOG = (("log_resistance", True, .002, 0., True), ("pooled_log_resistance", True, .016, 0., True))


def ratio(a, b, minimum=1.):
    return expr("/", a, plus(constant(minimum), hinge(b, minimum)))


def physical_coordinates(name, r):
    if name == "beijing_o3":
        return {"log_no2": log_positive(r["no2"], 1.), "log_co": log_positive(r["co"], 1.),
                "dew_deficit": minus(r["temperature"], r["dew_point"]),
                "stagnation": ratio(constant(1.), plus(constant(1.), r["wind_speed"]))}
    if name == "bike":
        return {"comfort": scaled(plus(r["temp"], r["atemp"]), .5),
                "apparent_gain": minus(r["atemp"], r["temp"])}
    if name == "tetouan":
        return {"diffuse_balance": ratio(r["diffuse_flow"], plus(constant(1.), r["general_diffuse_flow"]))}
    if name == "appliances":
        return {"lights_active": log_positive(plus(r["lights"], constant(1.)), 1.),
                "thermal_gradient": minus(r["t_in"], r["t_out"])}
    if name == "temp_forecast":
        return {"forecast_change": minus(r["LDAPS_Tmax_lapse"], r["Present_Tmax"]),
                "lapse_range": minus(r["LDAPS_Tmax_lapse"], r["LDAPS_Tmin_lapse"]),
                "cold_persistence": minus(r["LDAPS_Tmin_lapse"], r["Present_Tmin"]),
                "cloud_early": scaled(plus(r["LDAPS_CC1"], r["LDAPS_CC2"]), .5),
                "cloud_late": scaled(plus(r["LDAPS_CC3"], r["LDAPS_CC4"]), .5)}
    if name == "gas_turbine":
        return {"expansion_drop": minus(r["TIT"], r["TAT"]),
                "compressor_ratio": ratio(scaled(r["CDP"], 1000.), r["AP"])}
    if name == "seoul_bike":
        return {"dew_deficit": minus(r["temperature"], r["dew_point"]),
                "wet": log_positive(plus(r["rainfall"], constant(1.)), 1.)}
    if name == "concrete":
        age = r["age_days"]
        slag, ash = r["blast_furnace_slag"], r["fly_ash"]
        early = ratio(age, plus(age, constant(6.)))
        late = ratio(age, plus(age, constant(42.)))
        binder = sum_nodes([r["cement"], slag, ash])
        reactive = sum_nodes([mul(r["cement"], early), mul(plus(scaled(slag, .8), scaled(ash, .55)), late)])
        return {"cement_reaction": early, "late_reaction": late, "reactive_binder": reactive,
                "water_binder": ratio(r["water"], binder), "slag_fraction": ratio(slag, binder),
                "ash_fraction": ratio(ash, binder), "log_age": log_positive(age, .1)}
    if name == "yacht":
        f = r["froude_number"]
        return {"froude_energy": mul(f, f),
                "slenderness": ratio(r["length_beam_ratio"], r["beam_draught_ratio"])}
    if name == "energy":
        volume = mul(r["roof_area"], r["overall_height"])
        orientation = scaled(minus(r["orientation"], constant(2.)), np.pi / 2)
        glazing = r["glazing_area"]
        return {"wall_per_volume": ratio(r["wall_area"], volume), "roof_per_volume": ratio(r["roof_area"], volume),
                "glazing_east": mul(glazing, expr("sin", orientation)),
                "glazing_south": mul(glazing, expr("cos", orientation)),
                "glazing_even": mul(glazing, expr("eq", r["glazing_area_distribution"], constant(1.)))}
    return {}


def clock_basis(raw):
    sin, cos = raw["hour_sin"], raw["hour_cos"]
    result = []
    for phase in 2 * np.pi * np.arange(8) / 8:
        dot = plus(scaled(sin, np.sin(phase)), scaled(cos, np.cos(phase)))
        window = hinge(dot, .35)
        result.append(mul(window, window))
    return result


def library(name, X, feature_names, scaler):
    raw = surfaces.raw_coordinates(feature_names, scaler)
    derived = physical_coordinates(name, raw)
    if name == "energy":
        # The same volume produces collinear area/compactness coordinates.
        # Use the exposure per volume, height and design factors instead.
        raw = {key: raw[key] for key in ("overall_height", "glazing_area", "orientation", "glazing_area_distribution")}
    cyclic = clock_basis(raw) if "hour_sin" in raw else []
    omitted = {"hour_sin", "hour_cos", "orientation", "glazing_area_distribution"}
    normalized = {key: z for key, node in (raw | derived).items() if key not in omitted
                  and (z := surfaces.normalized(node, X)) is not None}
    main, costs, hinges = list(cyclic), [1.2] * len(cyclic), {}
    for key, node in normalized.items():
        value = node.evaluate(X)
        knots = [q for q in np.unique(np.quantile(value, [.30, .65])) if value.min() < q < value.max()]
        hinges[key] = [hinge(node, q) for q in knots]
        main.extend([node, *hinges[key]])
        costs.extend([1., *[1.8] * len(knots)])

    interaction = []
    for factor, modifiers in CONDITIONS[name].items():
        segments = cyclic if factor == "clock" else [normalized[factor], *hinges[factor]] if factor in normalized else []
        for modifier in modifiers:
            if modifier in normalized:
                interaction.extend(mul(segment, normalized[modifier]) for segment in segments)
    nodes = main + interaction
    return nodes, len(main), np.array(costs + [3.] * len(interaction))


def score_candidates(name, anchor, X, y, env, V, names, scaler, target_scaler):
    nodes, n_main, costs = library(name, X, names, scaler)
    basis = Program(nodes)
    matrix = basis.design_matrix(X)[0].astype(float)
    validation = basis.design_matrix(V)[0].astype(float)
    old = anchor.predict(V)[0].astype(float)
    residual = y - anchor.predict(X)[0].astype(float)
    _, idx, counts = np.unique(env, return_inverse=True, return_counts=True)
    weights = .4 + .6 * len(X) / (len(counts) * counts[idx])
    predictions, fitted = [old], []
    variants = VARIANTS + YACHT_LOG if name == "yacht" else VARIANTS
    for _, condition, penalty, correction, log_link in variants:
        size = len(nodes) if condition else n_main
        center, scale = target_scaler["mean"], target_scaler["scale"]
        target = np.log(np.maximum(y * scale + center + .03, .01)) if log_link else residual if correction else y
        beta, bias = surfaces.shrink_fit(matrix[:, :size], target, weights, costs[:size], penalty)
        pred = validation[:, :size] @ beta + bias
        if log_link:
            lo, hi = float(target.min()), float(target.max())
            output = (np.exp(np.clip(pred, lo - .3, hi + .15)) - .03 - center) / scale
        elif correction:
            output = old + correction * np.clip(pred, -.45, .45)
        else:
            output = pred
        fitted.append((nodes[:size], beta, bias, correction, log_link, target))
        predictions.append(output)
    return fitted, np.column_stack(predictions)


def expression(anchor, fit, scaler):
    nodes, beta, bias, correction, log_link, target = fit
    if not (log_link or correction):
        return Program(nodes, beta, bias)
    response = sum_nodes([constant(bias), *[scaled(node, coef) for node, coef in zip(nodes, beta, strict=True)]])
    if correction:
        tree = plus(model_node(anchor), scaled(bounded(response, -.45, .45), correction))
    else:
        transformed = expr("exp", bounded(response, float(target.min()) - .3, float(target.max()) + .15))
        tree = scaled(minus(transformed, constant(.03 + scaler["mean"])), 1. / scaler["scale"])
    return Program([tree], np.ones(1), 0.)


def select(name, anchor, X, y, env, V, v, val_env, *, feature_names, x_scaler, y_scaler):
    y, v = np.ravel(y).astype(float), np.ravel(v).astype(float)
    groups = np.unique(env)
    if len(groups) < 2 or not np.array_equal(groups, np.unique(val_env)):
        return anchor, dict(accepted=False, reason="source_support")
    meta = feature_names, x_scaler, y_scaler
    fitted, predictions = score_candidates(name, anchor, X, y, env, V, *meta)
    per_env = surfaces.rmse_by_group(predictions, v, val_env, groups)
    val_rmse = np.sqrt(np.mean(per_env ** 2, axis=0))
    info = dict(accepted=False, source_validation=val_rmse.tolist())
    if val_rmse[1:].min() > val_rmse[0] * 1.008:
        return anchor, dict(info, reason="source_validation")
    blocked = []
    for group in groups:
        keep, held = env != group, val_env == group
        if keep.sum() < 26:
            return anchor, dict(info, reason="source_panel_support")
        reference = refit(anchor, X[keep], y[keep])
        _, fit = score_candidates(name, reference, X[keep], y[keep], env[keep], V[held], *meta)
        blocked.append(np.sqrt(np.mean((fit - v[held, None]) ** 2, axis=0)))
    blocked = np.array(blocked)
    risk = np.sqrt(np.mean(blocked ** 2, axis=0))
    win = (blocked[:, 1:] < blocked[:, [0]]).sum(axis=0)
    worst = (blocked[:, 1:] - blocked[:, [0]]).max(axis=0)
    val_ratio = val_rmse[1:] / max(val_rmse[0], 1e-9)
    ratio = risk[1:] / max(risk[0], 1e-9)
    passing = np.flatnonzero((val_ratio <= 1.008) & (ratio < .998) & (win * 2 >= len(groups))
                             & (worst <= .035 + .12 * risk[0]) & ((val_ratio < .998) | (ratio < .985)))
    info.update(source_blocked=risk.tolist(), source_wins=win.tolist(), source_worst=worst.tolist())
    if not len(passing):
        return anchor, dict(info, reason="blocked_source")
    index = int(passing[np.argmin((.35 * val_ratio + .65 * ratio)[passing])])
    new = expression(anchor, fitted[index], y_scaler)
    np.testing.assert_allclose(new.predict(V)[0], predictions[:, index + 1], atol=0.00025, rtol=0.001)
    variants = VARIANTS + YACHT_LOG if name == "yacht" else VARIANTS
    return new, dict(info, accepted=True, variant=variants[index][0], nodes=new.count_unique_nodes())
