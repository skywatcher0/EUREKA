"""Source-selected, regularized terms for EUREKA's closed-form expressions."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Node, Program


def mul(a, b):
    return expr("*", a, b)


def plus(a, b):
    return expr("+", a, b)


def minus(a, b):
    return expr("-", a, b)


def scaled(a, scale):
    return mul(constant(scale), a)


def hinge(a, knot):
    return mul(minus(a, constant(knot)), expr("gt", a, constant(knot)))


def bounded(a, lo, hi):
    return minus(plus(constant(lo), hinge(a, lo)), hinge(a, hi))


def log_positive(a, floor):
    return expr("log", plus(constant(floor), hinge(a, floor)))


@dataclass(frozen=True)
class Terms:
    main: list[Node]
    interactions: list[Node]


# Interactions have declared covariate meanings and are estimated from source rows.
PAIR_TERMS = (
    ("temp", "hum"), ("atemp", "windspeed"),
    ("temperature", "humidity"), ("temperature", "dew_point"),
    ("temperature", "no2"), ("temperature", "wind_speed"),
    ("no2", "wind_speed"), ("no2", "co"), ("co", "so2"),
    ("TIT", "GTEP"), ("TIT", "TEY"), ("TAT", "TEY"),
    ("AT", "TEY"), ("AFDP", "CDP"), ("AH", "TIT"),
    ("LDAPS_Tmax_lapse", "Solar radiation"), ("LDAPS_Tmax_lapse", "DEM"),
    ("Present_Tmax", "LDAPS_CC1"), ("Present_Tmin", "LDAPS_RHmax"),
    ("DEM", "LDAPS_WS"), ("LDAPS_RHmin", "LDAPS_CC2"),
    ("overall_height", "roof_area"), ("roof_area", "glazing_area"),
    ("wall_area", "glazing_area"), ("overall_height", "glazing_area"),
    ("S5_CO2", "S5_CO2_Slope"), ("S6_PIR", "S7_PIR"),
    ("S1_Light", "S4_Light"), ("lights", "rh_in"),
    ("t_in", "t_out"), ("rh_in", "rh_out"),
    ("solar_radiation", "temperature"), ("rainfall", "humidity"),
    ("general_diffuse_flow", "diffuse_flow"),
)


def raw_coordinates(names, scaler):
    means = scaler.get("mean", np.zeros(len(names)))
    scales = scaler.get("scale", np.ones(len(names)))
    return {name: plus(constant(mu), scaled(variable(j), sd))
            for j, (name, mu, sd) in enumerate(zip(names, means, scales, strict=True))}


def normalized(node, X):
    values = node.evaluate(X).astype(float)
    lo, hi = np.quantile(values, [0.005, 0.995])
    width = hi - lo
    scale = max(values.std(), 1e-6)
    if width < 1e-7:
        return None
    stable = bounded(node, lo - .12 * width, hi + .12 * width)
    return scaled(minus(stable, constant(values.mean())), 1. / scale)


def polynomial_basis(z, X):
    values = z.evaluate(X)
    knots = np.unique(np.quantile(values, [.25, .5, .75]))
    return [z, *[hinge(z, q) for q in knots if values.min() < q < values.max()]]


def physics_terms(raw):
    """Derived coordinates are fixed functions of observed inputs."""
    derived = {}
    if "frequency_hz" in raw:
        lf = log_positive(raw["frequency_hz"], 1.)
        lv = log_positive(raw["free_stream_velocity_ms"], .01)
        ld = log_positive(raw["suction_side_displacement_thickness_m"], 1e-7)
        lc = log_positive(raw["chord_length_m"], 1e-7)
        derived.update(log_frequency=lf, log_thickness=ld,
                       thickness_frequency=minus(plus(lf, ld), lv),
                       chord_frequency=minus(plus(lf, lc), lv))
    if "froude_number" in raw:
        f = raw["froude_number"]
        f2 = mul(f, f)
        derived.update(froude_squared=f2, froude_fourth=mul(f2, f2))
    if "age_days" in raw:
        age = raw["age_days"]
        binder = plus(raw["cement"], plus(raw["blast_furnace_slag"], raw["fly_ash"]))
        derived.update(log_age=log_positive(age, .1),
                       age_saturation=expr("/", age, plus(age, constant(14.))),
                       water_binder=expr("/", raw["water"], plus(binder, constant(1.))),
                       slag_binder=expr("/", raw["blast_furnace_slag"], plus(binder, constant(1.))),
                       ash_binder=expr("/", raw["fly_ash"], plus(binder, constant(1.))),
                       binder_total=binder)
    if "glazing_area" in raw:
        derived["exterior_glazing"] = mul(raw["wall_area"], raw["glazing_area"])
        derived["roof_fraction"] = expr("/", raw["roof_area"], raw["surface_area"])
    if "no2" in raw:
        for name in ("no2", "co", "so2", "wind_speed"):
            derived["log_" + name] = log_positive(plus(raw[name], constant(1.)), 1.)
        derived["temperature_dew_gap"] = minus(raw["temperature"], raw["dew_point"])
    if "LDAPS_Tmax_lapse" in raw:
        derived["max_temperature_gap"] = minus(raw["LDAPS_Tmax_lapse"], raw["Present_Tmax"])
        derived["temperature_range"] = minus(raw["Present_Tmax"], raw["Present_Tmin"])
    if "TIT" in raw:
        derived["turbine_temperature_drop"] = minus(raw["TIT"], raw["TAT"])
    if "S5_CO2" in raw:
        derived["pir_sum"] = plus(raw["S6_PIR"], raw["S7_PIR"])
        derived["light_contrast"] = minus(raw["S1_Light"], raw["S4_Light"])
    if "t_in" in raw:
        derived["inside_outside_temperature"] = minus(raw["t_in"], raw["t_out"])
    return derived


def periodic_terms(raw, z):
    if "hour_sin" not in raw or "hour_cos" not in raw:
        return [], []
    sine, cosine = raw["hour_sin"], raw["hour_cos"]
    harmonics = [sine, cosine]
    def horner(coefficients):
        result = constant(coefficients[-1])
        for coefficient in coefficients[-2::-1]:
            result = plus(constant(coefficient), mul(cosine, result))
        return result

    for degree in range(2, 7):
        chebyshev = np.zeros(degree + 1)
        chebyshev[-1] = 1.
        cs = np.polynomial.chebyshev.cheb2poly(chebyshev)
        sn = np.arange(1, len(cs)) * cs[1:] / degree
        harmonics.extend((mul(sine, horner(sn)), horner(cs)))
    wave_interactions = []
    for name in ("weekend", "holiday", "temperature", "humidity", "t_in", "t_out",
                 "rainfall", "solar_radiation", "general_diffuse_flow", "lights", "pir_sum"):
        if name in z:
            order = 8 if name in ("weekend", "holiday") else 4
            wave_interactions.extend(mul(harmonic, z[name]) for harmonic in harmonics[:order])
    return harmonics, wave_interactions


def library(X, names, scaler):
    raw = raw_coordinates(names, scaler)
    raw.update(physics_terms(raw))
    coordinates = {name: node for name, leaf in raw.items()
                   if (node := normalized(leaf, X)) is not None}
    main = []
    for name, node in coordinates.items():
        if name in ("hour_sin", "hour_cos", "orientation", "glazing_area_distribution"):
            continue
        main.extend(polynomial_basis(node, X))
    for name in ("orientation", "glazing_area_distribution"):
        if name in raw:
            levels = np.unique(np.rint(raw[name].evaluate(X)))
            main.extend(expr("eq", raw[name], constant(level)) for level in levels[1:])

    pairs = list(PAIR_TERMS)
    for name in ("thickness_frequency", "chord_frequency", "log_frequency"):
        pairs.extend((name, geometry) for geometry in (
            "angle_of_attack_deg", "chord_length_m", "free_stream_velocity_ms", "log_thickness"))
    for name in ("froude_squared", "froude_fourth"):
        pairs.extend((name, geometry) for geometry in (
            "lcb", "prismatic_coefficient", "length_displacement_ratio", "beam_draught_ratio", "length_beam_ratio"))
    pairs.extend(("log_age", ingredient) for ingredient in (
        "water_binder", "binder_total", "slag_binder", "ash_binder", "superplasticizer"))
    interactions = [mul(coordinates[a], coordinates[b]) for a, b in pairs
                    if a in coordinates and b in coordinates]
    waves, varying = periodic_terms(raw, coordinates)
    main.extend(waves)
    interactions.extend(varying)
    return Terms(main, interactions)


def speed_terms(X):
    v = variable(0)
    x = X[:, 0].astype(float)
    scale = max(float(np.quantile(x, .95)), .01)
    speed = scaled(v, 1 / scale)
    square = mul(speed, speed)
    cubic = mul(square, speed)
    change = np.unique(np.quantile(x / scale, [.15, .35, .55, .75, .9]))
    width = max(.12 * np.ptp(x / scale), .025)
    local = []
    for knot in change:
        d = scaled(minus(speed, constant(knot)), 1 / width)
        local.append(expr("exp", scaled(mul(d, d), -.5)))
    hinges = [hinge(speed, knot) for knot in change]
    return Terms([speed, square, cubic], [*hinges, *local])


def shrink_fit(matrix, target, weights, costs, penalty):
    scale = np.maximum(matrix.std(axis=0), 1e-5)
    center = np.average(matrix, axis=0, weights=weights)
    design = (matrix - center) / scale
    offset = np.average(target, weights=weights)
    gram = design.T @ (weights[:, None] * design)
    gram.flat[::len(gram) + 1] += weights.sum() * penalty * costs
    coef = solve(gram, design.T @ (weights * (target - offset)), assume_a="pos")
    beta = coef / scale
    return beta, float(offset - center @ beta)


def add_formula(anchor, terms, beta, intercept):
    if anchor is None:
        return Program(terms, beta, intercept)
    if anchor.coefficient_mode == "conditional_affine":
        terms = [scaled(term, 1.) for term in terms]
    weights = anchor.coeffs if anchor.coeffs is not None else np.ones(anchor.coefficient_count())
    offset = anchor.intercept if anchor.coeffs is not None else 0.
    return Program(anchor.terms + terms, np.r_[weights, beta], offset + intercept, anchor.coefficient_mode)


VARIANTS = (
    ("additive", False, .014, 0.),
    ("interactions", True, .007, 0.),
    ("pooled_interactions", True, .055, 0.),
    ("residual_half", True, .024, .5),
    ("residual_full", True, .024, 1.),
)


def candidates(anchor, X, y, env, V, names, scaler):
    terms = speed_terms(X) if names == ("rpm_in_normalized",) else library(X, names, scaler)
    reference = anchor.predict(V)[0].astype(float)
    if not terms.main:
        return [anchor] * len(VARIANTS), np.column_stack([reference] * (len(VARIANTS) + 1))
    columns = terms.main + terms.interactions
    basis = Program(columns)
    design = basis.design_matrix(X)[0].astype(float)
    validation = basis.design_matrix(V)[0].astype(float)
    nmain = len(terms.main)
    costs = np.r_[np.ones(nmain), np.full(len(terms.interactions), 2.5)]
    _, idx, counts = np.unique(env, return_inverse=True, return_counts=True)
    weights = .5 + .5 * len(X) / (len(counts) * counts[idx])
    residual = y - anchor.predict(X)[0]
    forms, predictions = [], [reference]
    for _, interaction, penalty, correction in VARIANTS:
        size = len(columns) if interaction else nmain
        beta, bias = shrink_fit(design[:, :size], residual if correction else y, weights, costs[:size], penalty)
        if correction:
            forms.append(add_formula(anchor, columns[:size], correction * beta, correction * bias))
            predictions.append(reference + correction * (validation[:, :size] @ beta + bias))
        else:
            forms.append(add_formula(None, columns[:size], beta, bias))
            predictions.append(validation[:, :size] @ beta + bias)
    return forms, np.column_stack(predictions)


def rmse_by_group(predictions, y, env, groups):
    return np.sqrt(np.array([np.mean((predictions[env == g] - y[env == g, None]) ** 2, axis=0)
                             for g in groups]))


def select(anchor, X, y, env, V, v, val_env, *, feature_names, x_scaler=None):
    """Return one expression. All fitting and choice use source arrays only."""
    names = tuple(feature_names)
    scaler = x_scaler or {}
    y, v = np.asarray(y, float).ravel(), np.asarray(v, float).ravel()
    groups = np.unique(env)
    if len(names) != X.shape[1] or len(groups) < 2 or not np.array_equal(groups, np.unique(val_env)):
        return anchor, {"accepted": False, "reason": "source_support"}
    full, vp = candidates(anchor, X, y, env, V, names, scaler)
    per_group = rmse_by_group(vp, v, val_env, groups)
    validation = np.sqrt(np.mean(per_group ** 2, axis=0))
    report = {"accepted": False, "validation": validation.tolist()}
    if min(validation[1:]) >= .998 * validation[0]:
        return anchor, {**report, "reason": "source_validation"}
    folds = []
    for group in groups:
        tr, va = env != group, val_env == group
        if tr.sum() < 30:
            return anchor, {**report, "reason": "source_fold_support"}
        old = refit(anchor, X[tr], y[tr])
        _, P = candidates(old, X[tr], y[tr], env[tr], V[va], names, scaler)
        folds.append(np.sqrt(np.mean((P - v[va, None]) ** 2, axis=0)))
    folds = np.array(folds)
    blocked = np.sqrt(np.mean(folds ** 2, axis=0))
    wins = (folds[:, 1:] < folds[:, [0]]).sum(axis=0)
    eligible = np.flatnonzero((validation[1:] < .998 * validation[0])
                              & (blocked[1:] < .998 * blocked[0])
                              & (wins * 2 >= len(groups)))
    report.update(blocked=blocked.tolist(), source_fold_wins=wins.tolist(), source_folds=len(groups))
    if not len(eligible):
        return anchor, {**report, "reason": "blocked_source_validation"}
    loss = validation[1:] / max(validation[0], 1e-9) + blocked[1:] / max(blocked[0], 1e-9)
    winner = int(eligible[np.argmin(loss[eligible])])
    selected = full[winner]
    np.testing.assert_allclose(selected.predict(V)[0], vp[:, winner + 1], rtol=5e-4, atol=5e-5)
    return selected, {**report, "accepted": True, "variant": VARIANTS[winner][0],
                      "nodes": selected.count_unique_nodes()}
