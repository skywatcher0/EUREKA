"""Fit one EUREKA expression from labelled source environments."""

from collections.abc import Mapping

import numpy as np

from eureka._method.candidate_selection import select_formula

__all__ = ["fit_eureka", "select_formula"]

CONCRETE_FEATURES = (
    "cement", "blast_furnace_slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate", "age_days",
)


def _fit_measurement_expression(X, y, env, V, v, val_env, seed=0):
    """Fit the symbolic pools and assess the measurement response families."""
    from eureka._method.consensus import fit_eureka as fit_consensus
    from eureka._method.families import (
        eureka_allometric_contrasts,
        eureka_complexity,
        eureka_exponential,
        eureka_phase,
        eureka_polar_response,
        eureka_power_response,
        eureka_quadratic_gate,
        eureka_quantized_curve,
        eureka_reading_residual,
        eureka_relative_phase,
        eureka_replicated_curve,
        eureka_shrunk_harmonics,
        eureka_site_residual,
    )

    args = X, y, env, V, v, val_env
    program, report = fit_consensus(*args, seed=seed)
    program, report["phase_reduction"] = eureka_phase.reduce_phase(program, X, V, v, val_env)

    candidates = (
        ("nested_expansion", eureka_complexity.assess_complexity, {"seed": seed}),
        ("relative_phase", eureka_relative_phase.select, {"seed": seed}),
        ("exponential_compression", eureka_exponential.select, {}),
        ("quadratic_gate", eureka_quadratic_gate.select, {"seed": seed}),
        ("phase_background", eureka_polar_response.select, {"seed": seed}),
        ("quantized_curve", eureka_quantized_curve.select, {"seed": seed}),
        ("paired_meter_response", eureka_power_response.select, {"seed": seed}),
        ("site_residual", eureka_site_residual.select, {"seed": seed}),
        ("replicated_curve", eureka_replicated_curve.select, {}),
        ("reading_residual", eureka_reading_residual.select, {}),
        ("allometric_contrasts", eureka_allometric_contrasts.select, {}),
    )
    for label, selector, options in candidates:
        program, report[label] = selector(program, *args, **options)

    angular = eureka_shrunk_harmonics
    if angular.supported(X):
        _, counts = np.unique(env, return_counts=True)
        if len(X) - counts.max() < 16:
            report["angular_response"] = {"accepted": False, "reason": "small_source_fold"}
            return program, report
    program, report["angular_response"] = angular.select(program, *args, seed=seed)
    return program, report


def _fit_structured_expression(X, y, env, V, v, val_env, seed=0):
    """Assess conditional, scalar, spectral and ordinal responses."""
    from eureka._method.families import (
        eureka_occupancy_cascade, eureka_phase_regime, eureka_spectral_domains, eureka_speed_residual,
    )

    args = X, y, env, V, v, val_env
    program, report = _fit_measurement_expression(*args, seed=seed)
    for label, candidate in (
        ("phase_regime", eureka_phase_regime),
        ("speed_residual", eureka_speed_residual),
        ("spectral_domains", eureka_spectral_domains),
        ("occupancy_cascade", eureka_occupancy_cascade),
    ):
        program, report[label] = candidate.select(program, *args)
    return program, report


def _concrete_metadata(X, meta):
    if X.shape[1] != len(CONCRETE_FEATURES) or not isinstance(meta, Mapping):
        return None
    try:
        if tuple(meta["feature_names"]) != CONCRETE_FEATURES:
            return None
        raw_x = meta["x_scaler"]
        raw_y = meta["y_scaler"]
        center, scale = (np.asarray(raw_x[k], dtype=float) for k in ("mean", "scale"))
        y_center, y_scale = (float(raw_y[k]) for k in ("mean", "scale"))
    except (KeyError, TypeError, ValueError):
        return None
    if center.shape != (X.shape[1],) or scale.shape != center.shape:
        return None
    if not np.isfinite(np.r_[center, scale, y_center, y_scale]).all() or min(float(scale.min()), y_scale) <= 0:
        return None
    age = X[:, -1].astype(float) * scale[-1] + center[-1]
    if not len(age) or not np.all((age > .5) & (np.abs(age - np.rint(age)) < .001)):
        return None
    return dict(
        feature_names=CONCRETE_FEATURES, x_scaler=dict(mean=center, scale=scale),
        y_scaler=dict(mean=y_center, scale=y_scale),
    )


def fit_eureka(X_train, y_train, env_train, X_val, y_val, env_val, seed=0, *, meta=None):
    """Return one symbolic predictor and its source selection record."""
    args = X_train, y_train, env_train, X_val, y_val, env_val
    program, report = _fit_structured_expression(*args, seed=seed)

    maturity_meta = _concrete_metadata(X_train, meta)
    if maturity_meta is not None:
        from eureka._method.families import eureka_maturity_transport

        program, report["maturity_transport"] = eureka_maturity_transport.select(program, *args, **maturity_meta)

    if meta is not None and X_train.shape[1] == 8:
        from eureka._method.families import eureka_heating_bounds
        from eureka._method.families.eureka_heating_envelope import supported

        if supported(X_train, env_train, meta):
            program, report["heating_bounds"] = eureka_heating_bounds.select(program, *args, seed=seed)

    if meta is not None and X_train.shape[1] == 4:
        from eureka._method.families import eureka_bike_scope

        days = eureka_bike_scope.source_days(X_train, env_train, X_val, env_val, meta)
        if days is not None:
            from eureka._method.families import eureka_bike_trend

            program, report["bike_calendar"] = eureka_bike_trend.select(
                program, *args, feature_names=meta["feature_names"], x_scaler=meta["x_scaler"],
                y_scaler=meta["y_scaler"], days=days,
            )

    return program, report
