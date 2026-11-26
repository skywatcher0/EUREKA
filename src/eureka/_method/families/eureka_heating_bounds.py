"""Use a source geometry profile when exterior heating predictions are implausible."""

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families import eureka_heating_envelope as envelope
from eureka._method.families.eureka_design_neighbors import anchor_node, fit, model_node, profile, predict
from eureka._method.families.eureka_source_surfaces import bounded, minus
from eureka._symbolic.core.ast import Program


def extend(current, X, y, env, V, decision):
    structure = profile(X, np.ravel(env))
    if structure is None or structure[0] != 0 or current.coeffs is None:
        return current, dict(accepted=decision["accepted"], envelope=decision, reason="replicated_geometry_support")
    y = np.ravel(y).astype(float)
    lo, hi = np.quantile(y, [.025, .975])
    margin = .42 * (hi - lo)
    floor, cap = float(lo - margin), float(hi + margin)
    if not floor < cap:
        return current, dict(accepted=decision["accepted"], envelope=decision, reason="source_range")
    left, right = float(X[:, 0].min()), float(X[:, 0].max())
    estimate = fit(X, y, structure)
    previous = model_node(current)
    coordinate = variable(0)
    exterior = expr("+", expr("lt", coordinate, constant(left - 1e-4)), expr("gt", coordinate, constant(right + 1e-4)))
    extreme = expr("+", expr("lt", previous, constant(floor)), expr("gt", previous, constant(cap)))
    active = expr("*", exterior, extreme)
    replacement = bounded(anchor_node(estimate), floor, cap)
    correction = expr("ifelse", active, minus(replacement, previous), constant(0.))
    if current.coefficient_mode == "conditional_affine":
        correction = expr("*", constant(1.), correction)
    program = Program([*current.terms, correction], np.r_[current.coeffs, 1.], current.intercept, current.coefficient_mode)
    base = current.predict(V)[0]
    mask = ((V[:, 0] < left - 1e-4) | (V[:, 0] > right + 1e-4)) & ((base < floor) | (base > cap))
    expected = np.where(mask, np.clip(predict(estimate, V), floor, cap), base)
    np.testing.assert_allclose(program.predict(V)[0], expected, atol=1e-4, rtol=3e-4)
    return program, dict(accepted=True, envelope=decision, cutoff=[left, right], response_range=[floor, cap],
                         variant="exterior_source_bounds", nodes=program.count_unique_nodes())


def select(anchor, X, y, env, V, v, val_env, *, seed=0):
    current, decision = envelope.select(anchor, X, y, env, V, v, val_env, seed=seed)
    return extend(current, X, y, env, V, decision)
