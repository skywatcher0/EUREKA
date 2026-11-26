"""Use geometry-peer transport only beyond the source compactness envelope."""

from collections.abc import Mapping

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families import eureka_geometry_agreement as peers
from eureka._symbolic.core.ast import Program

FEATURES = ("relative_compactness", "surface_area", "wall_area", "roof_area", "overall_height",
            "orientation", "glazing_area", "glazing_area_distribution")
TARGET = "heating load (kWh/m2)"


def supported(X, env, meta):
    if X.shape[1] != len(FEATURES) or not isinstance(meta, Mapping):
        return False
    names = meta.get("feature_names", ())
    if not isinstance(names, (list, tuple)) or tuple(names) != FEATURES or meta.get("target_name") != TARGET:
        return False
    design = peers.neighbor.profile(X, np.ravel(env))
    return design is not None and design[0] == 0


def scoped(anchor, proposed, X, V, decision):
    if not decision["accepted"]:
        return anchor, dict(accepted=False, peer_selection=decision)
    assert proposed.coefficient_mode == anchor.coefficient_mode and len(proposed.terms) == len(anchor.terms) + 1
    lo, hi = float(X[:, 0].min()), float(X[:, 0].max())
    axis = variable(0)
    use = expr("+", expr("lt", axis, constant(lo - 1e-4)), expr("gt", axis, constant(hi + 1e-4)))
    delta = expr("ifelse", use, proposed.terms[-1], constant(0.))
    if proposed.coefficient_mode == "conditional_affine":
        delta = expr("*", constant(1.), delta)
    model = Program([*anchor.terms, delta], proposed.coeffs, proposed.intercept, proposed.coefficient_mode)
    outside = (V[:, 0] < lo - 1e-4) | (hi + 1e-4 < V[:, 0])
    expected = np.where(outside, proposed.predict(V)[0], anchor.predict(V)[0])
    np.testing.assert_allclose(model.predict(V)[0], expected, rtol=2e-4, atol=7e-5)
    return model, dict(accepted=True, compactness_support=[lo, hi], peer_selection=decision,
                       nodes=model.count_unique_nodes())


def select(anchor, X, y, env, V, v, val_env, *, seed=0):
    shape = peers.neighbor.profile(X, np.ravel(env))
    if shape is None or shape[0] != 0:
        return anchor, dict(accepted=False, reason="compactness_support")
    changed, decision = peers.select(anchor, X, y, env, V, v, val_env, seed=seed)
    return scoped(anchor, changed, X, V, decision)
