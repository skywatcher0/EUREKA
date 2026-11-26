"""Reuse a source-selected spectrum where the geometry has source support."""

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families import eureka_swept_response as sweep
from eureka._method.families.eureka_design_neighbors import model_node, sum_nodes
from eureka._symbolic.core.ast import Program


def support(X, env):
    if X.shape[1] != 5:
        return None
    sweep_axes = sweep.axes(X, env)
    if sweep_axes is None:
        return None
    axes = [j for j, column in enumerate(X.T) if j not in sweep_axes
            and 2 <= len(np.unique(column)) <= 32]
    if len(axes) != 2:
        return None
    keys = np.unique(X[:, axes], axis=0)
    return (axes, keys) if 6 <= len(keys) <= 48 else None


def select(anchor, X, y, env, V, v, val_env):
    profile = support(X, env)
    if profile is None or anchor.coeffs is None:
        return anchor, {"accepted": False, "reason": "source_support"}
    alternative, decision = sweep.select(anchor, X, y, env, V, v, val_env)
    if not decision["accepted"]:
        return anchor, {"accepted": False, "reason": "source_spectrum", "spectrum": decision}
    axes, keys = profile
    j, k = axes
    flags = [expr("*", expr("eq", variable(j), constant(a)), expr("eq", variable(k), constant(b)))
             for a, b in keys]
    seen = sum_nodes(flags)
    model = Program([expr("ifelse", seen, model_node(alternative), model_node(anchor))], np.ones(1), 0.)
    matches = np.any(np.all(np.abs(V[:, None, axes] - keys[None, :, :]) < 1e-4, axis=2), axis=1)
    expected = np.where(matches, alternative.predict(V)[0], anchor.predict(V)[0])
    np.testing.assert_allclose(model.predict(V)[0], expected, rtol=4e-4, atol=5e-5)
    return model, {"accepted": True, "geometry_axes": axes, "geometries": len(keys), "spectrum": decision,
                   "validation_seen_rows": int(matches.sum()), "nodes": model.count_unique_nodes()}
