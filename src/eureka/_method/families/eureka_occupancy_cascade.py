"""Try an ordinal response before a bounded residual correction for count sensors."""

import numpy as np

from eureka._method.families import eureka_ordinal_response as ordinal
from eureka._method.families import eureka_residual_calibration as calibration


def supported(X, y):
    return X.shape[1] == 18 and ordinal.supported(X, np.ravel(y))


def within_expression_budget(program):
    pending = list(program.terms)
    for _ in range(480):
        if not pending:
            return True
        node = pending.pop()
        pending.extend(node.children)
    return not pending


def select(anchor, X, y, env, V, v, val_env):
    if not supported(X, y):
        return anchor, dict(accepted=False, reason="count_sensor_support")
    if not within_expression_budget(anchor):
        return anchor, dict(accepted=False, reason="expression_budget")
    response, ordered = ordinal.select(anchor, X, y, env, V, v, val_env)
    if ordered["accepted"]:
        return response, dict(accepted=True, family="ordinal_response", source=ordered)
    corrected, residual = calibration.select(anchor, X, y, env, V, v, val_env)
    return corrected, dict(accepted=residual["accepted"], family="residual_calibration",
                           source=residual, ordinal=ordered)
