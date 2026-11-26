"""Regularized main effects and interactions for replicated, blocked designs."""
from itertools import combinations

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.eureka_factorial import mse, supported
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def design_key(X, env):
    if not supported(X) or len(np.unique(env)) < 2:
        return None
    for j, values in enumerate(X.T):
        levels, count = np.unique(values, return_counts=True)
        if 5 <= len(levels) <= 30 and np.min(count) >= 5:
            if all(len(np.unique(env[values == v])) == 1 for v in levels):
                return j
    return None


def surface(X, y, env, penalty):
    terms = []
    low = []
    for j, values in enumerate(X.T):
        node = variable(j)
        levels = np.unique(values)
        if len(levels) < 2:
            continue
        group_sizes = [len(np.unique(values[env == g])) for g in np.unique(env)]
        # Encode crossed settings with bounded contrasts. Ordered geometry keeps
        # its linear trend; unrestricted geometry indicators cannot extrapolate.
        if 2 < len(levels) <= 8 and min(group_sizes) == len(levels):
            nodes = [expr("eq", node, constant(v)) for v in levels[1:]]
        else:
            nodes = [node]
        terms.extend(nodes)
        low.append((nodes, len(levels) == 2))
    for (left, first_binary), (right, second_binary) in combinations(low, 2):
        if first_binary or second_binary:
            terms.extend(expr("*", a, b) for a in left for b in right)
    design = np.column_stack([n.evaluate(X) for n in terms]).astype(float)
    mean, scale = design.mean(axis=0), design.std(axis=0)
    scale[scale < 1e-6] = 1.
    Z = (design - mean) / scale
    gram = Z.T @ Z
    gram.flat[::len(gram)+1] += penalty * len(Z)
    y = np.ravel(y).astype(float)
    coef = np.linalg.solve(gram, Z.T @ (y-y.mean())) / scale
    return Program(terms, coef, float(y.mean()-mean@coef))


def select(base, X, y, env, V, v, val_env):
    env, val_env = np.ravel(env), np.ravel(val_env)
    key = design_key(X, env)
    if key is None:
        return base, dict(accepted=False, reason="no_replicated_blocks")
    penalties = (.001, .01, .1)
    full = [base, *(surface(X, y, env, a) for a in penalties)]
    val = np.sqrt([mse(m, V, v) for m in full])
    fold_errors = []
    for value in np.unique(X[:, key]):
        keep, held = X[:, key] != value, V[:, key] == value
        if not np.any(held):
            continue
        fitted = [refit(base, X[keep], y[keep]),
                  *(surface(X[keep], y[keep], env[keep], a) for a in penalties)]
        fold_errors.append([mse(m, V[held], v[held]) for m in fitted])
    if len(fold_errors) < 4:
        return base, dict(accepted=False, reason="insufficient_block_validation")
    errors = np.array(fold_errors)
    blocked = np.sqrt(errors.mean(axis=0))
    wins = (errors[:,1:] < errors[:,[0]]).sum(axis=0)
    eligible = np.flatnonzero((blocked[1:] < .98*blocked[0]) &
                              (wins > len(errors)/2) &
                              (val[1:] <= val[0] + .08*np.std(y)))
    report = dict(accepted=bool(len(eligible)), key=key, validation=val.tolist(),
                  blocked=blocked.tolist(), wins=wins.tolist(), folds=len(errors))
    if len(eligible):
        index = eligible[np.argmin(blocked[1:][eligible])]
        report["penalty"] = penalties[index]
        return full[index+1], report
    return base, report
