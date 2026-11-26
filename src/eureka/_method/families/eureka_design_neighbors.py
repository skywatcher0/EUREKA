"""Transfer crossed-factor effects among nearby replicated design settings."""
import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.eureka_design_response import design_key
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def profile(X, env):
    key = design_key(X, env)
    if key is None:
        return None
    levels, ids = np.unique(X[:, key], return_inverse=True)
    if not 6 <= len(levels) <= 20:
        return None
    fixed, crossed = [], []
    for j, column in enumerate(X.T):
        sizes = [len(np.unique(column[ids == k])) for k in range(len(levels))]
        count = len(np.unique(column))
        if count >= 2 and max(sizes) == 1:
            fixed.append(j)
        if 2 <= count <= 12 and min(sizes) >= 2:
            crossed.append(j)
    if len(fixed) < 2 or not crossed:
        return None
    return key, fixed, crossed


def index_terms(X, axes):
    terms = []
    for axis in axes:
        levels = np.unique(X[:, axis])
        terms.extend(expr("gt", variable(axis), constant((float(a)+float(b))/2))
                     for a, b in zip(levels, levels[1:]))
    return terms


def fit(X, y, structure):
    key, fixed, crossed = structure
    levels, index = np.unique(X[:, key], return_inverse=True)
    terms = index_terms(X, crossed)
    design = np.column_stack([np.ones(len(y)), *[t.evaluate(X) for t in terms]]).astype(float)
    centers, coefficients = [], []
    for group in range(len(levels)):
        mask = index == group
        A = design[mask]
        gram = A.T @ A
        gram.flat[::len(gram)+1] += np.r_[1e-8, np.full(len(terms), .01*len(A))]
        coefficients.append(np.linalg.solve(gram, A.T @ y[mask]))
        centers.append(X[mask][0, fixed])
    centers = np.asarray(centers, float)
    scale = np.maximum(centers.std(0), 1e-6)
    # Distinct binary regimes have a larger mismatch cost than ordered coordinates.
    scale /= np.array([2 if len(np.unique(c)) == 2 else 1 for c in centers.T])
    return fixed, centers, scale, terms, np.array(coefficients)


def predict(fitted, X):
    fixed, centers, scale, terms, coeff = fitted
    delta = (X[:, None, fixed].astype(float)-centers)/scale
    weights = (1+np.mean(delta**2, axis=-1)/.25)**-2
    effects = np.column_stack([np.ones(len(X)), *[t.evaluate(X) for t in terms]]) @ coeff.T
    return (weights*effects).sum(axis=1)/weights.sum(axis=1)


def sum_nodes(nodes):
    result = nodes[0]
    for node in nodes[1:]:
        result = expr("+", result, node)
    return result


def anchor_node(fitted):
    fixed, centers, scale, terms, coeff = fitted
    weights, products = [], []
    for center, beta in zip(centers, coeff):
        squared = []
        for axis, offset, sd in zip(fixed, center, scale):
            u = expr("*", constant(1/sd), expr("-", variable(axis), constant(offset)))
            squared.append(expr("*", u, u))
        distance = expr("+", constant(1.), expr("*", constant(4/len(fixed)), sum_nodes(squared)))
        weight = expr("/", constant(1.), expr("*", distance, distance))
        response = sum_nodes([constant(beta[0]), *[expr("*", constant(c), t) for c, t in zip(beta[1:], terms)]])
        weights.append(weight)
        products.append(expr("*", weight, response))
    return expr("/", sum_nodes(products), sum_nodes(weights))


def model_node(base):
    coefs = iter(base.coeffs)
    def embed(term):
        if base.coefficient_mode == "conditional_affine" and term.op == "ifelse":
            gate, left, right = term.children
            return expr("ifelse", gate, embed(left), embed(right))
        return expr("*", constant(next(coefs)), term)
    value = sum_nodes([constant(base.intercept), *[embed(term) for term in base.terms]])
    assert next(coefs, None) is None
    return value


def select(base, X, y, env, V, v, val_env):
    structure = profile(X, np.ravel(env))
    if structure is None or base.coeffs is None:
        return base, dict(accepted=False, reason="support")
    y, v = np.ravel(y).astype(float), np.ravel(v).astype(float)
    key = structure[0]
    levels = np.unique(X[:, key])
    folds = []
    for width in (1, 2):
        for start in range(len(levels)-width+1):
            held = levels[start:start+width]
            keep, valid = ~np.isin(X[:, key], held), np.isin(V[:, key], held)
            if not valid.any():
                continue
            anchor = predict(fit(X[keep], y[keep], structure), V[valid])
            current = refit(base, X[keep], y[keep]).predict(V[valid])[0]
            folds.append((current, anchor, v[valid]))
    if len(folds) < len(levels):
        return base, dict(accepted=False, reason="missing_folds")
    # Estimate the scale of an unsupported design from genuinely held geometries.
    deviation = np.concatenate([np.abs(anchor-truth) for _, anchor, truth in folds])
    fitted = fit(X, y, structure)
    anchor = predict(fitted, V)
    current = base.predict(V)[0]
    max_disagreement = max(np.max(np.abs(p-a)) for p, a, _ in folds)
    max_disagreement = max(max_disagreement, np.max(np.abs(current-anchor)))
    margins = np.r_[np.maximum(np.quantile(deviation, [.65, .90, .99]), .05*np.std(y)),
                    1.25*max_disagreement+1e-4]
    def alternatives(p, a):
        return [p, a, *[np.where(np.abs(p-a)>m, a, p) for m in margins]]
    def errors(p, a, truth):
        return [np.mean((trial-truth)**2) for trial in alternatives(p, a)]
    blocked = np.sqrt(np.mean([errors(*fold) for fold in folds], axis=0))
    validation = np.sqrt(errors(current, anchor, v))
    valid = np.flatnonzero((blocked[1:] <= blocked[0]+1e-7) &
                           (validation[1:] <= validation[0]+.02*np.std(y)))
    report = dict(accepted=bool(len(valid)), margins=margins.tolist(), blocked=blocked.tolist(),
                  validation=validation.tolist(), folds=len(folds))
    if not len(valid):
        return base, report
    # Break indistinguishable scores in favor of the widest correction threshold.
    best = min(blocked[1:][valid])
    choice = int(valid[blocked[1:][valid] <= best+1e-7][-1])
    node = anchor_node(fitted)
    if choice:
        incumbent = model_node(base)
        margin = constant(margins[choice-1])
        high = expr("gt", incumbent, expr("+", node, margin))
        low = expr("lt", incumbent, expr("-", node, margin))
        node = expr("ifelse", expr("+", high, low), node, incumbent)
    selected = Program([node], np.ones(1), 0.)
    for values in (X, V):
        expected = alternatives(base.predict(values)[0], predict(fitted, values))[choice+1]
        np.testing.assert_allclose(selected.predict(values)[0], expected, rtol=2e-4, atol=2e-4)
    return selected, dict(report, choice=choice, nodes=selected.count_unique_nodes())
