"""Estimate the conditional mean of a finite ordered response with sigmoid terms."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def supported(X, y):
    if not (350 <= len(X) <= 25000 and 3 <= X.shape[1] <= 30):
        return False
    levels, counts = np.unique(y, return_counts=True)
    if not 2 <= len(levels) <= 8 or counts.min() < 8:
        return False
    spacing = np.diff(levels.astype(float))
    return np.allclose(spacing, spacing.mean(), rtol=1e-4, atol=1e-7)


def basis(X, labels, weights):
    terms = []
    for j, column in enumerate(X.T):
        levels = np.unique(column)
        if len(levels) < 2:
            continue
        node = variable(j)
        if len(levels) <= 8:
            terms.extend(expr("gt", node, constant((float(lo)+float(hi))/2))
                         for lo, hi in zip(levels, levels[1:]))
            continue
        cut = np.quantile(column, [.01, .25, .5, .75, .99])
        low, high = float(cut[0]), float(cut[-1])
        if high-low <= 1e-6:
            continue
        x = expr("ifelse", expr("lt", node, constant(low)), constant(low),
                 expr("ifelse", expr("gt", node, constant(high)), constant(high), node))
        z = expr("*", constant(1/(high-low)), expr("-", x, constant(low)))
        terms.append(z)
        for k in np.unique(cut[1:-1]):
            if low < k < high:
                terms.append(expr("ifelse", expr("gt", node, constant(k)),
                                  expr("-", z, constant((k-low)/(high-low))), constant(0.)))
        # Class-conditional medians preserve thresholds for rare active states.
        medians = np.array([np.median(column[labels==k]) for k in np.unique(labels)])
        for threshold in np.unique((medians[:-1]+medians[1:])/2):
            if low < threshold < high:
                terms.append(expr("gt", node, constant(threshold)))
    if not terms:
        return [], np.empty((len(X), 0))
    A = np.column_stack([term.evaluate(X) for term in terms]).astype(float)
    center = weights @ A
    sd = np.sqrt(weights @ ((A-center)**2))
    good = np.flatnonzero(sd>1e-5)
    if not len(good):
        return [], np.empty((len(X), 0))
    scores = np.zeros(len(good))
    for k in np.unique(labels)[1:]:
        response = (labels >= k).astype(float)
        delta = response - weights @ response
        corr = (weights*delta) @ (A[:,good]-center[good])
        scores = np.maximum(scores, abs(corr)/(sd[good]*max(1e-8,np.sqrt(weights@(delta**2)))))
    keep = good[np.argsort(-scores, kind="stable")[:48]]
    primary = A[:,keep]
    main_terms = [terms[i] for i in keep]
    centered = primary-weights@primary
    gram = centered.T @ (weights[:,None]*centered)
    gram.flat[::len(gram)+1] += .0002
    ys = np.column_stack([labels>=k for k in np.unique(labels)[1:]]).astype(float)
    ys -= weights@ys
    residual = ys-centered@np.linalg.solve(gram, centered.T@(weights[:,None]*ys))
    proposals = []
    for i in range(min(32,len(main_terms))):
        products = primary[:,[i]]*primary[:,i+1:]
        if not products.shape[1]:
            continue
        products -= weights@products
        spread = np.sqrt(weights@(products**2))
        cov = np.max(abs(products.T@(weights[:,None]*residual)),axis=1)
        proposals.extend((cov[j]/spread[j],i,i+j+1) for j in np.flatnonzero(spread>1e-4))
    selected = sorted(proposals, key=lambda item:-item[0])[:28]
    if selected:
        terms_cross = [expr("*",main_terms[i],main_terms[j]) for _,i,j in selected]
        primary = np.column_stack([primary, *[primary[:,i]*primary[:,j] for _,i,j in selected]])
        main_terms.extend(terms_cross)
    return main_terms, primary


def fit(X, y, env, penalty):
    levels, labels = np.unique(y, return_inverse=True)
    _, group, count = np.unique(env, return_inverse=True, return_counts=True)
    weights = 1/(len(count)*count[group])
    terms, A = basis(X, labels, weights)
    center = weights @ A
    scale = np.maximum(np.sqrt(weights @ ((A-center)**2)), .02)
    Z = np.column_stack([np.ones(len(X)), (A-center)/scale])
    coefs = []
    for k in range(1,len(levels)):
        response = (labels>=k).astype(float)
        rate = np.clip(weights@response, 1e-7, 1-1e-7)
        start = np.zeros(Z.shape[1])
        start[0] = np.log(rate/(1-rate))
        def objective(theta):
            logit = Z @ theta
            loss = weights @ (np.logaddexp(0., logit)-response*logit) + .5*penalty*(theta[1:]@theta[1:])
            grad = Z.T @ (weights*(expit(logit)-response))
            grad[1:] += penalty*theta[1:]
            return loss, grad
        opt = minimize(objective, start, method="L-BFGS-B", jac=True,
                       options=dict(maxiter=200, ftol=1e-11))
        beta = opt.x[1:]/scale
        coefs.append((float(opt.x[0]-center@beta), beta))
    return levels, terms, coefs


def predict(fit, X):
    levels, terms, coefs = fit
    A = np.column_stack([t.evaluate(X) for t in terms]) if terms else np.empty((len(X),0))
    prediction = np.full(len(X), levels[0], dtype=float)
    for gap, (bias, beta) in zip(np.diff(levels), coefs):
        prediction += gap*expit(bias+A@beta)
    return prediction


def serialize(fit):
    levels, terms, coefs = fit
    probabilities = []
    for bias, beta in coefs:
        logit = constant(-bias)
        for c, t in zip(beta, terms):
            logit = expr("+", logit, expr("*", constant(-c), t))
        probabilities.append(expr("/", constant(1.), expr("+", constant(1.), expr("exp", logit))))
    return Program(probabilities, np.diff(levels).astype(float), float(levels[0]))


def select(old, X, y, env, V, v, val_env):
    y, v = np.ravel(y).astype(float), np.ravel(v).astype(float)
    env, val_env = np.ravel(env), np.ravel(val_env)
    groups = np.unique(env)
    if not supported(X, y) or len(groups)<2 or not np.array_equal(groups, np.unique(val_env)):
        return old, dict(accepted=False, reason="support")
    penalties = (.00003, .0003, .003)
    models = [fit(X, y, env, lam) for lam in penalties]
    vals = np.column_stack([old.predict(V)[0], *[predict(m, V) for m in models]])
    validation = np.sqrt(np.mean([np.mean((vals[val_env==g]-v[val_env==g,None])**2,axis=0) for g in groups],axis=0))
    if validation[1:].min()>=.995*validation[0]:
        return old, dict(accepted=False, reason="validation", validation=validation.tolist())
    folds = []
    for held in np.array_split(groups, min(len(groups),5)):
        keep, valid = ~np.isin(env, held), np.isin(val_env, held)
        if len(np.unique(y[keep])) < 2:
            return old, dict(accepted=False, reason="constant_training_fold")
        alternatives = [predict(fit(X[keep],y[keep],env[keep],lam), V[valid]) for lam in penalties]
        ps = np.column_stack([refit(old,X[keep],y[keep]).predict(V[valid])[0], *alternatives])
        err = (ps-v[valid,None])**2
        folds.extend(err[val_env[valid]==g].mean(axis=0) for g in held)
    losses = np.asarray(folds)
    blocked = np.sqrt(losses.mean(axis=0))
    wins = (losses[:,1:] < losses[:,[0]]).sum(axis=0)
    viable = np.flatnonzero((blocked[1:]<.99*blocked[0]) & (validation[1:]<.995*validation[0]) &
                            (wins >= (len(groups)+1)//2))
    report = dict(accepted=bool(len(viable)), validation=validation.tolist(), blocked=blocked.tolist(), wins=wins.tolist())
    if not len(viable):
        return old, report
    i = int(viable[np.argmin(blocked[1:][viable])])
    chosen = serialize(models[i])
    for H in (X,V):
        np.testing.assert_allclose(chosen.predict(H)[0],predict(models[i],H),atol=3e-5,rtol=2e-4)
    report.update(penalty=penalties[i], nodes=chosen.count_unique_nodes())
    return chosen, report
