"""Regularize weak angular effects around a dominant relative-angle harmonic."""
import numpy as np
from eureka._method.families import eureka_angular_response as angular
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program

supported = angular.supported


def identify(X, y):
    centered = y-y.mean()
    variance = centered @ centered
    best_loss, best_spec = .35*variance, None
    for sign in (-1, 1):
        for frequency in (1., 2., 3., 4.):
            angle = frequency * (X[:, 0].astype(float) + sign * X[:, 1])
            waves = np.column_stack([np.cos(angle), np.sin(angle)])
            waves -= waves.mean(0)
            residual = centered - waves @ np.linalg.lstsq(waves, centered, rcond=None)[0]
            loss = residual @ residual
            if loss < best_loss:
                best_loss, best_spec = loss, (0, 1, sign, frequency, ())
    return best_spec


def fit(basis, X, y, env, shrink):
    features = np.column_stack([wave.evaluate(X) for wave in basis]).astype(float)
    A = np.column_stack([np.ones(len(y)), features])
    _, labels, counts = np.unique(env, return_inverse=True, return_counts=True)
    weight = 1 / (len(counts) * counts[labels])
    # The intercept and leading sine/cosine stay free. Shrink the six nuisance
    # coefficients, especially when a held angle range leaves them collinear.
    penalty = np.r_[0., 1e-7, 1e-7, np.full(A.shape[1]-3, shrink)]
    def solve(w):
        gram = A.T @ (w[:, None]*A)
        gram.flat[::len(gram)+1] += penalty
        return np.linalg.solve(gram, A.T @ (w*y))
    beta = solve(weight)
    for _ in range(2):
        fitted = A @ beta
        residual = y-fitted
        groups = np.array_split(np.argsort(fitted, kind="stable"), min(6, len(X)//16))
        centers = [fitted[g].mean() for g in groups]
        scales = [1.4826*np.median(abs(residual[g]-np.median(residual[g]))) for g in groups]
        noise = np.maximum(np.interp(fitted, centers, scales), max(1e-6, np.std(residual)*.25))
        adjusted = weight * np.minimum(1., 2.5*noise/np.maximum(1e-8, abs(residual))) / noise
        beta = solve(adjusted/adjusted.sum())
    return Program(basis, beta[1:], float(beta[0]))


def select(old, X, y, env, V, v, val_env, seed=0):
    y, env, v, val_env = [np.asarray(a).ravel() for a in (y, env, v, val_env)]
    groups = np.unique(env)
    if not supported(X) or len(groups) < 2 or not np.array_equal(groups, np.unique(val_env)):
        return old, dict(accepted=False, reason="support")
    spec = identify(X, y)
    if spec is None:
        return old, dict(accepted=False, reason="weak_period")
    basis = angular.terms(spec, 1)
    strengths = [.001, .03, .3]
    models = [fit(basis, X, y, env, shrink) for shrink in strengths]
    def losses(models, mask):
        return [np.mean((m.predict(V[mask])[0].astype(float)-v[mask])**2) for m in models]
    validation = np.sqrt(np.mean([losses([old, *models], val_env==g) for g in groups], axis=0))
    folds = []
    for group in groups:
        keep = env != group
        if identify(X[keep], y[keep]) != spec:
            return old, dict(accepted=False, reason="unstable_period")
        competitors = [refit(old, X[keep], y[keep])]
        competitors.extend(fit(basis, X[keep], y[keep], env[keep], shrink) for shrink in strengths)
        folds.append(losses(competitors, val_env == group))
    blocked = np.sqrt(np.mean(folds, axis=0))
    # Select for across-environment extrapolation, with a relative validation
    # non-inferiority bound and a required improvement on blocked source means.
    viable = np.flatnonzero((validation[1:] <= 1.01*validation[0]) & (blocked[1:] < .99*blocked[0]))
    report = dict(accepted=bool(len(viable)), validation=validation.tolist(), blocked=blocked.tolist())
    if not len(viable):
        return old, report
    i = int(viable[np.argmin(blocked[1:][viable])])
    report.update(shrink=strengths[i], nodes=models[i].count_unique_nodes())
    return models[i], report
