"""Validate and export a nested trigonometric and piecewise-affine expansion."""
import numpy as np

from eureka._method.families.eureka_nested_projection import NestedProjection
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def surfaces(X,y,seed):
    return [NestedProjection(X,y,seed=seed,activation=name,iteration_limit=1000,hidden=(32,24),row_limit=6000)
            for name in ("sin","hinge")]


def prediction(fits,X):
    return .5*(fits[0].predict(X)+fits[1].predict(X))


def equation(fits):
    first,second = (f.program() for f in fits)
    return Program(first.terms+second.terms, .5*np.r_[first.coeffs,second.coeffs],
                   .5*(first.intercept+second.intercept))


def assess(incumbent,X,y,env,V,v,val_env,seed=0):
    y,v = np.asarray(y).ravel(),np.asarray(v).ravel()
    env,val_env = np.asarray(env).ravel(),np.asarray(val_env).ravel()
    fits = surfaces(X,y,seed)
    pred = prediction(fits,V)
    a,b = (incumbent.predict(V)[0].astype(float)-v)**2,(pred-v)**2
    gains = np.array([np.mean((a-b)[val_env==g]) for g in np.unique(val_env)])
    samples = gains if len(gains)>1 else a-b
    se = np.std(samples,ddof=1)/np.sqrt(len(samples)) if len(samples)>1 else 0.
    report = dict(accepted=False,by_environment=gains.tolist(),standard_error=float(se))
    if gains.min()<=0 or gains.mean()<=se or b.mean() >= .99**2*a.mean():
        return incumbent,report
    # Select a stress fold by source covariate shift, without using responses.
    groups = np.unique(env)
    if len(groups)>1:
        mean, scale = X.mean(0), X.std(0)
        z = (X-mean)/np.where(scale>1e-8,scale,1.)
        held = groups[np.argmax([np.sum(z[env==g].mean(0)**2) for g in groups])]
        mask = env==held
        va = val_env==held
        H,truth = np.vstack([X[mask],V[va]]),np.r_[y[mask],v[va]]
        original = refit(incumbent,X[~mask],y[~mask]).predict(H)[0]
        local = prediction(surfaces(X[~mask],y[~mask],seed),H)
        old = float(np.sqrt(np.mean((original.astype(float)-truth)**2)))
        new = float(np.sqrt(np.mean((local-truth)**2)))
        report.update(stress_environment=int(held),source_before=old,source_after=new)
        if new>=.99*old:
            return incumbent,report
    chosen = equation(fits)
    np.testing.assert_allclose(chosen.predict(V)[0],pred,rtol=1e-4,atol=1e-4)
    return chosen,dict(**{**report,"accepted":True},origin="nested_expansion")
