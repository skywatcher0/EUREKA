"""Reconstruct a response from paired magnitude and bounded fraction readings."""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


@dataclass
class Calibration:
    signal: int
    fraction: int
    zero: float
    maximum: float
    inverse_span: float
    slope: float
    offset: float

    def coordinates(self, X):
        x = X[:, self.signal].astype(float) - self.zero
        z = 1. + self.inverse_span * (X[:, self.fraction].astype(float) - self.maximum)
        return x, z

    def terms(self):
        x = expr("-", variable(self.signal), constant(self.zero))
        z = expr("+", constant(1.), expr("*", constant(self.inverse_span),
                 expr("-", variable(self.fraction), constant(self.maximum))))
        active = expr("*", expr("gt", x, constant(1e-7)), expr("*", expr("gt", z, constant(0.)),
                      expr("lt", z, constant(.999999))))
        inverse_root = expr("exp", expr("*", constant(-.5), expr("log",
                            expr("-", constant(1.), expr("*", z, z)))))
        prediction = expr("+", constant(self.offset), expr("*", constant(self.slope),
                          expr("*", x, expr("*", z, inverse_root))))
        return z, active, prediction


def fit_channels(X, y):
    X = np.asarray(X, np.float32)
    y = np.ravel(y).astype(float)
    n, dimension = X.shape
    if n < 300 or dimension < 4 or dimension > 24:
        return None
    low, high = X.min(0), X.max(0)
    varying = high > low + 1e-5
    signals = np.flatnonzero(varying & ((X == low).mean(0) > .05))
    fractions = np.flatnonzero(varying & ((X == high).mean(0) > .05))
    channels = []
    for j in fractions:
        span = float(high[j] - low[j])
        gap = (high[j] - X[:, j].astype(float)) / span
        for i in signals:
            if i == j:
                continue
            q = X[:, i].astype(float) - float(low[i])
            mask = (q > 1e-7) & (gap > .015) & (gap < .99)
            if mask.sum() < max(60, .06*n) or np.unique(gap[mask]).size < 14:
                continue
            x, g, t = q[mask], gap[mask], y[mask]
            sigma = np.std(t)
            if sigma < 1e-6:
                continue
            def feature(width):
                z = 1. - width*g
                return x*z / np.sqrt(np.maximum(1e-7, 1.-z*z))
            def affine(width):
                f = feature(width)
                c = f-f.mean()
                slope = c@(t-t.mean()) / max(c@c, 1e-12)
                offset = t.mean()-slope*f.mean()
                err = np.median(abs(offset+slope*f-t))
                return err, width, slope, offset
            error, width, slope, offset = min(affine(w) for w in (.4,.6,.8,1.))
            if error > .07*sigma or slope <= 0:
                continue
            def residual(p):
                return (p[2] + p[1]*feature(p[0])-t)/sigma
            solution = least_squares(residual, (width,slope,offset),
                                     bounds=((.12,0.,-np.inf),(1.01/g.max(),np.inf,np.inf)),
                                     loss="soft_l1",f_scale=.005,max_nfev=100)
            width,slope,offset = solution.x
            if np.quantile(abs(residual(solution.x)),.9) < .04:
                channels.append(Calibration(int(i),int(j),float(low[i]),float(high[j]),
                                            float(width/span),float(slope),float(offset)))
    best = None
    for a,b in combinations(channels,2):
        if len({a.signal,a.fraction,b.signal,b.fraction}) != 4:
            continue
        xa,za = a.coordinates(X)
        xb,zb = b.coordinates(X)
        first = (xa>1e-7)&(za>0)&(za<.999999)
        second = (xb>1e-7)&(zb>0)&(zb<.999999)
        if min((first&~second).sum(),(second&~first).sum()) < .02*n or np.mean(first|second)<.94:
            continue
        use_a = first & (~second | (za <= zb))
        unresolved = ~(first|second)
        background = float(np.mean(y[unresolved])) if unresolved.any() else float(y.mean())
        def pred(ch,x,z):
            return ch.offset + ch.slope*x*z / np.sqrt(np.maximum(1e-8,1.-z*z))
        values = np.where(use_a,pred(a,xa,za),np.where(second,pred(b,xb,zb),background))
        loss = np.mean((values-y)**2)
        if loss>.02*np.var(y) or (best is not None and loss>=best[0]):
            continue
        za,active_a,pa=a.terms()
        zb,active_b,pb=b.terms()
        both=expr("ifelse",expr("gt",za,zb),pb,pa)
        term=expr("ifelse",active_a,expr("ifelse",active_b,both,pa),
                  expr("ifelse",active_b,pb,constant(background)))
        best=loss,Program([term],np.ones(1))
    return best[1] if best else None


def select(incumbent,X,y,env,V,v,val_env,seed=0):
    env,val_env,v=np.ravel(env),np.ravel(val_env),np.ravel(v)
    groups=np.unique(env)
    if len(groups)<2:
        return incumbent,dict(accepted=False,reason="single_environment")
    model=fit_channels(X,y)
    if model is None:
        return incumbent,dict(accepted=False,reason="no_pair")
    def mse(p,rows):
        return np.mean((p.predict(V[rows])[0].astype(float)-v[rows])**2)
    scores=np.sqrt(np.mean([[mse(p,val_env==g) for p in (incumbent,model)]
                            for g in np.unique(val_env)],axis=0))
    if scores[1]>=.98*scores[0]:
        return incumbent,dict(accepted=False,reason="validation",validation=scores.tolist())
    folds=[]
    for held in np.array_split(groups,min(4,len(groups))):
        tr=~np.isin(env,held)
        candidate=fit_channels(X[tr],np.ravel(y)[tr])
        if candidate is None:
            return incumbent,dict(accepted=False,reason="fold_support")
        base=refit(incumbent,X[tr],np.ravel(y)[tr])
        folds.extend([mse(p,val_env==g) for p in (base,candidate)] for g in held if np.any(val_env==g))
    if not folds:
        return incumbent,dict(accepted=False,reason="no_matched_fold")
    blocked=np.sqrt(np.mean(folds,axis=0))
    accepted=bool(blocked[1]<.98*blocked[0])
    return (model if accepted else incumbent),dict(accepted=accepted,reason="blocked",validation=scores.tolist(),
             blocked=blocked.tolist(),nodes=model.count_unique_nodes())
