"""Source-estimated piecewise response along an observed circular coordinate."""
from itertools import combinations

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def orbit(X):
    axes = [j for j, c in enumerate(X.T) if 5 <= len(np.unique(np.round(c,5))) <= 64]
    for i,j in combinations(axes,2):
        pair = np.unique(np.round(X[:,[i,j]].astype(float),5),axis=0)
        if not 8 <= len(pair) <= 64:
            continue
        center = (pair.max(0) + pair.min(0))/2
        radius = np.ptp(pair,axis=0)/2
        coords = (pair-center)/radius
        if np.max(abs(np.linalg.norm(coords,axis=1)-1)) > .002:
            continue
        angles = np.sort(np.mod(np.arctan2(coords[:,1],coords[:,0]),2*np.pi))
        spacing = 2*np.pi/len(angles)
        gaps = np.diff(np.r_[angles,angles[0]+2*np.pi])
        if np.max(abs(gaps/spacing-1)) < .03:
            return (i,j),center,radius,angles
    return None


def fitted(X,y,cycle):
    (i,j),center,radius,angles = cycle
    axes = [expr("*",constant(1/r),expr("-",variable(k),constant(c)))
            for k,c,r in zip((i,j),center,radius)]
    states = []
    for a in angles:
        dot = expr("+",expr("*",constant(np.cos(a)),axes[0]),expr("*",constant(np.sin(a)),axes[1]))
        states.append(expr("gt",dot,constant(np.cos(np.pi/len(angles)))))
    main, moderators = [], []
    for j in [k for k in range(X.shape[1]) if k not in cycle[0]]:
        values = X[:,j]
        levels = np.unique(values)
        node = variable(j)
        if len(levels) == 1:
            continue
        if len(levels) <= 4:
            tags = [expr("gt",node,constant(float((a+b)/2))) for a,b in zip(levels,levels[1:])]
            main.extend(tags)
            moderators.extend(tags)
        else:
            lo,hi = np.quantile(values,[.01,.99])
            if hi-lo < 1e-5:
                continue
            clipped = expr("ifelse",expr("gt",node,constant(hi)),constant(hi),
                           expr("ifelse",expr("lt",node,constant(lo)),constant(lo),node))
            normalized = expr("*",constant(1./(hi-lo)),expr("-",clipped,constant(lo)))
            main.append(normalized)
            moderators.append(normalized)
            for cut in np.unique(np.quantile(values,[.15,.35,.55,.75,.9])):
                offset = expr("-",normalized,constant((cut-lo)/(hi-lo)))
                main.append(expr("ifelse",expr("gt",node,constant(cut)),offset,constant(0.)))
    y = np.ravel(y).astype(float)
    terms = [*states,*main]
    design = np.column_stack([t.evaluate(X) for t in terms])
    fits = []
    for interactions in (False,True):
        if interactions and not moderators:
            continue
        if interactions:
            extra = [expr("*",z,p) for z in moderators for p in states]
            terms += extra
            design = np.column_stack([design,*(t.evaluate(X) for t in extra)])
        mean = design.mean(0,dtype=float)
        Z = design - mean
        gram = Z.T @ Z
        penalty = np.full(len(terms),.5)
        penalty[len(states)+len(main):] = 2.
        gram.flat[::len(gram)+1] += penalty
        beta = np.linalg.solve(gram, Z.T@(y-y.mean()))
        fits.append(Program(terms.copy(),beta,float(y.mean()-mean@beta)))
    return fits


def source_losses(model,V,v,labels):
    residual = model.predict(V)[0].astype(float)-np.ravel(v)
    return np.asarray([np.mean(residual[labels==g]**2) for g in np.unique(labels)])


def select(incumbent,X,y,env,V,v,val_env):
    env,val_env = np.ravel(env),np.ravel(val_env)
    if len(X)<200 or X.shape[1]>32:
        return incumbent,dict(accepted=False,reason="sample_support")
    cycle = orbit(X)
    if cycle is None:
        return incumbent,dict(accepted=False,reason="no_cycle")
    candidates = fitted(X,y,cycle)
    losses = [source_losses(model,V,v,val_env) for model in candidates]
    choice = int(np.argmin([loss.mean() for loss in losses]))
    old = source_losses(incumbent,V,v,val_env)
    scores = np.sqrt([old.mean(),losses[choice].mean()])
    report = dict(accepted=False,axis=cycle[0],levels=len(cycle[-1]),interactions=bool(choice),validation=scores.tolist())
    if scores[1] >= .995*scores[0] or np.sum(losses[choice]<old) < len(old)/2:
        return incumbent,{**report,"reason":"validation"}
    groups = np.unique(env)
    if len(groups)<2:
        return incumbent,{**report,"reason":"single_environment"}
    folds = []
    for held in np.array_split(groups,min(3,len(groups))):
        tr, va = ~np.isin(env,held),np.isin(val_env,held)
        c = orbit(X[tr])
        if c is None or c[0] != cycle[0] or len(c[-1]) != len(cycle[-1]) or not va.any():
            return incumbent,{**report,"reason":"unsupported_fold"}
        fold = fitted(X[tr],y[tr],c)
        if choice>=len(fold):
            return incumbent,{**report,"reason":"unsupported_interaction"}
        base = refit(incumbent,X[tr],y[tr])
        folds.extend(np.column_stack([source_losses(m,V[va],v[va],val_env[va]) for m in (base,fold[choice])]))
    blocked = np.sqrt(np.mean(folds,axis=0))
    accepted = bool(blocked[1] < .995*blocked[0])
    return (candidates[choice] if accepted else incumbent),{**report,"accepted":accepted,"blocked":blocked.tolist(),"reason":"blocked"}
