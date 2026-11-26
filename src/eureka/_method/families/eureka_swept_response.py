"""Partially pooled response spectra on a repeated geometric sweep coordinate."""
import numpy as np
from scipy.spatial.distance import cdist

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.eureka_replicated_curve import linear_curve
from eureka._method.families.eureka_size_contrasts import clip, linear
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def axes(X, env):
    if not (100 <= len(X) <= 5000 and 3 <= X.shape[1] <= 10):
        return None
    groups = np.unique(env)
    if len(groups) < 2:
        return None
    controls = [j for j,x in enumerate(X.T) if len(np.unique(x)) == len(groups)
                and all(np.ptp(x[env == g]) == 0 for g in groups)]
    if len(controls) != 1:
        return None
    for j, x in enumerate(X.T):
        grid = np.unique(x)
        if not 9 <= len(grid) <= 45:
            continue
        spacing = np.diff(grid).astype(float)
        if spacing[-1] < 5*spacing[0] or np.corrcoef(np.arange(len(spacing)),np.log(spacing))[0,1] < .94:
            continue
        _, counts = np.unique(np.delete(X,j,axis=1),axis=0,return_counts=True)
        if len(counts) >= 8 and np.median(counts) >= .35*len(grid):
            return j, controls[0]
    return None


def geometry(X, sweep, control):
    grid = np.unique(X[:,sweep]).astype(float)
    mods = [j for j in range(X.shape[1]) if j not in (sweep,control) and np.ptp(X[:,j]) > 1e-6]
    raw = X[:,mods].astype(float)
    lower = raw.min(0)
    span = np.maximum(np.ptp(raw,axis=0),1e-6)
    take_log = (np.quantile(raw,.5,axis=0)-lower)/span < .18
    loc = (raw-lower)/span
    loc[:,take_log] = np.log(.03+loc[:,take_log])
    sd = np.maximum(loc.std(0),.04)
    pts = np.unique(loc/sd,axis=0)
    # Cover observed settings even when some sweeps contain more rows than others.
    ids = [int(np.argmin(np.sum((pts-np.median(pts,axis=0))**2,axis=1)))]
    nearest = np.full(len(pts), np.inf)
    for _ in range(min(40,len(pts))-1):
        nearest = np.minimum(nearest, np.sum((pts-pts[ids[-1]])**2,axis=1))
        ids.append(int(nearest.argmax()))
    return grid, mods, lower, span, take_log, sd, pts[ids], float(X[:,control].mean()), float(X[:,control].std())


def factors(X, axes_, geo, width):
    j, k = axes_
    grid,mods,low,span,take_log,sd,pts,mu,st = geo
    phase = np.interp(X[:,j], grid, np.linspace(0,1,len(grid)))
    curve = np.column_stack([np.ones(len(X)),*[np.clip(6*phase-h,0,1) for h in range(6)]])
    H = (X[:,mods].astype(float)-low)/span
    H[:,take_log] = np.log(np.maximum(.03+H[:,take_log],.005))
    H /= sd
    kernels = np.exp(-cdist(H,pts,"sqeuclidean")/(2*width**2))
    speed = (X[:,k]-mu)/st
    moderators = np.column_stack([np.ones(len(X)),H,kernels,speed,speed[:,None]*H])
    return curve, moderators


def estimates(X,y,axes_,geo,width):
    c,h = factors(X,axes_,geo,width)
    A = (c[:,:,None]*h[:,None,:]).reshape(len(X),-1)
    gram = A.T@A
    rhs = A.T@y
    out = []
    for penalty in (.00015,.0015):
        mat = gram.copy()
        mat.flat[::len(mat)+1] += len(X)*penalty
        out.append(np.linalg.solve(mat,rhs).reshape(c.shape[1],h.shape[1]))
    return out


def serialize(axes_,geo,width,beta):
    j,k = axes_
    grid,mods,low,span,take_log,sd,pts,mu,st = geo
    phase = linear_curve(variable(j),grid,np.linspace(0,1,len(grid)))
    curve = [constant(1.),*[clip(expr("-",expr("*",constant(6.),phase),constant(h)),0.,1.) for h in range(6)]]
    H = []
    for m,lo,s,log,scale in zip(mods,low,span,take_log,sd):
        z = expr("*",constant(1/s),expr("-",variable(m),constant(lo)))
        if log:
            z = expr("+",constant(.03),z)
            z = expr("log",expr("ifelse",expr("gt",z,constant(.005)),z,constant(.005)))
        H.append(expr("*",constant(1/scale),z))
    kernels = []
    for pt in pts:
        diffs = [expr("-",z,constant(p)) for z,p in zip(H,pt)]
        squares = [expr("*",z,z) for z in diffs]
        kernels.append(expr("exp",expr("*",constant(-.5/width**2),linear(squares,np.ones(len(H)),0))))
    speed = expr("*",constant(1/st),expr("-",variable(k),constant(mu)))
    mods = [constant(1.),*H,*kernels,speed,*[expr("*",speed,z) for z in H]]
    return Program([expr("*",c,linear(mods,b,0.)) for c,b in zip(curve,beta)],np.ones(len(curve)),0.)


def fit_block(X,y,V,env):
    chosen = axes(X,env)
    if chosen is None:
        return None
    geo = geometry(X,*chosen)
    if not geo[1] or geo[-1] <= 1e-8:
        return None
    models, preds = [], []
    for width in (.6,1.4):
        c,h = factors(V,chosen,geo,width)
        for b in estimates(X,y,chosen,geo,width):
            models.append((chosen,geo,width,b))
            preds.append(np.sum((c@b)*h,axis=1))
    return models, preds


def select(old,X,y,env,V,v,val_env):
    y,v = np.ravel(y).astype(float),np.ravel(v).astype(float)
    groups = np.unique(env)
    if len(groups)<3 or not np.array_equal(groups,np.unique(val_env)) or axes(X,env) is None:
        return old,dict(accepted=False,reason="support")
    full = fit_block(X,y,V,env)
    if full is None:
        return old,dict(accepted=False,reason="moderators")
    models,preds = full
    val = np.sqrt(np.mean((np.column_stack([old.predict(V)[0],*preds])-v[:,None])**2,axis=0))
    folds = []
    for g in groups:
        tr,va = env!=g,val_env==g
        block = fit_block(X[tr],y[tr],V[va],env[tr])
        if block is None or block[0][0][0] != models[0][0]:
            return old,dict(accepted=False,reason="fold_support")
        incumbent = refit(old,X[tr],y[tr]).predict(V[va])[0]
        folds.append(np.mean((np.column_stack([incumbent,*block[1]])-v[va,None])**2,axis=0))
    losses = np.asarray(folds)
    blocked = np.sqrt(losses.mean(0))
    choices = np.flatnonzero((val[1:] <= .99*val[0]) & (blocked[1:] <= .97*blocked[0])
                             & ((losses[:,1:]<losses[:,[0]]).sum(0)>len(groups)/2))
    report = dict(accepted=bool(len(choices)), axes=list(models[0][0]), validation=val.tolist(),
                  blocked=blocked.tolist(), folds=losses.tolist())
    if not len(choices):
        return old,report
    i = int(choices[np.argmin(blocked[1:][choices])])
    result = serialize(*models[i])
    np.testing.assert_allclose(result.predict(V)[0], preds[i],rtol=4e-4,atol=3e-5)
    return result,dict(report,config=i,nodes=result.count_unique_nodes())
