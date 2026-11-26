"""Shrink residual offsets for supported repeat readings, returning zero on unseen values."""
import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def coordinates(X):
    if len(X)<500 or not 2<=X.shape[1]<=24:
        return []
    result=[]
    for j,x in enumerate(X.T):
        levels,count=np.unique(x.astype(np.float32),return_counts=True)
        if 25<=len(levels)<=min(900,len(x)//4) and np.sum(count[count>=3])>=.75*len(x):
            result.append(j)
    return result


def offsets(x,residual,shrinkage):
    levels,idx,n=np.unique(x.astype(np.float32),return_inverse=True,return_counts=True)
    delta=np.bincount(idx,weights=residual)/(n+shrinkage)
    keep=n>=3
    return levels[keep],delta[keep]


def lookup(x,levels,delta):
    value=x.astype(np.float32)
    i=np.searchsorted(levels,value)
    clipped=np.minimum(i,len(levels)-1)
    hit=(i<len(levels))&(levels[clipped]==value)
    return np.where(hit,delta[clipped],0.),hit


def correction(axis,levels,values):
    x=variable(axis)
    def branch(start,stop):
        if stop-start==1:
            c=constant(levels[start])
            outside=expr("+",expr("lt",x,c),expr("gt",x,c))
            return expr("ifelse",outside,constant(0.),constant(values[start]))
        mid=(start+stop)//2
        return expr("ifelse",expr("gt",x,constant(levels[mid-1])),branch(mid,stop),branch(start,mid))
    return branch(0,len(levels))


def select(base,X,y,env,V,v,val_env):
    cols=coordinates(X)
    env,val_env,y,v=map(np.ravel,(env,val_env,y,v))
    groups=np.unique(env)
    if not cols or len(groups)<2 or not np.array_equal(groups,np.unique(val_env)):
        return base,dict(accepted=False,reason="no_repeated_coordinate")
    p=base.predict(V)[0].astype(float)
    residual=y-base.predict(X)[0].astype(float)
    masks=[val_env==g for g in groups]
    mse=lambda z,w:float(np.mean((z-w)**2))
    reference=np.array([mse(p[m],v[m]) for m in masks])
    proposals=[]
    for j in cols:
        for strength in (3.,12.):
            levels,delta=offsets(X[:,j],residual,strength)
            gain,hit=lookup(V[:,j],levels,delta)
            losses=np.array([mse((p+gain)[m],v[m]) for m in masks])
            if losses.mean()<=.97**2*reference.mean() and np.sum(losses<reference)>len(groups)/2:
                proposals.append((j,strength,levels,delta,losses,float(hit.mean())))
    if not proposals:
        return base,dict(accepted=False,reason="source_validation")
    fold_scores=[]
    for g in groups:
        keep=env!=g
        held=val_env==g
        fitted=refit(base,X[keep],y[keep])
        prediction=fitted.predict(V[held])[0].astype(float)
        training_residual=y[keep]-fitted.predict(X[keep])[0].astype(float)
        row=[mse(prediction,v[held])]
        for j,strength,*_ in proposals:
            levels,delta=offsets(X[keep,j],training_residual,strength)
            extra=lookup(V[held,j],levels,delta)[0] if len(levels) else 0.
            row.append(mse(prediction+extra,v[held]))
        fold_scores.append(row)
    folds=np.array(fold_scores)
    blocked=np.sqrt(folds.mean(0))
    wins=(folds[:,1:]<folds[:,[0]]).sum(0)
    eligible=np.flatnonzero((blocked[1:]<=.985*blocked[0])&(wins>len(groups)/2))
    if not len(eligible):
        return base,dict(accepted=False,reason="blocked_source",blocked=blocked.tolist())
    i=eligible[np.argmin(blocked[1:][eligible])]
    j,strength,levels,delta,losses,coverage=proposals[i]
    new=correction(j,levels,delta)
    if base.coefficient_mode=="conditional_affine":
        new=expr("*",constant(1.),new)
    coefs=base.coeffs if base.coeffs is not None else np.ones(base.coefficient_count())
    model=Program([*base.terms,new],np.r_[coefs,1.],base.intercept if base.coeffs is not None else 0.,base.coefficient_mode)
    np.testing.assert_allclose(model.predict(V)[0],p+lookup(V[:,j],levels,delta)[0],rtol=1e-4,atol=2e-5)
    return model,dict(accepted=True,axis=int(j),strength=strength,levels=len(levels),coverage=coverage,
                      blocked=[float(blocked[0]),float(blocked[i+1])],wins=int(wins[i]),
                      validation=[float(np.sqrt(reference.mean())),float(np.sqrt(losses.mean()))],
                      nodes=model.count_unique_nodes())
