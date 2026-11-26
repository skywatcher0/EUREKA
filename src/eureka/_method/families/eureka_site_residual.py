"""Partially pool site-specific residuals around the incumbent expression."""
import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.eureka_site_calibration import site_coordinate
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def adjustments(base,X,y,env,key):
    residual=np.ravel(y)-base.predict(X)[0].astype(float)
    levels,site=np.unique(X[:,key],return_inverse=True)
    centers=np.bincount(site,weights=np.ravel(y))/np.bincount(site)
    dynamic=[]
    for j,x in enumerate(X.T):
        if len(np.unique(x))<=len(levels):
            continue
        mean=np.bincount(site,weights=x)/np.bincount(site)
        deviation=x-mean[site]
        association=abs(deviation@(np.ravel(y)-centers[site]))/max(1e-8,np.linalg.norm(deviation))
        dynamic.append((association,int(j)))
    top=[idx for _,idx in sorted(dynamic,reverse=True)[:3]]
    tags=[expr("eq",variable(key),constant(v)) for v in levels]
    _,group,group_counts=np.unique(env,return_inverse=True,return_counts=True)
    weights=1./group_counts[group]
    weights/=weights.mean()
    observed=base.coeffs if base.coeffs is not None else np.ones(base.coefficient_count())
    models=[]
    for count in (0,1,3):
        terms=[]
        for label,tag in enumerate(tags):
            mask=site==label
            cols=[np.ones(mask.sum())]
            nodes=[tag]
            for j in top[:count]:
                x=X[mask,j].astype(float)
                mu,sd=x.mean(),max(x.std(),1e-5)
                cols.append((x-mu)/sd)
                nodes.append(expr("*",tag,expr("*",constant(1./sd),expr("-",variable(j),constant(mu)))))
            A=np.column_stack(cols)
            w=weights[mask]
            gram=A.T@(w[:,None]*A)
            rhs=A.T@(w*residual[mask])
            terms.append((nodes,gram,rhs))
        nodes=[t for group,_,_ in terms for t in group]
        for penalty in (10.,80.):
            coeffs=np.concatenate([np.linalg.solve(G+penalty*np.eye(len(h)),h) for _,G,h in terms])
            models.append(Program([*base.terms,*nodes],np.r_[observed,coeffs],
                                  base.intercept if base.coeffs is not None else 0.,base.coefficient_mode))
    return models


def select(base,X,y,env,V,v,val_env,seed=0):
    env,val_env,v=np.ravel(env),np.ravel(val_env),np.ravel(v)
    groups=np.unique(env)
    if len(X)<400 or X.shape[1]>40 or len(groups)<2:
        return base,dict(accepted=False,reason="source_support")
    key=site_coordinate(X,env)
    if key is None:
        return base,dict(accepted=False,reason="no_site")
    def losses(models,mask):
        return [np.mean((m.predict(V[mask])[0].astype(float)-v[mask])**2) for m in models]
    full=adjustments(base,X,y,env,key)
    val=np.sqrt(np.mean([losses([base,*full],val_env==g) for g in np.unique(val_env)],axis=0))
    eligible=np.flatnonzero(val[1:]<.9975*val[0])
    if not len(eligible):
        return base,dict(accepted=False,reason="validation",validation=val.tolist())
    errors=[]
    for held in np.array_split(groups,min(len(groups),5)):
        train=~np.isin(env,held)
        if site_coordinate(X[train],env[train])!=key:
            return base,dict(accepted=False,reason="unstable_site")
        incumbent=refit(base,X[train],np.ravel(y)[train])
        choices=adjustments(incumbent,X[train],np.ravel(y)[train],env[train],key)
        errors.extend(losses([incumbent,*choices],val_env==g) for g in held if np.any(val_env==g))
    if not errors:
        return base,dict(accepted=False,reason="no_matched_fold")
    blocked=np.sqrt(np.mean(errors,axis=0))
    winner=int(eligible[np.argmin(blocked[1:][eligible])])
    accept=bool(blocked[winner+1]<.995*blocked[0])
    return (full[winner] if accept else base),dict(accepted=accept,reason="fold",site=key,
            adjustment=winner,validation=[val[0],val[winner+1]],blocked=[blocked[0],blocked[winner+1]])
