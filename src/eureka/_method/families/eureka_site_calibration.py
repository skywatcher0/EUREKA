"""Partially pooled response surfaces at repeated, identifiable measurement sites."""
from itertools import combinations_with_replacement

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def site_coordinate(X, env):
    axes=[]
    for j,col in enumerate(X.T):
        levels,index,counts=np.unique(col,return_inverse=True,return_counts=True)
        if 6<=len(levels)<=64 and counts.min()>=4 and all(
                np.unique(index[env==e]).size>=.9*len(levels) for e in np.unique(env)):
            axes.append((j,index,len(levels)))
    for n,(j,index,size) in enumerate(axes):
        for _,other,k in axes[n+1:]:
            if k==size and np.unique(index*size+other).size==size:
                return j
    return None


def candidates(X,y,site):
    y=np.ravel(y).astype(float)
    levels,idx=np.unique(X[:,site],return_inverse=True)
    site_nodes=[expr("eq",variable(site),constant(v)) for v in levels]
    linear,splines,associations=[],[],[]
    for j,values in enumerate(X.T):
        lo,hi=np.quantile(values,[.005,.995])
        if hi-lo<1e-6 or len(np.unique(np.column_stack([idx,values]),axis=0))==len(levels):
            continue
        node=variable(j)
        bounded=expr("ifelse",expr("gt",node,constant(hi)),constant(hi),
                     expr("ifelse",expr("lt",node,constant(lo)),constant(lo),node))
        center=float(np.mean(values,dtype=float))
        scale=float(np.std(values,dtype=float))
        norm=expr("*",constant(1./scale),expr("-",bounded,constant(center)))
        linear.append(norm)
        associations.append(abs(float(np.mean((values-center)*(y-y.mean()))))/scale)
        for t in np.unique(np.quantile(values,[.15,.35,.65,.85])):
            if lo<t<hi:
                splines.append(expr("ifelse",expr("gt",node,constant(t)),
                               expr("-",norm,constant((t-center)/scale)),constant(0.)))
    if not linear:
        return []
    ordered=np.argsort(-np.asarray(associations),kind="stable")[:8]
    products=[expr("*",linear[i],linear[j]) for i,j in combinations_with_replacement(ordered,2)]
    global_terms=linear+splines+products
    local_terms=[expr("*",tag,linear[j]) for tag in site_nodes for j in ordered[:4]]
    terms=[*global_terms,*site_nodes,*local_terms]
    mat=np.column_stack([node.evaluate(X) for node in terms]).astype(float)
    mean=mat.mean(0)
    design=mat-mean
    gram=design.T@design
    rhs=design.T@(y-y.mean())
    fits=[]
    for shrink in (4.,30.,120.):
        penalty=np.full(len(terms),.002*len(y))
        penalty[len(global_terms):]=shrink
        fit=gram.copy()
        fit.flat[::len(fit)+1]+=penalty
        beta=np.linalg.solve(fit,rhs)
        fits.append(Program(terms,beta,float(y.mean()-mean@beta)))
    return fits


def select(incumbent,X,y,env,V,v,val_env,seed=0):
    env,val_env,v=np.ravel(env),np.ravel(val_env),np.ravel(v)
    groups=np.unique(env)
    if len(X)<400 or X.shape[1]>40 or len(groups)<2:
        return incumbent,dict(accepted=False,reason="sample_support")
    site=site_coordinate(X,env)
    if site is None:
        return incumbent,dict(accepted=False,reason="no_site_key")
    models=candidates(X,y,site)
    if not models:
        return incumbent,dict(accepted=False,reason="constant_covariates")
    def mse(p,m):
        return np.mean((p.predict(V[m])[0].astype(float)-v[m])**2)
    val=np.sqrt(np.mean([[mse(p,val_env==g) for p in [incumbent,*models]] for g in np.unique(val_env)],axis=0))
    if val[1:].min()>=.99*val[0]:
        return incumbent,dict(accepted=False,reason="validation",validation=val.tolist())
    losses=[]
    for held in np.array_split(groups,min(len(groups),4)):
        tr=~np.isin(env,held)
        fold_site=site_coordinate(X[tr],env[tr])
        if fold_site!=site:
            return incumbent,dict(accepted=False,reason="unstable_key")
        fold=candidates(X[tr],np.ravel(y)[tr],site)
        if len(fold)!=len(models):
            return incumbent,dict(accepted=False,reason="fold_covariates")
        old=refit(incumbent,X[tr],np.ravel(y)[tr])
        losses.extend([mse(p,val_env==g) for p in [old,*fold]] for g in held if np.any(val_env==g))
    if not losses:
        return incumbent,dict(accepted=False,reason="no_matched_fold")
    blocked=np.sqrt(np.mean(losses,axis=0))
    valid=np.flatnonzero(val[1:]<.99*val[0])
    winner=int(valid[np.argmin(blocked[1:][valid])])
    accept=bool(blocked[winner+1]<.99*blocked[0])
    return (models[winner] if accept else incumbent),dict(accepted=accept,reason="blocked",site=site,
           shrink=(4.,30.,120.)[winner],validation=[val[0],val[winner+1]],blocked=[blocked[0],blocked[winner+1]],
           nodes=models[winner].count_unique_nodes())
