"""A polynomial sensor calibration with a smooth correction for rounded readings."""
import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._method.families.quantized_measurements import fit_curve, response_step
from eureka._symbolic.core.ast import Program


def calibration(curve):
    _,j,center,scale,_,_,params,step,origin=curve
    z=expr("*",constant(1./scale),expr("-",variable(j),constant(center)))
    poly=constant(params[-2])
    for coef in params[-3::-1]:
        poly=expr("+",constant(coef),expr("*",z,poly))
    angle=expr("*",constant(2*np.pi/step),expr("-",poly,constant(origin)))
    terms,beta=[poly],[1.]
    for k in range(1,13):
        weight=step*(-1.)**k*np.exp(-2*(np.pi*k*params[-1])**2)/(np.pi*k)
        if abs(weight)<1e-7*step:
            break
        terms.append(expr("sin",expr("*",constant(k),angle)))
        beta.append(weight)
    return Program(terms,np.asarray(beta))


def select(incumbent,X,y,env,V,v,val_env,seed=0):
    env,val_env,v=np.ravel(env),np.ravel(val_env),np.ravel(v)
    groups=np.unique(env)
    if len(X)<300 or len(groups)<2 or response_step(np.ravel(y).astype(float)) is None:
        return incumbent,dict(accepted=False,reason="no_response_grid")
    curve=fit_curve(X,y)
    if curve is None:
        return incumbent,dict(accepted=False,reason="no_calibration")
    model=calibration(curve)
    def losses(p,mask):
        return np.mean((p.predict(V[mask])[0].astype(float)-v[mask])**2)
    val=np.sqrt(np.mean([[losses(p,val_env==g) for p in (incumbent,model)] for g in np.unique(val_env)],axis=0))
    if val[1]>=.98*val[0]:
        return incumbent,dict(accepted=False,reason="validation",validation=val.tolist())
    folds=[]
    for held in np.array_split(groups,min(len(groups),5)):
        tr=~np.isin(env,held)
        fold=fit_curve(X[tr],np.ravel(y)[tr])
        if fold is None or fold[1]!=curve[1]:
            return incumbent,dict(accepted=False,reason="unstable_axis")
        base=refit(incumbent,X[tr],np.ravel(y)[tr])
        alternate=calibration(fold)
        for g in held:
            m=val_env==g
            if m.any():
                folds.append([losses(p,m) for p in (base,alternate)])
    if not folds:
        return incumbent,dict(accepted=False,reason="unmatched_sources")
    blocked=np.sqrt(np.mean(folds,axis=0))
    accept=bool(blocked[1]<.98*blocked[0])
    return (model if accept else incumbent),dict(accepted=accept,reason="blocked",axis=curve[1],nodes=model.count_unique_nodes(),
            step=float(curve[-2]),width=float(curve[-3][-1]),validation=val.tolist(),blocked=blocked.tolist())
