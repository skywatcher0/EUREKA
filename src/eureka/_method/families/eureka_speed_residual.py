"""Shrink smooth speed-dependent residual terms around the existing response."""
import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def supported(X):
    if X.shape[1]!=1 or not 240<=len(X)<=12000:
        return False
    x=X[:,0]
    return bool(x.min()>=0 and np.ptp(x)>1e-4 and np.unique(x).size>=60)


def matrix(x,knots,scale,width):
    u=x.astype(float)/scale
    return np.column_stack([np.ones(len(x)),u,u*u,np.exp(-.5*((u[:,None]-knots)/width)**2)])


def fits(old,X,y):
    x=X[:,0].astype(float)
    scale=max(np.quantile(x,.95),.001)
    u=x/scale
    knots=np.unique(np.quantile(u,np.linspace(.04,.96,7)))
    width=max(.16*np.ptp(u),.01)
    A=matrix(x,knots,scale,width)
    residual=y-old.predict(X)[0]
    models=[]
    for penalty in (.003,.03):
        # Penalize global slope changes more strongly than local corrections.
        gram=A.T@A
        gram.flat[::len(gram)+1]+=len(y)*penalty*np.r_[.12,4.,4.,np.ones(len(knots))]
        b=np.linalg.solve(gram,A.T@residual)
        for shrink in (.35,.7,1.):
            models.append((scale,knots,width,shrink*b))
    return models


def predict(old,models,X):
    baseline=old.predict(X)[0].astype(float)
    scale,knots,width,_=models[0]
    A=matrix(X[:,0],knots,scale,width)
    return np.column_stack([baseline,*[baseline+A@m[-1] for m in models]])


def serialize(old,model):
    scale,knots,width,coef=model
    u=expr("*",constant(1./scale),variable(0))
    terms=[u,expr("*",u,u)]
    for center in knots:
        dist=expr("*",constant(1./width),expr("-",u,constant(center)))
        terms.append(expr("exp",expr("*",constant(-.5),expr("*",dist,dist))))
    weights=old.coeffs if old.coeffs is not None else np.ones(old.coefficient_count())
    intercept=old.intercept if old.coeffs is not None else 0.
    return Program(old.terms+terms,np.r_[weights,coef[1:]],intercept+coef[0],old.coefficient_mode)


def select(old,X,y,env,V,v,val_env):
    y,v=np.ravel(y).astype(float),np.ravel(v).astype(float)
    groups=np.unique(env)
    if not supported(X) or len(groups)<3 or not np.array_equal(groups,np.unique(val_env)):
        return old,dict(accepted=False,reason="source_support")
    full=fits(old,X,y)
    ps=predict(old,full,V)
    val=np.sqrt(np.mean([np.mean((ps[val_env==g]-v[val_env==g,None])**2,axis=0) for g in groups],axis=0))
    folds=[]
    for g in groups:
        tr,va=env!=g,val_env==g
        anchor=refit(old,X[tr],y[tr])
        ps=predict(anchor,fits(anchor,X[tr],y[tr]),V[va])
        folds.append(np.mean((ps-v[va,None])**2,axis=0))
    errors=np.sqrt(np.array(folds))
    blocked=np.sqrt(np.mean(errors**2,axis=0))
    wins=(errors[:,1:]<errors[:,[0]]).sum(0)
    ok=np.flatnonzero((blocked[1:]<.999*blocked[0]) & (val[1:]<=1.001*val[0])
                      & (wins>=len(groups)/2))
    report=dict(accepted=bool(len(ok)),validation=val.tolist(),blocked=blocked.tolist(),wins=wins.tolist())
    if not len(ok):
        return old,report
    i=int(ok[np.argmin(blocked[1:][ok]/blocked[0]+val[1:][ok]/val[0])])
    result=serialize(old,full[i])
    np.testing.assert_allclose(result.predict(V)[0],predict(old,full,V)[:,i+1],rtol=2e-4,atol=2e-5)
    return result,dict(report,config=i,nodes=result.count_unique_nodes())
