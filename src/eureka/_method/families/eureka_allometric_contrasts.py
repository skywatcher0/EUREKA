"""Fit a smooth size trend and log-scale contrasts between positively correlated measurements."""
from itertools import combinations_with_replacement

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.eureka_size_contrasts import clip, linear, supported
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def columns(X,spec):
    origin,span,axis,mean,beta,scale,knots=spec
    logs=np.log(np.maximum((X.astype(float)-origin)/span,.01)+1e-8)
    x=logs[:,axis]
    d=np.clip((logs-mean-(x-mean[axis])[:,None]*beta)/scale,-3.,3.)
    zs=[d[:,j] for j in range(d.shape[1]) if j!=axis]
    steps=np.clip((x[:,None]-knots[:-1])/np.diff(knots),0.,1.)
    trend=np.clip((x-knots[0])/(knots[-1]-knots[0]),0.,1.)
    return np.column_stack([x,steps,*zs,*[z*trend for z in zs],
                            *[a*b for a,b in combinations_with_replacement(zs,2)]])


def fit(X,y,offset,penalty):
    X=np.asarray(X,float)
    low,high=np.quantile(X,[.01,.99],axis=0)
    span=np.maximum(high-low,1e-6)
    origin=np.minimum(X.min(0),low-offset*span)
    logs=np.log(np.maximum((X-origin)/span,.01)+1e-8)
    mean=logs.mean(0)
    centered=logs-mean
    pc=np.linalg.eigh(np.corrcoef(logs.T))[1][:,-1]
    axis=int(np.argmax(abs(pc)))
    size=centered[:,axis]
    beta=size@centered/(size@size)
    scale=np.maximum((centered-size[:,None]*beta).std(0),.015)
    knots=np.unique(np.quantile(logs[:,axis],np.linspace(.02,.98,6)))
    spec=origin,span,axis,mean,beta,scale,knots
    A=columns(X,spec)
    center=A.mean(0)
    spread=np.maximum(A.std(0),.02)
    Z=(A-center)/spread
    gram=Z.T@Z
    gram.flat[::len(gram)+1]+=penalty*len(X)
    coef=np.linalg.solve(gram,Z.T@(y-y.mean()))/spread
    return spec,coef,float(y.mean()-center@coef)


def serialize(model):
    (origin,span,axis,mean,beta,scale,knots),coef,bias=model
    logs=[]
    for j,(a,s) in enumerate(zip(origin,span)):
        z=expr("*",constant(1./s),expr("-",variable(j),constant(a)))
        logs.append(expr("log",expr("ifelse",expr("gt",z,constant(.01)),z,constant(.01))))
    x=logs[axis]
    delta=expr("-",x,constant(mean[axis]))
    contrasts=[]
    for j,reading in enumerate(logs):
        if j!=axis:
            residual=expr("-",reading,linear([delta],[beta[j]],mean[j]))
            contrasts.append(clip(expr("*",constant(1./scale[j]),residual),-3.,3.))
    def ramp(lo,hi):
        return clip(expr("*",constant(1./(hi-lo)),expr("-",x,constant(lo))),0.,1.)
    trend=ramp(knots[0],knots[-1])
    terms=[x,*[ramp(a,b) for a,b in zip(knots[:-1],knots[1:])],*contrasts,
           *[expr("*",z,trend) for z in contrasts],
           *[expr("*",a,b) for a,b in combinations_with_replacement(contrasts,2)]]
    return Program(terms,coef,bias)


def select(base,X,y,env,V,v,val_env):
    y,v=np.ravel(y).astype(float),np.ravel(v).astype(float)
    env,val_env=np.ravel(env),np.ravel(val_env)
    groups=np.unique(env)
    if not supported(X) or len(groups)<2 or not np.array_equal(groups,np.unique(val_env)):
        return base,dict(accepted=False,reason="source_support")
    configs=[(offset,ridge) for offset in (.02,.12,.3) for ridge in (.002,.02)]
    folds=[]
    for g in groups:
        tr,va=env!=g,val_env==g
        if not supported(X[tr]):
            return base,dict(accepted=False,reason="fold_support")
        predictions=[refit(base,X[tr],y[tr]).predict(V[va])[0]]
        for config in configs:
            s,c,b=fit(X[tr],y[tr],*config)
            predictions.append(columns(V[va],s)@c+b)
        folds.append([np.mean((p-v[va])**2) for p in predictions])
    blocked=np.sqrt(np.mean(folds,axis=0))
    models=[fit(X,y,*cfg) for cfg in configs]
    preds=[base.predict(V)[0],*[columns(V,s)@c+b for s,c,b in models]]
    val=np.sqrt(np.mean([[np.mean((p[val_env==g]-v[val_env==g])**2) for p in preds] for g in groups],axis=0))
    allowed=np.flatnonzero((blocked[1:]<.995*blocked[0]) & (val[1:]<=1.003*val[0]))
    report=dict(accepted=bool(len(allowed)),blocked=blocked.tolist(),validation=val.tolist())
    if not len(allowed):
        return base,report
    i=allowed[np.argmin(blocked[1:][allowed])]
    model=serialize(models[i])
    for Z in (X,V):
        s,c,b=models[i]
        np.testing.assert_allclose(model.predict(Z)[0],columns(Z,s)@c+b,rtol=2e-4,atol=3e-5)
    report.update(config=configs[i],nodes=model.count_unique_nodes())
    return model,report
