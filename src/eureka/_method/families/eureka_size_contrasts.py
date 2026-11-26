"""Vary the slopes of correlated measurements over their common size coordinate."""
from itertools import combinations_with_replacement

import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def supported(X):
    if not 4<=X.shape[1]<=12 or not 350<=len(X)<=8000:
        return False
    if any(np.unique(x).size<20 for x in X.T):
        return False
    corr=np.corrcoef(X.T)
    return bool(np.min(corr[np.triu_indices(len(corr),1)])>.45 and
                np.linalg.eigvalsh(corr)[-1]>.75*len(corr))


def ramps(x,knots):
    return np.clip((x[:,None]-knots[:-1])/np.diff(knots),0.,1.)


def design(X,spec):
    mu,sd,axis,knots,center,scale,location,spread,second=spec
    z=(X.astype(float)-mu)/sd
    size=z[:,axis]
    basis=ramps(size,knots)
    common=np.column_stack([np.ones(len(z)),size,basis])
    contrasts=np.clip((z-common@center)/scale,-4.,4.)
    trend=np.clip((size-location)/spread,-2.,2.)
    blocks=[common[:,1:],contrasts,contrasts*trend[:,None]]
    if second:
        blocks.append(np.column_stack([contrasts[:,i]*contrasts[:,j]
                                       for i,j in combinations_with_replacement(range(z.shape[1]),2)]))
    return np.column_stack(blocks)


def fit(X,y,n,second,penalty):
    X=np.asarray(X,float)
    mu,sd=X.mean(0),X.std(0)
    z=(X-mu)/sd
    pc=np.linalg.eigh(z.T@z)[1][:,-1]
    axis=int(np.argmax(abs(pc)))
    size=z[:,axis]
    knots=np.unique(np.quantile(size,np.linspace(.02,.98,n)))
    common=np.column_stack([np.ones(len(z)),size,ramps(size,knots)])
    center=np.linalg.lstsq(common,z,rcond=1e-7)[0]
    scale=np.maximum((z-common@center).std(0),.04)
    spec=mu,sd,axis,knots,center,scale,size.mean(),size.std(),second
    A=design(X,spec)
    mean=A.mean(0)
    spread=np.maximum(A.std(0),.05)
    Z=(A-mean)/spread
    gram=Z.T@Z
    gram.flat[::len(gram)+1]+=penalty*len(X)
    coef=np.linalg.solve(gram,Z.T@(y-y.mean()))/spread
    return spec,coef,float(y.mean()-mean@coef)


def linear(nodes,coefs,bias=0.):
    total=constant(bias)
    for node,coef in zip(nodes,coefs,strict=True):
        total=expr("+",total,expr("*",constant(coef),node))
    return total


def clip(node,lo,hi):
    return expr("ifelse",expr("lt",node,constant(lo)),constant(lo),
                expr("ifelse",expr("gt",node,constant(hi)),constant(hi),node))


def serialize(fit):
    (mu,sd,axis,knots,center,scale,location,spread,second),coef,bias=fit
    z=[expr("*",constant(1./s),expr("-",variable(j),constant(m))) for j,(m,s) in enumerate(zip(mu,sd))]
    size=z[axis]
    # Share the same measured size coordinate and ramps in every contrast.
    basis=[clip(expr("*",constant(1./(hi-lo)),expr("-",size,constant(lo))),0.,1.) for lo,hi in zip(knots[:-1],knots[1:])]
    common=[size,*basis]
    contrasts=[]
    for j,node in enumerate(z):
        center_j=linear(common,center[1:,j],center[0,j])
        contrasts.append(clip(expr("*",constant(1./scale[j]),expr("-",node,center_j)),-4.,4.))
    trend=clip(expr("*",constant(1./spread),expr("-",size,constant(location))),-2.,2.)
    terms=[*common,*contrasts,*[expr("*",trend,node) for node in contrasts]]
    if second:
        terms.extend(expr("*",a,b) for a,b in combinations_with_replacement(contrasts,2))
    return Program(terms,coef,bias)


def select(base,X,y,env,V,v,val_env):
    y,v,env,val_env=map(np.ravel,(y,v,env,val_env))
    y,v=y.astype(float),v.astype(float)
    groups=np.unique(env)
    if not supported(X) or len(groups)<2 or not np.array_equal(groups,np.unique(val_env)):
        return base,dict(accepted=False,reason="source_support")
    configs=[(n,second,p) for n in (4,7) for second in (False,True) for p in (.001,.02)]
    masks=[val_env==g for g in groups]
    def predictions(tr,va,incumbent):
        fitted=[fit(X[tr],y[tr],*cfg) for cfg in configs]
        return [incumbent.predict(V[va])[0],*[design(V[va],s)@c+b for s,c,b in fitted]],fitted
    folded=[]
    for g,va in zip(groups,masks):
        tr=env!=g
        if not supported(X[tr]):
            return base,dict(accepted=False,reason="fold_support")
        p,_=predictions(tr,va,refit(base,X[tr],y[tr]))
        folded.append([np.mean((z-v[va])**2) for z in p])
    blocked=np.sqrt(np.mean(folded,axis=0))
    p,models=predictions(np.ones(len(X),bool),np.ones(len(V),bool),base)
    val=np.sqrt(np.mean([[np.mean((z[m]-v[m])**2) for z in p] for m in masks],axis=0))
    allowed=np.flatnonzero((blocked[1:]<.995*blocked[0])&(val[1:]<.998*val[0]))
    report=dict(accepted=bool(len(allowed)),blocked=blocked.tolist(),validation=val.tolist())
    if not len(allowed):
        return base,report
    i=allowed[np.argmin(blocked[1:][allowed])]
    model=serialize(models[i])
    s,c,b=models[i]
    for Z in (X,V):
        np.testing.assert_allclose(model.predict(Z)[0],design(Z,s)@c+b,rtol=3e-4,atol=5e-5)
    report.update(config=configs[i],nodes=model.count_unique_nodes())
    return model,report
