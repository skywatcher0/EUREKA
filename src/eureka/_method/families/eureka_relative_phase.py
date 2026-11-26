"""Fit a short relative-phase law with optional amplitude coordinates."""
from itertools import combinations
import numpy as np
from eureka._method.eureka import variable,constant,expr
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def design(X, spec):
    i,j,sign,freq,amp = spec
    theta = freq*(X[:,i]+sign*X[:,j])
    waves = np.column_stack([np.cos(theta),np.sin(theta)])
    if not amp:
        return np.column_stack([np.ones(len(X)),waves])
    a = X[:,amp]
    return np.column_stack([np.ones(len(X)),a,a*waves[:,0,None],a*waves[:,1,None]])


def search(X,y,seed):
    rng=np.random.default_rng(seed)
    ids=np.sort(rng.choice(len(X),min(512,len(X)),replace=False))
    sample,truth=X[ids],y[ids]
    axes=[j for j,col in enumerate(sample.T) if np.unique(col).size>=8 and
          np.abs(col).max()<=2*np.pi+.01 and .5<=np.ptp(col)<=2*np.pi+.01]
    if len(axes)>12:
        axes=sorted(axes,key=lambda j:-abs(np.cov(sample[:,j],truth)[0,1]))[:12]
    bound=[j for j,col in enumerate(X.T) if col.min()>=0 and col.max()<=1 and 1<np.unique(col).size<=12]
    best_error=.12*np.var(truth)
    best=None
    for i,j in combinations(axes,2):
        amp=tuple(k for k in bound if k not in (i,j))[:8]
        for group in ([(),amp] if amp else [()]):
            for sign in (-1,1):
                for freq in (1.,2.,3.,4.):
                    spec=i,j,sign,freq,group
                    A=design(sample,spec)
                    coef=np.linalg.lstsq(A,truth,rcond=1e-7)[0]
                    err=float(np.mean((A@coef-truth)**2))
                    if err<best_error:
                        best_error,best=err,spec
    return best


def program(spec,beta):
    i,j,sign,freq,amp=spec
    phase=expr("*",constant(freq),expr("+" if sign==1 else "-",variable(i),variable(j)))
    cosine,sine=expr("cos",phase),expr("sin",phase)
    a=[variable(k) for k in amp]
    terms=a+[expr("*",t,cosine) for t in a]+[expr("*",t,sine) for t in a] if a else [cosine,sine]
    return Program(terms,beta[1:],float(beta[0]))


def select(incumbent,X,y,env,V,v,val_env,seed=0):
    X,V=np.asarray(X,float),np.asarray(V,float)
    y,v,env,val_env=[np.asarray(a).ravel() for a in (y,v,env,val_env)]
    spec=search(X,y,seed)
    if spec is None:
        return incumbent,dict(accepted=False,reason="screen")
    train=design(X,spec)
    model=program(spec,np.linalg.lstsq(train,y,rcond=1e-7)[0])
    # The source validation gate precedes the more expensive structure-refit check.
    def mse(p,h,t):
        return np.mean((p.predict(h)[0].astype(float)-t)**2)
    groups=np.unique(val_env)
    val=np.mean([[mse(p,V[val_env==g],v[val_env==g]) for p in (incumbent,model)] for g in groups],axis=0)
    report=dict(accepted=False,validation=np.sqrt(val).tolist(),spec=[*spec[:4],list(spec[4])])
    if val[1]>=1.01**2*val[0]:
        return incumbent,{**report,"reason":"validation"}
    losses=[]
    train_groups=np.unique(env)
    if len(train_groups)>1:
        for held in np.array_split(train_groups,min(5,len(train_groups))):
            mask=~np.isin(env,held)
            fit=search(X[mask],y[mask],seed)
            if fit is None:
                return incumbent,{**report,"reason":"fold_screen"}
            other=program(fit,np.linalg.lstsq(design(X[mask],fit),y[mask],rcond=1e-7)[0])
            base=refit(incumbent,X[mask],y[mask])
            for g in held:
                m=val_env==g
                if m.any():
                    losses.append([mse(p,V[m],v[m]) for p in (base,other)])
    folds=np.mean(losses,axis=0) if losses else val
    accept=folds[1]<.99**2*folds[0]
    return (model if accept else incumbent),{**report,"reason":"fold","accepted":bool(accept),"blocked":np.sqrt(folds).tolist()}
