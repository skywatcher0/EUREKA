"""Conditional periodic response with partial pooling of covariate slopes."""
import numpy as np
from scipy.linalg import solve

from eureka._method.eureka import constant, expr, variable
from eureka._method.families import eureka_cyclic_smoothing as cyclic
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def support(X):
    if not 600 <= len(X) <= 30000 or not 4 <= X.shape[1] <= 16:
        return None
    cycle=cyclic.orbit(X)
    if cycle is None:
        return None
    states=[]
    for j,x in enumerate(X.T):
        if j in cycle[0]:
            continue
        levels,n=np.unique(x,return_counts=True)
        if len(n)==2 and n.min()>=.08*len(x):
            states.append((n.min(),-j,j,float(levels.mean(dtype=float))))
    if not states:
        return None
    _,_,axis,cut=max(states)
    return cycle,axis,cut


def fits(X,y,env,params):
    cycle,j,cut=params
    bounds=[b for b in cyclic.readings(X,cycle) if b[0]!=j]
    if not bounds:
        return []
    phase,A=cyclic.design(X,cycle,bounds)
    k,p=len(cycle[-1]),A.shape[1]
    cell=phase+k*(X[:,j]>cut)
    counts=np.bincount(cell,minlength=2*k)
    if counts.min()<4 or 2*k*p>1800:
        return []
    _,ix,ng=np.unique(env,return_inverse=True,return_counts=True)
    w=len(y)/(len(ng)*ng[ix])
    d=np.eye(k)-np.roll(np.eye(k),1,axis=0)
    cycle_penalty=np.kron(np.eye(2),d.T@d)
    state_penalty=np.kron([[1.,-1.],[-1.,1.]],np.eye(k))
    slope=np.diag(np.r_[.003,np.ones(p-1)])
    ridge=np.diag(np.tile(np.r_[.001,np.ones(p-1)],2*k))

    def estimate(weight,penalty):
        gram=penalty.copy()
        rhs=np.empty((2*k,p))
        for h in range(2*k):
            mask=cell==h
            part=A[mask]
            sl=slice(h*p,(h+1)*p)
            gram[sl,sl]+=part.T@(weight[mask,None]*part)
            rhs[h]=part.T@(weight[mask]*y[mask])
        return solve(gram,rhs.ravel(),assume_a="pos").reshape(2,k,p)

    models=[]
    for smooth in (.06,.35):
        penalty=len(y)/(2*k)*(np.kron(smooth*cycle_penalty+.12*state_penalty,slope)+.012*ridge)
        beta=estimate(w,penalty)
        models.append((params,bounds,beta))
        # Downweight exceptional observations relative to their own phase and regime.
        residual=y-np.sum(A*beta.reshape(2*k,p)[cell],axis=1)
        scales=np.array([max(.12,1.4826*np.median(np.abs(residual[cell==h]))) for h in range(2*k)])
        robust=np.minimum(1.,2.5*scales[cell]/np.maximum(np.abs(residual),1e-9))
        models.append((params,bounds,estimate(w*robust,penalty)))
    return models


def predict(model,X):
    (cycle,j,cut),bounds,beta=model
    phase,A=cyclic.design(X,cycle,bounds)
    return np.sum(A*beta[(X[:,j]>cut).astype(int),phase],axis=1)


def serialize(model):
    (cycle,j,cut),bounds,beta=model
    a,b=[cyclic.serialize((cycle,bounds,coef)).terms[0] for coef in beta]
    return Program([expr("ifelse",expr("gt",variable(j),constant(cut)),b,a)],np.ones(1),0.)


def select(old,X,y,env,V,v,val_env):
    params=support(X)
    groups=np.unique(env)
    if params is None or len(groups)<2 or not np.array_equal(groups,np.unique(val_env)):
        return old,dict(accepted=False,reason="source_support")
    y,v=np.ravel(y).astype(float),np.ravel(v).astype(float)
    full=fits(X,y,env,params)
    if not full:
        return old,dict(accepted=False,reason="cell_support")
    pred=np.column_stack([old.predict(V)[0],*[predict(m,V) for m in full]])
    val=np.sqrt(np.mean([np.mean((pred[val_env==g]-v[val_env==g,None])**2,0) for g in groups],0))
    if val[1:].min()>=.995*val[0]:
        return old,dict(accepted=False,reason="validation",validation=val.tolist())
    folds=[]
    for g in groups:
        tr,va=env!=g,val_env==g
        local=support(X[tr])
        if local is None or local[0][0]!=params[0][0] or local[1]!=params[1]:
            return old,dict(accepted=False,reason="fold_support")
        choices=fits(X[tr],y[tr],env[tr],local)
        if not choices:
            return old,dict(accepted=False,reason="fold_cells")
        base=refit(old,X[tr],y[tr]).predict(V[va])[0]
        ps=np.column_stack([base,*[predict(m,V[va]) for m in choices]])
        folds.append(np.mean((ps-v[va,None])**2,0))
    losses=np.array(folds)
    blocked=np.sqrt(losses.mean(0))
    wins=(losses[:,1:]<losses[:,[0]]).sum(0)
    eligible=np.flatnonzero((blocked[1:]<.995*blocked[0])&(val[1:]<.995*val[0])&(wins>=len(groups)/2))
    report=dict(accepted=bool(len(eligible)),axis=params[1],blocked=blocked.tolist(),validation=val.tolist(),wins=wins.tolist())
    if not len(eligible):
        return old,report
    i=int(eligible[np.argmin(blocked[1:][eligible])])
    out=serialize(full[i])
    np.testing.assert_allclose(out.predict(V)[0],predict(full[i],V),rtol=2e-4,atol=1e-5)
    return out,dict(report,config=i,nodes=out.count_unique_nodes())
