"""Estimate nonlinear weather responses with cyclic smoothing over observed phases."""
import numpy as np
from scipy.linalg import solve

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.eureka_cyclic_table import orbit
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def supported(X):
    return 500 <= len(X) <= 15000 and 4 <= X.shape[1] <= 18 and orbit(X) is not None


def readings(X, cycle):
    bounds = []
    for j, col in enumerate(X.T):
        levels = np.unique(col)
        if j in cycle[0] or len(levels)<2:
            continue
        if len(levels)<=5:
            bounds.extend((j,float((a+b)/2),None) for a,b in zip(levels,levels[1:]))
        else:
            knots=np.unique(np.quantile(col,[.02,.25,.55,.8,.98]))
            bounds.extend((j,float(a),float(b)) for a,b in zip(knots,knots[1:]) if b-a>1e-4)
    return bounds


def design(X,cycle,bounds):
    axes,center,radius,angles=cycle
    xy=(X[:,axes].astype(float)-center)/radius
    phase=np.argmax(xy@np.stack([np.cos(angles),np.sin(angles)]),axis=1)
    A=np.column_stack([np.ones(len(X)),*[(X[:,j]>lo).astype(float) if hi is None else
                  np.clip((X[:,j].astype(float)-lo)/(hi-lo),0.,1.) for j,lo,hi in bounds]])
    return phase,A


def surfaces(X,y,env,cycle):
    bounds=readings(X,cycle)
    if not bounds:
        return []
    phase,A=design(X,cycle,bounds)
    k=len(cycle[-1])
    p=A.shape[1]
    _,labels,count=np.unique(env,return_inverse=True,return_counts=True)
    w=len(y)/(len(count)*count[labels])
    gram=np.zeros((k*p,k*p))
    rhs=np.zeros((k,p))
    for h in range(k):
        mask=phase==h
        if mask.sum()<p:
            return []
        part=A[mask]
        sl=slice(h*p,(h+1)*p)
        gram[sl,sl]=part.T@(w[mask,None]*part)
        rhs[h]=part.T@(w[mask]*y[mask])
    difference=np.eye(k)-np.roll(np.eye(k),1,axis=0)
    penalty=np.kron(difference.T@difference,np.eye(p))
    shrink=np.tile(np.r_[.02,np.ones(p-1)],k)
    models=[]
    for smooth,ridge in ((.03,.001),(.15,.003),(.6,.012)):
        G=gram+(len(y)/k)*smooth*penalty
        G.flat[::len(G)+1]+=(len(y)/k)*ridge*shrink
        beta=solve(G,rhs.ravel(),assume_a="pos").reshape(k,p)
        models.append((cycle,bounds,beta))
    return models


def predict(model,X):
    cycle,bounds,beta=model
    indices,A=design(X,cycle,bounds)
    return np.sum(A*beta[indices],axis=1)


def serialize(model):
    cycle,bounds,beta=model
    axes,center,radius,angles=cycle
    xy=[expr("*",constant(1./s),expr("-",variable(j),constant(c))) for j,c,s in zip(axes,center,radius)]
    basis=[]
    for j,lo,hi in bounds:
        node=variable(j)
        term=expr("gt",node,constant(lo))
        if hi is not None:
            ramp=expr("*",constant(1./(hi-lo)),expr("-",node,constant(lo)))
            term=expr("ifelse",expr("gt",node,constant(hi)),constant(1.),
                      expr("ifelse",term,ramp,constant(0.)))
        basis.append(term)
    # A sector is the intersection of two half-planes through the origin.
    # Using actual angle midpoints also supports inputs between observed phases.
    edges=(angles+np.r_[angles[-1]-2*np.pi,angles[:-1]])/2
    def side(angle):
        return expr("+",expr("*",constant(-np.sin(angle)),xy[0]),
                    expr("*",constant(np.cos(angle)),xy[1]))
    def branch(start,stop):
        if stop-start==1:
            coeff=beta[start]
            total=constant(coeff[0])
            for c,node in zip(coeff[1:],basis):
                total=expr("+",total,expr("*",constant(c),node))
            return total
        mid=(start+stop)//2
        gate=expr("*",expr("gt",side(edges[start]),constant(0.)),
                  expr("lt",side(edges[mid]),constant(0.)))
        return expr("ifelse",gate,branch(start,mid),branch(mid,stop))
    return Program([branch(0,len(angles))],np.ones(1),0.)


def select(base,X,y,env,V,v,val_env):
    y,v,env,val_env=map(np.ravel,(y,v,env,val_env))
    cycle=orbit(X) if 500<=len(X)<=15000 and 4<=X.shape[1]<=18 else None
    groups=np.unique(env)
    if cycle is None or len(groups)<2 or not np.array_equal(groups,np.unique(val_env)):
        return base,dict(accepted=False,reason="source_support")
    full=surfaces(X,y.astype(float),env,cycle)
    if not full:
        return base,dict(accepted=False,reason="phase_support")
    preds=[base.predict(V)[0],*[predict(m,V) for m in full]]
    val=np.sqrt(np.mean([[np.mean((p[val_env==g]-v[val_env==g])**2) for p in preds] for g in groups],axis=0))
    if np.min(val[1:])>=.995*val[0]:
        return base,dict(accepted=False,reason="validation",validation=val.tolist())
    folded=[]
    for g in groups:
        keep,held=env!=g,val_env==g
        c=orbit(X[keep])
        if c is None or c[0]!=cycle[0] or len(c[-1])!=len(cycle[-1]):
            return base,dict(accepted=False,reason="fold_orbit")
        models=surfaces(X[keep],y[keep].astype(float),env[keep],c)
        if not models:
            return base,dict(accepted=False,reason="fold_support")
        p=[refit(base,X[keep],y[keep]).predict(V[held])[0],*[predict(m,V[held]) for m in models]]
        folded.append([np.mean((a-v[held])**2) for a in p])
    losses=np.array(folded)
    blocked=np.sqrt(losses.mean(0))
    wins=(losses[:,1:]<losses[:,[0]]).sum(0)
    eligible=np.flatnonzero((blocked[1:]<.99*blocked[0])&(val[1:]<.995*val[0])&(wins>=len(groups)/2))
    record=dict(accepted=bool(len(eligible)),validation=val.tolist(),blocked=blocked.tolist(),wins=wins.tolist())
    if not len(eligible):
        return base,record
    choice=int(eligible[np.argmin(blocked[1:][eligible])])
    program=serialize(full[choice])
    for Z in (X,V):
        np.testing.assert_allclose(program.predict(Z)[0],predict(full[choice],Z),rtol=2e-4,atol=1e-4)
    record.update(choice=choice,nodes=program.count_unique_nodes())
    return program,record
