"""Check pairwise gate screening with the same quadratic response class on both sides."""
from itertools import combinations, combinations_with_replacement

import numpy as np

from eureka._method.eureka import variable, constant, expr
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def sparse_fit(A, y, budget=3):
    if not len(y):
        # A collapsed gate has a zero least-squares fit on its empty side.
        return [0] * budget, np.zeros(budget + 1)
    scale = np.maximum(A.std(0), 1e-7)
    z = (A-A.mean(0))/scale
    cols, residual = [], y-y.mean()
    best = np.mean(residual**2)
    for _ in range(budget):
        scores = abs(z.T@residual)
        scores[cols] = 0.
        j = int(np.argmax(scores))
        proposed = [*cols, j]
        beta = np.linalg.lstsq(z[:,proposed], y-y.mean(), rcond=1e-6)[0]
        residual = y-y.mean()-z[:,proposed]@beta
        loss = np.mean(residual**2)
        if loss >= best*.995:
            break
        cols,best = proposed,loss
    design = np.column_stack([np.ones(len(y)),A[:,cols]])
    coeffs = np.linalg.lstsq(design,y,rcond=1e-6)[0]
    return cols,coeffs


def fit_gate(X,y,seed):
    y = np.ravel(y).astype(float)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(y),min(640,len(y)),replace=False)
    sample,target=X[rows].astype(float),y[rows]
    linear=(sample-sample.mean(0))/np.maximum(sample.std(0),1e-6)
    interactions=linear[:,:,None]*linear[:,None,:]
    rel=abs(linear.T@(target-target.mean()))
    nonlin=abs(np.einsum("nij,n->ij",interactions,target-target.mean()))/np.maximum(interactions.std(0),1e-6)
    support=np.argsort(-(rel+nonlin.max(1)),kind="stable")[:5].tolist()
    pairs=list(combinations_with_replacement(support,2))
    def matrix(x):
        return np.column_stack([x[:,support],*(x[:,i]*x[:,j] for i,j in pairs)])
    design=matrix(sample)
    # Rank gates by the residual from separate quadratic fits on their two sides.
    scan=np.column_stack([np.ones(len(rows)),design])
    scores=[]
    for i,j in combinations(support,2):
        for slope in (-2.,-1.,-.5,.5,1.,2.):
            z=sample[:,i]+slope*sample[:,j]
            for cut in np.quantile(z,[.3,.5,.7]):
                mask=z>cut
                loss=0.
                for side in (mask,~mask):
                    B,t=scan[side],target[side]
                    beta=np.linalg.lstsq(B,t,rcond=1e-6)[0]
                    loss+=np.sum((B@beta-t)**2)
                scores.append((loss,i,j,slope,cut))
    A=matrix(X.astype(float))
    terms=[variable(i) for i in support]+[expr("*",variable(i),variable(j)) for i,j in pairs]
    best=None
    for _,i,j,slope,cut in sorted(scores)[:3]:
        z=X[:,i]+slope*X[:,j]
        mask=z>cut
        fits=[sparse_fit(A[m],y[m]) for m in (mask,~mask)]
        preds=[coefs[0]+A[:,cols]@coefs[1:] for cols,coefs in fits]
        order=np.argsort(z,kind="stable")
        errors=np.column_stack([(p[order]-y[order])**2 for p in preds])
        total=errors[:,0].sum()+np.cumsum(errors[:,1]-errors[:,0])
        n=len(y)
        possible=np.flatnonzero((z[order][1:]>z[order][:-1])&(np.arange(n-1)>=.15*n)&(np.arange(n-1)<=.85*n))
        if not len(possible):
            continue
        k=possible[np.argmin(total[possible])]
        refined=float((float(z[order[k]])+float(z[order[k+1]]))/2)
        mask=z>refined
        fits=[sparse_fit(A[m],y[m]) for m in (mask,~mask)]
        err=sum(np.sum((coefs[0]+A[m][:,cols]@coefs[1:]-y[m])**2)
                for (cols,coefs),m in zip(fits,(mask,~mask)))
        if best is None or err<best[0]:
            gate=expr("gt",expr("+",variable(i),expr("*",constant(slope),variable(j))),constant(refined))
            branches=[]
            for cols,coefs in fits:
                node=constant(coefs[0])
                for col,coef in zip(cols,coefs[1:]):
                    node=expr("+",node,expr("*",constant(coef),terms[col]))
                branches.append(node)
            best=err,Program([expr("ifelse",gate,*branches)],np.ones(1)),[i,j,slope,refined]
    return best[1:] if best else (None,None)


def select(incumbent,X,y,env,V,v,val_env,seed=0):
    env,val_env=np.ravel(env),np.ravel(val_env)
    groups=np.unique(env)
    if len(X)<300 or not 2<=X.shape[1]<=32 or len(groups)<2:
        return incumbent,dict(accepted=False,reason="support")
    def losses(p,mask):
        return np.mean((p.predict(V[mask])[0].astype(float)-np.ravel(v[mask]))**2)
    initial=np.sqrt([losses(incumbent,val_env==g) for g in np.unique(val_env)])
    if max(initial)<=.035*np.std(y) or np.sum(np.std(X,axis=0)>1e-6)<2:
        return incumbent,dict(accepted=False,reason="small_residual_or_constant")
    model,gate=fit_gate(X,y,seed)
    if model is None:
        return incumbent,dict(accepted=False,reason="screen")
    val=np.asarray([initial,np.sqrt([losses(model,val_env==g) for g in np.unique(val_env)])])
    if np.any(val[1]>=val[0]) or np.mean(val[1]**2)>=.98**2*np.mean(val[0]**2):
        return incumbent,dict(accepted=False,reason="validation",validation=val.tolist())
    folds=[]
    for held in np.array_split(groups,min(3,len(groups))):
        tr=~np.isin(env,held)
        candidate,_=fit_gate(X[tr],y[tr],seed)
        if candidate is None:
            return incumbent,dict(accepted=False,reason="fold_screen")
        fixed=refit(incumbent,X[tr],y[tr])
        for group in held:
            va=val_env==group
            if va.any():
                folds.append([losses(p,va) for p in (fixed,candidate)])
    if not folds:
        return incumbent,dict(accepted=False,reason="unmatched_groups")
    blocked=np.sqrt(np.mean(folds,axis=0))
    accepted=bool(np.all(val[1]<val[0]) and blocked[1]<.95*blocked[0])
    return (model if accepted else incumbent),dict(accepted=accepted,gate=gate,validation=val.tolist(),blocked=blocked.tolist(),nodes=model.count_unique_nodes())
