"""A shared response curve with low-order, partially pooled specimen effects."""
import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def axis(X, y):
    if not (35<=len(X)<=5000 and 2<=X.shape[1]<=12):
        return None
    denominator = max(np.var(y),1e-10)
    candidates = []
    for j,x in enumerate(X.T):
        grid,index,count = np.unique(x,return_inverse=True,return_counts=True)
        if not 6<=len(grid)<=40 or count.min()<2:
            continue
        _,specimen,repeats = np.unique(np.delete(X,j,axis=1),axis=0,return_inverse=True,return_counts=True)
        if len(repeats)<4 or np.median(repeats)<.6*len(grid):
            continue
        center = np.bincount(index,weights=y)/count
        residual = np.mean((y-center[index])**2)/denominator
        if residual<.12:
            candidates.append((residual,j,specimen))
    return min(candidates,key=lambda t:(t[0],t[1]))[1:] if candidates else None


def fit(X,y,j,penalty,degree):
    grid,idx,count = np.unique(X[:,j],return_inverse=True,return_counts=True)
    curve = np.bincount(idx,weights=y)/count
    span = max(float(np.ptp(curve)),1e-7)
    amplitude = (curve-curve[0])/span
    axes = [k for k in range(X.shape[1]) if k!=j and np.std(X[:,k])>1e-6]
    H = X[:,axes].astype(float)
    mean,sd = H.mean(0),H.std(0)
    H = (H-mean)/sd
    blocks = [H*amplitude[idx,None]**d for d in range(1,degree+1)]
    slopes = np.column_stack(blocks)
    # Partial out speed means before ridge fitting the specimen terms.
    centers = np.array([slopes[idx==k].mean(0) for k in range(len(grid))])
    Z = slopes-centers[idx]
    gram = Z.T@Z
    gram.flat[::gram.shape[0]+1] += penalty*len(y)
    beta = np.linalg.solve(gram,Z.T@(y-curve[idx]))
    return grid,curve-centers@beta,amplitude,axes,mean,sd,beta.reshape(degree,len(axes))


def predict(fit,X,j):
    grid,base,amplitude,axes,mean,sd,beta = fit
    result = np.interp(X[:,j],grid,base)
    H = (X[:,axes].astype(float)-mean)/sd
    for d,b in enumerate(beta,start=1):
        result += np.interp(X[:,j],grid,amplitude**d)*(H@b)
    return result


def linear_curve(axis_node, grid, values):
    slopes = np.diff(values)/np.diff(grid)
    node = constant(values[0])
    for level,coef in zip(grid,np.diff(np.r_[0.,slopes,0.])):
        positive = expr("ifelse",expr("gt",axis_node,constant(level)),
                        expr("-",axis_node,constant(level)),constant(0.))
        node = expr("+",node,expr("*",constant(coef),positive))
    return node


def serialize(fit,j):
    grid,base,amplitude,axes,mean,sd,beta = fit
    speed = variable(j)
    terms = [linear_curve(speed,grid,base)]
    for d,b in enumerate(beta,start=1):
        hull = constant(-mean@(b/sd))
        for k,c in zip(axes,b/sd):
            hull = expr("+",hull,expr("*",constant(c),variable(k)))
        terms.append(expr("*",linear_curve(speed,grid,amplitude**d),hull))
    return Program(terms,np.ones(len(terms)),0.)


def select(base,X,y,env,V,v,val_env):
    y,v = np.ravel(y).astype(float),np.ravel(v).astype(float)
    support = axis(X,y)
    if support is None:
        return base,dict(accepted=False,reason="no_replicated_curve")
    j,block = support
    hull = np.delete(X,j,axis=1)
    validation_hull = np.delete(V,j,axis=1)
    configs = [(a,d) for a in (.001,.01,.1) for d in (1,2)]
    losses = []
    for k in np.unique(block):
        keep = block!=k
        held = np.all(validation_hull==hull[~keep][0],axis=1)
        if not np.any(held) or len(np.unique(X[keep,j]))!=len(np.unique(X[:,j])):
            continue
        incumbent = refit(base,X[keep],y[keep])
        predictions = [incumbent.predict(V[held])[0]]
        for penalty,degree in configs:
            model = fit(X[keep],y[keep],j,penalty,degree)
            predictions.append(predict(model,V[held],j))
        losses.append([np.mean((p-v[held])**2) for p in predictions])
    if len(losses)<4:
        return base,dict(accepted=False,reason="fold_support")
    folds = np.asarray(losses)
    blocked = np.sqrt(folds.mean(0))
    old = np.sqrt(np.mean((base.predict(V)[0]-v)**2))
    full = [fit(X,y,j,a,d) for a,d in configs]
    val = np.array([np.sqrt(np.mean((predict(model,V,j)-v)**2)) for model in full])
    eligible = np.flatnonzero((blocked[1:]<.98*blocked[0]) & (val<=1.05*old) &
                              ((folds[:,1:]<folds[:,[0]]).sum(0)>=len(folds)/2))
    report = dict(accepted=bool(len(eligible)),axis=j,blocks=len(losses),
                  blocked=blocked.tolist(),validation=[float(old),*val.tolist()])
    if not len(eligible):
        return base,report
    i = eligible[np.argmin(blocked[1:][eligible])]
    result = serialize(full[i],j)
    for Z in (X,V):
        np.testing.assert_allclose(result.predict(Z)[0],predict(full[i],Z,j),rtol=2e-4,atol=1e-5)
    report.update(penalty=configs[i][0],degree=configs[i][1],nodes=result.count_unique_nodes())
    return result,report
