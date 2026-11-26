"""Source selection of phase responses with a shared measurement background."""
import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._method.families import eureka_relative_phase as phase
from eureka._method.families.blocked_selection import refit
from eureka._symbolic.core.ast import Program


def multiply(a, b):
    return expr("*", a, b)


def basis(spec, order, modulation):
    i, j, sign, w, amplitude = spec
    theta = multiply(constant(w), expr("+" if sign == 1 else "-", variable(i), variable(j)))
    cos, sin = expr("cos", theta), expr("sin", theta)
    if amplitude:
        # At extinction the leading amplitude term vanishes, identifying a shared offset
        # even when source blocking leaves a single illumination level.
        response = [expr("+", constant(1.), cos), sin]
    else:
        response = [cos, sin]
    for k in range(2, order + 1):
        harmonic = multiply(constant(k), theta)
        cosine = expr("cos", harmonic)
        if amplitude:
            cosine = expr("-", cosine, constant((-1.)**k))
        response.extend([cosine, expr("sin", harmonic)])
    if modulation:
        for axis in (i, j):
            angle = multiply(constant(w), variable(axis))
            for fn in ("cos", "sin"):
                local = expr(fn, angle)
                response.extend([multiply(local, expr("+",constant(1.),cos)), multiply(local, sin)])
    if amplitude:
        return [multiply(variable(a), term) for a in amplitude for term in response]
    return response


def fit(terms, X, y):
    mat = np.column_stack([term.evaluate(X) for term in terms]).astype(float)
    y = np.ravel(y).astype(float)
    mean = mat.mean(0)
    centered = mat - mean
    gram = centered.T @ centered
    gram.flat[::len(gram)+1] += .002 * len(y)
    beta = np.linalg.solve(gram, centered.T @ (y-y.mean()))
    return Program(terms, beta, float(y.mean()-mean@beta))


def select(incumbent, X, y, env, V, v, val_env, seed=0):
    env, val_env, v = map(np.ravel, (env, val_env, v))
    groups = np.unique(env)
    if len(X)<80 or len(groups)<2 or X.shape[1]<2:
        return incumbent, dict(accepted=False,reason="support")
    spec = phase.search(np.asarray(X,float), np.ravel(y), seed)
    if spec is None:
        return incumbent,dict(accepted=False,reason="phase_screen")
    i,j,sign,w,amp=spec
    # Keep one column per physically indistinguishable amplitude setting.
    unique=[]
    for k in amp:
        if all(not np.array_equal(X[:,k],X[:,other]) for other in unique):
            unique.append(k)
    spec=(i,j,sign,w,tuple(unique))
    def mse(p,mask):
        return float(np.mean((p.predict(V[mask])[0].astype(float)-v[mask])**2))
    masks=[val_env==g for g in np.unique(val_env)]
    old=np.sqrt(np.mean([mse(incumbent,m) for m in masks]))
    proposals=[]
    for order in (1,2,3):
        for modulated in (False,True):
            terms=basis(spec,order,modulated)
            model=fit(terms,X,y)
            val=np.sqrt(np.mean([mse(model,m) for m in masks]))
            if val<=1.02*old:
                proposals.append((terms,model,order,modulated,val))
    if not proposals:
        return incumbent,dict(accepted=False,reason="source_validation",before=old)
    scores=[[] for _ in proposals]
    reference=[]
    for held in np.array_split(groups,min(5,len(groups))):
        tr=~np.isin(env,held)
        source_spec=phase.search(np.asarray(X[tr],float),np.ravel(y)[tr],seed)
        if source_spec is None or source_spec[:4]!=spec[:4]:
            return incumbent,dict(accepted=False,reason="unstable_phase")
        valid=[val_env==g for g in held if np.any(val_env==g)]
        base=refit(incumbent,X[tr],np.ravel(y)[tr])
        reference.extend(mse(base,m) for m in valid)
        for losses,(terms,*_) in zip(scores,proposals):
            model=fit(terms,X[tr],np.ravel(y)[tr])
            losses.extend(mse(model,m) for m in valid)
    if not reference:
        return incumbent,dict(accepted=False,reason="no_matched_source")
    blocked=np.sqrt(np.mean(scores,axis=1))
    choice=int(np.argmin(blocked))
    _,model,order,modulated,val=proposals[choice]
    base_rmse=float(np.sqrt(np.mean(reference)))
    accepted=bool(blocked[choice]<.985*base_rmse)
    return (model if accepted else incumbent),dict(accepted=accepted,reason="blocked",order=order,modulation=modulated,
        validation=[float(old),float(val)],blocked=[base_rmse,float(blocked[choice])],nodes=model.count_unique_nodes())
