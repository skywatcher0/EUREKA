"""Fit one closed-form EUREKA predictor to source environments."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from eureka._method.families.eureka_oblique import gate_candidates
from eureka._method.families.blocked_selection import SETTINGS, choose, loeo, max_abs_parameter, refit, score_pool
from eureka._symbolic.core.ast import Node, Program
from eureka._symbolic.search.trace_repair import forward_path


def variable(j):
    return Node("var", feature_idx=int(j))


def constant(x):
    return Node("const", constant=float(x))


def expr(op, *children):
    return Node(op, children=list(children))


def smooth_paths(X, y):
    """Build sparse paths over hinges, stable nonlinear terms and their interactions."""
    order = np.argsort(-np.abs((X-X.mean(0)).T @ (y-y.mean())), kind="stable")[:12]
    factors, nodes = [], []
    waves = []
    for j in order:
        x, leaf = X[:, j], variable(j)
        if x.std() < 1e-8:
            continue
        nodes.append(leaf)
        factors.append(leaf)
        for cut in np.unique(np.quantile(x, [.1,.3,.5,.7,.9])):
            delta = expr("-", leaf, constant(cut))
            hinge = expr("*", delta, expr("gt", leaf, constant(cut)))
            nodes.append(hinge)
            factors.append(hinge)
        square = expr("*", leaf, leaf)
        nodes.extend([square, expr("exp", expr("*", constant(.5), leaf)),
                      expr("exp", expr("*", constant(-.5), leaf)),
                      expr("/", leaf, expr("+", constant(1), square))])
        for frequency in np.pi*np.arange(1,33):
            phase = expr("*", constant(frequency), leaf)
            sine, cosine = expr("sin", phase), expr("cos", phase)
            cols = np.column_stack([sine.evaluate(X), cosine.evaluate(X)]).astype(float)
            cols -= cols.mean(0)
            coeff = np.linalg.lstsq(cols, y-y.mean(), rcond=1e-8)[0]
            waves.append((float(np.sum((cols@coeff)**2)), sine, cosine))
    if not nodes:
        return []
    proposals=[]
    for _,sine,cosine in sorted(waves, key=lambda w:-w[0])[:4]:
        nodes.extend([sine,cosine])
        proposals.extend([Program([sine]),Program([cosine]),Program([sine,cosine])])
    # Screen product terms by association with the additive residual on source rows.
    C = np.column_stack([t.evaluate(X) for t in nodes]).astype(float)
    additive = forward_path(C,y,12,lambda cols: True)
    if additive and factors:
        fit = refit(Program([nodes[k] for k in additive[-1][0]]),X,y)
        residual = y - fit.predict(X)[0]
        F = np.column_stack([t.evaluate(X) for t in factors]).astype(float)
        hits=[]
        for i,node in enumerate(factors):
            partners=[k for k in range(i+1,len(factors))
                      if factors[k].collect_feature_indices()!=node.collect_feature_indices()]
            if not partners:
                continue
            block=F[:,partners]*F[:,i,None]
            block-=block.mean(0)
            score=np.abs(block.T@residual)/np.maximum(1e-8,np.linalg.norm(block,axis=0))
            hits.extend((value,i,k) for value,k in zip(score,partners))
        for _,i,k in sorted(hits,reverse=True)[:40]:
            nodes.append(expr("*",factors[i],factors[k]))
    C = np.column_stack([t.evaluate(X) for t in nodes]).astype(float)
    for indices,_ in forward_path(C,y,min(30,max(3,len(y)//16)),lambda cols: True):
        if len(indices) in (1,2,3,4,6,8,12,16,20,24,30):
            proposals.append(Program([nodes[k] for k in indices]))
    return proposals


def projection_paths(X, y, rng):
    """Learn smooth ridge functions, prune each direction, then refit its shape."""
    if len(y)<60:
        return []
    mean, scale = X.mean(0), X.std(0)
    active=np.flatnonzero(scale>1e-5)
    if not len(active):
        return []
    X=(X[:,active]-mean[active])/scale[active]
    order=np.argsort(-np.abs(X.T@(y-y.mean())),kind="stable")[:12]
    features=active[order]
    X=X[:,order]
    d=len(features)
    proposals=[]
    for width in (4,10):
        if len(y)<8*width:
            continue
        W=rng.normal(size=(d,width))/np.sqrt(d)
        bias=rng.normal(scale=.5,size=width)
        weights=rng.normal(scale=.1,size=width)
        linear=np.zeros(d)
        offset=np.mean(y)
        mask=np.ones_like(W)

        def unpack(theta):
            return (theta[:d*width].reshape(d,width)*mask, theta[d*width:(d+1)*width],
                    theta[(d+1)*width:(d+2)*width],theta[(d+2)*width:-1],theta[-1])

        def objective(theta):
            w,b,a,v,c=unpack(theta)
            z=expit(X@w+b)
            err=z@a+X@v+c-y
            deriv=err/len(y)
            local=deriv[:,None]*a*z*(1-z)
            reg=1e-4
            grad=np.concatenate([((X.T@local+reg*w)*mask).ravel(),local.sum(0),
                                 z.T@deriv+reg*a,X.T@deriv+reg*v,[deriv.sum()]])
            loss=.5*(err@err/len(y)+reg*(np.sum(w*w)+a@a+v@v))
            return loss,grad

        theta=np.r_[W.ravel(),bias,weights,linear,offset]
        for dense in (True,False):
            result=minimize(objective,theta,method="L-BFGS-B",jac=True,
                            options={"maxiter":100,"ftol":1e-10,"maxcor":6})
            theta=result.x
            W,bias,weights,linear,offset=unpack(theta)
            if dense:
                keep=np.argsort(-np.abs(W),axis=0,kind="stable")[:3]
                mask=np.zeros_like(W)
                np.put_along_axis(mask,keep,1,axis=0)
        terms=[variable(j) for j in features]
        for h in range(width):
            selected=np.flatnonzero(mask[:,h])
            coeff=W[selected,h]/scale[features[selected]]
            bias_raw=bias[h]-coeff@mean[features[selected]]
            # sigmoid(z) = 1/(1+exp(-z)), using the portable historical AST.
            arg=constant(-bias_raw)
            for k,v in zip(selected,coeff):
                arg=expr("+",arg,expr("*",constant(-v),variable(features[k])))
            terms.append(expr("/",constant(1.),expr("+",constant(1.),expr("exp",arg))))
        proposals.append(Program(terms))
    return proposals


def fit_eureka(X_train, y_train, env_train, X_val, y_val, env_val, seed=0):
    X=np.asarray(X_train,np.float32)
    y=np.asarray(y_train,float).reshape(-1)
    V=np.asarray(X_val,np.float32)
    v=np.asarray(y_val,float).reshape(-1)
    env,ev=np.asarray(env_train).reshape(-1),np.asarray(env_val).reshape(-1)
    rng=np.random.default_rng(seed)
    indices=np.sort(rng.choice(len(X),min(len(X),1536),replace=False))
    xs,ys=X[indices].astype(float),y[indices]
    programs,table=score_pool(gate_candidates(X,y,seed),X,y,env,V,v,ev,seed)
    for row in table:
        if row["origin"]=="search":
            row["origin"]="oblique"
    seen={p.structural_hash() for p in programs}
    baseline=choose(table,"loeo")
    core=programs[baseline["chosen"]["index"]]
    smooth=smooth_paths(xs,ys)
    projections=projection_paths(xs,ys,rng)
    features={}
    for candidate in smooth+projections:
        for term in candidate.terms:
            features[term.structural_signature()]=term
    existing={t.structural_signature() for t in core.terms}
    extra=[t for key,t in features.items() if key not in existing]
    corrected=[]
    if extra:
        cols=np.column_stack([t.evaluate(xs) for t in extra]).astype(float)
        residual=ys-core.predict(xs)[0]
        for selected,_ in forward_path(cols,residual,12,lambda cols: True):
            if len(selected) in (1,2,4,8,12):
                corrected.append(Program(core.terms+[extra[j] for j in selected],
                                         coefficient_mode=core.coefficient_mode))
    for family, paths in (("harmonic_spline",smooth),("sparse_projection",projections),
                          ("residual_completion",corrected)):
        for p in paths:
            key=p.structural_hash()
            if key in seen or p.count_unique_nodes()>160 or p.count_nodes()>480:
                continue
            seen.add(key)
            fitted=refit(p,X,y)
            val=float(np.sqrt(np.mean((fitted.predict(V)[0]-v)**2)))
            score=loeo(fitted,X,y,env,V,v,ev)[0] if len(np.unique(env))>=2 else val
            coef=max_abs_parameter(fitted)
            table.append(dict(index=len(programs),origin=family,nodes=p.count_unique_nodes(),
                              max_abs_coefficient=coef,source_val_rmse=val,loeo_rmse=score,
                              admissible=bool(np.isfinite(val+score) and coef<=SETTINGS.max_abs_coefficient)))
            programs.append(fitted)
    decision=choose(table,"loeo")
    chosen=decision["chosen"]
    return programs[chosen["index"]],dict(choices={"loeo":decision},
                predecessor=baseline["chosen"],candidates=len(table))
