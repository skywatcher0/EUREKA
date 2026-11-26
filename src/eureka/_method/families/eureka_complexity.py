"""Check whether a nested expansion earns its parameter cost on source data."""
import numpy as np

from eureka._method.families.eureka_composed import assess


def capacity(X):
    n = min(len(X), 6000)
    d = min(12, int(np.sum(np.std(X, axis=0, dtype=float)>1e-6)))
    h, k = min(32, max(2,n//32)), min(24, max(2,n//48))
    weights = 2*(d*h + h + h*k + k + k + d + 1)
    return n, weights


def stress_capacity(X, env):
    groups = np.unique(env)
    if len(groups)==1:
        return capacity(X)
    z = (X-X.mean(0))/np.where(X.std(0)>1e-8,X.std(0),1.)
    held = groups[np.argmax([np.sum(z[env==g].mean(0)**2) for g in groups])]
    return capacity(X[env!=held])


def log_gain(before, after):
    floor = np.finfo(float).tiny
    return float(np.log(max(floor, before))-np.log(max(floor, after)))


def assess_complexity(incumbent,X,y,env,V,v,val_env,seed=0):
    env, val_env = np.asarray(env).ravel(), np.asarray(val_env).ravel()
    n, params = capacity(X)
    ns, ps = stress_capacity(X,env)
    err = (incumbent.predict(V)[0].astype(float)-np.asarray(v).ravel())**2
    old = np.mean([np.mean(err[val_env==g]) for g in np.unique(val_env)])
    report = dict(accepted=False,train_rows=n,parameters=params,stress_rows=ns,stress_parameters=ps)
    if min(n-params,ns-ps)<0:
        return incumbent,{**report,"reason":"sample_count"}
    if np.sqrt(old)<=32*np.finfo(np.float32).eps*np.std(y):
        return incumbent,{**report,"reason":"roundoff"}
    candidate, evidence = assess(incumbent,X,y,env,V,v,val_env,seed=seed)
    if not evidence["accepted"]:
        return incumbent,{**report,"reason":"source_validation","nested":evidence}
    # Charge the full candidate parameter count even though the incumbent was also fitted.
    new = old - np.mean(evidence["by_environment"])
    supported = n*log_gain(old,new)-2*params
    report.update(validation_margin=float(supported),nested=evidence)
    if "stress_environment" in evidence:
        stress_margin = ns*log_gain(evidence["source_before"]**2,evidence["source_after"]**2)-2*ps
        report["stress_margin"] = float(stress_margin)
        supported = min(supported,stress_margin)
    if supported<=0:
        return incumbent,{**report,"reason":"parameter_cost"}
    return candidate,{**report,"accepted":True,"reason":"source_gain"}
