"""Public EUREKA fit entry point for the source-validated adaptive basis."""
import numpy as np

from eureka._method.eureka import fit_eureka as fit_pool
from eureka._method.families.eureka import fit_eureka as fit_core


def select_formula(core, expanded, core_score, expanded_score, X_val, y_val, env_val):
    """Accept the expanded winner only on a material fold gain and stable validation gain."""
    y=np.asarray(y_val,dtype=float).reshape(-1)
    env=np.asarray(env_val).reshape(-1)
    a=core.predict(X_val)[0].astype(float)
    b=expanded.predict(X_val)[0].astype(float)
    gains=(a-y)**2-(b-y)**2
    groups=np.unique(env)
    group_gains=np.array([np.mean(gains[env==group]) for group in groups])
    samples=group_gains if len(group_gains)>1 else gains
    se=float(np.std(samples,ddof=1)/np.sqrt(len(samples))) if len(samples)>1 else 0.
    gain=float(group_gains.mean())
    accept=bool(expanded_score < .99*core_score and gain>se and np.mean(group_gains>0)>.5)
    decision=dict(accepted=accept,source_score=expanded_score,core_score=core_score,
                  validation_gain=gain,standard_error=se,by_environment=group_gains.tolist())
    return (expanded if accept else core), decision


def fit_eureka(X_train, y_train, env_train, X_val, y_val, env_val, seed=0):
    """Return one executable Program and a record of its source-only selection."""
    args=(X_train,y_train,env_train,X_val,y_val,env_val)
    expanded,report=fit_pool(*args,seed=seed)
    row=report["choices"]["loeo"]["chosen"]
    core_row=report["predecessor"]
    if row["index"]==core_row["index"]:
        return expanded,dict(origin=row["origin"],accepted=False,
                             chosen_loeo=row["loeo_rmse"],identical_pool_winner=True)
    core,core_report=fit_core(*args,seed=seed)
    original=core_report["choices"]["loeo"]["chosen"]
    program,decision=select_formula(core,expanded,original["loeo_rmse"],row["loeo_rmse"],X_val,y_val,env_val)
    chosen=row if decision["accepted"] else original
    return program,dict(**decision,origin=chosen["origin"],chosen_loeo=chosen["loeo_rmse"])
