"""Range reduction for portable single-frequency programs in float32 arithmetic."""
import numpy as np

from eureka._method.eureka import constant, expr, variable
from eureka._symbolic.core.ast import Program


def harmonic(program):
    if program.coefficient_mode!="term" or len(program.terms)!=1:
        return None
    wave = program.terms[0]
    if wave.op not in ("sin","cos") or wave.children[0].op!="*":
        return None
    left,right = wave.children[0].children
    if left.op=="var":
        left,right = right,left
    if left.op!="const" or right.op!="var" or not np.isfinite(left.constant) or abs(left.constant)<1e-8:
        return None
    return wave.op,left.constant,right.feature_idx


def reduce_phase(program,X,V,v,val_env):
    specification = harmonic(program)
    if specification is None:
        return program,dict(accepted=False,reason="different_structure")
    op,w,j = specification
    period = 2*np.pi/abs(w)
    first,last = np.rint([X[:,j].min()/period,X[:,j].max()/period]).astype(int)
    if last-first<1 or last-first>=128:
        return program,dict(accepted=False,reason="range")
    x = variable(j)
    def branch(lo,hi):
        if lo==hi:
            center = float(np.float32(lo*period))
            correction = float(np.remainder(w*center+np.pi,2*np.pi)-np.pi)
            angle = expr("*",constant(w),expr("-",x,constant(center)))
            if abs(correction)>1e-12:
                angle = expr("+",angle,constant(correction))
            return expr(op,angle)
        mid = (lo+hi)//2
        split = expr("gt",x,constant((mid+.5)*period))
        return expr("ifelse",split,branch(mid+1,hi),branch(lo,mid))
    reduced = Program([branch(first,last)],program.coeffs,program.intercept)
    target = np.asarray(v).ravel()
    labels = np.asarray(val_env).ravel()
    before = (program.predict(V)[0].astype(float)-target)**2
    after = (reduced.predict(V)[0].astype(float)-target)**2
    gains = [float(np.mean((before-after)[labels==e])) for e in np.unique(labels)]
    accepted = bool(min(gains)>0 and after.mean()<.99**2*before.mean())
    return (reduced if accepted else program),dict(accepted=accepted,by_environment=gains,
            validation_rmse_before=float(np.sqrt(before.mean())),validation_rmse_after=float(np.sqrt(after.mean())))
