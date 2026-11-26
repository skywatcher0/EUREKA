"""Fit source-localized radial terms and export the selected surface as an AST."""
import numpy as np
from scipy.optimize import minimize
from eureka._method.eureka import constant, expr, variable
from eureka._method.families.blocked_selection import refit
from eureka._method.candidate_selection import select_formula
from eureka._symbolic.core.ast import Program
from eureka._symbolic.court.scorer import _fit_linear_coeffs


class RadialSurface:
    def __init__(self, X, y, seed):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        scale = X.std(0)
        active = np.flatnonzero(scale > 1e-6)
        self.axes = active[np.argsort(-np.abs((X[:,active]-X[:,active].mean(0)).T @ (y-y.mean()))
                                     /scale[active],kind="stable")[:8]]
        self.mean = X[:,self.axes].mean(0)
        self.scale = scale[self.axes]
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(len(X), min(len(X),1024), replace=False))
        z = (X[rows][:,self.axes]-self.mean)/self.scale
        target = y[rows]
        d, h = len(self.axes), min(18,max(2,len(rows)//12))
        def unpack(theta):
            return theta[:d*h].reshape(d,h),theta[d*h:(d+1)*h],theta[(d+1)*h:(d+2)*h],theta[(d+2)*h:-1],theta[-1]
        def objective(theta):
            w,b,a,linear,offset = unpack(theta)
            u = z @ w + b
            g = np.exp(-np.minimum(30.,u*u))
            error = g @ a + z @ linear + offset - target
            derivative = error / len(z)
            local = derivative[:,None] * a * (-2*u*g) * (u*u<30.)
            ridge = .0002
            objective_value = .5*(error @ derivative + ridge*(np.sum(w*w)+a@a+linear@linear))
            gradient = np.r_[(z.T@local+ridge*w).ravel(),local.sum(0),g.T@derivative+ridge*a,
                             z.T@derivative+ridge*linear,derivative.sum()]
            return objective_value,gradient
        initial = np.r_[rng.normal(size=d*h)/np.sqrt(max(d,1)),rng.normal(size=h),
                        rng.normal(scale=.2,size=h),np.zeros(d),target.mean()]
        result = minimize(objective,initial,jac=True,method="L-BFGS-B",options={"maxiter":160,"ftol":1e-9})
        self.weights,self.bias,_,_,_ = unpack(result.x)
        design = self.design(X)
        self.coef, self.offset = _fit_linear_coeffs(design, y, ridge=max(.01,len(y)*.0002))

    def design(self, X):
        z = np.asarray(X, dtype=float)[:, self.axes]
        directions = (z-self.mean)/self.scale @ self.weights + self.bias
        return np.column_stack([z,np.exp(-np.minimum(30.,directions**2))])

    def predict(self, X):
        return self.offset + self.design(X) @ self.coef

    def program(self):
        terms = [variable(k) for k in self.axes]
        for w,bias in zip(self.weights.T,self.bias):
            slopes = w / self.scale
            arg = constant(bias - self.mean @ slopes)
            for j,slope in zip(self.axes,slopes):
                arg = expr("+",arg,expr("*",constant(slope),variable(j)))
            terms.append(expr("exp",expr("*",constant(-1.),expr("*",arg,arg))))
        return Program(terms,self.coef,self.offset)


def assess_radial(incumbent, X, y, env, V, v, val_env, seed=0):
    y, v = np.asarray(y).ravel(), np.asarray(v).ravel()
    env, val_env = np.asarray(env).ravel(), np.asarray(val_env).ravel()
    groups = np.unique(env)
    surface = RadialSurface(X,y,seed)
    errors = []
    for held in np.array_split(groups,min(5,len(groups))):
        if len(groups)==1:
            base, local, H, truth, labels = incumbent, surface, V, v, val_env
        else:
            hold, valid = np.isin(env,held), np.isin(val_env,held)
            train, response = X[~hold], y[~hold]
            base, local = refit(incumbent,train,response), RadialSurface(train,response,seed)
            H, truth = np.vstack([X[hold],V[valid]]), np.r_[y[hold],v[valid]]
            labels = np.r_[env[hold],val_env[valid]]
        diff = np.column_stack([base.predict(H)[0].astype(float)-truth,local.predict(H)-truth])
        errors.extend(np.mean(diff[labels==label]**2,axis=0) for label in np.unique(labels))
    scores = np.sqrt(np.mean(errors,axis=0))
    if scores[1] >= .99*scores[0]:
        return incumbent,dict(accepted=False,source_score=float(scores[1]),core_score=float(scores[0]))
    candidate = surface.program()
    np.testing.assert_allclose(candidate.predict(V)[0],surface.predict(V),atol=1e-4,rtol=2e-4)
    return select_formula(incumbent,candidate,*scores,V,v,val_env)
