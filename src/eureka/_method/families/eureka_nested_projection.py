"""Learn a compact composition of sine projections and export it as a symbolic Program."""
import numpy as np
from scipy.optimize import minimize

from eureka._method.eureka import constant, expr, variable
from eureka._symbolic.core.ast import Program


def affine_node(inputs, weights, bias):
    node = constant(bias)
    for arg, w in zip(inputs, weights):
        if w:
            node = expr("+", node, expr("*", constant(w), arg))
    return node


class NestedProjection:
    def __init__(self,X,y,seed=0,activation="sin",iteration_limit=350,hidden=(16,12),row_limit=4096):
        X, y = np.asarray(X,float), np.asarray(y,float).ravel()
        sd = X.std(0)
        active = np.flatnonzero(sd>1e-6)
        z = (X[:,active]-X[:,active].mean(0))/sd[active]
        assoc = np.abs(z.T @ (y-y.mean()))
        self.axes = active[np.argsort(-assoc,kind="stable")[:12]]
        self.center, self.scale = X[:,self.axes].mean(0), sd[self.axes]
        rng = np.random.default_rng(seed)
        self.activation = activation
        if activation not in ("sin","hinge"):
            raise ValueError("Unknown projection activation")
        ids = np.sort(rng.choice(len(y),min(row_limit,len(y)),replace=False))
        Z = (X[ids][:,self.axes] - self.center)/self.scale
        target = y[ids]
        d = len(self.axes)
        h = min(hidden[0],max(2,len(ids)//32))
        k = min(hidden[1],max(2,len(ids)//48))
        shapes = [(d,h),(h,),(h,k),(k,),(k,),(d,),(1,)]
        cuts = np.cumsum([int(np.prod(s)) for s in shapes])[:-1]

        def unpack(theta):
            return [v.reshape(s) for v,s in zip(np.split(theta,cuts),shapes)]

        def objective(theta):
            w,b,L,c,a,v,m = unpack(theta)
            U = Z@w+b
            P = np.sin(U) if activation=="sin" else np.maximum(U,0.)
            T = P@L+c
            H = np.sin(T) if activation=="sin" else np.maximum(T,0.)
            err = H@a+Z@v+m[0]-target
            deriv = err/len(Z)
            upper = deriv[:,None]*a*(np.cos(T) if activation=="sin" else (T>0.))
            lower = (upper@L.T)*(np.cos(U) if activation=="sin" else (U>0.))
            lam = 1e-5
            penalty = np.sum(w*w)+np.sum(L*L)+a@a+v@v
            gradient = np.r_[(Z.T@lower+lam*w).ravel(),lower.sum(0),(P.T@upper+lam*L).ravel(),
                             upper.sum(0),H.T@deriv+lam*a,Z.T@deriv+lam*v,deriv.sum()]
            return .5*(err@deriv+lam*penalty),gradient

        theta = np.r_[rng.normal(size=d*h)/np.sqrt(max(1,d)),rng.uniform(-1.,1.,h),
                      rng.normal(size=h*k)/np.sqrt(h),rng.uniform(-1.,1.,k),rng.normal(scale=.15,size=k),
                      np.zeros(d),y.mean()]
        sol = minimize(objective,theta,jac=True,method="L-BFGS-B",options={"maxiter":iteration_limit,"ftol":1e-10,"maxcor":8})
        self.w,self.b,self.L,self.c,self.a,self.v,self.offset = unpack(sol.x)

    def predict(self,X):
        z = (np.asarray(X,float)[:,self.axes]-self.center)/self.scale
        activate = np.sin if self.activation=="sin" else lambda t:np.maximum(t,0.)
        return activate(activate(z@self.w+self.b)@self.L+self.c)@self.a+z@self.v+self.offset[0]

    def program(self):
        inputs = [variable(j) for j in self.axes]
        w = self.w/self.scale[:,None]
        def activate(node):
            return expr("sin",node) if self.activation=="sin" else expr("*",node,expr("gt",node,constant(0.)))
        first = [activate(affine_node(inputs,col,b))
                 for col,b in zip(w.T,self.b-self.center@w)]
        second = [activate(affine_node(first,col,c)) for col,c in zip(self.L.T,self.c)]
        linear = self.v/self.scale
        return Program(inputs+second,np.r_[linear,self.a],float(self.offset[0]-self.center@linear))
