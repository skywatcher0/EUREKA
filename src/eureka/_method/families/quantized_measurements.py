"""Source-only check of calibration on a repeated sensor coordinate."""

import numpy as np
from scipy.special import ndtr
from scipy.optimize import least_squares



def response_step(y):
    levels=np.unique(y)
    if len(levels)<10 or len(levels)>len(y)//2:
        return None
    step=float(np.quantile(np.diff(levels),.2))
    if step<=0:
        return None
    # Averaging many steps keeps single-precision label roundoff from accumulating.
    counts=np.rint((levels-levels[0])/step)
    step=np.polyfit(counts,levels,1)[0]
    if np.max(abs((levels-levels[0])/step-counts))>.02:
        return None
    return float(step),float(levels[0])


def expected_level(value,step,origin,width):
    unit=(value-origin)/step
    nearest=np.floor(unit+.5)
    frac=unit-nearest
    return origin+step*(nearest+ndtr((frac-.5)/width)-ndtr((-frac-.5)/width))


def fit_curve(X,y):
    y=np.ravel(y).astype(float)
    resolution=response_step(y)
    if resolution is None:
        return None
    step,origin=resolution
    best=None
    for j,col in enumerate(X.T):
        levels,indices,counts=np.unique(col,return_inverse=True,return_counts=True)
        if not 30<=len(levels)<=min(len(y)//2,4096):
            continue
        means=np.bincount(indices,weights=y)/counts
        floor=np.sqrt(np.mean((means[indices]-y)**2))
        if floor>.015*np.std(y):
            continue
        center=float(np.mean(col,dtype=float))
        scale=float(np.std(col,dtype=float))
        z=(levels.astype(float)-center)/scale
        matrix=np.polynomial.polynomial.polyvander(z,4)
        weight=np.sqrt(counts)
        initial=np.linalg.lstsq(weight[:,None]*matrix,weight*means,rcond=1e-9)[0]
        if np.sqrt(np.average((matrix@initial-means)**2,weights=counts))>.025*np.std(y):
            continue
        def residual(params):
            return weight*(expected_level(matrix@params[:-1],step,origin,params[-1])-means)
        solved=least_squares(residual,np.r_[initial,.35],bounds=(np.r_[np.full(5,-np.inf),.05],
                             np.r_[np.full(5,np.inf),1.]),max_nfev=120)
        pred=expected_level(matrix@solved.x[:-1],step,origin,solved.x[-1])
        values=(counts*means+4.*pred)/(counts+4.)
        score=float(np.sqrt(np.average((pred-means)**2,weights=counts)))
        if best is None or score<best[0]:
            best=(score,j,center,scale,levels.astype(float),values,solved.x.copy(),step,origin)
    return best


def predict(curve,X):
    _,j,center,scale,levels,values,params,step,origin=curve
    x=X[:,j].astype(float)
    y=expected_level(np.polynomial.polynomial.polyval((x-center)/scale,params[:-1]),step,origin,params[-1])
    inside=(x>=levels[0])&(x<=levels[-1])
    y[inside]=np.interp(x[inside],levels,values)
    return y
