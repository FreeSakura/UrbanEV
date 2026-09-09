"""Bounded predictive-quantile functionals and a small convex MSE adapter."""
from __future__ import annotations
import numpy as np


def quantile_design(quantiles, levels):
    q, t = np.asarray(quantiles, float), np.asarray(levels, float)
    if q.ndim != 2 or q.shape[0] == 0 or t.ndim != 1 or not len(t) or q.shape[1] != len(t):
        raise ValueError("Expected nonempty [samples, quantiles] and matching levels")
    if not np.isfinite(q).all() or not np.isfinite(t).all() or ((t<=0)|(t>=1)).any() or (np.diff(t)<=0).any():
        raise ValueError("Finite forecasts and strictly increasing interior levels required")
    repaired=np.sort(np.clip(q,0,1),axis=1)
    changed=float(np.mean(np.any(repaired!=q,axis=1)))
    x=np.column_stack((np.zeros(len(q)),repaired,np.ones(len(q))))
    knots=np.r_[0,t,1]; widths=np.diff(knots)
    lower=x[:,:-1]@widths;upper=x[:,1:]@widths
    weights=np.r_[widths[0]/2,(widths[:-1]+widths[1:])/2,widths[-1]/2]
    return x,weights,{"lower":lower,"upper":upper,"midpoint":x@weights,"repaired_fraction":changed}


def project_simplex(vector):
    v=np.asarray(vector,float)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():
        raise ValueError("Projection requires a nonempty finite vector")
    ordered=np.sort(v)[::-1]
    shifts=(np.cumsum(ordered)-1)/np.arange(1,len(v)+1)
    rho=np.flatnonzero(ordered>shifts)[-1]
    return np.maximum(v-shifts[rho],0)


def fit_quantile_head(quantiles, levels, target, ridge=.01, max_steps=5000, tolerance=1e-9):
    x,prior,_=quantile_design(quantiles,levels)
    y=np.asarray(target,float)
    if y.shape!=(len(x),) or not np.isfinite(y).all() or ((y<0)|(y>1)).any():
        raise ValueError("Training targets must be finite, bounded and one per sample")
    if not np.isfinite(ridge) or ridge<=0 or max_steps<1 or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError("Use positive ridge, steps and tolerance")
    gram=x.T@x/len(x);linear=x.T@y/len(x)
    lipschitz=2*(np.linalg.eigvalsh(gram)[-1]+ridge)
    weights=prior.copy();converged=False
    for iteration in range(1,max_steps+1):
        gradient=2*(gram@weights-linear+ridge*(weights-prior))
        updated=project_simplex(weights-gradient/lipschitz)
        residual=float(lipschitz*np.linalg.norm(updated-weights))
        weights=updated
        if residual<=tolerance:
            converged=True;break
    return {"weights":weights,"levels":np.asarray(levels,float),"ridge":ridge,"iterations":iteration,
            "projected_gradient_residual":residual,"converged":converged}


def predict_quantile_head(quantiles, head):
    x,_,_=quantile_design(quantiles,head["levels"])
    weights=np.asarray(head["weights"],float)
    if weights.shape!=(x.shape[1],) or not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1):
        raise ValueError("Invalid simplex weights")
    return x@weights
