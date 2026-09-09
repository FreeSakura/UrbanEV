import numpy as np
import pytest
from urbanev_forecast.quantile_head import quantile_design,fit_quantile_head,predict_quantile_head


def test_uniform_distribution_mean_and_quantile_bound():
    levels=np.arange(.1,1,.1)
    x,w,result=quantile_design(levels[None,:],levels)
    assert result["lower"][0]==pytest.approx(.45)
    assert result["upper"][0]==pytest.approx(.55)
    assert result["midpoint"][0]==pytest.approx(.5)
    assert w.sum()==pytest.approx(1) and (w>=0).all()


def test_monotone_quantile_integral_within_bounds():
    rng=np.random.default_rng(51)
    values=np.sort(rng.uniform(size=(20,1001)),axis=1)
    levels=np.arange(.1,1,.1)
    q=np.quantile(values,levels,axis=1).T
    _,_,result=quantile_design(q,levels)
    mean=((values[:,:-1]+values[:,1:])*.0005).sum(axis=1)
    assert np.all(mean>=result["lower"]-1e-12) and np.all(mean<=result["upper"]+1e-12)


def test_head_learns_training_bias_and_is_bounded():
    rng=np.random.default_rng(4)
    q=np.sort(rng.uniform(.1,.8,(120,3)),axis=1)
    levels=[.1,.5,.9];target=.9*q[:,2]+.1
    head=fit_quantile_head(q,levels,target,ridge=1e-4,max_steps=20000,tolerance=1e-8)
    p=predict_quantile_head(q,head)
    _,_,base=quantile_design(q,levels)
    assert np.mean((p-target)**2)<np.mean((base["midpoint"]-target)**2)
    assert (p>=0).all() and (p<=1).all()
    assert head["converged"]


def test_repair_is_explicit_and_invalid_levels_fail():
    _,_,result=quantile_design([[1.2,-.1,.5]],[.1,.5,.9])
    assert result["repaired_fraction"]==1
    with pytest.raises(ValueError):quantile_design([[.1,.3]],[.9,.1])
