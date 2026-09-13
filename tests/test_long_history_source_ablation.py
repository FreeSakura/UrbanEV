import importlib.util
from pathlib import Path
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from urbanev_forecast import history_source as h
from urbanev_forecast import short_state as base

@pytest.fixture(autouse=True)
def two_threads():
    with threadpool_limits(limits=2):yield

def test_fixed_masks_and_duplicate_identities():
    assert {k:len(v) for k,v in h.MASKS.items()}=={'S':8,'O_LONG':174,'D_LONG':175,'OD_LONG':341,'O_LONG_D24':197,'O24_D_LONG':197,'O_LONG_D0':173,'O_LONG_DUP':341}
    assert len(set(h.MASKS['O_LONG_DUP']))==174
    mother=np.arange(341)[None,:]
    dup=mother[:,h.MASKS['O_LONG_DUP']]
    np.testing.assert_array_equal(dup[:,169:336],mother[:,:167])
    assert all(i<168 or i>=336 for i in h.MASKS['O_LONG_D0'])
    np.testing.assert_array_equal(mother[:,h.MASKS['O_LONG_D24']][:,168:192],mother[:,312:336])

def test_visible_history_and_D0_fitting_invariance():
    y=np.arange(600,dtype=float).reshape(300,2)/600;d=y/2;cap=np.array([2.,5.]);oo=np.arange(169,181)
    xx=base.features(y,d,cap,oo,'LONG_OD');other=base.features(y,1-d,cap,oo,'LONG_OD')
    z=base.transform(xx,base.fit_transform(xx));zz=base.transform(other,base.fit_transform(other));res=base.targets(y,oo)-y[oo-1].reshape(-1,1)
    g,b=h.sufficient_statistics(z,res);gg,bb=h.sufficient_statistics(zz,res)
    beta,_=h.solve_mask(g,b,h.MASKS['O_LONG_D0']);betb,_=h.solve_mask(gg,bb,h.MASKS['O_LONG_D0'])
    np.testing.assert_allclose(beta,betb,atol=1e-10)
    future=y.copy();future[192:]=999;df=d.copy();df[191:]=999
    np.testing.assert_array_equal(base.features(future,df,cap,[192],'LONG_OD'),base.features(y,d,cap,[192],'LONG_OD'))
    raw=base.features(y,d,cap,[192],'LONG_OD')[:,h.MASKS['O_LONG_D24']]
    np.testing.assert_array_equal(raw[0,168:192],d[167:191,0])

def test_duplicate_changes_penalty_without_information():
    z=np.linspace(-1,1,31)[:,None];target=z.copy();g,b=h.sufficient_statistics(z,target)
    single,_=h.solve_mask(g,b,[0]);dup,capacity=h.solve_mask(g,b,[0,0])
    expected=float(np.mean(z*z)/(np.mean(z*z)+.005))
    assert abs(dup[1:,0].sum()-expected)<1e-12
    assert dup[1:,0].sum()>single[1,0]
    assert 1<capacity['effective_df']<2

def test_normal_equations_and_training_penalized_objective():
    rng=np.random.default_rng(5);x=rng.normal(size=(40,341));z=base.transform(x,base.fit_transform(x));y=rng.normal(size=(40,12));g,b=h.sufficient_statistics(z,y)
    objectives=[]
    for name in ('S','O_LONG'):
        mask=h.MASKS[name];beta,meta=h.solve_mask(g,b,mask);p=beta[0]+z[:,mask]@beta[1:]
        objectives.append(np.sum((p-y)**2)/len(y)+.01*np.sum(beta[1:]**2))
        assert meta['normal_equation_relative_residual']<1e-10
    assert objectives[1]<=objectives[0]+1e-10

def test_loss_and_factorial_reconstruction_without_risk_monotonicity():
    y=np.array([[[.1,.8],[.2,.6]]]);a=np.array([[[.0,.4],[.3,.8]]]);b=np.array([[[.2,.7],[.1,.9]]])
    report,g1,g2=h.loss_attribution(y,a,b)
    for key,g in [('mae',g1),('mse',g2)]:
        assert abs(report[key]['positive_gain_sum']+report[key]['negative_gain_sum']-g.sum())<1e-12
    scores={k:float(i+1) for i,k in enumerate(h.MASKS)};out=h.factorial(scores)
    assert out['path_identity_max_error']<1e-12
    assert out['D_given_longO']<0  # Negative empirical gain remains a valid output.

def test_stage_budget_and_forbidden_imports(tmp_path):
    p=Path(__file__).resolve().parents[1]/'scripts/research/run_long_history_source_ablation.py'
    spec=importlib.util.spec_from_file_location('history_runner_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    reader=m.Reader({},tmp_path,[])
    with pytest.raises(RuntimeError):reader.read('SELECT')
    budget=m.FitBudget()
    for name in h.MASKS:budget.start(name)
    budget.close()
    with pytest.raises(RuntimeError):budget.start('S')
    with pytest.raises(RuntimeError):m.Forbidden().find_spec('torch')
