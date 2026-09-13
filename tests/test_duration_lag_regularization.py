import importlib.util
from pathlib import Path
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from urbanev_forecast import lag_regularization as l
from urbanev_forecast import history_source as h

@pytest.fixture(autouse=True)
def two_threads():
    with threadpool_limits(limits=2):yield

def test_difference_nullspace_spectrum_and_scale():
    active,dpos,diff,q,m=l.penalties(np.zeros(341,bool))
    assert diff.shape==(166,168) and m['unnormalized_trace']==996 and m['m']==168
    np.testing.assert_allclose(diff@np.ones(168),0,atol=1e-12);np.testing.assert_allclose(diff@np.arange(168),0,atol=1e-12)
    assert np.linalg.norm(diff@((-1.)**np.arange(168)))>0
    spike=np.zeros(168);spike[60]=1;assert np.linalg.norm(diff@spike)>0
    for mat in q.values():assert abs(np.trace(mat)-168)<1e-10 and np.all(mat[0]==0)
    np.testing.assert_allclose(np.linalg.eigvalsh(q[l.FAMILIES[0]]),np.linalg.eigvalsh(q[l.FAMILIES[1]]),atol=1e-10)

def test_missing_columns_not_bridged_or_used_as_auxiliary_coefficients():
    inactive=np.ones(341,bool);inactive[[0,1,*range(168,176)]]=False;inactive[171]=True
    active,dpos,diff,q,m=l.penalties(inactive)
    assert m['difference_rows']==3  # 168-170, 172-174, 173-175; no cross-gap triple.
    np.testing.assert_allclose(diff@np.asarray(m['active_D_features']),0,atol=1e-12)
    rng=np.random.default_rng(8);z=rng.normal(size=(24,341));z[:,inactive]=0;y=rng.normal(size=(24,2));g,b=h.sufficient_statistics(z,y)
    beta,_=l.solve(g,b,active,q[l.FAMILIES[0]],.1,np.mean(y*y,axis=0),diff,dpos)
    assert np.count_nonzero(beta[1:][inactive])==0

def test_no_triples_blocks_without_fallback():
    inactive=np.ones(341,bool);inactive[[0,168,170]]=False
    with pytest.raises(ValueError,match='triple'):l.penalties(inactive)

def test_zero_gamma_matches_ridge_and_penalty_decreases():
    inactive=np.ones(341,bool);inactive[[0,1,*range(168,176)]]=False
    active,dpos,diff,q,m=l.penalties(inactive);rng=np.random.default_rng(2);z=rng.normal(size=(40,341));z[:,inactive]=0;y=rng.normal(size=(40,3));g,b=h.sufficient_statistics(z,y);r2=np.mean(y*y,axis=0)
    regular,_=h.solve_mask(g,b,active.tolist());beta,details=l.solve(g,b,active,q[l.FAMILIES[0]],0.,r2,diff,dpos)
    np.testing.assert_allclose(beta[np.r_[0,active+1]],regular,atol=1e-12)
    beta2,details2=l.solve(g,b,active,q[l.FAMILIES[0]],1.,r2,diff,dpos)
    assert sum(r['actual_Q_penalty'] for r in details2['per_output'])<=sum(r['actual_Q_penalty'] for r in details['per_output'])+1e-12
    residual=y-np.column_stack([np.ones(len(z)),z])@beta
    np.testing.assert_allclose(np.mean(residual**2,axis=0),[r['data_mse'] for r in details['per_output']],atol=1e-12)

def test_selection_shared_zero_and_matched_control():
    scores={'BASE_OD':10.}|{l.fit_id(f,g):float(10+g) for f in l.FAMILIES for g in l.GAMMAS[1:]}
    assert len(scores)==13
    scores[l.fit_id('TIME_SMOOTH_D',.1)]=1.;scores[l.fit_id('SCRAMBLED_SMOOTH_D',1.)]=2.
    chosen,mapping=l.select(scores)
    assert mapping['SCRAMBLED_MATCHED']==l.fit_id('SCRAMBLED_SMOOTH_D',.1)
    assert mapping['SCRAMBLED_SELECTED']==l.fit_id('SCRAMBLED_SMOOTH_D',1.)
    scores={k:1. for k in scores};chosen,mapping=l.select(scores);assert set(mapping.values())=={'BASE_OD'}
    budget=l.Budget()
    for id in scores:budget.start(id)
    budget.close()
    with pytest.raises(RuntimeError):budget.start('BASE_OD')

def test_error_bins_include_boundaries_and_share_raw_baseline():
    y=np.zeros((1,1,5));base=np.array([0,.01,.05,.10,1.5]).reshape(1,1,5)
    np.testing.assert_array_equal(l.error_groups(y,base),np.array([0,1,2,3,3]).reshape(1,1,5))
    report,g1,g2=l.grouped_loss(y,base,np.clip(base,0,1),np.zeros_like(base))
    assert [r['sample_count'] for r in report['groups']]==[1,1,1,2]
    assert abs(sum(r['mse']['net_sum'] for r in report['groups'])-g2.sum())<1e-12
    empty,_,_=l.grouped_loss(y,base*0,base*0,base*0)
    assert empty['groups'][3]['sample_count']==0 and empty['groups'][3]['mse']['mean_gain'] is None

def test_runner_preserves_stage_guard(tmp_path):
    p=Path(__file__).resolve().parents[1]/'scripts/research/run_duration_lag_regularization.py'
    spec=importlib.util.spec_from_file_location('duration_runner_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    reader=m.Reader({},tmp_path,[])
    with pytest.raises(RuntimeError):reader.read('SELECT')
    with pytest.raises(RuntimeError):m.Forbidden().find_spec('chronos')
