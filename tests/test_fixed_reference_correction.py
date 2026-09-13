import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from urbanev_forecast import fixed_reference_correction as f
from urbanev_forecast import short_state as base

def test_time_boundaries_and_common_rows():
    train=np.arange(169,457);meta=f.meta_origins();mech=f.mechanism_origins()
    assert train[-1]+11==467 and meta[0]==480 and meta[-1]+11==1055
    assert len(meta)==543 and len(meta)*275==149325
    assert mech[0]==1224 and mech[-1]+11==1391 and len(mech)==14
    assert base.origins('SELECT')[-1]+11<mech[0]
    assert mech[-1]+11<base.origins('PREDICT')[0]

def test_checked_reference_uses_exact_prefix():
    from threadpoolctl import threadpool_limits
    y=(.3+.1*np.sin(np.arange(468)/13))[:,None];d=(.2+.05*np.cos(np.arange(455)/17))[:,None]
    with threadpool_limits(limits=2):model=f.fit_model(y,d,np.array([10.]),np.arange(169,457))
    assert model['normal_equation_relative_error']<1e-10 and model['max_training_label']==467
    with pytest.raises(ValueError):f.fit_model(np.r_[y,[[.1]]],d,np.array([10.]),np.arange(169,457))

def test_linear_repair_known_residual_and_product_counterexample():
    z=np.linspace(-1,1,51)[:,None];e=.2+.4*z
    beta,meta=f.linear_repair(z,e);expected=.4*np.mean(z*z)/(np.mean(z*z)+.01)
    np.testing.assert_allclose(beta[:,0],[.2,expected],atol=1e-12)
    assert meta['normal_equation_relative_error']<1e-10
    assert np.mean((e-beta[0]-z@beta[1:])**2)<np.mean(e**2)
    uv=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],dtype=float);prod=(uv[:,0]*uv[:,1])[:,None]
    beta,_=f.linear_repair(uv,prod);np.testing.assert_allclose(beta,0,atol=1e-12)

@pytest.mark.parametrize('scheme',f.MODELS)
def test_structure_zero_start_and_true_parameters(scheme):
    net=f.InteractionNet(scheme,27);assert sum(p.numel() for p in net.parameters())==9108
    x=torch.randn(8,341);target=torch.randn(8,12);assert net(x).count_nonzero()==0
    opt=torch.optim.AdamW(net.parameters(),lr=.001,foreach=False,fused=False)
    for _ in range(2):opt.zero_grad();((net(x)-target)**2).mean().backward();opt.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()>0 for p in net.parameters())

def test_fixed40_probe_does_not_follow_selected_zero():
    run={'checkpoints':[{'epoch':0,'file':'zero'},{'epoch':5,'file':'five'},{'epoch':40,'file':'forty'}]}
    selected=f.choose_checkpoint([{'epoch':0,'selection_rmse':1},{'epoch':40,'selection_rmse':2}])
    assert selected['epoch']==0 and f.mechanism_checkpoint(run)['file']=='forty'
    with pytest.raises(ValueError):f.mechanism_checkpoint({'checkpoints':[{'epoch':0}]})

def test_alignment_energy_and_zero_cases():
    y=np.ones((2,3));ref=np.zeros_like(y)
    for delta,sign_a,sign_g in [(.5,1,1),(3.,1,-1),(-1.,-1,-1)]:
        row=f.alignment(y,ref,np.full_like(y,delta))
        assert np.sign(row['alignment_A'])==sign_a and np.sign(row['mse_gain_G'])==sign_g
        assert row['mse_gain_G']==pytest.approx(2*row['alignment_A']-row['correction_energy_B'])
    assert f.alignment(y,ref,ref)['zero_correction']

def test_scale_is_raw_fixed_reference_only_and_frozen():
    e=np.array([.2,.3],dtype=np.float64);s=f.fit_scale(e)
    assert s['scale_float64']==np.sqrt(np.mean(e**2)) and not s['floor_applied']
    assert f.fit_scale(np.zeros(4))['floor_applied']
    with pytest.raises(ValueError):f.fit_scale(e.astype(np.float32))

def test_three_deterministic_nine_neural_order_and_closure():
    b=f.Budget();b.start_deterministic('BASE_FIXED');b.start_deterministic('LINEAR_REPAIR')
    with pytest.raises(RuntimeError):b.start_deterministic('RIDGE_FULL')
    for scheme in f.MODELS:
        for seed in f.SEEDS:b.start_neural(scheme,seed)
    b.start_deterministic('RIDGE_FULL');b.close()
    with pytest.raises(RuntimeError):b.start_neural(f.MODELS[0],f.SEEDS[0])
