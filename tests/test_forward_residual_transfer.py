import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from threadpoolctl import threadpool_limits
from urbanev_forecast import forward_residual_transfer as f

def test_forward_blocks_and_label_gaps():
    oo=f.meta_origins();assert len(oo)==543 and len(np.unique(oo))==543 and oo[-1]+11==1055
    assert len(oo)*275==149325 and (149325+4095)//4096==37
    for _,stop,ylimit,dlimit,left,right in f.BLOCKS:
        assert stop-1+12==ylimit and stop-2==dlimit
        assert left-ylimit==12 and right-left==181
        assert stop-1+11<left

def test_local_fit_cannot_see_later_rows_and_prediction_ignores_future():
    y=(.3+.1*np.sin(np.arange(1056)/13))[:,None];d=(.2+.03*np.cos(np.arange(1043)/19))[:,None];cap=np.array([10.])
    with threadpool_limits(limits=2):
        model=f.fit_model(y[:468],d[:455],cap,np.arange(169,457))
        yy=y.copy();dd=d.copy();yy[468:]=.95;dd[455:]=.7
        same=f.fit_model(yy[:468],dd[:455],cap,np.arange(169,457))
        np.testing.assert_array_equal(model['beta'],same['beta'])
        for key in model['stats']:np.testing.assert_array_equal(model['stats'][key],same['stats'][key])
        with pytest.raises(ValueError):f.fit_model(y,d,cap,np.arange(169,457))
        p=f.predict_model(model,y,d,cap,np.array([480]));yy=y.copy();dd=d.copy();yy[480:]=99;dd[479:]=99
        np.testing.assert_array_equal(p,f.predict_model(model,yy,dd,cap,np.array([480])))

def test_residual_transport_common_scale_and_base_context():
    y=np.array([[.2,.4],[.6,.8]],dtype=np.float64);qf=y*.8;qold=y*.3;eis=y-qf;efwd=y-qold
    np.testing.assert_allclose(efwd-eis,qf-qold,atol=1e-15)
    # Correcting the FWD label for the base difference would erase the experiment.
    np.testing.assert_allclose(efwd+(qold-qf),eis)
    s=f.fit_scale(eis)['scale_float64'];assert s==np.sqrt(np.mean(eis**2))
    z=np.zeros((2,341));anchor=np.array([.1,.2]);a=f.augment(z,qf,anchor,s)
    assert a.dtype==np.float32;np.testing.assert_allclose(a[:,341:],(qf-anchor[:,None])/s,rtol=1e-7)
    np.testing.assert_allclose(qf+s*((y-qf)/s),y)

@pytest.mark.parametrize('scheme',f.MODELS)
def test_zero_context_parameters_and_gradients(scheme):
    net=f.InteractionNet(scheme,31);assert sum(p.numel() for p in net.parameters())==f.parameter_count(scheme)
    if net.context is not None:assert net.context.weight.shape==(24,12) and net.context.weight.count_nonzero()==0
    x=torch.randn(9,353);target=torch.randn(9,12);assert net(x).count_nonzero()==0
    opt=torch.optim.AdamW(net.parameters(),lr=.001,foreach=False,fused=False)
    for _ in range(2):opt.zero_grad();((net(x)-target)**2).mean().backward();opt.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()>0 for p in net.parameters())

def test_matching_initial_functions_and_context_has_effect_after_learning():
    a=f.InteractionNet('P_IS_X',19);b=f.InteractionNet('P_FWD_BC',19)
    for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key])
    x=torch.randn(5,353);xx=x.clone();xx[:,341:]+=4
    with torch.no_grad():a.head.weight.fill_(.1);b.head.weight.fill_(.1)
    torch.testing.assert_close(a(x),b(x));torch.testing.assert_close(a(x),a(xx))
    with torch.no_grad():b.context.weight.fill_(.1)
    assert not torch.allclose(b(x),b(xx))

def test_weak_base_transfer_counterexample_and_context_correction():
    x=np.array([.2,.4,.7]);early=np.zeros_like(x);full=x.copy();trained=x.copy()
    np.testing.assert_allclose(early+trained,x);assert np.mean((full+trained-x)**2)>0
    s=.3
    for qb in (early,full):np.testing.assert_allclose(qb+s*((x-qb)/s),x)

def test_exact_fit_budget_and_epoch_zero_selection():
    b=f.Budget()
    with pytest.raises(RuntimeError):b.start_ridge('BF')
    for name in ('B1','B2','B3','BF'):b.start_ridge(name)
    for scheme in f.MODELS:
        for seed in f.SEEDS:b.start_neural(scheme,seed)
    b.close()
    with pytest.raises(RuntimeError):b.start_neural('P_IS_X',f.SEEDS[0])
    assert f.choose_checkpoint([{'epoch':0,'selection_rmse':1},{'epoch':40,'selection_rmse':1}])['epoch']==0
