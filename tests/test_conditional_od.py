import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from urbanev_forecast.conditional_od import InteractionNet,MODELS,SEEDS,Budget,learning_rate,choose_checkpoint,state_groups,grouped_losses

@pytest.mark.parametrize('kind',MODELS)
def test_counts_zero_start_and_all_parameters_receive_gradients(kind):
    net=InteractionNet(kind,7);assert sum(p.numel() for p in net.parameters())==9108
    x=torch.randn(9,341);target=torch.randn(9,12)
    assert torch.count_nonzero(net(x))==0
    opt=torch.optim.AdamW(net.parameters(),lr=.001,foreach=False,fused=False)
    for _ in range(2):
        opt.zero_grad();((net(x)-target)**2).mean().backward();opt.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)>0 for p in net.parameters())

def four_inputs():
    rng=np.random.default_rng(3);o1,o2,d1,d2=[rng.normal(size=168) for _ in range(4)];c=rng.normal(size=5)
    return torch.tensor(np.stack([np.r_[o1,d1,c],np.r_[o1,d2,c],np.r_[o2,d1,c],np.r_[o2,d2,c]]),dtype=torch.float64)

def test_raw_mixed_difference_separability_and_product_identity():
    x=four_inputs()
    for kind in ('OD_PRODUCT','OD_SEPARABLE'):
        net=InteractionNet(kind,4).double()
        with torch.no_grad():net.head.weight.normal_();net.head.bias.normal_()
        q=net(x);mixed=q[0]-q[1]-q[2]+q[3]
        if kind=='OD_SEPARABLE':torch.testing.assert_close(mixed,torch.zeros(12,dtype=torch.float64),atol=1e-12,rtol=0)
        else:
            u,v=net.encoded(x);expected=((u[0]-u[2])*(v[0]-v[1]))@net.head.weight[:,48:].T
            torch.testing.assert_close(mixed,expected,atol=1e-12,rtol=1e-12);assert mixed.abs().max()>1e-5

def test_product_and_additive_counterexamples():
    u=np.array([-1,-1,1,1]);v=np.array([-1,1,-1,1]);product=.5+.1*u*v
    assert np.mean((product-.5)**2)==pytest.approx(.01)
    np.testing.assert_allclose(np.column_stack([np.ones(4),u,v])@np.linalg.lstsq(np.column_stack([np.ones(4),u,v]),product,rcond=None)[0],.5)
    additive=.5+.1*u+.1*v;assert additive[0]-additive[1]-additive[2]+additive[3]==pytest.approx(0)

def test_shared_initial_encoders_and_OO_extra_module():
    nets=[InteractionNet(k,9).double() for k in ('OD_PRODUCT','OD_SEPARABLE','OO_PRODUCT')]
    for net in nets[1:]:
        torch.testing.assert_close(net.first.weight,nets[0].first.weight);torch.testing.assert_close(net.second.weight,nets[0].second.weight)
    net=nets[-1]
    with torch.no_grad():net.head.weight.fill_(.1)
    x=four_inputs();torch.testing.assert_close(net(x)[0],net(x)[1]);torch.testing.assert_close(net(x)[2],net(x)[3])

def test_learning_rate_selection_and_fixed_budget():
    assert learning_rate(1)==pytest.approx(.001) and learning_rate(40)==pytest.approx(.0001)
    assert choose_checkpoint([{'epoch':5,'selection_rmse':1,'mae':0},{'epoch':0,'selection_rmse':1,'mae':99}])['epoch']==0
    b=Budget();b.start_ridge()
    for k in MODELS:
        for seed in SEEDS:b.start_neural(k,seed)
    b.close()
    with pytest.raises(RuntimeError):b.start_neural(MODELS[0],SEEDS[0])

def test_state_groups_boundaries_empty_groups_and_loss_restore():
    p=np.array([[.4,.4,.5,.5]]);change=np.array([[-1.,0.,-1.,0.]]);groups=state_groups(p,change);np.testing.assert_array_equal(groups,[[0,1,2,3]])
    y=np.zeros((1,2,4));reference=np.ones_like(y)*.2;candidate=np.ones_like(y)*.1
    report,g1,g2=grouped_losses(y,reference,candidate,groups)
    assert sum(r['sample_count'] for r in report['groups'])==8
    assert sum(r['mae']['net_sum'] for r in report['groups'])==pytest.approx(g1.sum())
    empty,_,_=grouped_losses(y,reference,candidate,np.zeros_like(groups));assert empty['groups'][3]['mae']['mean_gain'] is None
