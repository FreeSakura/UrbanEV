import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from urbanev_forecast import reference_harm as h

def test_loss_identities_and_positive_part_before_mean():
    e=torch.tensor([1.,1.],dtype=torch.float64);f=torch.tensor([.4,-.4],dtype=torch.float64)
    p=h.loss_parts(e,f,.2,'PRODUCT_POSREG');mix=h.loss_parts(e,f,.2,'PRODUCT_MIX');l1=h.loss_parts(e,f,.2,'PRODUCT_L1')
    assert p['harm'].item()==pytest.approx(.2)
    assert torch.relu(((e-f).abs()-e.abs()).mean()).item()==pytest.approx(0)
    assert (p['harm']-p['benefit']).item()==pytest.approx(((e-f).abs()-e.abs()).mean().item())
    assert mix['objective']<=p['objective']<=l1['objective']
    torch.manual_seed(8);e=torch.randn(19,12,dtype=torch.float64);f=torch.randn_like(e)
    p=h.loss_parts(e,f,.2,'PRODUCT_POSREG');mix=h.loss_parts(e,f,.2,'PRODUCT_MIX');l1=h.loss_parts(e,f,.2,'PRODUCT_L1')
    assert mix['objective']<=p['objective']<=l1['objective']

def test_zero_subgradient_and_scale_float_semantics():
    e=torch.tensor([.2,-.3]);f=torch.zeros_like(e,requires_grad=True)
    for scheme in ('PRODUCT_POSREG','PRODUCT_L1'):
        p=h.loss_parts(e,f,.3,scheme);extra=p['objective']-p['mse'];assert extra.item()==0
        grad=torch.autograd.grad(extra,f,retain_graph=True)[0];assert torch.count_nonzero(grad)==0
    x=np.array([.1,.2],dtype=np.float32);s=h.fit_scale(x)
    assert s['scale_float64']==float(np.sqrt(np.mean(x.astype(np.float64)**2)))
    assert s['tau_float32']==float(np.float32(.5*s['scale_float64']))
    assert h.fit_scale(np.zeros(3,dtype=np.float32))['floor_applied']
    with pytest.raises(ValueError):h.fit_scale(x.astype(np.float64))

def test_bernoulli_positive_harm_does_not_guarantee_mae():
    e=torch.tensor([0.]*9+[1.],dtype=torch.float64);tau=.5*np.sqrt(.1)
    q=(.2-.9*tau)/2;f=torch.full_like(e,q,requires_grad=True)
    p=h.loss_parts(e,f,tau,'PRODUCT_POSREG');zero=h.loss_parts(e,torch.zeros_like(e),tau,'PRODUCT_POSREG')
    assert p['objective']<zero['objective'] and p['mse']<zero['mse']
    assert (e-f).abs().mean()>e.abs().mean()
    assert q==pytest.approx(.02884875264621145)
    assert abs(torch.autograd.grad(p['objective'],f)[0].sum())<1e-12

@pytest.mark.parametrize('scheme',h.MODELS)
def test_same_architecture_initialization_and_two_step_gradient(scheme):
    net=h.InteractionNet(scheme,123);reference=h.InteractionNet('PRODUCT_MSE' if scheme.startswith('PRODUCT') else 'CONCAT_MSE',123)
    assert sum(p.numel() for p in net.parameters())==9108
    for p,q in zip(net.parameters(),reference.parameters()):torch.testing.assert_close(p,q)
    x=torch.randn(10,341);e=torch.randn(10,12);assert net(x).count_nonzero()==0
    opt=torch.optim.AdamW(net.parameters(),lr=.001,foreach=False,fused=False)
    for _ in range(2):opt.zero_grad();h.loss_parts(e,net(x),.1,scheme)['objective'].backward();opt.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)>0 for p in net.parameters())

def test_relative_disagreement_is_scale_invariant_and_zero_not_success():
    a=np.random.default_rng(12).normal(size=(3,14,275));r=h.relative_stability(a);scaled=h.relative_stability(7*a)
    assert r['eta']==pytest.approx(scaled['eta'])
    assert r['V']==pytest.approx(np.var(a,axis=0,ddof=0).mean())
    assert scaled['T']==pytest.approx(49*r['T'])
    assert h.relative_stability(np.zeros_like(a))['eta'] is None

def test_fixed_budget_and_selection_ignores_risk_diagnostics():
    b=h.Budget();b.start_ridge()
    for scheme in h.MODELS:
        for seed in h.SEEDS:b.start_neural(scheme,seed)
    b.close()
    with pytest.raises(RuntimeError):b.start_neural(h.MODELS[0],h.SEEDS[0])
    assert h.choose_checkpoint([{'epoch':0,'selection_rmse':1,'harm':99},{'epoch':5,'selection_rmse':1,'harm':0}])['epoch']==0
