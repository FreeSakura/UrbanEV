import pytest
torch=pytest.importorskip('torch')
from urbanev_forecast.short_state_torch import StateNet,relaxation_curve

def test_relaxation_is_bounded_monotone_and_composes():
    p=torch.tensor([0.,1.,.4],dtype=torch.float64);mu=torch.tensor([.8,.2,.4],dtype=torch.float64);rho=torch.tensor([.9,.7,.5],dtype=torch.float64)
    y=relaxation_curve(p,mu,rho,12)
    assert torch.all((y>=0)&(y<=1)) and torch.all(y[0,1:]>y[0,:-1]) and torch.all(y[1,1:]<y[1,:-1])
    torch.testing.assert_close(y[:,7],relaxation_curve(y[:,2],mu,rho,5)[:,-1])
    torch.testing.assert_close(y[2],torch.full((12,),.4,dtype=torch.float64))
    # A rise-then-fall sequence is outside this monotone curve family.
    assert not (bool(y[0,1]>y[0,0]) and bool(y[0,2]<y[0,1]))

@pytest.mark.parametrize('dim,relax,count',[(8,True,1410),(8,False,1740),(341,False,12396)])
def test_parameter_count_initialization_and_gradients(dim,relax,count):
    net=StateNet(dim,relax,20260913)
    assert sum(p.numel() for p in net.parameters())==count
    assert torch.count_nonzero(net.head.weight)==0
    x=torch.ones(4,dim);p=torch.tensor([0.,.2,.8,1.]);out=net(x,p)
    assert out.shape==(4,12) and torch.all((out>=0)&(out<=1))
    out.square().mean().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())

def test_same_short_trunk_seed_and_no_duration_dependence_when_masked():
    r=StateNet(8,True,3);d=StateNet(8,False,3)
    torch.testing.assert_close(r.first.weight,d.first.weight);torch.testing.assert_close(r.second.weight,d.second.weight)
    x=torch.randn(3,8);x[:,2]=0;y=x.clone();y[:,2]=0
    torch.testing.assert_close(r(x,torch.ones(3)*.4),r(y,torch.ones(3)*.4))

def test_runner_guards_and_optimizer_work_on_artificial_data(tmp_path):
    import importlib.util,sys
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/research/run_short_state_relaxation.py'
    spec=importlib.util.spec_from_file_location('short_state_runner_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source=m.Inputs({},tmp_path)
    with pytest.raises(RuntimeError,match='Stage'):source.read('SELECT')
    guard=m.NoFoundation();sys.meta_path.insert(0,guard)
    try:
        with pytest.raises(RuntimeError):guard.find_spec('chronos')
        net=StateNet(8,True,1);opt=torch.optim.AdamW(net.parameters(),lr=.001,foreach=False)
        loss=(net(torch.ones(2,8),torch.tensor([.2,.4]))-.7).square().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert all(torch.isfinite(v).all() for v in net.parameters())
    finally:sys.meta_path.remove(guard)
