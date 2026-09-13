from pathlib import Path
import sys
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from urbanev_forecast import comparison_core as c

def author_equivalent_local_class():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models/timexer'))
    from TimeXer import Model
    return Model

def test_origin_batches_cover_all_rows_without_dropping_tail():
    batches=c.effective_batches(876,20260915,1)
    assert len(batches)==55 and len(batches[-1])==12
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)),np.arange(876))
    assert sum(len(c.microbatches(b)) for b in batches)==438
    assert all(sum(w for _,w in c.microbatches(b))==pytest.approx(1) for b in batches)

@pytest.mark.parametrize('n',[12,16])
def test_microbatch_weighted_gradients_equal_full_batch(n):
    torch.manual_seed(7);model=torch.nn.Linear(3,2).double();x=torch.randn(n,3,dtype=torch.float64);y=torch.randn(n,2,dtype=torch.float64)
    ((model(x)-y)**2).mean().backward();expected=[p.grad.clone() for p in model.parameters()];model.zero_grad()
    for indices,weight in c.microbatches(np.arange(n)):(((model(x[indices])-y[indices])**2).mean()*weight).backward()
    for p,q in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,q,atol=1e-12,rtol=1e-12)

def test_local_channel_order_marker_and_global_information_contract():
    raw=torch.arange(2*3*341,dtype=torch.float32).reshape(2,3,341)
    local,marker=c.timexer_inputs(raw,'TIMEXER_LOCAL_OD')
    assert local.shape==(6,168,2) and marker.shape==(6,168,5)
    torch.testing.assert_close(local[0,:,0],raw[0,0,168:336]);torch.testing.assert_close(local[0,:,1],raw[0,0,:168])
    torch.testing.assert_close(marker[0,5],raw[0,0,336:]);torch.testing.assert_close(marker[:,0],marker[:,-1])
    glob,gm=c.timexer_inputs(raw,'TIMEXER_GLOBAL_O');assert glob.shape==(2,168,3) and gm is None
    changed=raw.clone();changed[:,:,168:]=999
    torch.testing.assert_close(c.timexer_inputs(changed,'TIMEXER_GLOBAL_O')[0],glob)

@pytest.mark.parametrize('mode',['TIMEXER_LOCAL_OD','TIMEXER_GLOBAL_O'])
def test_output_shape_and_author_denormalization(mode):
    torch.set_num_threads(2);adapter=c.TimeXerAdapter(mode,17,author_equivalent_local_class(),channels=3).eval()
    raw=torch.rand(2,3,341);raw[:,:,:168]=torch.tensor([.2,.5,.8])[None,:,None];raw[:,:,168:336]=7.
    with torch.no_grad():adapter.model.head.linear.weight.zero_();adapter.model.head.linear.bias.zero_()
    result=adapter(raw);assert result.shape==(2,12,3)
    expected=torch.tensor([.2,.5,.8])[None,None,:].expand(2,12,3)
    torch.testing.assert_close(result,expected,atol=1e-6,rtol=1e-6)

def test_budget_requires_all_core_methods_and_closes():
    budget=c.Budget();budget.ridge('RIDGE_O');budget.ridge('RIDGE_OD')
    for model in c.MODELS:
        for seed in c.SEEDS:budget.start_neural(model,seed)
    budget.close()
    with pytest.raises(RuntimeError):budget.start_neural(c.MODELS[0],c.SEEDS[0])
