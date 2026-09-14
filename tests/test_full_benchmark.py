import hashlib
from pathlib import Path

import numpy as np
import pytest

from urbanev_forecast.full_benchmark_data import (
    FoldReader, streaming_transform, streaming_statistics, ridge_solution,
    predict_ridge, TensorWindows, read_exact_prefix,
)
from urbanev_forecast.short_state import features, fit_transform, transform


@pytest.fixture
def small_history():
    rng=np.random.default_rng(17)
    return rng.uniform(.05,.9,(240,3)),rng.uniform(.05,.8,(240,3)),np.array([4.,7.,10.]),np.arange(169,218)


def test_streamed_statistics_and_ridge_match_dense_design(small_history):
    y,d,cap,oo=small_history
    raw=features(y,d,cap,oo,'LONG_OD');expected=fit_transform(raw)
    stats=streaming_transform(y,d,cap,oo)
    np.testing.assert_allclose(stats['mean'],expected['mean'],rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(stats['std'],expected['std'],rtol=1e-12,atol=1e-13)
    z=transform(raw,stats);x=np.column_stack((np.ones(len(z)),z))
    target=y[oo[:,None]+np.arange(3)].transpose(0,2,1).reshape(-1,3)
    anchor=y[oo-1].reshape(-1)
    g,b=streaming_statistics(y,d,cap,oo,3,stats)
    np.testing.assert_allclose(g,x.T@x/len(x),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(b,x.T@(target-anchor[:,None])/len(x),rtol=1e-12,atol=1e-12)
    mask=list(range(168))
    beta,error=ridge_solution(g,b,mask,.01)
    ix=np.r_[0,np.asarray(mask)+1];dense=g[np.ix_(ix,ix)]+np.diag([0]+[.01]*len(mask))
    np.testing.assert_allclose(beta,np.linalg.solve(dense,b[ix]),atol=1e-10,rtol=1e-10)
    assert error<1e-10
    q=predict_ridge(y,d,cap,oo,3,stats,mask,beta)
    np.testing.assert_allclose(q,(anchor[:,None]+x[:,ix]@beta).reshape(len(oo),3,3).transpose(0,2,1))


def test_reader_rejects_test_construction_before_training_freeze():
    reader=FoldReader(Path('not-read'),{'columns':[]},1,3,[])
    with pytest.raises(RuntimeError,match='Freeze'):
        reader.read('predict')
    with pytest.raises(ValueError,match='scoring is separate'):
        reader.read('test')
    reader.close_training()
    with pytest.raises(RuntimeError,match='closed'):
        reader.read('train')
    with pytest.raises(ValueError):
        read_exact_prefix(Path('not-read'),'occupancy.csv',4345)


def test_tensor_windows_match_past_feature_contract_and_ignore_future(small_history):
    torch=pytest.importorskip('torch')
    y,d,cap,oo=small_history
    stats=streaming_transform(y,d,cap,oo)
    chosen=oo[:4]
    tensor=TensorWindows(y,d,cap,stats,device='cpu')
    actual=tensor.inputs(chosen).numpy()
    expected=features(y,d,cap,chosen,'LONG_OD').reshape(len(chosen),3,341)
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
    yy=y.copy();dd=d.copy();yy[chosen[-1]:]=999;dd[chosen[-1]-1:]=999
    altered=TensorWindows(yy,dd,cap,stats,device='cpu')
    torch.testing.assert_close(tensor.inputs(chosen),altered.inputs(chosen),rtol=0,atol=0)
    with pytest.raises(ValueError):tensor.inputs([168])


def test_residual_horizon_shapes_and_zero_reference_alias():
    torch=pytest.importorskip('torch')
    from urbanev_forecast.full_benchmark_models import ResidualForecaster
    for h in (3,6,9,12):
        for kind in ('OD_PRODUCT','OD_SEPARABLE','OD_CONCAT_MLP'):
            net=ResidualForecaster(kind,h,7)
            output=net(torch.randn(2,3,341))
            assert output.shape==(2,h,3)
            assert torch.count_nonzero(output)==0
            output.square().sum().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
