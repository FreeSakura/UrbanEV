import importlib.util
from pathlib import Path
import numpy as np
import pytest
from urbanev_forecast.foundation import validate_prediction


def test_foundation_point_shape_and_semantics():
    q=np.zeros((3,275,9));p=np.zeros((3,275))
    validate_prediction(q,p,3,275)
    with pytest.raises(ValueError):validate_prediction(q,p+1,3,275)
    with pytest.raises(ValueError):validate_prediction(q[:,:32],p[:,:32],3,275)


def test_cache_passes_only_each_origins_history(tmp_path):
    script=Path(__file__).resolve().parents[1]/"scripts/research/cache_foundation.py"
    spec=importlib.util.spec_from_file_location("foundation_cache",script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    values=np.repeat(np.arange(180,dtype=np.float32)[:,None],275,axis=1)
    origins=np.array([168,169],dtype=np.int64);calls=[]
    class Backend:
        def predict(self,x,horizon):
            calls.append(x.copy())
            q=np.full((horizon,275,9),x[-1,0],np.float32)
            return q,q[...,4]
    result=module.fill_cache(Backend(),values,origins,3,tmp_path/"cache")
    assert result["windows"]==2
    assert np.array_equal(calls[0],values[:168]) and np.array_equal(calls[1],values[1:169])
    assert np.load(tmp_path/"cache/native.npy")[0,0,0]==167
    calls.clear()
    resumed=module.fill_cache(Backend(),values,origins,3,tmp_path/"cache",resume=True)
    assert not calls and resumed["new_windows_this_session"]==0
