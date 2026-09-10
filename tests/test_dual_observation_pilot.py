import importlib.util
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip('torch')

def module():
    path=Path(__file__).resolve().parents[1]/'scripts/research/dual_observation_pilot.py'
    spec=importlib.util.spec_from_file_location('dual_observation_pilot',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_lagged_duration_inputs_and_two_target_conventions():
    m=module();o=np.repeat(np.arange(648,dtype=np.float32)[:,None],275,axis=1);d=o+1000
    x,y,left,right=m.examples(o,d,[169],3)
    assert x.shape==(275,336)
    assert x[0,:168].tolist()==list(range(1,169))
    assert x[0,168:].tolist()==list(range(1000,1168))
    assert y[0].tolist()==[169,170,171]
    assert left[0].tolist()==[1168,1169,1170]
    assert right[0].tolist()==[1169,1170,1171]

def test_soft_operator_is_bounded_and_parameter_matched():
    m=module();free=m.ObservationModel(12,False);op=m.ObservationModel(12,True)
    assert sum(p.numel() for p in free.parameters())==sum(p.numel() for p in op.parameters())
    endpoint,average,occupied,active=op(torch.rand(5,336))
    assert endpoint.shape==average.shape==(5,12)
    assert torch.all(active<=occupied) and torch.all(active>=0) and torch.all(occupied<=1)
    (endpoint.mean()+average.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in op.parameters())
