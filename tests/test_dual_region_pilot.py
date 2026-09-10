import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module():
    path=Path(__file__).resolve().parents[1]/'scripts/research/dual_region_pilot.py'
    spec=importlib.util.spec_from_file_location('dual_region_pilot',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_balanced_partition_covers_each_region_once():
    m=module();rng=np.random.default_rng(42)
    groups=m.partition(m.affinity(rng.normal(size=(40,275))))
    m.validate_groups(groups)
    assert [len(g) for g in groups]==[32]*8+[19]
    with pytest.raises(ValueError):m.validate_groups([[0]*32]*9)

def test_joint_swap_loss_counts_both_groups_on_global_scale():
    m=module();calls=[]
    class Backend:
        def predict(self,x,groups,h):
            calls.append((x.shape,h,sum(map(len,groups))))
            return np.zeros((h,sum(map(len,groups))))
    value=m.pair_loss(Backend(),np.ones((648,275)),[list(range(32)),list(range(32,64))])
    assert value==pytest.approx(64/275)
    assert len(calls)==8 and all(c[0]==(168,275) and c[2]==64 for c in calls)
