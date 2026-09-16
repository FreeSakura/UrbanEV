"""Check causal feature availability, including artificial masking."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn')
SPEC=importlib.util.spec_from_file_location('shared_pilot',Path(__file__).resolve().parents[1]/'scripts/research/run_shared_missing_p0.py')
pilot=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(pilot)


def test_origin_features_never_use_present_or_future_targets():
    values=np.arange(300,dtype=float)/10
    dates=pd.date_range('2013-03-01',periods=300,freq='h')
    config={'history':168,'threshold':75}
    before=pilot.features(values,dates,config,10,0,1)
    changed=values.copy();changed[200:]=1e6
    after=pilot.features(changed,dates,config,10,0,1)
    np.testing.assert_array_equal(before[:201],after[:201])


def test_hidden_truth_cannot_influence_masked_input_features():
    values=np.arange(300,dtype=float)/10
    dates=pd.date_range('2013-03-01',periods=300,freq='h')
    hidden=np.arange(180,210)
    alternate=values.copy();alternate[hidden]=-1e6
    values[hidden]=np.nan;alternate[hidden]=np.nan
    config={'history':168,'threshold':75}
    np.testing.assert_array_equal(pilot.features(values,dates,config,10,0,1),pilot.features(alternate,dates,config,10,0,1))


def test_panel_bounds_contain_known_artificial_truth():
    predictions={'a':np.array([.1,.8]),'b':np.array([.6,.3]),'c':np.array([.4,.4])}
    rows=pilot.compare_panel('toy','one','artificial','test',2,1,np.array([0,-1,0]),predictions,0,truth=np.array([0,1,0]))
    assert len(rows)==3 and all(r['truth_covered'] for r in rows)
    assert rows[0]['new_strict_decision']
