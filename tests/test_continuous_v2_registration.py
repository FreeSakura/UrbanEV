import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def setup():
    spec=importlib.util.spec_from_file_location('verify_v2',ROOT/'scripts/research/verify_continuous_v2_registration.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,json.loads((ROOT/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_REGISTRATION.json').read_text())

def test_registered_targets_stay_in_each_time_partition():
    m,c=setup();r=m.validate_registration(c)
    assert r['calibration_fit_counts']==[44,72,100]
    assert r['tail_fit_count']==114 and r['tail_origins_per_horizon']==15
    assert max(v['evaluation_last_target'] for v in r['target_boundary_checks'])==1739

def test_rejects_extra_h3_tail_origin_and_training_boundary_crossing():
    m,c=setup();bad=copy.deepcopy(c);bad['development_tail']['evaluation_origins'].append(1740)
    with pytest.raises(ValueError,match='Tail must use common15'):m.validate_registration(bad)
    bad=copy.deepcopy(c);bad['calibration_windows'][0]['fit_origins'].append(720)
    with pytest.raises(ValueError,match='Calibration origin list'):m.validate_registration(bad)

def test_rejects_changes_to_reserved_validation_range():
    m,c=setup();c['closed_ranges']['third_fold_validation'][0]=1740
    with pytest.raises(ValueError,match='Closed validation/test'):m.validate_registration(c)
