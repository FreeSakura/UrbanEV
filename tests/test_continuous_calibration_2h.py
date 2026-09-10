import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from urbanev_forecast.continuous_calibration_2h import guarded_load,execute_systems,explain_gates
from urbanev_forecast.continuous_step import score_fixed_alpha

ROOT=Path(__file__).resolve().parents[1]
AUTH=json.loads((ROOT/'configs/research/CONTINUOUS_V2_CALIBRATION_2H_AUTHORIZATION.json').read_text())
REG=json.loads((ROOT/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_REGISTRATION.json').read_text())
CFG=json.loads((ROOT/AUTH['solver_config_path']).read_text())


def manifest():
    return {'authorization_id':AUTH['authorization_id'],'stage':'calibrate_h3_h12','horizons':[3,12],
            'maximum_semantic_row_stop':1560,'new_foundation_inference':0,'automatic_stage_advance':False,
            'systems':REG['systems'],'cells':[{'cut':cut,'horizon':h,'fit_origins':list(range(192,cut,12)),
                'calibration_origins':list(range(cut,cut+168,12)),'fit_target_stop':cut,'calibration_target_stop':cut+168}
                for h in (3,12) for cut in (720,1056,1392)]}


def observed():return {'accepted_solver_commit':AUTH['accepted_solver_commit'],
    'solver_config_sha256':AUTH['solver_config_sha256'],'code_identity':AUTH['accepted_code_identity']}


@pytest.mark.parametrize('kind',['stage','config_hash','code_hash','target_boundary','control_block'])
def test_invalid_admission_stops_before_data_reader(kind):
    plan=manifest();identity=copy.deepcopy(observed());stage='calibrate_h3_h12';statuses=None
    if kind=='stage':stage='tail'
    elif kind=='config_hash':identity['solver_config_sha256']='0'*64
    elif kind=='code_hash':identity['code_identity']['continuous_step_except_authorizer_AST']='0'*64
    elif kind=='target_boundary':plan['cells'][-1]['calibration_origins'].append(1560)
    elif kind=='control_block':statuses={s:'OK' for s in REG['systems']};statuses['duplicate']='NUMERICAL_BLOCKED'
    loader=Mock(side_effect=AssertionError('Real reader must not run'))
    result=guarded_load(REG,CFG,AUTH,stage,plan,identity,loader,statuses)
    assert result['admission']['status']=='NOT_AUTHORIZED'
    loader.assert_not_called()


def test_valid_stage_accepts_old_pending_config_only_with_new_explicit_review():
    assert CFG['solver_code_review']=='pending'
    loader=Mock(return_value='synthetic payload')
    result=guarded_load(REG,CFG,AUTH,'calibrate_h3_h12',manifest(),observed(),loader)
    assert result['admission']['status']=='AUTHORIZED' and result['payload']=='synthetic payload'
    assert result['admission']['automatic_stage_advance'] is False
    loader.assert_called_once()


def cells_by_system():
    return {s:[{'id':{'stage':'calibration','block':cut,'horizon':h},'p':np.array([.25]),'delta':np.array([.1]),'y':np.array([.75])}
               for h in (3,12) for cut in (720,1056,1392)] for s in REG['systems']}


def test_legal_zero_outer_status_uses_inner_score_and_fails_scientific_gate():
    def zero(cells,config):return {'status':'OPTIMAL_ALPHA_ZERO','alpha':0.,'score':score_fixed_alpha(cells,0.)}
    result=execute_systems(cells_by_system(),CFG,REG,solver=zero)
    assert result['status']=='CALIBRATION_2H_INFORMATION_NO_GO'
    assert result['gate']['information_gate'] is False
    assert explain_gates(result,REG)['criteria']['alpha_positive'] is False


def test_required_numerical_block_stops_remaining_systems_and_never_gates():
    solver=Mock(return_value={'status':'NUMERICAL_BLOCKED','reason':'synthetic boundary unresolved'})
    result=execute_systems(cells_by_system(),CFG,REG,solver=solver)
    assert result['status']=='CALIBRATION_2H_BLOCKED' and result['blocking_system']=='bias'
    assert solver.call_count==1 and result['gate']=='NOT_RUN'
    assert explain_gates(result,REG)['system_statuses']['orthogonal_duration']=='NOT_RUN'


def test_numeric_label_invalidity_stops_before_solver():
    data=cells_by_system();data['native'][0]['y']=np.array([1.1]);solver=Mock()
    result=execute_systems(data,CFG,REG,solver=solver)
    assert result['status']=='CALIBRATION_2H_BLOCKED';solver.assert_not_called()


def test_cache_hash_failure_precedes_label_parsing(tmp_path,monkeypatch):
    import sys
    sys.path.insert(0,str(ROOT/'scripts/research'))
    spec=importlib.util.spec_from_file_location('calibration_entry',ROOT/'scripts/research/run_continuous_calibration_2h.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    metadata=tmp_path/'result.json';metadata.write_text(json.dumps({'backend':{'numpy_version':np.__version__}}))
    op=tmp_path/'private_origins_h3.npy';bp=tmp_path/'private_base_h3.npy';op.write_bytes(b'fake origin metadata');bp.write_bytes(b'fake prediction bytes')
    plan={'v1_result_sha256':entry.digest(metadata),'cache':[{'horizon':3,'origins_sha256':entry.digest(op),'prediction_sha256':'0'*64}]}
    read=Mock(side_effect=AssertionError('Labels must not be parsed'))
    monkeypatch.setattr(entry,'load_rate_prefix',read)
    with pytest.raises(ValueError,match='Cache hash changed'):
        entry.load_payload(SimpleNamespace(cache=tmp_path,csv=tmp_path/'nonexistent',duration=tmp_path/'nonexistent',info=tmp_path/'nonexistent',output=tmp_path),plan,[])
    read.assert_not_called()
