import copy
import json
from pathlib import Path
from urbanev_forecast import continuous_step as m

ROOT=Path(__file__).resolve().parents[1]
REG=json.loads((ROOT/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_REGISTRATION.json').read_text())
CFG=json.loads((ROOT/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_SOLVER_REPAIR.json').read_text())

def scores():
    alphas={s:(0 if s=='native' else 1) for s in REG['systems']};out={}
    shifts={'native':0,'bias':0,'occupancy':0,'duplicate':0,'raw_duration':.1,'orthogonal_duration':.2,'permuted_orthogonal':0}
    for system in REG['systems']:
        cells=[{'id':{'stage':'calibration','block':block,'horizon':h},'p':[0.],'delta':[shifts[system]],'y':[1.]} for block in (720,1056,1392) for h in (3,12)]
        out[system]=m.score_fixed_alpha(cells,alphas[system])
    return out,alphas

def test_gates_after_selection_have_one_global_reference_and_no_solver_call(monkeypatch):
    out,alphas=scores()
    def fail(*args,**kwargs):raise AssertionError('Gate called solver')
    monkeypatch.setattr(m,'solve_continuous_alpha',fail)
    r=m.evaluate_registered_gates(out,alphas,REG)
    assert r['status']=='OK' and r['information_gate'] and r['structure_gate']
    assert r['Cstar']=='native' and r['positive_blocks']==3
    assert r['third_fold_validation_admitted'] is False

def test_missing_or_blocked_control_blocks_whole_stage():
    out,alphas=scores();out['duplicate']={'status':'NUMERICAL_BLOCKED'}
    assert m.evaluate_registered_gates(out,alphas,REG)['status']=='STAGE_BLOCKED'

def test_registration_change_and_real_calibration_are_blocked():
    assert m.validate_solver_registration(REG,CFG)
    changed=copy.deepcopy(REG);changed['information_gate']['macro_rmse_gain_vs_Cstar_percent_at_least']=.1
    assert not m.validate_solver_registration(changed,CFG)
    out,alphas=scores()
    assert m.evaluate_registered_gates(out,alphas,changed)['status']=='INPUT_BLOCKED'
    assert m.authorize_real_calibration(REG,CFG)['status']=='NOT_AUTHORIZED'

def test_tail_main_metric_pools_by_horizon_not_block_rmse():
    cells=[{'id':{'stage':'tail','block':block,'horizon':h},'p':[p],'delta':[0.],'y':[0.]} for block,p in [(0,0),(1,0),(2,1)] for h in (3,6,9,12)]
    s=m.score_fixed_alpha(cells,0)
    assert abs(s['macro']['rmse']-(1/3)**.5)<1e-15
    assert s['macro']['rmse']!=1/3

def test_zero_reference_is_no_go_and_different_targets_block():
    out,alphas=scores()
    for s in out:
        out[s]['macro']={'rmse':0.,'mae':0.}
        for row in out[s]['cells']:row.update(rmse=0.,mae=0.,mse=0.)
    r=m.evaluate_registered_gates(out,alphas,REG)
    assert r['status']=='OK' and not r['information_gate']
    out,alphas=scores();out['bias']['cells'][0]['data_fingerprint']='changed'
    assert m.evaluate_registered_gates(out,alphas,REG)['status']=='INPUT_BLOCKED'
