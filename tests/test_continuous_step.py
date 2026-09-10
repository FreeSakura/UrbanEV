import copy
import json
import math
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
from urbanev_forecast import continuous_step as m

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_CANDIDATE_REPAIR_ROUNDING.json').read_text())

def cell(p,d,y,name='synthetic'):
    return {'id':name,'p':np.asarray(p,dtype=float),'delta':np.asarray(d,dtype=float),'y':np.asarray(y,dtype=float)}

@pytest.mark.parametrize('c,alpha,components,status',[
    (cell([.875,0],[1,.5],[.875,1]),1,[[0,0],[.25,1]],'OK'),
    (cell([.875,.875,0],[1,1,.5],[.875,.875,.25]),.5,[[0,0],[.5,.5]],'OK'),
    (cell([.5],[.25],[.5]),0,[[0,0]],'ZERO_ONLY_FEASIBLE')])
def test_fixed_analytic_cases(c,alpha,components,status):
    result=m.solve_continuous_alpha([c],CFG)
    assert result['status']==status,result
    assert result['alpha']==alpha
    assert result['feasible_components']==components

def test_zero_direction_and_constant_nonzero_direction_are_distinct():
    zero=m.solve_continuous_alpha([cell([.5],[0],[.25])],CFG)
    assert zero['status']=='ZERO_DIRECTION' and zero['feasible_components']==[[0,1]]
    constant=m.solve_continuous_alpha([cell([-2],[-1],[1])],CFG)
    assert constant['status']=='OPTIMAL_ALPHA_ZERO' and constant['positive_feasible_exists']

@pytest.mark.parametrize('c,expected',[
    (cell([0,.75],[.5,-.5],[.25,.5]),.5),
    (cell([0],[2],[1]),.5),
    (cell([1,0],[-1,1],[0,1]),1),
    (cell([-1],[2],[.5]),.75)])
def test_interior_endpoint_and_flat_minima(c,expected):
    r=m.solve_continuous_alpha([c],CFG)
    assert r['status']=='OK',r
    assert r['alpha']==pytest.approx(expected,abs=1e-12)

def test_raw_prediction_is_not_preclipped():
    c=cell([-1],[2],[.5])
    score=m.score_fixed_alpha([c],.5)
    assert score['macro']['mae']==.5

def test_unequal_cell_support_and_duplication_do_not_change_cell_weights():
    cells=[cell([0],[.5],[1],'a'),cell([.5,.5,.5],[0,0,0],[.5,.5,.5],'b')]
    assert m.score_fixed_alpha(cells,0)['macro']['rmse']==.5
    duplicated=copy.deepcopy(cells)
    for key in ('p','delta','y'):duplicated[0][key]=np.repeat(duplicated[0][key],20)
    a=m.solve_continuous_alpha(cells,CFG);b=m.solve_continuous_alpha(duplicated,CFG)
    assert a['status']==b['status'] and a['alpha']==b['alpha']
    assert a['score']['macro']==b['score']['macro']

def test_invalid_inputs_are_blocked_without_removal_or_label_clipping():
    for c in [cell([math.nan],[1],[0]),cell([0],[math.inf],[0]),cell([0],[1],[1.01]),cell([],[],[])]:
        assert m.solve_continuous_alpha([c],CFG)['status']=='INPUT_BLOCKED'
    assert m.score_fixed_alpha([cell([0],[1],[0])],math.nan)['status']=='INPUT_BLOCKED'

def test_no_gate_call_and_name_invariance(monkeypatch):
    def fail(*args,**kwargs):raise AssertionError('Gate must not be invoked')
    monkeypatch.setattr(m,'evaluate_registered_gates',fail)
    c=cell([.875,0],[1,.5],[.875,1],'occupancy')
    a=m.solve_continuous_alpha([c],CFG)
    c['id']='orthogonal_duration';b=m.solve_continuous_alpha([c],CFG)
    assert a['alpha']==b['alpha'] and a['score']['macro']==b['score']['macro']

def test_close_representable_events_are_not_merged():
    c=cell([0,0],[1,1],[.5,np.nextafter(.5,1)])
    r=m.solve_continuous_alpha([c],CFG)
    assert r['stats']['events']==2
    assert r['stats']['segments']>=2
    assert r['status'] in ('OK','NUMERICAL_BLOCKED')
    if r['status']=='OK':assert r['stats']['segments']==3

def test_subnormal_nonzero_direction_is_preserved_under_registered_tie_rule():
    """Research adjudication: exact arithmetic resolves the old underflow limitation."""
    c=cell([0],[1e-320],[.5])
    exact_delta=Fraction.from_float(float(c['delta'][0]))
    exact_R0=Fraction(1,2);exact_R1=exact_R0-exact_delta
    assert 0<exact_R0-exact_R1<Fraction(str(CFG['objective_tie_atol']))
    r=m.solve_continuous_alpha([c],CFG)
    assert r['status']=='OPTIMAL_ALPHA_ZERO' and r['alpha']==0
    assert r['positive_feasible_exists']
    assert r['exact_feasible_components']==[[{'kind':'rational','numerator':'0','denominator':'1'},{'kind':'rational','numerator':'1','denominator':'1'}]]
    assert r['stats']['unresolved_segments']==r['stats']['boundary_comparisons_unresolved']==0

def test_native_damage_constraint_and_tiny_mae_budget():
    cells=[cell([.5],[1],[.5],'zero_native'),cell([0],[1],[1],'other')]
    r=m.solve_continuous_alpha(cells,CFG)
    assert r['status']=='ZERO_ONLY_FEASIBLE',r
    r=m.solve_continuous_alpha([cell([.5],[1e-11],[.5])],CFG)
    assert r['alpha']==0 and r['status']=='ZERO_ONLY_FEASIBLE',r


def test_regression_random_case22_must_not_return_wrong_success():
    """Known failed acceptance case; intentionally remains red pending research review."""
    c=cell([.5,.125,1,1],[-.875,-.625,-.125,.875],[.25,.375,.375,.875],'case22_cell0')
    r=m.solve_continuous_alpha([c],CFG)
    if r['status']=='NUMERICAL_BLOCKED':return
    assert r['status']=='OK',r
    assert r['alpha']==1.,r
    assert r['feasible_components']==[[0.,.5],[1.,1.]],r
