from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
import pytest
from urbanev_forecast import continuous_step as api
from urbanev_forecast.continuous_exact import _bounded_single_repair
from urbanev_forecast.continuous_boundaries import BoundaryUnresolved

CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_CANDIDATE_REPAIR_ROUNDING.json').read_text())


@pytest.mark.parametrize('original,direction,expected',[
    (.17189523021473727,-1,.17189523011473729),
    (.2,1,.2000000001)])
def test_inward_rounding_enforces_actual_float_step(original,direction,expected):
    repaired,receipt=_bounded_single_repair(original,1e-10,direction)
    actual=direction*(F.from_float(repaired)-F.from_float(original))
    assert 0<actual<=F(1,10**10)
    assert repaired==expected
    assert receipt['inward_rounding_used'] and receipt['nextafter_count']==1


@pytest.mark.parametrize('direction',[-1,1])
def test_exactly_representable_move_needs_no_inward_rounding(direction):
    distance=2.**-40
    repaired,receipt=_bounded_single_repair(.5,distance,direction)
    assert direction*(F.from_float(repaired)-F(1,2))==F.from_float(distance)
    assert receipt['nextafter_count']==0


@pytest.mark.parametrize('direction',[-1,1])
def test_unrepresentable_positive_move_blocks_without_retries(direction):
    with pytest.raises(BoundaryUnresolved,match='NO_REPRESENTABLE_BOUNDED_SINGLE_REPAIR'):
        _bounded_single_repair(.5,1e-20,direction)


@pytest.mark.parametrize('args',[(float('nan'),1e-10,1),(.5,0,1),(.5,1e-10,0)])
def test_invalid_repair_inputs_are_rejected(args):
    with pytest.raises(BoundaryUnresolved,match='INVALID_SINGLE_REPAIR_INPUT'):
        _bounded_single_repair(*args)


def test_decimal_cap_is_not_replaced_by_the_larger_binary_parameter():
    repaired,receipt=_bounded_single_repair(.2,2e-10,1)
    assert 0<F.from_float(repaired)-F.from_float(.2)<=F(1,10**10)
    assert receipt['allowed_step_exact']=={'numerator':'1','denominator':'10000000000'}


def test_case35_scores_only_the_final_inward_rounded_point(monkeypatch):
    rng=np.random.default_rng(20260910)
    for index in range(36):
        cells=[]
        for j in range(int(rng.integers(1,4))):
            n=int(rng.integers(4,8))
            cells.append({'id':f'case{index}_cell{j}','p':rng.integers(-4,13,n)/8,
                          'delta':rng.integers(-8,9,n)/8,'y':rng.integers(0,9,n)/8})
    calls=[];original=api._score_prepared
    def observe(c,alpha):
        calls.append(alpha)
        return original(c,alpha)
    monkeypatch.setattr(api,'_score_prepared',observe)
    result=api.solve_continuous_alpha(cells,CFG)
    assert result['status']=='OK',result
    assert result['original_selected_alpha']==.17189523021473727
    assert result['alpha']==.17189523011473729
    assert result['stats']['boundary_repairs']==1
    step=abs(F.from_float(result['alpha'])-F.from_float(result['original_selected_alpha']))
    assert 0<step<=F(1,10**10)
    assert result['repair_diagnostics']['nearest_rounded_alpha'] not in calls
    assert result['alpha'] in calls and result['stats']['full_array_direct_checks']<=8
