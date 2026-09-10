from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
from urbanev_forecast.continuous_boundaries import Boundary,quadratic_roots,compare_boundaries,sign_at,intersect_polynomial
from urbanev_forecast.continuous_step import solve_continuous_alpha

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_SOLVER_REPAIR.json').read_text())


def test_quadratic_singleton_and_exact_perturbations():
    domain=(Boundary.of(0),Boundary.of(1));epsilon=F(1,2**2000)
    point=intersect_polynomial(domain,(F(1),F(-1),F(1,4)))
    assert point[0].rational==point[1].rational==F(1,2)
    assert intersect_polynomial(domain,(F(1),F(-1),F(1,4)+epsilon)) is None
    narrow=intersect_polynomial(domain,(F(1),F(-1),F(1,4)-epsilon))
    assert compare_boundaries(*narrow)=='LESS'
    assert narrow[0].identity()!=narrow[1].identity()
    assert narrow[0].approximate()==narrow[1].approximate()==.5


def test_normalized_root_identity_and_reduced_polynomial_sign():
    root=quadratic_roots(1,0,-2)[1];same=quadratic_roots(6,0,-12)[1]
    assert compare_boundaries(root,same)=='EQUAL'
    assert sign_at((F(1),F(0),F(-2)),root)=='ZERO'
    assert sign_at((F(2),F(3),F(-4)),root)=='POSITIVE'
    assert compare_boundaries(root,Boundary.of(F(3,2)))=='LESS'


def test_unresolvable_distinct_roots_never_merge():
    root=quadratic_roots(1,0,-2)[1]
    near=quadratic_roots(F(1),F(0),-F(2)-F(1,2**2000))[1]
    stats={}
    assert root.identity()!=near.identity()
    assert compare_boundaries(root,near,stats)=='UNRESOLVED'
    assert stats['max_root_refinement_bits']==512
    assert stats['boundary_comparisons_unresolved']==1


def case22(direction):
    return {'id':'case22','p':[.5,.125,1,1],'delta':[-.875,-.625,direction,.875],
            'y':[.25,.375,.375,.875]}


def test_ulp_toward_zero_does_not_accept_false_equal_float_mae():
    result=solve_continuous_alpha([case22(np.nextafter(-.125,0))],CFG)
    assert result['status']=='OK',result
    assert result['alpha']!=1
    assert all(pair[1]!={'kind':'rational','numerator':'1','denominator':'1'} for pair in result['exact_feasible_components'])


def test_ulp_away_preserves_narrow_feasible_component():
    result=solve_continuous_alpha([case22(np.nextafter(-.125,-np.inf))],CFG)
    # The contract requires retaining the interval, not forcing a float answer:
    # the prescribed quarter-width repair can round back to the same float.
    assert result['status'] in ('OK','NUMERICAL_BLOCKED'),result
    if result['status']=='NUMERICAL_BLOCKED':
        assert result['reason']=='NO_REPRESENTABLE_SINGLE_REPAIR_INSIDE_COMPONENT'
    else:
        assert result['alpha']>0
    lo,hi=result['exact_feasible_components'][-1]
    assert lo!=hi and hi=={'kind':'rational','numerator':'1','denominator':'1'}
    width=F(int(hi['numerator']),int(hi['denominator']))-F(int(lo['numerator']),int(lo['denominator']))
    assert 0<width<F(1,10**12)
    assert result['stats']['segments_total']==result['stats']['segments_classified']
    assert result['stats']['unresolved_segments']==0


def test_unrepresentable_winning_singleton_blocks_instead_of_using_runner_up():
    c={'id':'Bscaled','p':[.875,.875,0],'delta':[1.5,1.5,.75],'y':[.875,.875,.25]}
    result=solve_continuous_alpha([c],CFG)
    assert result['status']=='NUMERICAL_BLOCKED',result
    assert result['reason']=='UNREPRESENTABLE_SELECTED_SINGLETON'
    assert result['selected_exact_point']=={'kind':'rational','numerator':'1','denominator':'3'}


def test_case22_complete_classification_and_direct_check_budget():
    result=solve_continuous_alpha([case22(-.125)],CFG)
    assert result['status']=='OK' and result['alpha']==1
    stats=result['stats']
    assert stats['segments_total']==stats['segments_classified']==4
    assert stats['unresolved_segments']==stats['boundary_comparisons_unresolved']==stats['event_continuity_failures']==0
    assert stats['full_array_direct_checks']<=8
    assert result['segment_trace_prefix'][-1]['classification']=='CERTIFIED_FEASIBLE_SINGLETON'
