from fractions import Fraction as F
import math
import pytest
import json
from pathlib import Path
import numpy as np
from urbanev_forecast.continuous_candidates import Candidate,reduce_candidates,CandidateReductionBlocked
from urbanev_forecast.continuous_boundaries import Boundary
from urbanev_forecast.continuous_step import solve_continuous_alpha


def candidate(alpha,bounds,segment):
    return Candidate(Boundary.of(alpha),tuple(map(F,bounds)),segment)


def root_bounds(bits):
    scale=1<<bits;n=math.isqrt((1<<(2*bits))//8)
    return F(n,scale),F(n+1,scale)


def test_refines_definite_possible_minimum_as_well_as_unresolved_candidate():
    tolerance=F(1,10**12);offset=tolerance-F(1,2**100);low,high=root_bounds(64)
    cases=[candidate(1,(low,high),0),candidate(0,(low+offset,high+offset),1)]
    calls=[];stats={}
    def refine(indices,bits):
        calls.append((indices,bits));a,b=root_bounds(bits)
        return {'bounds':{i:((a,b) if i==0 else (a+offset,b+offset)) for i in indices},'refined_segment_count':len(indices)}
    result=reduce_candidates(cases,refine,stats=stats)
    assert calls==[((0,1),128)]
    assert result['selected_index']==1
    assert result['diagnostics']['rounds'][0]['in_count']==1
    assert result['diagnostics']['rounds'][1]['in_count']==2
    assert result['diagnostics']['rounds'][1]['unresolved_count']==0
    assert stats['max_objective_refinement_bits']==128


@pytest.mark.parametrize('count',[600,1200,2400])
def test_all_ties_have_linear_classification_visits(count):
    cases=[candidate(F(count-1-i,count-1),(1,1),i) for i in range(count)]
    def forbidden(*args):raise AssertionError('Exact ties do not need refinement')
    result=reduce_candidates(cases,forbidden)
    assert result['selected_index']==count-1
    assert result['diagnostics']['classification_visits']==count
    assert result['diagnostics']['minimum_bound_visits']==count
    assert result['diagnostics']['rounds'][0]['in_count']==count


def test_all_but_one_are_excluded_without_changing_positions():
    cases=[candidate(1,(0,0),0),candidate(0,(1,1),1),candidate(F(1,2),(2,2),2)]
    result=reduce_candidates(cases,lambda *args:None)
    assert result['selected_index']==0
    assert result['diagnostics']['rounds'][0]['out_count']==2
    assert [c.point.rational for c in cases]==[F(1),F(0),F(1,2)]


def test_equal_objectives_with_different_initial_enclosures():
    cases=[candidate(1,(F(49,100),F(51,100)),0),candidate(0,(F(1,2),F(1,2)),1)]
    def refine(indices,bits):return {'bounds':{i:(F(1,2),F(1,2)) for i in indices},'refined_segment_count':len(indices)}
    result=reduce_candidates(cases,refine)
    assert result['selected_index']==1
    assert len(result['diagnostics']['rounds'])==2


def test_budget_exhaustion_records_remaining_candidates():
    cases=[candidate(0,(0,1),0),candidate(1,(0,1),1)]
    def refine(indices,bits):return {'bounds':{i:(F(0),F(1)) for i in indices},'refined_segment_count':len(indices)}
    with pytest.raises(CandidateReductionBlocked,match='FINITE_CANDIDATE_TIE_COMPARISON_UNRESOLVED') as caught:
        reduce_candidates(cases,refine)
    assert [r['precision_bits'] for r in caught.value.diagnostics['rounds']]==[64,128,256,512]
    assert caught.value.diagnostics['unresolved_indices']==[0,1]


def test_disjoint_new_enclosure_blocks_and_does_not_expand_old_bounds():
    cases=[candidate(0,(0,1),0),candidate(1,(0,1),1)]
    def refine(indices,bits):return {'bounds':{i:(F(2),F(3)) for i in indices},'refined_segment_count':len(indices)}
    with pytest.raises(CandidateReductionBlocked,match='CANDIDATE_ENCLOSURE_CONTRADICTION') as caught:
        reduce_candidates(cases,refine)
    assert caught.value.diagnostics['contradiction_candidate_index']==0


def test_broader_refinement_never_widens_an_enclosure():
    cases=[candidate(0,(F(0),F(1)),0)]
    def refine(indices,bits):return {'bounds':{0:(F(-1),F(2))},'refined_segment_count':1}
    with pytest.raises(CandidateReductionBlocked) as caught:reduce_candidates(cases,refine)
    for row in caught.value.diagnostics['rounds']:
        assert row['min_lower']=={'numerator':'0','denominator':'1'}
        assert row['min_upper']=={'numerator':'1','denominator':'1'}


@pytest.mark.parametrize('n',[63,255,1023])
def test_dense_tie_family_retains_all_candidates_with_linear_classification(n):
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_CANDIDATE_REPAIR_ROUNDING.json').read_text())
    t=np.arange(1,n+1,dtype=np.float64)/(n+1);d=2.**-36
    cell={'id':'dense_tie','p':np.r_[.75,.5-d*t],'delta':np.r_[0.,np.full(n,d)],'y':np.full(n+1,.5)}
    result=solve_continuous_alpha([cell],config)
    assert result['status']=='OPTIMAL_ALPHA_ZERO',result
    assert result['alpha']==0 and result['positive_feasible_exists']
    assert result['feasible_components']==[[0.,1.]]
    assert result['stats']['events']==n and result['stats']['segments_total']==n+1
    reduction=result['candidate_reduction']
    assert reduction['candidate_count']==3*(n+1)
    assert reduction['classification_visits']==reduction['candidate_count']
    assert len(reduction['rounds'])==1 and reduction['rounds'][0]['in_count']==reduction['candidate_count']
