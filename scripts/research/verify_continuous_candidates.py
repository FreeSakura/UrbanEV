"""Synthetic candidate-reduction regression receipts; no research arrays loaded."""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from verify_continuous_step_synthetic import ROOT,monitored_solve
from urbanev_forecast.continuous_candidates import Candidate,reduce_candidates,CandidateReductionBlocked
from urbanev_forecast.continuous_boundaries import Boundary


def candidate(alpha,bounds,segment):return Candidate(Boundary.of(alpha),bounds,segment)


def root_bounds(bits):
    scale=1<<bits;z=math.isqrt((1<<(2*bits))//8)
    return F(z,scale),F(z+1,scale)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError('Fresh output required')
    start=time.perf_counter();tolerance=F(1,10**12);offset=tolerance-F(1,2**100);lo,hi=root_bounds(64)
    cases=[candidate(1,(lo,hi),0),candidate(0,(lo+offset,hi+offset),1)];requests=[];stats={}
    def refine(indices,bits):
        requests.append({'indices':list(indices),'bits':bits});l,u=root_bounds(bits)
        return {'bounds':{i:((l,u) if i==0 else (l+offset,u+offset)) for i in indices},'refined_segment_count':len(indices)}
    result=reduce_candidates(cases,refine,stats=stats)
    assert result['selected_index']==1 and requests==[{'indices':[0,1],'bits':128}]
    linear=[]
    def forbidden(*args):raise AssertionError('Unexpected refinement')
    for count in (600,1200,2400):
        inputs=[candidate(F(count-1-i,count-1),(F(1),F(1)),i) for i in range(count)]
        outcome=reduce_candidates(inputs,forbidden)
        assert outcome['selected_index']==count-1 and outcome['diagnostics']['classification_visits']==count
        linear.append({'candidate_count':count,'classification_visits':outcome['diagnostics']['classification_visits'],
                       'minimum_bound_visits':outcome['diagnostics']['minimum_bound_visits'],
                       'historical_list_membership_triangular_count_not_rerun':count*(count+1)//2})
    failures=[]
    for name,new in [('budget_exhaustion',(F(0),F(1))),('enclosure_contradiction',(F(2),F(3)))]:
        def callback(indices,bits):return {'bounds':{i:new for i in indices},'refined_segment_count':len(indices)}
        try:reduce_candidates([candidate(0,(F(0),F(1)),0),candidate(1,(F(0),F(1)),1)],callback)
        except CandidateReductionBlocked as error:
            failures.append({'case':name,'reason':str(error),'diagnostics':error.diagnostics})
        else:raise AssertionError('Expected block did not occur')
    report={'status':'PASS','isolated_minimum_dependency_case':{'diagnostics':result['diagnostics'],'refinement_requests':requests,'stats':stats},
            'all_tie_linear_counts':linear,'explicit_block_cases':failures,'elapsed_seconds':time.perf_counter()-start,
            'real_calibration_run':False,'tail_scoring':False,'new_foundation_inference':False,'solver_code_review':'pending'}
    config=json.loads((ROOT/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_CANDIDATE_REPAIR_ROUNDING.json').read_text())
    monitoring={k:0 for k in ('builtin_open','path_open','numpy_load','model_init','gate_from_solver')};dense=[]
    for n in (63,255,1023):
        t=np.arange(1,n+1,dtype=np.float64)/(n+1);direction=2.**-36
        c={'id':'dense_tie','p':np.r_[.75,.5-direction*t],'delta':np.r_[0.,np.full(n,direction)],'y':np.full(n+1,.5)}
        began=time.perf_counter();value=monitored_solve([c],config,monitoring);elapsed=time.perf_counter()-began
        assert value['status']=='OPTIMAL_ALPHA_ZERO' and value['alpha']==0 and value['positive_feasible_exists']
        assert value['feasible_components']==[[0.,1.]] and value['stats']['events']==n
        assert value['candidate_reduction']['classification_visits']==value['candidate_reduction']['candidate_count']==3*(n+1)
        bound=F(1,2**72)*F(math.isqrt(n+1),2)
        assert bound<F(1,10**12)
        dense.append({'n':n,'samples':n+1,'rmse_variation_upper_bound':str(bound),'elapsed_seconds':elapsed,
                      'status':value['status'],'alpha':value['alpha'],'exact_feasible_components':value['exact_feasible_components'],
                      'stats':value['stats'],'candidate_reduction':value['candidate_reduction']})
    dense_report={'status':'PASS','cases':dense,'observed_forbidden_call_counts':monitoring,
                  'scope':'synthetic all-tie workload; elapsed time covers each solver invocation only',
                  'real_calibration_run':False,'tail_scoring':False,'new_foundation_inference':False}
    a.output_dir.mkdir(parents=True)
    for name,obj in [('candidate_reduction_verification.json',report),('dense_tie_verification.json',dense_report)]:
        (a.output_dir/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','linear_counts':linear,'dense_n':[v['n'] for v in dense],
                      'observed_forbidden_call_counts':monitoring}))


if __name__=='__main__':main()
