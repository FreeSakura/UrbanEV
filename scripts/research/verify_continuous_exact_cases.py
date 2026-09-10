"""Materialize exact-predicate and repair traces using synthetic inputs only."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from verify_continuous_step_synthetic import ROOT,cell,monitored_solve
from urbanev_forecast import continuous_exact as exact
from urbanev_forecast.continuous_step import _prepare,solve_continuous_alpha
from urbanev_forecast.continuous_boundaries import (Boundary,BoundaryUnresolved,quadratic_roots,
    compare_boundaries,sign_at,intersect_polynomial)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError('Fresh verification directory required')
    cfg=json.loads((ROOT/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_SOLVER_REPAIR.json').read_text())
    counts={k:0 for k in ('builtin_open','path_open','numpy_load','model_init','gate_from_solver')}
    results={};inputs={}
    for name,d in [('original',-.125),('ulp_toward_zero',np.nextafter(-.125,0)),('ulp_away',np.nextafter(-.125,-np.inf))]:
        c=cell([.5,.125,1,1],[-.875,-.625,d,.875],[.25,.375,.375,.875],name);inputs[name]=c
        results[name]=monitored_solve([c],cfg,counts)
    scaled=cell([.875,.875,0],[1.5,1.5,.75],[.875,.875,.25],'B_scaled')
    results['B_scaled']=monitored_solve([scaled],cfg,counts)
    assert results['original']['status']=='OK' and results['original']['alpha']==1
    assert results['ulp_toward_zero']['status']=='OK' and results['ulp_toward_zero']['alpha']!=1
    assert results['B_scaled']['reason']=='UNREPRESENTABLE_SELECTED_SINGLETON'
    assert results['ulp_away']['exact_feasible_components'][-1][0]!=results['ulp_away']['exact_feasible_components'][-1][1]
    def exact_gap(c):
        p,d,y=([F.from_float(float(v)) for v in c[k]] for k in ('p','delta','y'))
        clip=lambda v:max(F(0),min(F(1),v))
        return sum((abs(clip(a+b)-c)-abs(clip(a)-c) for a,b,c in zip(p,d,y)),F(0))/len(p)
    gaps={name:str(exact_gap(c)) for name,c in inputs.items()}
    assert exact_gap(inputs['ulp_toward_zero'])==F(1,2**58)
    stats={'max_exact_integer_bits':0,'event_continuity_failures':0,'event_groups_processed':0}
    sweep=exact.Sweep(_prepare([inputs['original']]),stats)
    trace=[]
    for segment in sweep.segments():
        trace.append({'left':segment.left.encode(),'right':segment.right.encode(),
                      'global_mse_coefficients':[str(v) for v in segment.mse(0)],
                      'global_mae_coefficients':[str(v) for v in segment.mae],
                      'native_mae':str(segment.native_mae)})
    assert trace[-1]['global_mae_coefficients']==['-1/32','11/32']
    assert trace[-1]['native_mae']=='5/16'
    domain=(Boundary.of(0),Boundary.of(1));epsilon=F(1,2**2000)
    tangent=intersect_polynomial(domain,(F(1),F(-1),F(1,4)))
    empty=intersect_polynomial(domain,(F(1),F(-1),F(1,4)+epsilon))
    narrow=intersect_polynomial(domain,(F(1),F(-1),F(1,4)-epsilon))
    assert tangent[0].rational==tangent[1].rational==F(1,2) and empty is None
    assert compare_boundaries(*narrow)=='LESS'
    root=quadratic_roots(1,0,-2)[1];same=quadratic_roots(6,0,-12)[1]
    assert compare_boundaries(root,same)=='EQUAL' and sign_at((1,0,-2),root)=='ZERO'
    near=quadratic_roots(F(1),F(0),-F(2)-F(1,2**2000))[1];bstats={}
    assert compare_boundaries(root,near,bstats)=='UNRESOLVED'
    def inject(*args,**kwargs):raise BoundaryUnresolved('INJECTED_UNRESOLVED_BOUNDARY')
    with patch.object(exact,'_feasible',inject):
        blocked=solve_continuous_alpha([inputs['original']],cfg)
    assert blocked['status']=='NUMERICAL_BLOCKED' and blocked['stats']['unresolved_segments']==1
    report={'status':'PASS','scope':'synthetic exact-boundary verification, not research performance',
        'predicate_cases':{'tangent':[v.encode() for v in tangent],'positive_perturbation':'CERTIFIED_EMPTY',
            'negative_perturbation':[v.encode() for v in narrow],'same_root':'EQUAL','polynomial_at_own_root':'ZERO',
            'different_close_roots':'UNRESOLVED_WITHIN_REGISTERED_BUDGET','refinement':bstats},
        'propagation_injection':blocked,'observed_forbidden_call_counts':counts,
        'real_calibration_run':False,'tail_scoring':False,'new_foundation_inference':False,
        'third_fold_validation_opened':False,'third_fold_test_opened':False,'solver_code_review':'pending'}
    case_report={'status':'EXPECTED_FIXED_AND_REPRESENTATION_CASES_VERIFIED','results':results,
        'exact_mae_gaps_at_alpha1':gaps,'case22_global_trace':trace,
        'narrow_interval_note':'Exact component retained; registered quarter-width float repair may be unrepresentable, returning explicit NUMERICAL_BLOCKED instead of choosing another alpha',
        'observed_forbidden_call_counts':counts,'real_data':False}
    a.output_dir.mkdir(parents=True)
    for name,obj in [('boundary_predicate_verification.json',report),('case22_repair_trace.json',case_report)]:
        (a.output_dir/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','case22_alpha':results['original']['alpha'],'ulp_toward_zero_gap':gaps['ulp_toward_zero'],
                      'ulp_away_status':results['ulp_away']['status'],'B_scaled_status':results['B_scaled']['status'],'observed_forbidden_call_counts':counts}))


if __name__=='__main__':main()
