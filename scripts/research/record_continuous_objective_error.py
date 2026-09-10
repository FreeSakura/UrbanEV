"""Decompose only the largest recorded synthetic objective discrepancy."""
import argparse
from decimal import Decimal,localcontext
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--verification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh diagnostic output required')
    report=json.loads(a.verification.read_text())
    if report['verification_status']!='PASS' or report['random_cases_executed']!=200:
        raise ValueError('Requires the complete passed synthetic run')
    row=max(report['random_cases'],key=lambda x:x['objective_abs_error']);result=row['production'];reference=row['reference']
    read=lambda value:F(int(value['numerator']),int(value['denominator']))
    selection=result['selected_objective_bounds'];minimum=result['finite_candidate_minimum_bounds']
    gap=read(selection['upper'])-read(minimum['lower'])
    step=abs(F.from_float(result['alpha'])-F.from_float(result['original_selected_alpha']))
    repair_gain=result['post_repair_objective']-result['pre_repair_objective']
    assert gap<=F(1,10**12)
    assert step<=F(1,10**10)
    assert repair_gain<=1e-10
    with localcontext() as ctx:
        ctx.prec=100
        dd=lambda f:Decimal(f.numerator)/Decimal(f.denominator)
        ref=Decimal(reference['objective_decimal80'])
        error_pre=Decimal.from_float(result['pre_repair_objective'])-ref
        reference_inside=dd(read(selection['lower']))<=ref<=dd(read(selection['upper']))
    out={'case_id':row['case'],'selected_exact_point':result['selected_exact_point'],
         'selected_objective_bounds':selection,'finite_candidate_minimum_bounds':minimum,
         'original_selected_alpha':result['original_selected_alpha'],'returned_alpha':result['alpha'],
         'boundary_repairs':result['stats']['boundary_repairs'],'bisection_iterations':result['stats']['bisection_iterations'],
         'pre_repair_objective':result['pre_repair_objective'],'post_repair_objective':result['post_repair_objective'],
         'reference_objective':reference['objective'],'reference_objective_decimal80':reference['objective_decimal80'],
         'reference_minimum_decimal80':reference['finite_minimum_decimal80'],
         'selection_gap_upper_exact':str(gap),'selection_gap_upper':float(gap),'objective_tie_budget':1e-12,'objective_search_budget':1e-12,
         'reference80_inside_selected_enclosure':reference_inside,
         'pre_repair_direct_minus_reference80':str(error_pre),
         'post_minus_pre':repair_gain,'post_minus_reference':row['objective_abs_error'],'repair_objective_budget':1e-10,
         'actual_float_alpha_change_exact':str(step),'actual_float_alpha_change':float(step),
         'actual_change_within_exact_decimal_budget':step<=F(1,10**10),
         'repair_diagnostics':result['repair_diagnostics'],'source_verification_sha256':hashlib.sha256(a.verification.read_bytes()).hexdigest(),
         'scope':'largest synthetic case only; separates finite-candidate selection, float scoring and one bounded repair; not research performance',
         'real_calibration_run':False,'tail_scoring':False,'new_foundation_inference':False}
    a.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'case_id':row['case'],'selection_gap_upper':float(gap),'post_minus_pre':repair_gain,
                      'actual_alpha_change':float(step),'actual_step_within_cap':True,'returned_alpha':result['alpha']}))


if __name__=='__main__':main()
