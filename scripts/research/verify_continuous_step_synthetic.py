"""Synthetic-only acceptance runner; stops on the first failed required random case."""
import argparse
import builtins
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast import continuous_step as production
from urbanev_forecast.foundation import FoundationBackend
spec=importlib.util.spec_from_file_location('decimal_oracle',ROOT/'tests/continuous_step_reference.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)


def cell(p,d,y,ident):
    return {'id':ident,'p':np.asarray(p,dtype=float),'delta':np.asarray(d,dtype=float),'y':np.asarray(y,dtype=float)}


def monitored_solve(cells,config,counts):
    def forbidden(kind):
        def stop(*args,**kwargs):
            counts[kind]+=1
            raise AssertionError('Forbidden dependency called: '+kind)
        return stop
    with patch.object(builtins,'open',forbidden('builtin_open')),patch.object(Path,'open',forbidden('path_open')),patch.object(np,'load',forbidden('numpy_load')),patch.object(FoundationBackend,'__init__',forbidden('model_init')),patch.object(production,'evaluate_registered_gates',forbidden('gate_from_solver')):
        return production.solve_continuous_alpha(cells,config)


def peak_rss():
    if os.name=='nt':
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
        kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=ctypes.c_void_p
        psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):raise OSError('Peak RSS unavailable')
        return int(counters.PeakWorkingSetSize)
    import resource
    raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform=='darwin' else raw*1024)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['random','scale2','scale4'],default='random');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use fresh output')
    config_path=ROOT/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_SOLVER.json'
    config=json.loads(config_path.read_text());rng=np.random.default_rng(20260910)
    monitoring={name:0 for name in ('builtin_open','path_open','numpy_load','model_init','gate_from_solver')}
    report={'phase':a.phase,'seed':20260910,'scope':'synthetic arrays only','config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'production_sha256':hashlib.sha256((ROOT/'src/urbanev_forecast/continuous_step.py').read_bytes()).hexdigest(),
            'reference_sha256':hashlib.sha256((ROOT/'tests/continuous_step_reference.py').read_bytes()).hexdigest(),
            'monitor_scope':'all production solve invocations; imports and report writes outside guarded region',
            'observed_forbidden_call_counts':monitoring,'real_calibration_run':False,'tail_scoring':False,'new_foundation_inference':False,
            'third_fold_validation_opened':False,'third_fold_test_opened':False,'solver_code_review':'pending',
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.system()}}
    start=time.perf_counter();failed=False
    if a.phase=='random':
        fixed=[]
        for name,pred,d,y,expected,components in [('A',[.875,0],[1,.5],[.875,1],1.,[[0,0],[.25,1]]),('B',[.875,.875,0],[1,1,.5],[.875,.875,.25],.5,[[0,0],[.5,.5]]),('C',[.5],[.25],[.5],0.,[[0,0]])]:
            inputs=[cell(pred,d,y,name)];actual=monitored_solve(inputs,config,monitoring);ref=oracle.reference_solve(inputs)
            okay=actual.get('alpha')==expected and actual.get('feasible_components')==components
            fixed.append({'case':name,'passed':okay,'production':actual,'reference':ref})
            if not okay:failed=True;break
        report['fixed_cases']=fixed;rows=[]
        if not failed:
            for index in range(200):
                cells=[]
                for j in range(int(rng.integers(1,4))):
                    n=int(rng.integers(4,8))
                    cells.append(cell(rng.integers(-4,13,n)/8,rng.integers(-8,9,n)/8,rng.integers(0,9,n)/8,f'case{index}_cell{j}'))
                actual=monitored_solve(cells,config,monitoring);ref=oracle.reference_solve(cells)
                row={'case':index,'production_status':actual['status'],'reference':ref,'production':actual}
                if actual['status'] in ('INPUT_BLOCKED','NUMERICAL_BLOCKED'):
                    row.update(passed=False,reason='Required ordinary randomized case blocked; needs research/code review')
                else:
                    component_count=len(actual['feasible_components'])==len(ref['components'])
                    endpoint_error=max((abs(x-y) for pair,rpair in zip(actual['feasible_components'],ref['components']) for x,y in zip(pair,rpair)),default=0.) if component_count else None
                    objective_error=abs(actual['score']['macro']['rmse']-ref['objective'])
                    native=production.score_fixed_alpha(cells,0)
                    violation=max([actual['score']['macro']['mae']-native['macro']['mae']]+[c['rmse']-1.01*n['rmse'] for c,n in zip(actual['score']['cells'],native['cells'])])
                    passed=component_count and endpoint_error<=1e-10 and objective_error<=1e-10 and violation<=0
                    row.update(passed=bool(passed),component_count_equal=component_count,endpoint_max_abs_error=endpoint_error,objective_abs_error=objective_error,final_direct_constraint_violation=violation)
                rows.append(row)
                if not row['passed']:
                    row['reproducible_synthetic_inputs']=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in cells]
                    failed=True;break
        report.update(random_cases_planned=200,random_cases_executed=len(rows),random_cases=rows,
                      verification_status='FAILED_STOPPED' if failed else 'PASS',
                      scale_tests_status='NOT_RUN_IN_THIS_PHASE',
                      synthetic_only_results_not_research_metrics=True)
    else:
        hs=[3,12] if a.phase=='scale2' else [3,6,9,12]
        cells=[]
        for b in range(3):
            for h in hs:
                shape=(14,h,275)
                pp=rng.uniform(.1,.9,shape);dd=rng.uniform(-1,1,shape);yy=rng.uniform(0,1,shape)
                cells.append(cell(pp,dd,yy,{'stage':'calibration','block':b,'horizon':h}))
        result=monitored_solve(cells,config,monitoring)
        failed=result['status'] in ('INPUT_BLOCKED','NUMERICAL_BLOCKED')
        report.update(sample_count=sum(c['p'].size for c in cells),cell_count=len(cells),result=result,
                      process_peak_rss_bytes=peak_rss(),memory_scope='fresh runner process peak resident/working-set memory including imports and synthetic input creation',
                      scan_counter_definition='logical initialization/event/scoring phases, not individual NumPy ufunc traversals; event replay has no per-event full sample scoring',
                      verification_status='FAILED_STOPPED' if failed else 'PASS')
    report['elapsed_seconds']=time.perf_counter()-start
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ('phase','verification_status','random_cases_executed','elapsed_seconds','sample_count','process_peak_rss_bytes','observed_forbidden_call_counts')}))
    if failed:raise SystemExit(2)


if __name__=='__main__':main()
