"""Fair AP1 language/compiler/kernel benchmark with explicit cutoff records."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_audit.persistent_events import automaton,_extrema,suffix_extrema
from urbanev_audit.regular_event_baseline import compile_event,canonical,generic_edge_paths


def timed(fn,repeats=3):
    samples=[];answer=None
    for _ in range(repeats):
        start=time.perf_counter();answer=fn();samples.append(time.perf_counter()-start)
    return float(np.median(samples)),answer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);rows=[];compilers=[]
    cfg=json.loads((ROOT/'configs/research/SHARED_MISSING_EVENTS_AP1_20260916.json').read_text())
    bc=cfg['benchmark'];start=time.perf_counter();cpu=time.process_time()
    o=np.array([-1,0,1]*4,np.int8);c=np.ones(9);edges,accept,_=automaton(4,3)
    first_calls={}
    for name,fn in [('state_kernel',lambda:_extrema(o,c,4,edges,accept)),('edge_kernel',lambda:generic_edge_paths(o,c,4,edges,accept)),('suffix_kernel',lambda:suffix_extrema(o,c,4,3))]:
        before=time.perf_counter();fn();first_calls[name]=time.perf_counter()-before
    dc=cfg['datasets']['household'];end=dc['calibration_end']+max(bc['T'])
    with zipfile.ZipFile(a.data/'household_original.zip') as archive:
        name=next(n for n in archive.namelist() if n.endswith('household_power_consumption.txt'))
        with archive.open(name) as f:values=pd.read_csv(f,sep=';',nrows=end,usecols=['Global_active_power'],na_values=['?'])['Global_active_power'].to_numpy(float)[dc['calibration_end']:]
    natural=np.where(np.isnan(values),-1,values>=3).astype(np.int8)
    with threadpool_limits(limits=2):
        for k in bc['K']:
            for ell in sorted({1,min(3,k),min(5,k),k,k//2}):
                compile_event.cache_clear();automaton.cache_clear()
                before=time.perf_counter();manual,event,_=automaton(k,ell);manual_seconds=time.perf_counter()-before
                try:
                    generic,flags,meta=compile_event(k,ell,bc['compiler_timeout_seconds']);status='COMPLETE'
                    ca,cf=canonical(manual,event)
                    if not np.array_equal(ca,generic) or not np.array_equal(cf,flags):raise AssertionError('Generic minimal DFA differs from specialized graph')
                    compiler_seconds=meta['construction_seconds']
                except TimeoutError:
                    status='CONSTRUCTION_BUDGET_CUTOFF';compiler_seconds=time.perf_counter()-before;generic=flags=None
                expected=1+ell+sum(min(ell,d) for d in range(1,k-ell+1))
                if len(manual)!=expected:raise AssertionError('Reachable-state formula mismatch')
                compilers.append({'K':k,'L':ell,'specialized_states':len(manual),'generic_states':len(generic) if generic is not None else '',
                                  'state_formula':expected,'specialized_construct_seconds':manual_seconds,'generic_construct_seconds':compiler_seconds,'generic_status':status,
                                  **({key:meta[key] for key in ['determinize_seconds','minimize_seconds','union_product_seconds']} if generic is not None else {})})
                for t in bc['T']:
                    for layout in ('iid30','blocks30','natural_power'):
                        rng=np.random.default_rng(20260916+k*1000+ell+t)
                        observed=rng.integers(0,2,t,dtype=np.int8)
                        if layout=='iid30':observed[rng.random(t)<.3]=-1
                        elif layout=='blocks30':
                            block=max(1,k//4)
                            for anchor in rng.choice(t,max(1,int(.3*t/block)),replace=False):observed[anchor:anchor+block]=-1
                        else:observed=natural[:t].copy()
                        weights=rng.uniform(-2,2,t-k+1)
                        sp,expected_bounds=timed(lambda:_extrema(observed,weights,k,manual,event))
                        edge,_=timed(lambda:generic_edge_paths(observed,weights,k,manual,event))
                        ge='';error=''
                        if generic is not None:
                            ge,actual=timed(lambda:_extrema(observed,weights,k,generic,flags))
                            error=max(abs(actual[0]-expected_bounds[0]),abs(actual[1]-expected_bounds[1]))
                            if error>1e-9:raise AssertionError('Compiled DFA bounds mismatch')
                        old='';old_status='NOT_RUN_K_GT_16'
                        if k<=bc['suffix_max_K']:
                            old,actual=timed(lambda:suffix_extrema(observed,weights,k,ell));old_status='COMPLETE'
                            if max(abs(actual[0]-expected_bounds[0]),abs(actual[1]-expected_bounds[1]))>1e-9:raise AssertionError('Suffix bounds mismatch')
                        rows.append({'T':t,'K':k,'L':ell,'layout':layout,'missing_rate':float(np.mean(observed==-1)),
                                     'specialized_state_kernel_seconds':sp,'generic_dfa_same_kernel_seconds':ge,'generic_rd_edge_kernel_seconds':edge,
                                     'suffix_kernel_seconds':old,'suffix_status':old_status,'generic_status':status,'generic_bound_error':error,
                                     'rolling_extrema_bytes':4*len(manual)*8,'graph_array_bytes':manual.nbytes+event.nbytes})
                pd.DataFrame(rows).to_csv(a.output/'algorithm_benchmark.csv',index=False)
                pd.DataFrame(compilers).to_csv(a.output/'compiler_benchmark.csv',index=False)
                print(json.dumps({'K':k,'L':ell,'states':len(manual),'generic_status':status,'construct_seconds':compiler_seconds}),flush=True)
    pd.DataFrame(compilers).to_csv(a.output/'compiler_stage_profiles.csv',index=False)
    accounting=pd.DataFrame(rows).merge(pd.DataFrame(compilers)[['K','L','specialized_construct_seconds','generic_construct_seconds']],on=['K','L'])
    accounting['specialized_cold_graph_warm_kernel_seconds']=accounting.specialized_construct_seconds+accounting.specialized_state_kernel_seconds
    accounting['generic_cold_graph_warm_kernel_seconds']=accounting.generic_construct_seconds+accounting.generic_dfa_same_kernel_seconds
    accounting.to_csv(a.output/'end_to_end_component_accounting.csv',index=False)
    receipt={'status':'COMPLETE','kernel_cases':len(rows),'compiler_cases':len(compilers),'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
             'first_call_or_disk_cache_load_seconds':first_calls,'timing_policy':'Three warm kernel repetitions for every implementation; compiler construction separately cold per K/L; disk JIT caches may exist',
             'memory_scope':'Explicit rolling DP and graph array bytes, not whole-process peak RSS','terminal_policy':'All final states legal; acceptance used only as per-step output cost'}
    (a.output/'benchmark_execution.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
