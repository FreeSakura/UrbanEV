"""A-P0 exhaustive correctness and matched-kernel timing; no real data reads."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from urbanev_audit.persistent_events import weighted_bounds, paired_bounds, suffix_extrema, automaton, _extrema
from urbanev_audit.paired_events import joint_brier_bounds


def brute(o,c,k,ell):
    unknown=np.flatnonzero(np.asarray(o)==-1)
    minimum,maximum=np.inf,-np.inf
    for assignment in product((0,1),repeat=len(unknown)):
        z=np.asarray(o).copy();z[unknown]=assignment
        labels=[any(all(z[j:j+ell]) for j in range(s,s+k-ell+1)) for s in range(len(z)-k+1)]
        value=float(np.dot(c,labels));minimum=min(minimum,value);maximum=max(maximum,value)
    return minimum,maximum


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();cpu=time.process_time();rng=np.random.default_rng(20260916)
    count=0;max_error=0.;v3_count=0
    for t in range(1,6):
        for obs in product((-1,0,1),repeat=t):
            for k in range(1,t+1):
                for ell in range(1,k+1):
                    c=rng.uniform(-2,2,t-k+1)
                    r=weighted_bounds(obs,c,k,ell);expected=brute(obs,c,k,ell)
                    err=max(abs(r['lower_sum']-expected[0]),abs(r['upper_sum']-expected[1]))
                    if err>1e-10:raise AssertionError((obs,k,ell,r,expected))
                    max_error=max(max_error,err);count+=1
    for _ in range(250):
        t=int(rng.integers(12,50));k=int(rng.integers(1,min(t,12)+1));ell=int(rng.integers(1,k+1))
        obs=rng.choice([-1,0,1],t);x,y=rng.random((2,t-k+1))
        new=paired_bounds(obs,x,y,k,ell);old=joint_brier_bounds(obs,x,y,k,ell)
        err=max(abs(new[key]-old[key]) for key in ['lower_sum','upper_sum','independent_lower_sum','independent_upper_sum'])
        if err>1e-10:raise AssertionError('V3 differential mismatch')
        max_error=max(max_error,err);v3_count+=1
    print(json.dumps({'exhaustive_cases':count,'v3_cases':v3_count,'max_error':max_error}),flush=True)
    timings=[]
    for k in [4,8,12,16,24,72,180]:
        ell=3;obs=rng.choice(np.array([-1,0,1],np.int8),2048,p=[.3,.35,.35]);c=rng.uniform(-2,2,len(obs)-k+1)
        edges,event,states=automaton(k,ell)
        _extrema(obs,c,k,edges,event)
        times=[]
        for _ in range(3):
            t0=time.perf_counter();new=_extrema(obs,c,k,edges,event);times.append(time.perf_counter()-t0)
        row={'T':len(obs),'K':k,'L':ell,'reachable_states':len(states),'compressed_slots_upper':ell*(k-ell+2),
             'suffix_slots':2**(k-1),'compressed_median_seconds':float(np.median(times)),
             'suffix_median_seconds':'','suffix_status':'NOT_RUN_K_GT_16','bound_error':''}
        if k<=16:
            suffix_extrema(obs,c,k,ell);times=[]
            for _ in range(3):
                t0=time.perf_counter();old=suffix_extrema(obs,c,k,ell);times.append(time.perf_counter()-t0)
            row.update(suffix_median_seconds=float(np.median(times)),suffix_status='MEASURED',bound_error=max(abs(new[0]-old[0]),abs(new[1]-old[1])))
            if row['bound_error']>1e-10:raise AssertionError('Matched timing implementations disagree')
        timings.append(row)
    with (a.output/'algorithm_timings.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=timings[0]);writer.writeheader();writer.writerows(timings)
    summary={'exhaustive_ternary_sequences_max_T':5,'exhaustive_cases':count,'v3_differential_cases':v3_count,
             'max_abs_error':max_error,'timing_repeats':3,'jit_excluded_from_kernel_timings':True,
             'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'status':'PASS'}
    (a.output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
