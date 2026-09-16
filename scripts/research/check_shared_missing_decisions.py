"""Independently verify every natural decision change with binary linear programs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from urbanev_audit.persistent_events import event_labels


def solve(observed,coefficient,k,ell):
    t=len(observed);runs=t-ell+1;windows=t-k+1;variables=t+runs+windows
    clauses=[]
    for j in range(runs):
        for i in range(j,j+ell):clauses.append(({t+j:1,i:-1},0))
        clauses.append(({**{i:1 for i in range(j,j+ell)},t+j:-1},ell-1))
    for s in range(windows):
        event=t+runs+s
        for j in range(s,s+k-ell+1):clauses.append(({t+j:1,event:-1},0))
        clauses.append(({event:1,**{t+j:-1 for j in range(s,s+k-ell+1)}},0))
    matrix=lil_matrix((len(clauses),variables),dtype=float)
    rhs=np.empty(len(clauses))
    for row,(terms,bound) in enumerate(clauses):
        for col,value in terms.items():matrix[row,col]=value
        rhs[row]=bound
    lower=np.zeros(variables);upper=np.ones(variables)
    ix=np.flatnonzero(observed!=-1);lower[ix]=upper[ix]=observed[ix]
    objective=np.zeros(variables);objective[t+runs:]=coefficient
    answers=[]
    for sign in (1,-1):
        result=milp(sign*objective,integrality=np.ones(variables),bounds=Bounds(lower,upper),
                    constraints=LinearConstraint(matrix.tocsr(),-np.inf,rhs),
                    options={'mip_rel_gap':0.,'time_limit':60})
        if not result.success:raise RuntimeError(result.message)
        witness=np.rint(result.x[:t]).astype(np.int8)
        if not np.array_equal(witness[ix],observed[ix]):raise AssertionError('Inconsistent witness')
        labels=event_labels(witness,k,ell)
        value=float(coefficient@labels)
        if abs(value-sign*result.fun)>1e-8:raise AssertionError('MILP event linearization mismatch')
        answers.append((value,witness,float(result.mip_gap)))
    return answers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','private','results'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('pilot',ROOT/'scripts/research/run_shared_missing_p0.py')
    pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
    config=json.loads((ROOT/'configs/research/SHARED_MISSING_EVENTS_AP0_20260916.json').read_text(encoding='utf-8'))
    series,_,_=pilot.read_data(a.data,config)
    frame=pd.read_csv(a.results/'paired_bounds.csv')
    cases=frame[(frame['mask']=='natural') & frame['new_strict_decision']]
    records=[];started=time.perf_counter()
    for _,row in cases.iterrows():
        dc=config['datasets'][row['dataset']];k=int(row.K);ell=int(row.L);start=int(row.panel_start);n=int(row.windows)
        values=series[row['dataset'],row['series']][0][start:start+n+k-1]
        observed=pilot.observed_bits(values,dc['threshold'])
        with np.load(a.private/f"{row['dataset']}_{row['series']}_K{k}_predictions.npz") as pred:
            offset=start-dc['calibration_end'];pa=pred[row.model_A][offset:offset+n];pb=pred[row.model_B][offset:offset+n]
        coefficient=2*(pb-pa);constant=float(np.sum(pa*pa-pb*pb));solutions=solve(observed,coefficient,k,ell)
        lo=(solutions[0][0]+constant)/n;hi=(solutions[1][0]+constant)/n
        error=max(abs(lo-row.joint_lower),abs(hi-row.joint_upper))
        if error>1e-8:raise AssertionError('MILP and compressed DP disagree')
        record={'dataset':row['dataset'],'series':row['series'],'K':k,'panel_start':start,'model_A':row.model_A,'model_B':row.model_B,
                'missing_bits':int(np.sum(observed==-1)),'milp_lower':lo,'milp_upper':hi,'max_abs_error':error,
                'mip_gaps':[s[2] for s in solutions],'witness_sha256':[hashlib.sha256(s[1].tobytes()).hexdigest() for s in solutions]}
        records.append(record)
    report={'status':'PASS','cases':len(records),'max_abs_error':max((r['max_abs_error'] for r in records),default=0.),
            'wall_seconds':time.perf_counter()-started,'method':'Independent binary ILP with AND run variables and OR window-event variables; zero requested MIP gap',
            'records':records}
    (a.results/'independent_milp_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
