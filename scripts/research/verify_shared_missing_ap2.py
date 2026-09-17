"""Independent MILP checks of new natural decisions and fixed negative controls."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/research'/name)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','private','results'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();pilot=load('run_shared_missing_p0.py');checker=load('check_shared_missing_decisions.py')
    c=json.loads((ROOT/'configs/research/SHARED_MISSING_EVENTS_AP2_20260917.json').read_text())
    data,_,_=pilot.read_data(a.data,c);f=pd.read_csv(a.results/'new_year_hierarchy.csv')
    natural=f[f['mask']=='natural'];selection=set(natural[natural.joint_new].index)
    for _,r in natural[natural.higher_order_new].iterrows():
        selection.update(natural[(natural.dataset==r.dataset)&(natural.series==r.series)&(natural.K==r.K)&(natural.scope==r.scope)&(natural.origin_start==r.origin_start)].index)
    negatives=natural[(natural.scope=='origin_week')&natural.independent_undirected&~natural.joint_new]
    selection.update(negatives.sort_values(['series','origin_start']).groupby(['dataset','K']).head(1).index)
    selected=f.loc[sorted(selection)];selected.to_csv(a.results/'decision_milp_plan.csv',index=False)
    rows=[];startclock=time.perf_counter()
    for _,r in selected.iterrows():
        k,ell,s,n=map(int,(r.K,r.L,r.origin_start,r.N));dc=c['datasets'][r.dataset]
        meta={key:r[key] for key in ['dataset','series','K','L','scope','origin_start','N','model_A','model_B','higher_order_new']}
        constraints=(n+k-ell)*(ell+1)+n*(k-ell+2)
        if constraints>300000:
            rows.append({**meta,'status':'DECLARED_SIZE_CUTOFF','constraint_rows':constraints});continue
        o=pilot.observed_bits(data[r.dataset,r.series][0][s:s+n+k-1],dc['threshold'])
        with np.load(a.private/f'{r.dataset}_{r.series}_K{k}_predictions.npz') as z:
            offset=s-dc['calibration_end'];pa=z[r.model_A][offset:offset+n];pb=z[r.model_B][offset:offset+n]
        solutions=checker.solve(o,2*(pb-pa),k,ell);base=float(np.sum(pa*pa-pb*pb))
        low=(base+solutions[0][0])/n;high=(base+solutions[1][0])/n
        error=max(abs(low-r.joint_lower),abs(high-r.joint_upper))
        if error>1e-8:raise AssertionError('MILP mismatch')
        rows.append({**meta,'status':'OPTIMAL_MATCH','constraint_rows':constraints,'milp_lower':low,'milp_upper':high,'max_abs_error':error,
            'lower_missing_bits':''.join(map(str,solutions[0][1][o==-1])),
            'upper_missing_bits':''.join(map(str,solutions[1][1][o==-1]))})
        pd.DataFrame(rows).to_csv(a.results/'decision_milp.csv',index=False)
    pd.DataFrame(rows).to_csv(a.results/'decision_milp.csv',index=False)
    pilot.dump(a.results/'decision_milp_summary.json',{'selection':'All new natural decisions; all pairs of higher-order-new panel; first undirected negative week per dataset/K. Saved before MILP.',
        'status_counts':pd.DataFrame(rows).status.value_counts().to_dict(),'max_abs_error':max(r.get('max_abs_error',0) for r in rows),
        'wall_seconds':time.perf_counter()-startclock})


if __name__=='__main__':main()
