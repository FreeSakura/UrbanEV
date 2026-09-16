"""Deterministic, outcome-stratified MILP checks, including negatives and cutoffs."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_audit.persistent_events import paired_bounds


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/research'/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','ap0','private','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();pilot=load_script('run_shared_missing_p0.py');checker=load_script('check_shared_missing_decisions.py')
    frames=[pd.read_csv(a.output/'ap0_hierarchy.csv'),pd.read_csv(a.output/'new_year_hierarchy.csv')]
    frame=pd.concat(frames,ignore_index=True)
    frame['has_ambiguity']=frame.ambiguous_windows>0
    frame['has_tightening']=(frame.Gamma_minus+frame.Gamma_plus)/frame.N>1e-10
    strata=['phase','dataset','K','scope','mask','has_ambiguity','has_tightening','independent_undirected']
    pool=frame.sort_values(['series','origin_start','model_A','model_B']).groupby(strata,sort=True).first().reset_index()
    buckets=defaultdict(list)
    for record in pool.to_dict('records'):buckets[record['phase'],record['dataset'],record['K']].append(record)
    chosen=[];seen=set()
    while len(chosen)<24 and any(buckets.values()):
        for key in sorted(buckets):
            if not buckets[key]:continue
            row=buckets[key].pop(0);ident=tuple(row[k] for k in ['phase','dataset','K','scope','mask','series','origin_start'])
            if ident not in seen:chosen.append(row);seen.add(ident)
            if len(chosen)==24:break
    pd.DataFrame(chosen).to_csv(a.output/'milp_sample_plan.csv',index=False)
    limits={'max_full_constraint_rows':300000,'max_full_variables':50000,'optimization_seconds':60,
            'large_panel_diagnostic_prefix_origins':128,'sample_panels':len(chosen),
            'selection':'First panel per predeclared result/missingness stratum, round-robin phase/dataset/K; all model pairs. Plan saved before any MILP solve.'}
    (a.output/'milp_resource_policy.json').write_text(json.dumps(limits,indent=2)+'\n',encoding='utf-8')
    data_cache={};pred_cache={};records=[];started=time.perf_counter()
    for panel in chosen:
        phase,dataset,key,k=panel['phase'],panel['dataset'],panel['series'],int(panel['K']);ell=int(panel['L']);s=int(panel['origin_start']);n=int(panel['N'])
        if phase not in data_cache:
            config_name={'AP0':'SHARED_MISSING_EVENTS_AP0_20260916.json','AP0_EXTENSION':'SHARED_MISSING_EVENTS_AP0_POWER_EXTENSION_20260916.json','AP1':'SHARED_MISSING_EVENTS_AP1_20260916.json'}[phase]
            cfg=json.loads((ROOT/'configs/research'/config_name).read_text());series,_,_=pilot.read_data(a.data,cfg);data_cache[phase]=(cfg,series)
        cfg,series=data_cache[phase];dc=cfg['datasets'][dataset]
        private=a.private if phase=='AP1' else a.ap0/('run' if phase=='AP0' else 'run_power_extension')
        if panel['mask']!='natural':
            with np.load(private/f"artificial_{dataset}_{key}_K{k}_{panel['mask']}.npz") as pack:
                observed=pack['observed'];pred={m:pack[m] for m in ['constant','logistic','hist_gradient_boosting']}
        else:
            observed=pilot.observed_bits(series[dataset,key][0][s:s+n+k-1],dc['threshold'])
            ident=(phase,dataset,key,k)
            if ident not in pred_cache:
                with np.load(private/f'{dataset}_{key}_K{k}_predictions.npz') as pack:pred_cache[ident]={m:pack[m] for m in pack.files}
            offset=s-dc['calibration_end'];pred={m:arr[offset:offset+n] for m,arr in pred_cache[ident].items()}
        t=len(observed);runs=t-ell+1;estimated_rows=runs*(ell+1)+n*(k-ell+2);variables=t+runs+n
        extent='FULL_PANEL'
        if estimated_rows>limits['max_full_constraint_rows'] or variables>limits['max_full_variables']:
            records.append({**{k:panel[k] for k in ['phase','dataset','series','K','scope','mask','origin_start']},'extent':'FULL_PANEL','status':'DECLARED_SIZE_BUDGET_CUTOFF','N':n,
                            'constraint_rows':estimated_rows,'variables':variables,'model_A':'','model_B':'','seconds':0.,'max_abs_error':'','message':'Not attempted; no OOM or runtime inferred'})
            n=min(n,128);observed=observed[:n+k-1];pred={m:p[:n] for m,p in pred.items()};extent='FIRST_128_ORIGIN_DIAGNOSTIC'
        for ma,mb in [('constant','logistic'),('constant','hist_gradient_boosting'),('logistic','hist_gradient_boosting')]:
            pa,pb=pred[ma],pred[mb];coefficient=2*(pb-pa);constant=float(np.sum(pa*pa-pb*pb));clock=time.perf_counter()
            row={**{k:panel[k] for k in ['phase','dataset','series','K','scope','mask','origin_start']},'extent':extent,'status':'','N':n,
                 'constraint_rows':estimated_rows,'variables':variables,'model_A':ma,'model_B':mb,'seconds':0.,'max_abs_error':'','message':''}
            row['planned_full_constraint_rows']=estimated_rows;row['planned_full_variables']=variables
            actual_t=len(observed);actual_runs=actual_t-ell+1
            row['constraint_rows']=actual_runs*(ell+1)+n*(k-ell+2);row['variables']=actual_t+actual_runs+n
            try:
                solution=checker.solve(observed,coefficient,k,ell);expected=paired_bounds(observed,pa,pb,k,ell)
                error=max(abs((solution[0][0]+constant-expected['lower_sum'])/n),abs((solution[1][0]+constant-expected['upper_sum'])/n))
                if error>1e-8:raise AssertionError(f'MILP mismatch {error}')
                row.update(status='OPTIMAL_MATCH',max_abs_error=error)
            except RuntimeError as exc:row.update(status='SOLVER_CUTOFF_OR_FAILURE',message=str(exc))
            row['seconds']=time.perf_counter()-clock;records.append(row)
        pd.DataFrame(records).to_csv(a.output/'stratified_milp.csv',index=False)
        print(json.dumps({'event':'milp_panel','phase':phase,'dataset':dataset,'K':k,'extent':extent,'panels_done':len(records)}),flush=True)
    summary={'status':'COMPLETE','panels':len(chosen),'optimizations_pairs_attempted':sum(r['status']!='DECLARED_SIZE_BUDGET_CUTOFF' for r in records),
             'matched_pairs':sum(r['status']=='OPTIMAL_MATCH' for r in records),'size_cutoffs':sum(r['status']=='DECLARED_SIZE_BUDGET_CUTOFF' for r in records),
             'solver_cutoffs_or_failures':sum(r['status']=='SOLVER_CUTOFF_OR_FAILURE' for r in records),'wall_seconds':time.perf_counter()-started,
             'max_abs_error':max((r['max_abs_error'] for r in records if r['status']=='OPTIMAL_MATCH'),default=0.)}
    (a.output/'milp_execution.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
