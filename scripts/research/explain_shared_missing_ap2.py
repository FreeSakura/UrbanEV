"""Explain every natural comparison, with completion and structural witnesses."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('pilot',ROOT/'scripts/research/run_shared_missing_p0.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
from urbanev_audit.event_witnesses import (completion_witness,exact_brier_difference,
    pairwise_pattern,structural_certificate,pattern_feasible)
from urbanev_audit.persistent_events import event_labels
from urbanev_audit.event_relaxations import PairwiseLP


def run(a):
    phase=a.phase
    cfg=ROOT/f'configs/research/SHARED_MISSING_EVENTS_{phase}_202609{16 if phase=="AP1" else 17}.json'
    c=json.loads(cfg.read_text());data,_,_=pilot.read_data(a.data,c)
    frame=pd.read_csv(a.results/'new_year_hierarchy.csv');frame=frame[frame['mask']=='natural']
    predictions={};rows=[];components=[];panelrows=[];endpoints=[];witness_count=0
    witnessfile=a.output/f'{phase.lower()}_completion_witnesses.jsonl'
    patternfile=a.output/f'{phase.lower()}_unreachable_patterns.jsonl'
    with witnessfile.open('w',encoding='utf-8') as witnessout,patternfile.open('w',encoding='utf-8') as patternout:
        for ident,g in frame.groupby(['dataset','series','K','L','scope','origin_start','N'],sort=True):
            dataset,key,k,ell,scope,start,n=ident;k,ell,start,n=map(int,(k,ell,start,n))
            dc=c['datasets'][dataset];o=pilot.observed_bits(data[dataset,key][0][start:start+n+k-1],dc['threshold'])
            cachekey=(dataset,key,k)
            if cachekey not in predictions:
                with np.load(a.private/f'{dataset}_{key}_K{k}_predictions.npz') as z:predictions[cachekey]={m:z[m] for m in z.files}
            offset=start-dc['calibration_end'];pred={m:p[offset:offset+n] for m,p in predictions[cachekey].items()}
            meta=dict(phase=phase,dataset=dataset,series=key,K=k,L=ell,scope=scope,origin_start=start,N=n)
            panel_id=f'{phase}_{dataset}_{key}_K{k}_{scope}_{start}'
            structure,parts=structural_certificate(o,k,ell)
            for i,part in enumerate(parts):components.append({**meta,'component':i,**part})
            lp=PairwiseLP(o,k,ell) if len(parts) else None
            counterexamples=0
            for _,row in g.iterrows():
                a_prob,b_prob=pred[row.model_A],pred[row.model_B];coef=2*(b_prob-a_prob)
                rid=panel_id+'_'+row.model_A+'_'+row.model_B
                record={**meta,'comparison_id':rid,'model_A':row.model_A,'model_B':row.model_B,
                    'independent_direction':row.independent_direction,'pairwise_direction':row.pairwise_direction,
                    'joint_direction':row.joint_direction,'ambiguous_windows':int(row.ambiguous_windows),
                    'structure_status':structure,'witness_lower':None,'witness_upper':None,'information_insufficient_proved':False,
                    'lower_tightening':max(0.,row.joint_lower-row.pairwise_lower),
                    'upper_tightening':max(0.,row.pairwise_upper-row.joint_upper),
                    'needed_lower_shift_to_zero':max(0.,-row.joint_lower),'needed_upper_shift_to_zero':max(0.,row.joint_upper)}
                if row.ambiguous_windows:
                    signs=[]
                    for side in ('lower','upper'):
                        bits=completion_witness(o,coef,k,ell,side)
                        labels=event_labels(bits,k,ell);exact=exact_brier_difference(a_prob,b_prob,labels)
                        record['witness_'+side]=exact['delta'];signs.append(exact['sign'])
                        if abs(exact['delta']-row['joint_'+side])>1e-8:raise AssertionError('Witness disagrees with recorded DP')
                        missing=np.flatnonzero(o==-1)
                        payload={'comparison_id':rid,'endpoint':side,'missing_offsets':missing.tolist(),
                            'assigned_bits':''.join(map(str,bits[missing])),**exact,
                            'observed_bits_sha256':hashlib.sha256(o.tobytes()).hexdigest(),
                            'predictions_sha256':hashlib.sha256(a_prob.tobytes()+b_prob.tobytes()).hexdigest(),
                            'labels_sha256':hashlib.sha256(labels.tobytes()).hexdigest()}
                        witnessout.write(json.dumps(payload)+'\n');witness_count+=1
                        pattern=pairwise_pattern(lp,coef,side)
                        feasible=pattern_feasible(o,pattern,k,ell)
                        value=float(np.mean(a_prob*a_prob-b_prob*b_prob+coef*pattern))
                        endpoint={**meta,'comparison_id':rid,'endpoint':side,'pairwise_pattern_feasible':feasible,
                            'pairwise_pattern_value':value,'pairwise_reported_bound':row['pairwise_'+side],
                            'witness_delta':exact['delta'],'absolute_pattern_DP_gap':abs(value-exact['delta']),
                            'equality_status':'REALIZABLE_LP_OPTIMUM_NUMERIC' if feasible and abs(value-exact['delta'])<=1e-10 else
                            'NUMERIC_ENDPOINT_MATCH_ONLY' if abs(value-exact['delta'])<=1e-10 else 'OBJECTIVE_GAP_NUMERIC',
                            'independent_conflict':';'.join(map(str,lp.preference_conflict(coef,side)))}
                        endpoints.append(endpoint)
                        if not feasible:
                            counterexamples+=1
                            patternout.write(json.dumps({'comparison_id':rid,'endpoint':side,
                                'ambiguous_origins':lp.ambiguous.tolist(),'required_bits':''.join(map(str,pattern[lp.ambiguous])),
                                'pairwise_valid':True,'trajectory_feasible':False})+'\n')
                    record['information_insufficient_proved']=signs[0]<=0 and signs[1]>=0
                else:
                    exact=exact_brier_difference(a_prob,b_prob,event_labels(o==1,k,ell))
                    record.update(witness_lower=exact['delta'],witness_upper=exact['delta'],information_insufficient_proved=exact['sign']==0)
                record['reason']=('NO_AMBIGUOUS_LABELS' if not row.ambiguous_windows else
                    'INDEPENDENT_ALREADY_DIRECTED' if row.independent_direction in ('A','B') else
                    'PAIRWISE_REQUIRED_FOR_DIRECTION' if row.pairwise_direction in ('A','B') else
                    'HIGHER_REQUIRED_FOR_DIRECTION' if row.joint_direction in ('A','B') else
                    'OBSERVATION_INFORMATION_INSUFFICIENT' if record['information_insufficient_proved'] else 'NUMERIC_BOUNDARY_UNRESOLVED')
                rows.append(record)
            if counterexamples:structure='HIGHER_STRUCTURE'
            for record in rows[-len(g):]:record['structure_status']=structure
            panelrows.append({**meta,'structure_status':structure,'components':len(parts),'unreachable_LP_patterns':counterexamples})
            if len(panelrows)%100==0:print(json.dumps({'phase':phase,'panels':len(panelrows),'witnesses':witness_count}),flush=True)
    result=pd.DataFrame(rows);result.to_csv(a.output/f'{phase.lower()}_explanations.csv',index=False)
    pd.DataFrame(components).to_csv(a.output/f'{phase.lower()}_components.csv',index=False)
    pd.DataFrame(panelrows).to_csv(a.output/f'{phase.lower()}_panel_structure.csv',index=False)
    pd.DataFrame(endpoints).to_csv(a.output/f'{phase.lower()}_endpoint_witnesses.csv',index=False)
    # Preserve the AP1 published width-gap definition (including both ends).
    tightened=result[(result.lower_tightening+result.upper_tightening)>1e-10]
    tightened.to_csv(a.output/f'{phase.lower()}_higher_order_gaps.csv',index=False)
    summary={'phase':phase,'comparisons':len(rows),'witness_endpoints':witness_count,
        'reasons':result.reason.value_counts().to_dict(),'tightened_comparisons':len(tightened),
        'panel_structure':pd.DataFrame(panelrows).structure_status.value_counts().to_dict(),
        'endpoint_status':pd.DataFrame(endpoints).equality_status.value_counts().to_dict()}
    pilot.dump(a.output/f'{phase.lower()}_explanation_summary.json',summary);print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','private','results','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['AP1','AP2'],required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();cpu=time.process_time()
    with threadpool_limits(limits=2):run(a)
    pilot.dump(a.output/f'{a.phase.lower()}_explanation_execution.json',{'status':'COMPLETE',
        'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu})


if __name__=='__main__':main()
