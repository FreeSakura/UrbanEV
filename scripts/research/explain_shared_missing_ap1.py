"""Classify endpoint conflict sizes without fitting or selecting any predictor."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_audit.event_relaxations import PairwiseLP
from urbanev_audit.persistent_events import endpoint_certificate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','ap0','private','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('ap0',ROOT/'scripts/research/run_shared_missing_p0.py');pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
    frame=pd.concat([pd.read_csv(a.output/'ap0_hierarchy.csv'),pd.read_csv(a.output/'new_year_hierarchy.csv')],ignore_index=True)
    data={};cache={};records=[]
    for ident,group in frame.groupby(['phase','dataset','series','K','L','scope','mask','origin_start','N'],sort=True):
        phase,dataset,key,k,ell,scope,mask,s,n=ident;k,ell,s,n=map(int,(k,ell,s,n))
        if phase not in data:
            file={'AP0':'SHARED_MISSING_EVENTS_AP0_20260916.json','AP0_EXTENSION':'SHARED_MISSING_EVENTS_AP0_POWER_EXTENSION_20260916.json','AP1':'SHARED_MISSING_EVENTS_AP1_20260916.json'}[phase]
            config=json.loads((ROOT/'configs/research'/file).read_text());series,_,_=pilot.read_data(a.data,config);data[phase]=(config,series)
        config,series=data[phase];dc=config['datasets'][dataset]
        private=a.private if phase=='AP1' else a.ap0/('run' if phase=='AP0' else 'run_power_extension')
        if mask=='natural':
            observed=pilot.observed_bits(series[dataset,key][0][s:s+n+k-1],dc['threshold']);id=(phase,dataset,key,k)
            if id not in cache:
                with np.load(private/f'{dataset}_{key}_K{k}_predictions.npz') as z:cache[id]={m:z[m] for m in z.files}
            offset=s-dc['calibration_end'];pred={m:v[offset:offset+n] for m,v in cache[id].items()}
        else:
            with np.load(private/f'artificial_{dataset}_{key}_K{k}_{mask}.npz') as z:
                observed=z['observed'];pred={m:z[m] for m in ['constant','logistic','hist_gradient_boosting']}
        lp=None
        for _,row in group.iterrows():
            coef=2*(pred[row.model_B]-pred[row.model_A])
            for side,gamma in [('lower',row.Gamma_minus),('upper',row.Gamma_plus)]:
                result={'phase':phase,'dataset':dataset,'series':key,'K':k,'scope':scope,'mask':mask,'origin_start':s,
                        'model_A':row.model_A,'model_B':row.model_B,'endpoint':side,'Gamma':gamma,'core_size':0,'size_lower_bound':0,
                        'kind':'NUMERIC_ZERO','window_indices':''}
                if gamma/n>1e-10:
                    if lp is None:lp=PairwiseLP(observed,k,ell)
                    pair=lp.preference_conflict(coef,side)
                    if pair:result.update(core_size=2,size_lower_bound=2,kind='BINARY_IMPLICATION',window_indices=';'.join(map(str,pair)))
                    else:
                        explain=len(lp.ambiguous)<=32 and len(observed)<=512
                        cert=endpoint_certificate(observed,coef,k,ell,side,explain=explain)
                        if cert['independent_endpoint_attainable']:result.update(kind='NUMERIC_GAP_ONLY')
                        else:
                            core=cert['conflict_core'];result.update(core_size=len(core) if explain else '',size_lower_bound=3,
                              kind='HIGHER_DELETION_MINIMAL' if explain else 'HIGHER_SIZE_NOT_EXTRACTED',
                              window_indices=';'.join(str(x['window_start']) for x in core))
                records.append(result)
    pd.DataFrame(records).to_csv(a.output/'endpoint_conflict_classification.csv',index=False)
    summary=pd.DataFrame(records).groupby(['phase','kind']).size().reset_index(name='endpoints')
    summary.to_csv(a.output/'endpoint_conflict_summary.csv',index=False);print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
