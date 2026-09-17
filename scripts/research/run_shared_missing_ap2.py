"""One fixed AP2 replication, loading AP1 selected models without fitting."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
SPEC=importlib.util.spec_from_file_location('ap1',ROOT/'scripts/research/run_shared_missing_ap1.py')
ap1=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(ap1)
from urbanev_audit.event_comparison_ap1 import score_panel,origin_blocks


def run(a,c):
    hashes=[];models={}
    for dataset,dc in c['datasets'].items():
        for k in dc['windows']:
            path=a.ap1/f'{dataset}_K{k}_models.joblib'
            models[dataset,k]=joblib.load(path)
            hashes.append({'dataset':dataset,'K':k,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    ap1.ap0.dump(a.output/'frozen_models.json',{'new_fits':0,'model_files':hashes})
    series,qualification,sources=ap1.ap0.read_data(a.data,c)
    ap1.save_csv(a.output/'data_qualification.csv',qualification);ap1.ap0.dump(a.output/'data_sources.json',sources)
    rows=[];sets=[];skipped=[]
    for dataset,dc in c['datasets'].items():
        keys=sorted(key for d,key in series if d==dataset)
        for station_id,key in enumerate(keys):
            values,dates=series[dataset,key];start=dc['calibration_end'];stop=len(values)
            assert dates[start]==pd.Timestamp(dc['evaluation_start'])
            assert dates[-1]+(dates[1]-dates[0])==pd.Timestamp(dc['evaluation_end'])
            median=float(np.nanmedian(values[:dc['train_end']]))
            x=ap1.ap0.features(values,dates,dc,median,station_id,len(keys))
            for k in dc['windows']:
                ell=dc['run_length'];bundle=models[dataset,k]
                def predict(x):
                    return {'constant':np.full(len(x),bundle['constant']),
                            **{name:model.predict_proba(x)[:,1] for name,model in bundle['models'].items()}}
                pred=predict(x[start:stop-k+1])
                np.savez_compressed(a.private/f'{dataset}_{key}_K{k}_predictions.npz',**pred)
                obs=ap1.ap0.observed_bits(values,dc['threshold'])
                panels=[('annual',start,stop-k+1,stop)]+[('origin_week',s,b,end) for s,b,end in origin_blocks(start,stop,k,dc['panel_length'])]
                for scope,s,b,end in panels:
                    probs={m:p[s-start:b-start] for m,p in pred.items()}
                    scored,candidates=score_panel(obs[s:end],probs,k,ell,ap1.metadata('AP2',dataset,key,scope,'natural',s,dates))
                    rows.extend(scored);sets.append(candidates)
                clean_start,clean_end=ap1.ap0.longest_complete(values,start,dc['artificial_max_length'])
                if clean_end-clean_start>=k:
                    truth=(values[clean_start:clean_end]>=dc['threshold']).astype(np.int8)
                    for mask in c['artificial']['masks']:
                        rng=np.random.default_rng(c['artificial']['seed']+station_id+(0 if dataset=='beijing' else 1000))
                        hidden=np.zeros(clean_end-clean_start,bool)
                        if mask.startswith('iid'):hidden=rng.random(len(hidden))<int(mask.split('_')[1])/100
                        else:
                            length=max(ell,k//4)
                            for anchor in rng.choice(len(hidden),max(1,int(.1*len(hidden)/length)),replace=False):hidden[anchor:anchor+length]=True
                        masked=values.copy();masked[np.flatnonzero(hidden)+clean_start]=np.nan
                        features=ap1.ap0.features(masked,dates,dc,median,station_id,len(keys))
                        probs=predict(features[clean_start:clean_end-k+1])
                        observed=ap1.ap0.observed_bits(masked[clean_start:clean_end],dc['threshold'])
                        imputed=(pd.Series(masked).ffill().fillna(median).to_numpy()[clean_start:clean_end]>=dc['threshold']).astype(np.int8)
                        scored,candidates=score_panel(observed,probs,k,ell,ap1.metadata('AP2',dataset,key,'artificial',mask,clean_start,dates),truth,imputed)
                        rows.extend(scored);sets.append(candidates)
                        np.savez_compressed(a.private/f'artificial_{dataset}_{key}_K{k}_{mask}.npz',observed=observed,truth=truth,**probs)
                else:skipped.append({'dataset':dataset,'series':key,'K':k,'reason':'No eligible complete run'})
                ap1.save_csv(a.output/'new_year_hierarchy.csv',rows);ap1.save_csv(a.output/'new_year_candidate_sets.csv',sets)
                print(json.dumps({'event':'scored','dataset':dataset,'series':key,'K':k,'comparisons':len(rows)}),flush=True)
    ap1.ap0.dump(a.output/'skipped_artificial.json',skipped)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','ap1','private','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.private.mkdir(parents=True,exist_ok=True)
    path=ROOT/'configs/research/SHARED_MISSING_EVENTS_AP2_20260917.json';c=json.loads(path.read_text())
    started=time.perf_counter();cpu=time.process_time()
    with threadpool_limits(limits=c['resources']['threads']):run(a,c)
    ap1.ap0.dump(a.output/'execution.json',{'status':'COMPLETE','new_fits':0,
        'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-started,
        'cpu_seconds':time.process_time()-cpu,'future_beyond_ap2_used':False})


if __name__=='__main__':main()
