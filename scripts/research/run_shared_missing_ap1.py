"""Execute AP1: AP0 hierarchy replay, fixed old-data selection and new-year tests."""
import argparse
from itertools import combinations
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_audit.persistent_events import label_bounds,event_labels
from urbanev_audit.event_comparison_ap1 import score_panel,origin_blocks
SPEC=importlib.util.spec_from_file_location('ap0',ROOT/'scripts/research/run_shared_missing_p0.py')
ap0=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(ap0)


def save_csv(path,rows):
    if rows:pd.DataFrame(rows).to_csv(path,index=False)


def metadata(phase,dataset,station,scope,mask,start,dates):
    return {'phase':phase,'dataset':dataset,'series':station,'scope':scope,'mask':mask,'origin_start':int(start),
            'time_start':str(dates[start]),'quarter':str(dates[start].to_period('Q')),
            'cluster':f'{dataset}_{dates[start].isoformat()}'}


def replay_ap0(args):
    all_rows=[];candidates=[];start=time.perf_counter()
    for ext in (False,True):
        cfg='SHARED_MISSING_EVENTS_AP0_POWER_EXTENSION_20260916.json' if ext else 'SHARED_MISSING_EVENTS_AP0_20260916.json'
        config=json.loads((ROOT/'configs/research'/cfg).read_text(encoding='utf-8'))
        series,_,_=ap0.read_data(args.data,config)
        public=ROOT/'artifacts/summaries/shared_missing_events_ap0'
        if ext:public=public/'power_extension'
        private=args.ap0/('run_power_extension' if ext else 'run')
        frame=pd.read_csv(public/'paired_bounds.csv');cache={}
        group_columns=['dataset','series','scope','mask','K','L','panel_start','windows']
        groups=frame[frame['mask']=='natural'].groupby(group_columns,sort=True)
        for gi,(ident,group) in enumerate(groups):
            dataset,station,scope,mask,k,ell,s,n=ident;k,ell,s,n=map(int,(k,ell,s,n))
            key=(dataset,station,k);dc=config['datasets'][dataset]
            if key not in cache:
                with np.load(private/f'{dataset}_{station}_K{k}_predictions.npz') as package:cache[key]={m:package[m] for m in package.files}
            values,dates=series[dataset,station];offset=s-dc['calibration_end']
            pred={m:p[offset:offset+n] for m,p in cache[key].items()}
            observed=ap0.observed_bits(values[s:s+n+k-1],dc['threshold'])
            meta=metadata('AP0_EXTENSION' if ext else 'AP0',dataset,station,scope,mask,s,dates)
            rows,sets=score_panel(observed,pred,k,ell,meta)
            # Preserve exactly the old scored origins, not a retroactive AP1 block.
            for row in rows:
                old=group[(group.model_A==row['model_A'])&(group.model_B==row['model_B'])].iloc[0]
                if max(abs(row['joint_lower']-old.joint_lower),abs(row['joint_upper']-old.joint_upper))>1e-9:
                    raise AssertionError('AP0 replay changed original scores')
            all_rows.extend(rows);candidates.append(sets)
        save_csv(args.output/'ap0_hierarchy.csv',all_rows);save_csv(args.output/'ap0_candidate_sets.csv',candidates)
        print(json.dumps({'event':'ap0_replayed','extension':ext,'comparisons':len(all_rows),'seconds':time.perf_counter()-start}),flush=True)


def selection_rates(dataset,key,values,dates,lo,hi,dc,k):
    records=[]
    for role,start,stop in [('fit',dc['history'],dc['train_end']-k+1),('select',dc['train_end'],dc['selection_end']-k+1)]:
        origins=np.arange(start,stop);known=lo[origins]==hi[origins]
        state=np.where(np.isnan(values[origins-1]),'missing',np.where(values[origins-1]>=dc['threshold'],'high','low'))
        quarter=np.asarray(dates[origins].to_period('Q').astype(str))
        for q in sorted(set(quarter)):
            for st in ('missing','low','high'):
                ix=(quarter==q)&(state==st)
                if not ix.any():continue
                records.append({'dataset':dataset,'series':key,'K':k,'role':role,'quarter':q,'origin_state':st,
                                'windows':int(ix.sum()),'identified':int(known[ix].sum()),
                                'identified_positive':int(np.sum(lo[origins][ix & known])),
                                'identified_negative':int(np.sum(1-lo[origins][ix & known]))})
    return records


def new_year(args,config):
    series,qualification,downloads=ap0.read_data(args.data,config)
    save_csv(args.output/'data_qualification.csv',qualification);ap0.dump(args.output/'data_sources.json',downloads)
    rows=[];candidate_rows=[];selection=[];rates=[];frozen=[];skipped=[]
    for dataset,dc in config['datasets'].items():
        keys=sorted(key for d,key in series if d==dataset)
        for key in keys:
            _,dates=series[dataset,key]
            if dates[dc['calibration_end']]!=pd.Timestamp(dc['evaluation_start']) or dates[-1]+(dates[1]-dates[0])!=pd.Timestamp(dc['evaluation_end']):
                raise ValueError('Calendar scope differs from the written AP1 plan')
        medians={key:float(np.nanmedian(series[dataset,key][0][:dc['train_end']])) for key in keys}
        xs={key:ap0.features(*series[dataset,key],dc,medians[key],i,len(keys)) for i,key in enumerate(keys)}
        for k in dc['windows']:
            ell=dc['run_length'];X=[];Y=[];V=[];W=[]
            for key in keys:
                values,dates=series[dataset,key];lo,hi=label_bounds(ap0.observed_bits(values,dc['threshold']),k,ell)
                train=np.arange(dc['history'],dc['train_end']-k+1);valid=np.arange(dc['train_end'],dc['selection_end']-k+1)
                train=train[lo[train]==hi[train]];valid=valid[lo[valid]==hi[valid]]
                X.append(xs[key][train]);Y.append(lo[train]);V.append(xs[key][valid]);W.append(lo[valid])
                rates.extend(selection_rates(dataset,key,values,dates,lo,hi,dc,k))
            X,Y,V,W=map(np.concatenate,(X,Y,V,W));constant=float((Y.sum()+1)/(len(Y)+2))
            selected={};params=config['predictors'];bundle=args.private/f'{dataset}_K{k}_models.joblib'
            if bundle.exists():
                saved=joblib.load(bundle);selected=saved['models'];selection.extend(saved['selection']);constant=saved['constant']
            else:
                local=[]
                for family,grid in [('logistic',params['logistic_C']),('hist_gradient_boosting',params['hgb_max_leaf_nodes'])]:
                    trials=[]
                    for value in grid:
                        model=(make_pipeline(StandardScaler(),LogisticRegression(C=value,**params['logistic_fixed'])) if family=='logistic'
                               else HistGradientBoostingClassifier(max_leaf_nodes=value,**params['hgb_fixed']))
                        started=time.perf_counter()
                        with warnings.catch_warnings(record=True) as messages:
                            warnings.simplefilter('always');model.fit(X,Y)
                        brier=float(np.mean((model.predict_proba(V)[:,1]-W)**2))
                        trials.append((brier,value,model))
                        local.append({'dataset':dataset,'K':k,'model':family,'value':value,'validation_brier':brier,
                                      'fit_windows':len(Y),'validation_windows':len(W),'seconds':time.perf_counter()-started,
                                      'warnings':'; '.join(str(m.message) for m in messages),'selected':False})
                    chosen=min(trials,key=lambda t:(t[0],grid.index(t[1])));selected[family]=chosen[2]
                    for record in local:
                        if record['model']==family:record['selected']=record['value']==chosen[1]
                local.append({'dataset':dataset,'K':k,'model':'constant','value':constant,'validation_brier':float(np.mean((constant-W)**2)),
                              'fit_windows':len(Y),'validation_windows':len(W),'seconds':0.,'warnings':'','selected':True})
                selection.extend(local);joblib.dump({'models':selected,'selection':local,'constant':constant},bundle)
            frozen.append({'dataset':dataset,'K':k,'constant':constant,'validation_only_selection':True,'new_year_scores_seen_before_selection':False})
            save_csv(args.output/'selection.csv',selection);save_csv(args.output/'label_identification_rates.csv',rates)
            print(json.dumps({'event':'selected_models','dataset':dataset,'K':k}),flush=True)
            def predict(features):return {'constant':np.full(len(features),constant),**{name:model.predict_proba(features)[:,1] for name,model in selected.items()}}
            for station_id,key in enumerate(keys):
                values,dates=series[dataset,key];start=dc['calibration_end'];stop=len(values)
                predfile=args.private/f'{dataset}_{key}_K{k}_predictions.npz'
                if predfile.exists():
                    with np.load(predfile) as package:pred={m:package[m] for m in package.files}
                else:
                    pred=predict(xs[key][start:stop-k+1]);np.savez_compressed(predfile,**pred)
                obs=ap0.observed_bits(values,dc['threshold'])
                for scope,s,b,end in [('annual',start,stop-k+1,stop)]+[('origin_week',s,b,end) for s,b,end in origin_blocks(start,stop,k,dc['panel_length'])]:
                    probs={name:p[s-start:b-start] for name,p in pred.items()}
                    scored,sets=score_panel(obs[s:end],probs,k,ell,metadata('AP1',dataset,key,scope,'natural',s,dates))
                    rows.extend(scored);candidate_rows.append(sets)
                clean_start,clean_end=ap0.longest_complete(values,start,dc['artificial_max_length'])
                if clean_end-clean_start>=k:
                    truth=(values[clean_start:clean_end]>=dc['threshold']).astype(np.int8)
                    for mask in config['artificial']['masks']:
                        rng=np.random.default_rng(20260916+station_id+(0 if dataset=='beijing' else 1000));hidden=np.zeros(clean_end-clean_start,bool)
                        if mask.startswith('iid'):hidden=rng.random(len(hidden))<int(mask.split('_')[1])/100
                        else:
                            length=max(ell,k//4)
                            for anchor in rng.choice(len(hidden),max(1,int(.1*len(hidden)/length)),replace=False):hidden[anchor:anchor+length]=True
                        masked=values.copy();masked[np.flatnonzero(hidden)+clean_start]=np.nan
                        features=ap0.features(masked,dates,dc,medians[key],station_id,len(keys))
                        probs=predict(features[clean_start:clean_end-k+1]);observed=ap0.observed_bits(masked[clean_start:clean_end],dc['threshold'])
                        # LOCF label point estimate uses only the masked measurement stream.
                        imputed=(pd.Series(masked).ffill().fillna(medians[key]).to_numpy()[clean_start:clean_end]>=dc['threshold']).astype(np.int8)
                        scored,sets=score_panel(observed,probs,k,ell,metadata('AP1',dataset,key,'artificial',mask,clean_start,dates),truth,imputed)
                        rows.extend(scored);candidate_rows.append(sets)
                        np.savez_compressed(args.private/f'artificial_{dataset}_{key}_K{k}_{mask}.npz',observed=observed,truth=truth,**probs)
                else:skipped.append({'dataset':dataset,'series':key,'K':k,'reason':'No eligible naturally complete run'})
                save_csv(args.output/'new_year_hierarchy.csv',rows);save_csv(args.output/'new_year_candidate_sets.csv',candidate_rows)
                print(json.dumps({'event':'series_scored','dataset':dataset,'K':k,'series':key,'comparisons':len(rows)}),flush=True)
    ap0.dump(args.output/'frozen_selection.json',frozen);ap0.dump(args.output/'skipped_artificial.json',skipped)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','ap0','private','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['ap0','new'],required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);a.private.mkdir(parents=True,exist_ok=True)
    c=json.loads((ROOT/'configs/research/SHARED_MISSING_EVENTS_AP1_20260916.json').read_text(encoding='utf-8'))
    start=time.perf_counter();cpu=time.process_time()
    with threadpool_limits(limits=c['resources']['threads']):
        if a.phase=='ap0':replay_ap0(a)
        else:new_year(a,c)
    ap0.dump(a.output/f'{a.phase}_execution.json',{'status':'COMPLETE','phase':a.phase,'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
                                                'old_evidence_overwritten':False,'ap2_reserved_data_used':False})
    print('COMPLETE '+a.phase,flush=True)


if __name__=='__main__':main()
