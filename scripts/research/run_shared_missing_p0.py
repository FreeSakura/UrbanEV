"""A-P0: causal light predictors and exact comparisons on two public prefixes."""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import combinations
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from urbanev_audit.persistent_events import paired_bounds, label_bounds, event_labels


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csvout(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def read_data(data_root,config):
    series={};qualification=[];downloads=[]
    for dataset,c in config['datasets'].items():
        prefix=data_root/'household_development_prefix.txt'
        original=data_root/('beijing_original.zip' if dataset=='beijing' else 'household_original.zip')
        if dataset=='household' and prefix.is_file() and not (original.exists() and zipfile.is_zipfile(original)):
            raw=prefix.read_bytes()
            downloads.append({'dataset':dataset,'uci_id':235,'prefix_sha256':hashlib.sha256(raw).hexdigest(),
                              'prefix_bytes':len(raw),'prefix_records':c['prefix_rows_per_series'],
                              'source_type':'Exact first records decoded from the official ZIP stream; not an imputed or generated dataset',
                              'download':'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip',
                              'official_page':'https://archive.ics.uci.edu/dataset/235','license':'CC-BY-4.0'})
            frame=pd.read_csv(io.BytesIO(raw),sep=';',nrows=c['prefix_rows_per_series'],na_values=['?',''],usecols=['Date','Time',c['variable']])
            dates=pd.DatetimeIndex(pd.to_datetime(frame['Date']+' '+frame['Time'],format='%d/%m/%Y %H:%M:%S'))
            series[dataset,'household']=(frame[c['variable']].to_numpy(float),dates)
            continue
        path=original if original.exists() and zipfile.is_zipfile(original) else data_root/f"uci_{c['uci_id']}.zip"
        raw=path.read_bytes()
        downloads.append({'dataset':dataset,'uci_id':c['uci_id'],'archive_sha256':hashlib.sha256(raw).hexdigest(),'archive_bytes':len(raw),
                          'official_page':f"https://archive.ics.uci.edu/dataset/{c['uci_id']}",'license':'CC-BY-4.0'})
        with zipfile.ZipFile(io.BytesIO(raw)) as outer:
            if dataset=='beijing':
                nested=[name for name in outer.namelist() if name.lower().endswith('.zip')]
                archive=zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) if nested else outer
                names=sorted(name for name in archive.namelist() if Path(name).name.startswith('PRSA_Data_') and name.endswith('.csv'))
                if len(names)!=12:raise ValueError('Expected the original 12 Beijing station files')
                for name in names:
                    with archive.open(name) as f:
                        frame=pd.read_csv(f,nrows=c['prefix_rows_per_series'],na_values=['NA'])
                    dates=pd.DatetimeIndex(pd.to_datetime(frame[['year','month','day','hour']]))
                    key=str(frame['station'].iloc[0])
                    series[dataset,key]=(frame[c['variable']].to_numpy(float),dates)
            else:
                name=next(n for n in outer.namelist() if n.endswith('household_power_consumption.txt'))
                with outer.open(name) as f:
                    frame=pd.read_csv(f,sep=';',nrows=c['prefix_rows_per_series'],na_values=['?',''],usecols=['Date','Time',c['variable']])
                dates=pd.DatetimeIndex(pd.to_datetime(frame['Date']+' '+frame['Time'],format='%d/%m/%Y %H:%M:%S'))
                series[dataset,'household']=(frame[c['variable']].to_numpy(float),dates)
    for (dataset,key),(values,dates) in series.items():
        c=config['datasets'][dataset];freq='h' if dataset=='beijing' else 'min'
        expected=pd.date_range(dates[0],periods=c['prefix_rows_per_series'],freq=freq)
        if not dates.equals(expected):raise ValueError(f'Non-regular or incomplete time grid: {dataset}/{key}')
        if np.isinf(values).any():raise ValueError('Infinite measurement values')
        missing=np.isnan(values)
        qualification.append({'dataset':dataset,'series':key,'rows':len(values),'start':str(dates[0]),'end':str(dates[-1]),
                              'missing':int(missing.sum()),'missing_rate':float(missing.mean()),
                              'train_missing':int(missing[:c['train_end']].sum()),
                              'evaluation_missing':int(missing[c['calibration_end']:].sum()),
                              'negative_measurements':int(np.sum(values<0)),'regular_grid':True})
    return series,qualification,downloads


def features(values,dates,c,median,station_id,station_count):
    """X[o] uses values at indices <o only; missing-value handling is causal."""
    series=pd.Series(values);filled=series.ffill().fillna(median)/c['threshold']
    lags=[1,2,3,6,12,24,48,168] if c['history']==168 else [1,2,5,15,30,60,180,1440]
    spans=[6,24,168] if c['history']==168 else [15,60,1440]
    columns=[filled.shift(lag).fillna(median/c['threshold']).to_numpy() for lag in lags]
    for span in spans:
        roll=filled.shift(1).rolling(span,min_periods=1)
        columns.extend([roll.mean().fillna(0).to_numpy(),roll.std(ddof=0).fillna(0).to_numpy(),
                        roll.min().fillna(0).to_numpy(),roll.max().fillna(0).to_numpy(),
                        series.isna().astype(float).shift(1).rolling(span,min_periods=1).mean().fillna(1).to_numpy()])
    valid=~np.isnan(values)
    last=np.maximum.accumulate(np.where(valid,np.arange(len(values)),-1))
    ages=np.arange(len(values))-np.r_[-1,last[:-1]]
    columns.extend([np.minimum(ages,c['history'])/c['history'],series.isna().shift(1,fill_value=True).to_numpy(float)])
    hour=np.asarray(dates.hour+dates.minute/60)
    day=np.asarray(dates.dayofweek)
    columns.extend([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*day/7),np.cos(2*np.pi*day/7)])
    columns.extend([np.full(len(values),float(station_id==i)) for i in range(station_count)])
    return np.column_stack(columns).astype(np.float32)


def observed_bits(values,threshold):
    return np.where(np.isnan(values),-1,(values>=threshold).astype(int)).astype(np.int8)


def label_winner(lower,upper):
    if lower>1e-12:return 'B'
    if upper< -1e-12:return 'A'
    if abs(lower)<=1e-12 and abs(upper)<=1e-12:return 'TIE'
    return 'UNRESOLVED'


def compare_panel(dataset,station,scope,mask,k,ell,observed,predictions,panel_start,truth=None):
    rows=[]
    for first,second in combinations(predictions,2):
        begin=time.perf_counter()
        result=paired_bounds(observed,predictions[first],predictions[second],k,ell)
        n=result['window_count']
        row={'dataset':dataset,'series':station,'scope':scope,'mask':mask,'K':k,'L':ell,'panel_start':panel_start,
             'model_A':first,'model_B':second,'windows':n,'missing_snapshots':result['missing_snapshots'],
             'ambiguous_windows':result['ambiguous_windows'],
             'independent_lower':result['independent_lower_sum']/n,'independent_upper':result['independent_upper_sum']/n,
             'joint_lower':result['lower_sum']/n,'joint_upper':result['upper_sum']/n,'reachable_states':result['reachable_states'],
             'seconds':time.perf_counter()-begin,'truth_delta':'','truth_covered':''}
        row['independent_width']=row['independent_upper']-row['independent_lower']
        row['joint_width']=row['joint_upper']-row['joint_lower']
        row['width_reduction']=row['independent_width']-row['joint_width']
        row['relative_width_reduction']=row['width_reduction']/row['independent_width'] if row['independent_width']>1e-12 else 0.
        row['independent_winner']=label_winner(row['independent_lower'],row['independent_upper'])
        row['joint_winner']=label_winner(row['joint_lower'],row['joint_upper'])
        row['new_strict_decision']=row['independent_winner']=='UNRESOLVED' and row['joint_winner'] in ['A','B']
        if truth is not None:
            labels=event_labels(truth,k,ell)
            delta=float(np.mean((predictions[first]-labels)**2-(predictions[second]-labels)**2))
            row.update(truth_delta=delta,truth_covered=row['joint_lower']-1e-10<=delta<=row['joint_upper']+1e-10)
            if not row['truth_covered']:raise AssertionError('Artificial masking truth outside exact bounds')
        rows.append(row)
    return rows


def longest_complete(values,start,max_length):
    best=(start,start);begin=start
    for end in range(start,len(values)+1):
        if end==len(values) or np.isnan(values[end]):
            if end-begin>best[1]-best[0]:best=(begin,end)
            begin=end+1
    return best[0],min(best[1],best[0]+max_length)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--private',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/research/SHARED_MISSING_EVENTS_AP0_20260916.json')
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'))
    a.output.mkdir(parents=True,exist_ok=True);a.private.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();cpu=time.process_time()
    series,qualification,downloads=read_data(a.data,c)
    csvout(a.output/'data_qualification.csv',qualification);dump(a.output/'data_sources.json',downloads)
    scores=[];model_metrics=[];training=[];skipped=[]
    with threadpool_limits(limits=c['compute']['threads']):
        for dataset,dc in c['datasets'].items():
            keys=sorted(key for d,key in series if d==dataset)
            medians={key:float(np.nanmedian(series[dataset,key][0][:dc['train_end']])) for key in keys}
            feature_arrays={key:features(*series[dataset,key],dc,medians[key],i,len(keys)) for i,key in enumerate(keys)}
            for k in dc['windows']:
                ell=dc['run_length'];X_train=[];y_train=[]
                for key in keys:
                    values,_=series[dataset,key];lo,hi=label_bounds(observed_bits(values,dc['threshold']),k,ell)
                    ix=np.arange(dc['history'],dc['train_end']-k+1);ix=ix[lo[ix]==hi[ix]]
                    X_train.append(feature_arrays[key][ix]);y_train.append(lo[ix])
                X_train=np.concatenate(X_train);y_train=np.concatenate(y_train)
                if len(np.unique(y_train))!=2:raise ValueError('Fixed pilot has insufficient training classes; report without changing thresholds')
                model_started=time.perf_counter()
                lp=c['predictors']['logistic'];hp=c['predictors']['hist_gradient_boosting']
                models={'logistic':make_pipeline(StandardScaler(),LogisticRegression(**lp)),
                        'hist_gradient_boosting':HistGradientBoostingClassifier(**hp,random_state=20260916)}
                for model in models.values():model.fit(X_train,y_train)
                frequency=float((y_train.sum()+1)/(len(y_train)+2))
                training.append({'dataset':dataset,'K':k,'L':ell,'training_windows':len(y_train),'positive_training_windows':int(y_train.sum()),
                                 'features':X_train.shape[1],'constant_probability':frequency,'fit_seconds':time.perf_counter()-model_started})
                print(json.dumps({'event':'models_fit','dataset':dataset,'K':k,'windows':len(y_train)}),flush=True)
                def predict(X):
                    return {'constant':np.full(len(X),frequency),**{name:model.predict_proba(X)[:,1] for name,model in models.items()}}
                for station_id,key in enumerate(keys):
                    values,dates=series[dataset,key];start=dc['calibration_end'];n=len(values)-start
                    probabilities=predict(feature_arrays[key][start:len(values)-k+1])
                    observed=observed_bits(values[start:],dc['threshold'])
                    scores.extend(compare_panel(dataset,key,'full_station','natural',k,ell,observed,probabilities,start))
                    lo,hi=label_bounds(observed,k,ell);known=lo==hi
                    for name,pred in probabilities.items():
                        model_metrics.append({'dataset':dataset,'series':key,'K':k,'model':name,'identified_windows':int(known.sum()),
                                              'brier_on_identified':float(np.mean((pred[known]-lo[known])**2)) if known.any() else '',
                                              'mean_prediction':float(pred.mean())})
                    np.savez_compressed(a.private/f'{dataset}_{key}_K{k}_predictions.npz',**probabilities)
                    for offset in range(0,n,dc['panel_length']):
                        end=min(n,offset+dc['panel_length'])
                        if end-offset<k:continue
                        pp={name:v[offset:end-k+1] for name,v in probabilities.items()}
                        scores.extend(compare_panel(dataset,key,'fixed_panel','natural',k,ell,observed[offset:end],pp,start+offset))
                    clean_start,clean_end=longest_complete(values,start,dc['artificial_max_length'])
                    if clean_end-clean_start<k:
                        skipped.append({'dataset':dataset,'series':key,'K':k,'reason':'No naturally complete run of K or longer'})
                        continue
                    truth=(values[clean_start:clean_end]>=dc['threshold']).astype(np.int8)
                    for mask in c['artificial_evaluation']['masks']:
                        rng=np.random.default_rng(20260916+station_id+(0 if dataset=='beijing' else 1000))
                        hidden=np.zeros(clean_end-clean_start,dtype=bool)
                        if mask.startswith('iid'):
                            hidden=rng.random(len(hidden))<int(mask.split('_')[1])/100
                        else:
                            length=max(ell,k//4)
                            for anchor in rng.choice(len(hidden),size=max(1,int(.1*len(hidden)/length)),replace=False):
                                hidden[anchor:min(len(hidden),anchor+length)]=True
                        masked=values.copy();masked[np.flatnonzero(hidden)+clean_start]=np.nan
                        XX=features(masked,dates,dc,medians[key],station_id,len(keys))
                        pp=predict(XX[clean_start:clean_end-k+1])
                        obs=observed_bits(masked[clean_start:clean_end],dc['threshold'])
                        scores.extend(compare_panel(dataset,key,'artificial',mask,k,ell,obs,pp,clean_start,truth))
                if time.process_time()-cpu>c['compute']['cpu_hour_budget']*3600:
                    raise RuntimeError('Pilot CPU budget exhausted; keep completed outputs')
                csvout(a.output/'paired_bounds.csv',scores);csvout(a.output/'predictor_metrics.csv',model_metrics);csvout(a.output/'training_summary.csv',training)
    # Pool separate station sequences by summing extrema, using the same window denominator.
    pooled=[];grouped=defaultdict(list)
    for row in scores:
        if row['scope']=='full_station':grouped[row['dataset'],row['K'],row['model_A'],row['model_B']].append(row)
    for (dataset,k,ma,mb),group in grouped.items():
        total=sum(r['windows'] for r in group)
        row={'dataset':dataset,'K':k,'model_A':ma,'model_B':mb,'series_count':len(group),'windows':total}
        for metric in ['independent_lower','independent_upper','joint_lower','joint_upper']:
            row[metric]=sum(r[metric]*r['windows'] for r in group)/total
        row['independent_winner']=label_winner(row['independent_lower'],row['independent_upper'])
        row['joint_winner']=label_winner(row['joint_lower'],row['joint_upper'])
        pooled.append(row)
    csvout(a.output/'pooled_bounds.csv',pooled);dump(a.output/'skipped_panels.json',skipped)
    summary={'study_id':c['study_id'],'status':'COMPLETE','datasets':list(c['datasets']),
             'probability_predictors':3,'training_configurations':len(training),'fitted_classifiers':2*len(training),
             'reported_pair_panels':len(scores),'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
             'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
             'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
             'prediction_use':'fixed per panel; masked observations are removed before recomputing features',
             'confidence_interval':False,'threshold_or_model_selection_on_evaluation':False}
    for scope in ['full_station','fixed_panel','artificial']:
        selected=[r for r in scores if r['scope']==scope]
        ambiguous=[r for r in selected if r['independent_width']>1e-12]
        summary[scope]={'comparisons':len(selected),'with_ambiguous_labels':len(ambiguous),
                        'strictly_tighter':sum(r['width_reduction']>1e-12 for r in selected),
                        'new_strict_decisions':sum(r['new_strict_decision'] for r in selected),
                        'median_relative_tightening_when_ambiguous':float(np.median([r['relative_width_reduction'] for r in ambiguous])) if ambiguous else 0.}
    summary['artificial_truth_violations']=sum(r['truth_covered'] is False for r in scores if r['scope']=='artificial')
    dump(a.output/'experiment_summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
