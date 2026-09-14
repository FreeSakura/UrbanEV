"""Score only a complete frozen prediction set; no fitting or model selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast.benchmark_contract import origins,fold_plan,origin_digest,macro24,score_arrays
from urbanev_forecast.full_benchmark_data import read_exact_prefix
from urbanev_forecast.bounded_predictor_csv import parse_prefix
from urbanev_forecast.full_benchmark_guard import verify_execution,consume_or_validate_claim


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):d.update(block)
    return d.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def csvout(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def expected_tasks(c):
    systems=[(n,'fixed') for n in ['LAST','DAY','WEEK']+[s['name'] for s in c['ridge_models']]]
    systems += [(s['name'],str(seed)) for s in c['neural_models'] for seed in c['training']['seeds']]
    systems += [(name.upper()+'_'+variant,'pretrained') for name in c['foundation_models'] for variant in ('POINT','Q05')]
    return {(name,seed,f,h) for name,seed in systems for f in c['folds'] for h in c['horizons']}


def verify_prediction_set(c,records,root):
    keys=[]
    tracks={s['name']:s['information_track'] for s in c['ridge_models']+c['neural_models']}
    tracks.update({name:'LOCAL_O_HISTORICAL' for name in ('LAST','DAY','WEEK')})
    tracks.update({name.upper()+'_'+variant:'GLOBAL_O_168_PRETRAINED' for name in c['foundation_models'] for variant in ('POINT','Q05')})
    for row in records:
        key=(row['model'],str(row['seed']),row['fold'],row['horizon']);keys.append(key)
        p=(root/row['file']).resolve()
        if root.resolve() not in p.parents or not p.is_file() or sha(p)!=row['sha256']:
            raise ValueError('Prediction artifact changed or missing')
        oo=origins(row['fold'],'test',168,row['horizon'],contract='MATCHED_TERMINAL_168')
        if (row['origins'],row['origin_first'],row['origin_last'])!=(len(oo),int(oo[0]),int(oo[-1])):
            raise ValueError('Incorrect prediction support metadata')
        if row['information_track']!=tracks[row['model']]:raise ValueError('Prediction information identity mismatch')
        array=np.load(p,allow_pickle=False,mmap_mode='r')
        if array.shape!=(len(oo),row['horizon'],275) or not np.isfinite(array).all():
            raise ValueError('Invalid prediction payload before score-target access')
        del array
    if len(set(keys))!=len(keys) or set(keys)!=expected_tasks(c):
        raise ValueError('Complete unique registered prediction set required; no partial averages')


def circular_indices(n,repetitions,block_length,seed):
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(repetitions,(n+block_length-1)//block_length))
    return ((starts[:,:,None]+np.arange(block_length))%n).reshape(repetitions,-1)[:,:n]


def verify_training_budget(c,root):
    """Fail before opening score targets if any registered training trial is incomplete."""
    total_steps=0;trials=0;ridges=0
    for f in c['folds']:
        for h in c['horizons']:
            cell=root/f'fold{f}_h{h}';linear=load(cell/'linear_complete.json')
            if set(linear['models'])!={v['name'] for v in c['ridge_models']}:raise ValueError('Missing ridge identity')
            ridges+=len(linear['models'])
            expected_steps=((len(origins(f,'train',168,h,contract='MATCHED_TERMINAL_168'))+15)//16)*20
            for spec in c['neural_models']:
                for seed in c['training']['seeds']:
                    trial=load(cell/f"{spec['name']}_s{seed}_lr0.001_complete.json")
                    if (trial['model'],trial['fold'],trial['horizon'],trial['seed'])!=(spec['name'],f,h,seed):
                        raise ValueError('Training task key mismatch')
                    if trial['optimizer_steps']!=expected_steps or [v['epoch'] for v in trial['curves']]!=list(range(1,21)):
                        raise ValueError('Incomplete fixed training budget')
                    if [v['epoch'] for v in trial['validations']]!=[0,5,10,15,20]:raise ValueError('Validation checkpoint set changed')
                    total_steps+=trial['optimizer_steps'];trials+=1
    if (ridges,trials,total_steps)!=(72,432,1011960):raise ValueError('Training coverage is incomplete')


def uncertainty(c,losses):
    primary=sorted({key[0] for key in losses if not key[0].endswith('_Q05')})
    reps={};points={};B=c['bootstrap']['replicates']
    for h in c['horizons']:
        for model in primary:
            reps[model,h]=np.zeros((B,2));points[model,h]=np.zeros(2)
        for f in c['folds']:
            n=len(origins(f,'test',168,h,contract='MATCHED_TERMINAL_168'))
            ix=circular_indices(n,B,c['bootstrap']['block_length'],np.random.SeedSequence([c['bootstrap']['seed'],f,h]))
            for model in primary:
                arrays=[a for (m,_,ff,hh),a in losses.items() if (m,ff,hh)==(model,f,h)]
                if not arrays:raise ValueError('Missing bootstrap cell')
                for a in arrays:
                    mse=a[:,0]/a[:,2];mae=a[:,1]/a[:,2]
                    reps[model,h][:,0]+=np.sqrt(mse[ix].mean(axis=1))/(6*len(arrays))
                    reps[model,h][:,1]+=mae[ix].mean(axis=1)/(6*len(arrays))
                    points[model,h]+=np.array([np.sqrt(mse.mean()),mae.mean()])/(6*len(arrays))
    rows=[]
    for model in primary:
        for ref in c['bootstrap']['references']:
            for h in c['horizons']:
                for i,metric in enumerate(('rmse','mae')):
                    delta=reps[model,h][:,i]-reps[ref,h][:,i]
                    lo,hi=np.quantile(delta,[.025,.975])
                    rows.append({'model':model,'reference':ref,'horizon':h,'metric':metric,
                                 'candidate_minus_reference':float(points[model,h][i]-points[ref,h][i]),
                                 'percentile_low':float(lo),'percentile_high':float(hi),'replicates':B,
                                 'block_length_origins':c['bootstrap']['block_length'],
                                 'scope':'raw terminal; six-fold equal metric mean; seed metrics averaged',
                                 'interpretation':'conditional time-resampling sensitivity, not global SOTA certification'})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','data','private','public'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();c=load(a.config)
    verify_execution(c,ROOT);consume_or_validate_claim(a.private,a.config,'score')
    if (a.public/'execution_receipt.json').exists():raise RuntimeError('Completed scoring receipt exists; do not overwrite')
    supervised=load(a.private/'supervised_prediction_freeze.json');training=load(a.private/'training_freeze.json')
    if supervised['config_sha256']!=sha(a.config) or training['config_sha256']!=sha(a.config):raise ValueError('Config mismatch')
    if supervised['training_freeze_sha256']!=sha(a.private/'training_freeze.json'):raise ValueError('Training freeze changed')
    for item in training['files']:
        if sha(a.private/item['file'])!=item['sha256']:raise ValueError('Training artifact changed after freeze')
    verify_training_budget(c,a.private)
    records=list(supervised['predictions']);foundations={}
    for name in c['foundation_models']:
        value=load(a.private/name/'foundation_complete.json')
        if value['config_sha256']!=sha(a.config) or value['origin_tasks']!=5960:raise ValueError('Foundation stage incomplete')
        if value['supervised_prediction_freeze_sha256']!=sha(a.private/'supervised_prediction_freeze.json'):raise ValueError('Prediction stage linkage changed')
        foundations[name]=value;records.extend(value['predictions'])
    verify_prediction_set(c,records,a.private)
    a.public.mkdir(parents=True,exist_ok=True)
    dump(a.public/'all_prediction_freeze.json',{'config_sha256':sha(a.config),'predictions':records,
         'all_required_predictions_complete_before_scoring':True,'payloads':len(records)})
    # The only truth loader in the scoring process; no training/inference imports.
    import pandas as pd
    from urbanev_forecast.comparability_bridge import zone_capacities
    inf=a.data/'inf.csv'
    if sha(inf)!=c['data_identity']['static_sha256']:raise ValueError('Capacity file changed')
    columns=c['data_identity']['columns']
    cap=zone_capacities(pd.read_csv(inf,usecols=['station_id','TAZID','charge_count'],dtype={'TAZID':str}),columns)
    rows=[];losses={};reads=[]
    for f in c['folds']:
        stop=fold_plan(f).test_stop;blob=read_exact_prefix(a.data,'occupancy.csv',stop)
        raw,read=parse_prefix(blob,stop,columns,c['data_identity']['prefixes']['occupancy.csv'][str(stop)],nonnegative=True)
        y=(raw.astype(np.float32)/cap.astype(np.float32)[None,:]).astype(np.float64)
        if np.any(y<0) or np.any(y>1):raise ValueError('Invalid score target; no repair')
        reads.append({'stage':'SCORE','fold':f,**read})
        for record in records:
            if record['fold']!=f:continue
            h=record['horizon'];oo=origins(f,'test',168,h,contract='MATCHED_TERMINAL_168')
            target=y[oo[:,None]+np.arange(h)];q=np.load(a.private/record['file'],allow_pickle=False).astype(np.float64)
            if q.shape!=(len(oo),h,275) or not np.isfinite(q).all():raise ValueError('Invalid frozen prediction')
            error=q[:,-1]-target[:,-1]
            losses[record['model'],str(record['seed']),f,h]=np.column_stack(((error**2).sum(axis=1),np.abs(error).sum(axis=1),np.full(len(oo),275)))
            for scope in ('terminal','path'):
                for post in ('raw','clip_0_1'):
                    score=score_arrays(q,target,scope=scope,postprocess=post)
                    rows.append({'model':record['model'],'seed':str(record['seed']),'number':f,'horizon':h,
                                 'contract':'MATCHED_TERMINAL_168','split':'test','history':168,
                                 'information_track':record['information_track'],'prediction_mode':'horizon_specific',
                                 'unit':'occupancy_rate','origin_sha256':origin_digest(oo),
                                 'config_sha256':sha(a.config),**score})
    csvout(a.public/'cell_scores.csv',rows)
    keys=sorted({(v['model'],v['seed'],v['scope'],v['postprocess']) for v in rows})
    macro=[];per_h=[]
    for model,seed,scope,post in keys:
        group=[r for r in rows if (r['model'],r['seed'],r['scope'],r['postprocess'])==(model,seed,scope,post)]
        for field in ('information_track','prediction_mode','unit','config_sha256'):
            if len({r[field] for r in group})!=1:raise ValueError('Mixed full benchmark identity')
        score=macro24({(r['number'],r['horizon']):r for r in group})
        macro.append({'model':model,'seed':seed,'information_track':group[0]['information_track'],**score})
        for h in c['horizons']:
            sub=[r for r in group if r['horizon']==h]
            if len(sub)!=6:raise ValueError('Incomplete per-horizon table')
            per_h.append({'model':model,'seed':seed,'horizon':h,'scope':scope,'postprocess':post,
                          'rmse':float(np.mean([r['rmse'] for r in sub])),
                          'mae':float(np.mean([r['mae'] for r in sub]))})
    csvout(a.public/'seed_macro_scores.csv',macro);csvout(a.public/'per_horizon_scores.csv',per_h)
    summary=[]
    for model in sorted({r['model'] for r in macro}):
        for scope in ('terminal','path'):
            for post in ('raw','clip_0_1'):
                group=[r for r in macro if (r['model'],r['scope'],r['postprocess'])==(model,scope,post)]
                row={'model':model,'scope':scope,'postprocess':post,'instances':len(group),
                     'information_track':group[0]['information_track']}
                for metric in ('rmse','mae'):
                    values=[r[metric] for r in group]
                    row[metric+'_mean']=float(np.mean(values));row[metric+'_min']=min(values);row[metric+'_max']=max(values)
                summary.append(row)
    csvout(a.public/'model_summary.csv',summary)
    csvout(a.public/'paired_block_bootstrap.csv',uncertainty(c,losses))
    # Private origin losses support independent recalculation without publishing predictions.
    lossdir=a.private/'origin_losses';lossdir.mkdir(exist_ok=True)
    for (name,seed,f,h),array in losses.items():np.save(lossdir/f'{name}_{seed}_f{f}_h{h}.npy',array)
    dump(a.public/'validation_selection.json',{'selections':training['selections'],'config_sha256':sha(a.config)})
    trials=[];ridge_seconds=0.;ridge_fits=0;curves=[];validation=[];ridge_diagnostics=[]
    for f in c['folds']:
        for h in c['horizons']:
            cell=a.private/f'fold{f}_h{h}';linear=load(cell/'linear_complete.json')
            ridge_seconds+=linear['seconds'];ridge_fits+=len(linear['models'])
            ridge_diagnostics.extend({'fold':f,'horizon':h,**v} for v in linear['search'])
            for spec in c['neural_models']:
                for seed in c['training']['seeds']:
                    for lr in spec['learning_rates']:
                        trial=load(cell/f"{spec['name']}_s{seed}_lr{lr:g}_complete.json")
                        trials.append({k:trial[k] for k in ('model','fold','horizon','seed','lr','parameters','optimizer_steps','seconds','gpu_peak_allocated','gpu_peak_reserved')})
                        curves.extend({'model':spec['name'],'fold':f,'horizon':h,'seed':seed,**v} for v in trial['curves'])
                        validation.extend({'model':spec['name'],'fold':f,'horizon':h,'seed':seed,**v} for v in trial['validations'])
    csvout(a.public/'training_costs.csv',trials)
    csvout(a.public/'training_curves.csv',curves);csvout(a.public/'validation_scores.csv',validation)
    csvout(a.public/'ridge_diagnostics.csv',ridge_diagnostics)
    if len(trials)!=432 or ridge_fits!=72 or sum(t['optimizer_steps'] for t in trials)!=1011960:
        raise ValueError('Execution budget completeness mismatch')
    dump(a.public/'execution_receipt.json',{'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),
         'status':'MATCHED_SIX_FOLD_COMPARISON_COMPLETE_REVIEW_REQUIRED','ridge_fits':ridge_fits,'neural_runs':len(trials),
         'optimizer_steps':sum(t['optimizer_steps'] for t in trials),'prediction_payloads':len(records),
         'score_rows':len(rows),'foundation_origin_tasks':sum(v['origin_tasks'] for v in foundations.values()),
         'neural_run_seconds':sum(t['seconds'] for t in trials),'ridge_preparation_seconds':ridge_seconds,
         'foundation_prediction_seconds':{k:v['seconds'] for k,v in foundations.items()},
         'foundation_point_q05_aliases':{k:v['point_q05_alias_for_all_outputs'] for k,v in foundations.items()},
         'private_origin_loss_columns':['squared_error_sum','absolute_error_sum','region_count'],
         'score_reads':reads,'global_baseline_exhaustiveness':False,'sota_claim':'requires_result_and_comparability_review',
         'new_foundation_weight_downloads':0,'automation_enabled':False,'quota_reset_used':False})
    print('All registered complete predictions scored; result review required',flush=True)


if __name__=='__main__':main()
