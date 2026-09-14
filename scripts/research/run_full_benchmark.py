"""Resumable train -> freeze -> predict -> score runner for a frozen six-fold plan.

Each phase is explicit. This entrypoint never downloads weights or launches a
timer. Private arrays/checkpoints are kept outside the public summary directory.
"""
from __future__ import annotations

import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast.benchmark_contract import origins, fold_plan, score_arrays
from urbanev_forecast.bounded_predictor_csv import parse_prefix
from urbanev_forecast.full_benchmark_data import (
    FoldReader, streaming_transform, streaming_statistics, ridge_solution,
    predict_ridge, target_windows, TensorWindows, read_exact_prefix,
)
from urbanev_forecast.full_benchmark_models import (
    load_verified_author, AuthorForecaster, ResidualForecaster,
)
from urbanev_forecast.full_benchmark_guard import verify_execution,consume_or_validate_claim


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    os.replace(temp,path)


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def csvout(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def record(path,root):return {'file':Path(path).relative_to(root).as_posix(),'sha256':sha(path)}


def checked(root,item):
    p=(root/item['file']).resolve()
    if root.resolve() not in p.parents or sha(p)!=item['sha256']:raise ValueError('Frozen artifact identity mismatch')
    return p


def progress(**values):print(json.dumps(values,ensure_ascii=False),flush=True)


def setup():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)


def cell_path(a,f,h):
    p=a.private/f'fold{f}_h{h}';p.mkdir(parents=True,exist_ok=True);return p


def make_model(spec,h,seed,c,roots,classes):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if spec['kind']=='residual':return ResidualForecaster(spec['architecture'],h,seed).cuda()
    source=spec['source']
    if source not in classes:classes[source]=load_verified_author(roots[source],c['author_sources'][source])
    return AuthorForecaster(spec['architecture'],h,classes[source],spec['model_config']).cuda()


def predict_net(net,windows,oo,micro,standardized,h):
    net.eval();outputs=[]
    with torch.no_grad():
        for start in range(0,len(oo),micro):
            q=net(windows.inputs(oo[start:start+micro],standardized))
            if not torch.isfinite(q).all():raise ValueError('Nonfinite prediction')
            outputs.append(q.cpu().numpy().astype(np.float64))
    output=np.concatenate(outputs)
    if output.shape!=(len(oo),h,275):raise ValueError('Incomplete prediction axes')
    return output


def prepare_cell(a,c,f,h):
    p=cell_path(a,f,h);done=p/'linear_complete.json'
    if done.exists():
        result=load(done)
        if result['config_sha256']!=sha(a.config):raise ValueError('Configuration changed on resume')
        checked(a.private,result['stats'])
        for item in result['models'].values():checked(a.private,item['weights'])
        return result
    started=time.perf_counter();ledger=[];reader=FoldReader(a.data,c['data_identity'],f,h,ledger)
    y,d,cap=reader.read('train');oo=origins(f,'train',168,h,contract='MATCHED_TERMINAL_168')
    stats=streaming_transform(y,d,cap,oo);gram,rhs=streaming_statistics(y,d,cap,oo,h,stats)
    statfile=p/'private_stats.npz';np.savez(statfile,**stats)
    yy,dd,_=reader.read('validation');vo=origins(f,'validation',168,h,contract='MATCHED_TERMINAL_168')
    truth=target_windows(yy,vo,h);models={};search=[]
    for spec in c['ridge_models']:
        options=[]
        for regularization in spec['lambdas']:
            beta,normal_error=ridge_solution(gram,rhs,spec['mask'],regularization)
            pred=predict_ridge(yy,dd,cap,vo,h,stats,spec['mask'],beta)
            metric=score_arrays(pred,truth,scope='terminal',postprocess='raw')
            entry={'model':spec['name'],'lambda':regularization,'validation_rmse':metric['rmse'],
                   'validation_mae':metric['mae'],'normal_equation_error':normal_error}
            search.append(entry);options.append((metric['rmse'],regularization,beta,entry))
        _,regularization,beta,entry=min(options,key=lambda row:(row[0],row[1]))
        bp=p/f"private_{spec['name']}.npy";np.save(bp,beta)
        models[spec['name']]={'weights':record(bp,a.private),'mask':spec['mask'],'lambda':regularization,
                              'information_track':spec['information_track'],'selection':entry}
    result={'config_sha256':sha(a.config),'fold':f,'horizon':h,'stats':record(statfile,a.private),
            'models':models,'search':search,'reads':ledger,'seconds':time.perf_counter()-started,
            'training_origins':len(oo),'validation_origins':len(vo)}
    dump(done,result);progress(event='linear_complete',fold=f,horizon=h,seconds=result['seconds'])
    return result


def train_one(a,c,spec,f,h,seed,lr,linear,classes,roots):
    p=cell_path(a,f,h);tag=f"{spec['name']}_s{seed}_lr{lr:g}"
    done=p/(tag+'_complete.json')
    if done.exists():
        result=load(done)
        if result['config_sha256']!=sha(a.config):raise ValueError('Changed config on resume')
        checked(a.private,result['best_weights']);return result
    # A pending marker prevents silently restarting an interrupted trial with new weights.
    meta_file=p/(tag+'_resume.json')
    cfg=c['training'];started=time.perf_counter();ledger=[]
    reader=FoldReader(a.data,c['data_identity'],f,h,ledger)
    y,d,cap=reader.read('train');yy,dd,_=reader.read('validation')
    oo=origins(f,'train',168,h,contract='MATCHED_TERMINAL_168')
    vo=origins(f,'validation',168,h,contract='MATCHED_TERMINAL_168')
    stats=dict(np.load(checked(a.private,linear['stats']),allow_pickle=False))
    train_y=target_windows(y,oo,h);valid_y=target_windows(yy,vo,h)
    base_train=np.zeros_like(train_y);base_valid=np.zeros_like(valid_y)
    residual=spec['kind']=='residual'
    if residual:
        base=linear['models'][spec['base']];beta=np.load(checked(a.private,base['weights']),allow_pickle=False)
        base_train=predict_ridge(y,d,cap,oo,h,stats,base['mask'],beta)
        base_valid=predict_ridge(yy,dd,cap,vo,h,stats,base['mask'],beta)
    train_windows=TensorWindows(y,d,cap,stats);valid_windows=TensorWindows(yy,dd,cap,stats)
    e32=(train_y-base_train).astype(np.float32)
    residual_rms=float(np.sqrt(np.mean(e32.astype(np.float64)**2)))
    tau=float(np.float32(.5*max(residual_rms,1e-6)))
    targets=torch.as_tensor(e32,device='cuda')
    net=make_model(spec,h,seed,c,roots,classes)
    optimizer=torch.optim.AdamW([
        {'params':[p for p in net.parameters() if p.ndim>=2],'weight_decay':cfg['weight_decay']},
        {'params':[p for p in net.parameters() if p.ndim<2],'weight_decay':0.}],
        lr=lr,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    torch.cuda.reset_peak_memory_stats()
    best=None;curves=[];validations=[];checkpoints=[];start_epoch=1;total_steps=0;prior_seconds=0.
    if meta_file.exists():
        meta=load(meta_file)
        state_file=p/meta['checkpoint_file']
        if meta['config_sha256']!=sha(a.config) or meta['checkpoint_sha256']!=sha(state_file):raise ValueError('Resume identity mismatch')
        state=torch.load(state_file,map_location='cpu',weights_only=False)
        net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_epoch=meta['epoch']+1;curves=meta['curves'];checkpoints=meta['checkpoints']
        total_steps=meta['steps'];prior_seconds=meta['seconds']
        for checkpoint in checkpoints:checked(a.private,checkpoint['weights'])

    def save_checkpoint(epoch):
        path=p/f'{tag}_e{epoch}.pt'
        torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},path)
        checkpoints.append({'epoch':epoch,'weights':record(path,a.private)})

    def validate(checkpoint):
        nonlocal best
        epoch=checkpoint['epoch']
        net.load_state_dict(torch.load(checked(a.private,checkpoint['weights']),map_location='cpu',weights_only=True))
        pred=predict_net(net,valid_windows,vo,cfg['microbatch'],residual,h)+base_valid
        score=score_arrays(pred,valid_y,scope='terminal',postprocess='raw')
        row={'epoch':epoch,'validation_rmse':score['rmse'],'validation_mae':score['mae']}
        validations.append(row)
        if best is None or (row['validation_rmse'],epoch)<(best['validation_rmse'],best['epoch']):
            best={**row,'weights':checkpoint['weights']}

    if start_epoch==1 and 0 in cfg['checkpoints']:save_checkpoint(0)
    for epoch in range(start_epoch,cfg['epochs']+1):
        net.train();current_lr=lr*(cfg['minimum_lr_fraction']+(1-cfg['minimum_lr_fraction'])*(1+math.cos(math.pi*(epoch-1)/max(1,cfg['epochs']-1)))/2)
        for group in optimizer.param_groups:group['lr']=current_lr
        order=np.random.default_rng(np.random.SeedSequence([seed,f,h,epoch])).permutation(len(oo))
        total_loss=torch.zeros(2,device='cuda');n_seen=0
        for start in range(0,len(order),cfg['batch_origins']):
            indices=order[start:start+cfg['batch_origins']];optimizer.zero_grad(set_to_none=True)
            for i in range(0,len(indices),cfg['microbatch']):
                ix=indices[i:i+cfg['microbatch']]
                q=net(train_windows.inputs(oo[ix],residual));target=targets[ix]
                mse=((q-target)**2).mean()
                loss=mse
                if spec.get('objective')=='POSREG':
                    loss=mse+tau*torch.relu((target-q).abs()-target.abs()).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                (loss*len(ix)/len(indices)).backward()
                total_loss+=torch.stack((mse.detach(),loss.detach()))*len(ix);n_seen+=len(ix)
            torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();total_steps+=1
        totals=(total_loss/n_seen).cpu().tolist()
        curves.append({'epoch':epoch,'learning_rate':current_lr,'path_training_mse':totals[0],
                       'training_objective':totals[1],'optimizer_steps':total_steps})
        if epoch in cfg['checkpoints']:save_checkpoint(epoch)
        # Save a deterministic epoch-boundary restart, including optimizer and RNG.
        # Distinct files retain the old valid checkpoint if interrupted before metadata commit.
        state_file=p/f'{tag}_resume_e{epoch}.pt'
        temporary=state_file.with_suffix('.tmp')
        torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temporary)
        os.replace(temporary,state_file)
        dump(meta_file,{'config_sha256':sha(a.config),'checkpoint_sha256':sha(state_file),
                        'checkpoint_file':state_file.name,'epoch':epoch,'checkpoints':checkpoints,'curves':curves,
                        'steps':total_steps,'seconds':prior_seconds+time.perf_counter()-started})
        if epoch in cfg['checkpoints']:progress(event='checkpoint',model=spec['name'],fold=f,horizon=h,seed=seed,lr=lr,**curves[-1])
    if total_steps!=math.ceil(len(oo)/cfg['batch_origins'])*cfg['epochs']:
        raise ValueError('Optimizer-step budget mismatch')
    # All twenty epochs finish before any checkpoint is scored on validation.
    for checkpoint in checkpoints:validate(checkpoint)
    torch.cuda.synchronize()
    result={'config_sha256':sha(a.config),'model':spec['name'],'fold':f,'horizon':h,'seed':seed,'lr':lr,
            'best':best,'best_weights':best['weights'],'curves':curves,'validations':validations,
            'checkpoints':checkpoints,'training_residual_rms':residual_rms,'posreg_tau_float32':tau if spec.get('objective')=='POSREG' else None,
            'optimizer_steps':total_steps,'parameters':sum(p.numel() for p in net.parameters()),
            'seconds':prior_seconds+time.perf_counter()-started,'gpu_peak_allocated':torch.cuda.max_memory_allocated(),
            'gpu_peak_reserved':torch.cuda.max_memory_reserved(),'reads':ledger}
    dump(done,result);progress(event='neural_complete',model=spec['name'],fold=f,horizon=h,seed=seed,seconds=result['seconds'])
    del net,optimizer,targets,train_windows,valid_windows;torch.cuda.empty_cache()
    return result


def train_phase(a,c):
    if (a.private/'training_freeze.json').exists():
        freeze=load(a.private/'training_freeze.json')
        if freeze['config_sha256']!=sha(a.config):raise ValueError('Configuration changed')
        for item in freeze['files']:checked(a.private,item)
        return freeze
    roots=load(a.author_roots);classes={};selections=[];files=[]
    for f in c['folds']:
        for h in c['horizons']:
            linear=prepare_cell(a,c,f,h);files.append(record(cell_path(a,f,h)/'linear_complete.json',a.private))
            files.append(linear['stats']);files.extend(v['weights'] for v in linear['models'].values())
            for spec in c['neural_models']:
                for seed in c['training']['seeds']:
                    trials=[train_one(a,c,spec,f,h,seed,lr,linear,classes,roots) for lr in spec['learning_rates']]
                    chosen=min(trials,key=lambda t:(t['best']['validation_rmse'],t['lr'],t['best']['epoch']))
                    selections.append({'model':spec['name'],'fold':f,'horizon':h,'seed':seed,
                                       'lr':chosen['lr'],'selected_epoch':chosen['best']['epoch'],
                                       'validation_rmse':chosen['best']['validation_rmse'],'weights':chosen['best_weights'],
                                       'information_track':spec['information_track'],'base':spec.get('base')})
                    files.extend(t['best_weights'] for t in trials)
    freeze={'config_sha256':sha(a.config),'selections':selections,'files':files,
            'test_scores_seen':False,'all_registered_supervised_training_complete':True}
    dump(a.private/'training_freeze.json',freeze)
    return freeze


def predict_phase(a,c):
    frozen=a.private/'training_freeze.json'
    if not frozen.exists():raise RuntimeError('Complete all training before prediction')
    freeze=load(frozen)
    if freeze['config_sha256']!=sha(a.config):raise ValueError('Frozen configuration mismatch')
    for item in freeze['files']:checked(a.private,item)
    roots=load(a.author_roots);classes={};records=[]
    for f in c['folds']:
        for h in c['horizons']:
            p=cell_path(a,f,h);done=p/'prediction_complete.json'
            if done.exists():
                saved=load(done)
                if saved['training_freeze_sha256']!=sha(frozen):raise ValueError('Prediction freeze mismatch')
                for item in saved['predictions']:checked(a.private,item)
                records.extend(saved['predictions']);continue
            linear=load(p/'linear_complete.json');stats=dict(np.load(checked(a.private,linear['stats']),allow_pickle=False))
            reader=FoldReader(a.data,c['data_identity'],f,h,[]);reader.close_training();y,d,cap=reader.read('predict')
            oo=origins(f,'test',168,h,contract='MATCHED_TERMINAL_168');windows=TensorWindows(y,d,cap,stats)
            predictions={};cell_records=[]
            def save_prediction(name,seed,q,information):
                if q.shape!=(len(oo),h,275) or not np.isfinite(q).all():raise ValueError('Bad complete prediction')
                file=p/f'private_test_{name}_{seed}.npy';np.save(file,q)
                cell_records.append({**record(file,a.private),'model':name,'seed':seed,'fold':f,'horizon':h,
                                     'origins':len(oo),'origin_first':int(oo[0]),'origin_last':int(oo[-1]),
                                     'information_track':information})
            for name,item in linear['models'].items():
                q=predict_ridge(y,d,cap,oo,h,stats,item['mask'],np.load(checked(a.private,item['weights']),allow_pickle=False))
                predictions[name]=q;save_prediction(name,'fixed',q,item['information_track'])
            index=oo[:,None]+np.arange(h)
            for name,q in [('LAST',np.repeat(y[oo-1,None,:],h,axis=1)),('DAY',y[index-24]),('WEEK',y[index-168])]:
                save_prediction(name,'fixed',q,'LOCAL_O_HISTORICAL')
            for item in freeze['selections']:
                if (item['fold'],item['horizon'])!=(f,h):continue
                spec=next(v for v in c['neural_models'] if v['name']==item['model'])
                net=make_model(spec,h,item['seed'],c,roots,classes)
                net.load_state_dict(torch.load(checked(a.private,item['weights']),map_location='cpu',weights_only=True))
                q=predict_net(net,windows,oo,c['training']['microbatch'],spec['kind']=='residual',h)
                if spec['kind']=='residual':q+=predictions[spec['base']]
                save_prediction(spec['name'],item['seed'],q,spec['information_track']);del net
            dump(done,{'training_freeze_sha256':sha(frozen),'predictions':cell_records,'reads':reader.ledger})
            records.extend(cell_records);del windows;torch.cuda.empty_cache()
            progress(event='supervised_predictions_frozen',fold=f,horizon=h)
    dump(a.private/'supervised_prediction_freeze.json',{'config_sha256':sha(a.config),
         'training_freeze_sha256':sha(frozen),'predictions':records,'test_scores_seen':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','data','private','author-roots'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['train','predict'],required=True)
    a=p.parse_args();c=load(a.config)
    verify_execution(c,ROOT)
    consume_or_validate_claim(a.private,a.config,a.phase)
    if c['folds']!=list(range(1,7)) or c['horizons']!=[3,6,9,12]:raise ValueError('Full registered cell coverage required')
    a.private.mkdir(parents=True,exist_ok=True);setup()
    with threadpool_limits(limits=2):
        if a.phase=='train':train_phase(a,c)
        else:predict_phase(a,c)


if __name__=='__main__':main()
