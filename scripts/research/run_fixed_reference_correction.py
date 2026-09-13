"""Frozen-base equal-parameter O-D interaction study on the registered CUDA device."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,csv,hashlib,importlib.abc,io,json,platform,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from run_long_history_source_ablation import Reader,dump,csvout,sha,trajectory
from urbanev_forecast import short_state as base
from urbanev_forecast import fixed_reference_correction as core
from urbanev_forecast.benchmark_metrics import scoped_scores

class NoFoundation(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('chronos','timesfm','transformers') or fullname=='urbanev_forecast.foundation':raise RuntimeError('Foundation call forbidden')

def setup_cuda():
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name(0):raise RuntimeError('Registered RTX4060 cuda:0 required; no fallback')

def metric_rows(model,seed,pred,y,stage,hs=base.HORIZONS):
    return [{'stage':stage,'model':model,'seed':str(seed),'horizon':h,'status':'AVAILABLE',**scoped_scores(pred[:,:h],y[:,:h],target_scope=scope,postprocess=post)} for h in hs for post in ('raw','clip_0_1') for scope in ('terminal_H','path_1_to_H')]

def infer(net,x32,interaction=False):
    torch.cuda.synchronize();clock=time.perf_counter();values=[];parts=[];net.eval()
    with torch.no_grad():
        for start in range(0,len(x32),4096):
            x=torch.as_tensor(x32[start:start+4096],dtype=torch.float32,device='cuda:0');v=net(x)
            if not torch.isfinite(v).all():raise ValueError('Nonfinite neural output')
            values.append(v.cpu().numpy().astype(np.float64))
            if interaction:parts.append(net.interaction_component(x).cpu().numpy().astype(np.float64))
    torch.cuda.synchronize()
    return np.concatenate(values),np.concatenate(parts) if interaction else None,time.perf_counter()-clock

def train_one(kind,seed,x32,residual,a,budget,curves,receipt):
    budget.start_neural(kind,seed);receipt['neural_training_runs_started']=len(budget.neural);random.seed(seed);np.random.seed(seed)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0);torch.cuda.synchronize();clock=time.perf_counter()
    net=core.InteractionNet(kind,seed).cuda(0)
    if sum(p.numel() for p in net.parameters())!=core.parameter_count(kind):raise ValueError('Parameter count mismatch')
    opt=torch.optim.AdamW([{'params':[p for p in net.parameters() if p.ndim>=2],'weight_decay':1e-4},{'params':[p for p in net.parameters() if p.ndim<2],'weight_decay':0.}],lr=.001,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    x=torch.as_tensor(x32,dtype=torch.float32,device='cuda:0');y=torch.as_tensor(residual,dtype=torch.float32,device='cuda:0');checkpoints=[];steps=0
    def save(epoch):
        path=a.private_output/f'private_{kind}_s{seed}_e{epoch}.pt';torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},path);checkpoints.append({'epoch':epoch,'file':path.name,'sha256':sha(path)})
    save(0)
    curves.append({'model':kind,'seed':seed,'epoch':0,'lr':0.,'training_mse':float(np.mean(residual.astype(np.float64)**2)),'optimizer_steps':0,'selection_rmse':None,'log_scope':'epoch0 fixed parameters','frozen_fit_mse':float(np.mean(residual.astype(np.float64)**2))})
    for epoch in range(1,41):
        lr=core.learning_rate(epoch)
        for group in opt.param_groups:group['lr']=lr
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(x32));total=0.;net.train()
        for start in range(0,len(order),4096):
            ix=torch.as_tensor(order[start:start+4096],device='cuda:0');opt.zero_grad(set_to_none=True);f=net(x[ix]);loss=((f-y[ix])**2).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            if torch.stack([(~torch.isfinite(p)).any() for p in net.parameters()]).any():raise ValueError('Nonfinite parameter')
            steps+=1;receipt['optimizer_steps']=receipt.get('optimizer_steps',0)+1
            total+=float(loss.detach())*len(ix)
        curves.append({'model':kind,'seed':seed,'epoch':epoch,'lr':lr,'training_mse':total/len(x32),'optimizer_steps':steps,'selection_rmse':None,'log_scope':'during-traversal weighted batch means; frozen_fit_mse only at0/40','frozen_fit_mse':None})
        if epoch in core.CHECKPOINTS:save(epoch);print(json.dumps({'model':kind,'seed':seed,'epoch':epoch,'mse':total/len(x32)}),flush=True)
    frozen_f,_,fit_check_seconds=infer(net,x32)
    frozen_mse=float(np.mean((frozen_f-residual.astype(np.float64))**2));curves[-1]['frozen_fit_mse']=frozen_mse
    receipt['frozen_epoch40_fit_evaluations']=receipt.get('frozen_epoch40_fit_evaluations',0)+1
    torch.cuda.synchronize()
    result={'frozen_epoch40_fit_mse':frozen_mse,'fit_check_inference_seconds':fit_check_seconds,'model':kind,'seed':seed,'parameters':core.parameter_count(kind),'optimizer_steps':steps,'training_seconds_synchronized':time.perf_counter()-clock,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(0),'checkpoints':checkpoints}
    if steps!=1480:raise ValueError('Training step budget mismatch')
    del net,opt,x,y;torch.cuda.empty_cache();return result

def execute(a,c,receipt):
    reader=Reader(c,a.data_root,receipt['stage_reads']);budget=core.Budget();curves=[];runs=[]
    y,d,cap=reader.read('FIT');bases={};base_meta={};fit_times=[]
    def train_base(name,stop,ylimit,dlimit):
        budget.start_deterministic(name);receipt['base_ridge_fits']=receipt.get('base_ridge_fits',0)+1;clock=time.perf_counter()
        model=core.fit_model(y[:ylimit].copy(),d[:dlimit].copy(),cap,np.arange(169,stop));elapsed=time.perf_counter()-clock;fit_times.append(elapsed);bases[name]=model
        bp=a.private_output/f'private_base_{name}.npy';sp=a.private_output/f'private_transform_{name}.npz';np.save(bp,model['beta']);np.savez(sp,**model['stats'])
        base_meta[name]={'ridge':{'file':bp.name,'sha256':sha(bp),'coefficients':int(model['beta'].size)},'transform':{'file':sp.name,'sha256':sha(sp)},'origin_start':169,'origin_stop_exclusive':stop,'train_rows':(stop-169)*275,'max_label':model['max_training_label'],'O_local_prefix':ylimit,'D_local_prefix':dlimit,'fit_seconds':elapsed,'normal_equation_relative_error':model['normal_equation_relative_error']}
    train_base('BASE_FIXED',457,468,455);dump(a.public_output/'initial_reference_freeze.json',base_meta['BASE_FIXED'])
    oo=core.meta_origins();target=base.targets(y,oo);qfixed=core.predict_model(bases['BASE_FIXED'],y,d,cap,oo);e64=target-qfixed
    x=base.features(y,d,cap,oo,'LONG_OD');stats=base.fit_transform(x);z=base.transform(x,stats);del x
    nsp=a.private_output/'private_correction_transform.npz';np.savez(nsp,**stats);neural_hash=sha(nsp)
    scale=core.fit_scale(e64);a.scale=scale['scale_float64'];dump(a.public_output/'loss_scale.json',scale);scale_hash=sha(a.public_output/'loss_scale.json')
    budget.start_deterministic('LINEAR_REPAIR');receipt['linear_correction_fits']=1;clock=time.perf_counter();linear,lmeta=core.linear_repair(z,e64);receipt['linear_correction_seconds']=time.perf_counter()-clock
    lp=a.private_output/'private_linear_repair.npy';np.save(lp,linear);lmeta.update(file=lp.name,sha256=sha(lp))
    x32=z.astype(np.float32);residual=(e64/a.scale).astype(np.float32);del z,target,qfixed,e64
    for kind in core.MODELS:
        for seed in core.SEEDS:
            runs.append(train_one(kind,seed,x32,residual,a,budget,curves,receipt));receipt['neural_training_runs_completed']=len(runs)
    del x32,residual
    train_base('RIDGE_FULL',1045,1056,1043);budget.close();receipt['ridge_fit_seconds']=sum(fit_times)
    frozen_base={**base_meta['BASE_FIXED'],'bases':base_meta,'linear_repair':lmeta,'correction_transform':{'file':nsp.name,'sha256':neural_hash},'same_reference_all_stages':True,'full_ridge_is_reference_only':True}
    bp=a.private_output/base_meta['BASE_FIXED']['ridge']['file'];sp=a.private_output/base_meta['BASE_FIXED']['transform']['file']
    dump(a.public_output/'reference_manifest.json',frozen_base);base_hash=sha(a.public_output/'reference_manifest.json')
    curve_fields=['model','seed','epoch','lr','training_mse','optimizer_steps','selection_rmse','log_scope','frozen_fit_mse'];csvout(a.public_output/'training_curves.csv',curves,curve_fields)
    def prepare_inputs(yy,dd,cc,o):
        zz=base.transform(base.features(yy,dd,cc,o,'LONG_OD'),stats)
        qref=core.predict_model(bases['BASE_FIXED'],yy,dd,cc,o)
        det={'BASE_FIXED':qref,'LINEAR_REPAIR':qref+linear[0]+zz@linear[1:],'RIDGE_FULL':core.predict_model(bases['RIDGE_FULL'],yy,dd,cc,o)}
        return zz.astype(np.float32),det
    def stage_inputs(stage):
        if not budget.closed:raise RuntimeError('Training not closed')
        if sha(a.public_output/'loss_scale.json')!=scale_hash or sha(nsp)!=neural_hash or sha(lp)!=lmeta['sha256']:raise ValueError('Correction statistics/parameters changed')
        for metadata in base_meta.values():
            if sha(a.private_output/metadata['ridge']['file'])!=metadata['ridge']['sha256'] or sha(a.private_output/metadata['transform']['file'])!=metadata['transform']['sha256']:raise ValueError('Frozen reference changed')
        yy,dd,cc=reader.read(stage);o=base.origins(stage);xx,det=prepare_inputs(yy,dd,cc,o)
        return yy,xx,det,o
    y,x32,select_det,so=stage_inputs('SELECT');qbase=select_det['BASE_FIXED'];sy=trajectory(base.targets(y,so));base_select=[];selection_rows=[];selection=[];inference_seconds=0.
    for name,q in select_det.items():base_select.extend(metric_rows(name,'fixed',trajectory(q),sy,'SELECT'))
    for run in runs:
        kind,seed=run['model'],run['seed'];choices=[]
        for cp in run['checkpoints']:
            path=a.private_output/cp['file']
            if sha(path)!=cp['sha256']:raise ValueError('Checkpoint changed')
            if cp['epoch']==0:pred=trajectory(qbase.copy())
            else:
                net=core.InteractionNet(kind,seed).cuda(0);net.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));f,_,elapsed=infer(net,x32);inference_seconds+=elapsed;pred=trajectory(qbase+a.scale*f);del net
            rr=metric_rows(kind,seed,pred,sy,'SELECT');score=float(np.mean([v['rmse'] for v in rr if v['postprocess']=='raw' and v['target_scope']=='terminal_H']))
            selection_rows.extend([{'epoch':cp['epoch'],**v} for v in rr]);choices.append({**cp,'selection_rmse':score})
            next(v for v in curves if v['model']==kind and v['seed']==seed and v['epoch']==cp['epoch'])['selection_rmse']=score
        chosen=core.choose_checkpoint(choices);selection.append({'model':kind,'seed':seed,'selected':chosen,'prediction_alias':'BASE_FIXED' if chosen['epoch']==0 else None,'trainable_parameters':core.parameter_count(kind)})
    frozen={'base_sha256':base_hash,'selected':selection,'selection_rule':'raw four-H endpoint macro RMSE, exact ties earlier epoch including0; no best-seed or model-family selection','base_frozen':True,'no_refit':True,'no_ensemble':True}
    dump(a.public_output/'frozen_selection.json',frozen);selection_hash=sha(a.public_output/'frozen_selection.json')
    score_fields=['stage','model','seed','horizon','status','target_scope','postprocess','scored_values','rmse','mae']
    csvout(a.public_output/'base_selection_scores.csv',base_select,score_fields);csvout(a.public_output/'selection_scores.csv',selection_rows,['epoch']+score_fields);csvout(a.public_output/'training_curves.csv',curves,curve_fields)
    y,x32,dev_det,eo=stage_inputs('PREDICT');qbase=dev_det['BASE_FIXED'];groups=core.state_groups(y[eo-1],y[eo-1]-y[eo-25]);predictions={(name,'fixed'):trajectory(q) for name,q in dev_det.items()}
    mo=core.mechanism_origins();mx32,mech_det=prepare_inputs(y,reader.d,cap,mo);mqbase=mech_det['BASE_FIXED'];mechanism={(name,'fixed'):trajectory(q) for name,q in mech_det.items()}
    for item in selection:
        kind,seed=item['model'],item['seed'];path=a.private_output/item['selected']['file']
        if sha(path)!=item['selected']['sha256']:raise ValueError('Selected checkpoint changed')
        if item['selected']['epoch']==0:f=np.zeros_like(qbase)
        else:
            net=core.InteractionNet(kind,seed).cuda(0);net.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));f,_,elapsed=infer(net,x32);inference_seconds+=elapsed;del net
        predictions[kind,str(seed)]=trajectory(qbase+a.scale*f)
        run=next(r for r in runs if r['model']==kind and r['seed']==seed);fixed40=core.mechanism_checkpoint(run);mpath=a.private_output/fixed40['file']
        if sha(mpath)!=fixed40['sha256']:raise ValueError('Fixed40 checkpoint changed')
        net=core.InteractionNet(kind,seed).cuda(0);net.load_state_dict(torch.load(mpath,map_location='cuda:0',weights_only=True));mf,_,elapsed=infer(net,mx32);inference_seconds+=elapsed;del net
        mechanism[kind,str(seed)]=trajectory(mqbase+a.scale*mf)
    mech_records=[]
    for (kind,seed),pred in mechanism.items():
        path=a.private_output/f'private_MECH_FIXED40_{kind}_{seed}.npy';np.save(path,pred);mech_records.append({'model':kind,'seed':seed,'file':path.name,'sha256':sha(path),'epoch':40 if kind in core.MODELS else None})
    dump(a.public_output/'mechanism_prediction_freeze.json',{'selection_sha256':selection_hash,'fixed_epoch':40,'origins':mo.tolist(),'not_used_for_selection':True,'predictions':mech_records})
    records=[]
    for (kind,seed),pred in predictions.items():
        for post,value in [('raw',pred),('clip_0_1',np.clip(pred,0,1))]:
            path=a.private_output/f'private_DEV_{kind}_{seed}_{post}.npy';np.save(path,value);records.append({'model':kind,'seed':seed,'postprocess':post,'file':path.name,'sha256':sha(path)})
    dump(a.public_output/'prediction_freeze.json',{'selection_sha256':selection_hash,'predictions':records,'rolling_past_observations_available':True,'evaluation_losses_not_computed':True,'each_origin_uses_only_past_inputs':True})
    for record in records:
        if sha(a.private_output/record['file'])!=record['sha256']:raise ValueError('Saved prediction changed')
    y,d,cap=reader.read('SCORE');truth=y[eo[:,None]+np.arange(12)];np.save(a.private_output/'private_DEV_truth.npy',truth)
    if sha(a.public_output/'frozen_selection.json')!=selection_hash:raise ValueError('Selection changed after labels')
    scores=[]
    for (kind,seed),pred in predictions.items():scores.extend(metric_rows(kind,seed,pred,truth,'DEV_EVAL'))
    saved_score_error=0.
    for record in records:
        saved=np.load(a.private_output/record['file'],allow_pickle=False)
        if not np.isfinite(saved).all():raise ValueError('Nonfinite saved prediction')
        for h in base.HORIZONS:
            for scope in ('terminal_H','path_1_to_H'):
                check=scoped_scores(saved[:,:h],truth[:,:h],target_scope=scope,postprocess='raw')
                original=next(r for r in scores if r['model']==record['model'] and r['seed']==record['seed'] and r['postprocess']==record['postprocess'] and r['horizon']==h and r['target_scope']==scope)
                saved_score_error=max(saved_score_error,abs(check['rmse']-original['rmse']),abs(check['mae']-original['mae']))
    if saved_score_error>1e-10:raise ValueError('Saved prediction scoring mismatch')
    cache={};cache_error=0.
    if len(c['cache_files'])!=4:raise ValueError('Four cache entries required')
    for entry in c['cache_files']:
        h,s=entry['horizon'],entry['system'];name=f'private_truth_1392_h{h}.npy' if s=='truth' else f'private_1392_h{h}_native_a1.npy'
        if entry['cut']!=1392 or h not in (3,12) or s not in ('native','truth') or entry['file']!=name:raise ValueError('Cache not whitelisted')
        root=a.truth_root if s=='truth' else a.native_root;path=(root/name).resolve()
        if path.parent!=root.resolve() or sha(path)!=entry['sha256']:raise ValueError('Cache identity failed')
        arr=np.load(path,allow_pickle=False).astype(np.float64)
        if arr.shape!=(14,h,275) or not np.isfinite(arr).all():raise ValueError('Cache shape/value failure')
        cache[h,s]=arr;receipt['stage_reads'].append({'stage':'SCORE','file':name,'kind':'whitelisted_cache_decode','sha256':entry['sha256']})
    for h in (3,12):
        cache_error=max(cache_error,float(np.max(abs(cache[h,'truth']-truth[:,:h]))));scores.extend(metric_rows('NATIVE','cached',cache[h,'native'],truth,'DEV_EVAL',(h,)))
    for h in (6,9):
        for post in ('raw','clip_0_1'):
            for scope in ('terminal_H','path_1_to_H'):scores.append({'stage':'DEV_EVAL','model':'NATIVE','seed':'cached','horizon':h,'status':'NOT_AVAILABLE','target_scope':scope,'postprocess':post,'scored_values':0,'rmse':None,'mae':None})
    csvout(a.public_output/'development_scores.csv',scores,score_fields)
    loss_reports=[];origin_rows=[];reduction_error=0.
    for seed in core.SEEDS:
        main=predictions['FIXED_PRODUCT',str(seed)]
        for reference in ('BASE_FIXED','LINEAR_REPAIR','FIXED_SEPARABLE','FIXED_CONCAT','RIDGE_FULL','NATIVE'):
            for h in ((3,12) if reference=='NATIVE' else base.HORIZONS):
                pred_ref=cache[h,'native'] if reference=='NATIVE' else predictions[reference,'fixed' if reference in core.DETERMINISTIC else str(seed)]
                for scope in ('terminal_H','path_1_to_H'):
                    yy=truth[:,:h];qr=pred_ref[:,:h];qm=main[:,:h]
                    if scope=='terminal_H':yy=yy[:,-1:];qr=qr[:,-1:];qm=qm[:,-1:]
                    for post in ('raw','clip_0_1'):
                        rb=np.clip(qr,0,1) if post=='clip_0_1' else qr;mt=np.clip(qm,0,1) if post=='clip_0_1' else qm
                        report,g1,g2=core.grouped_losses(yy,rb,mt,groups);tags={'seed':seed,'candidate':'FIXED_PRODUCT','reference':reference,'horizon':h,'target_scope':scope,'postprocess':post};loss_reports.append({**tags,**report})
                        ar=next(r for r in scores if r['model']=='FIXED_PRODUCT' and r['seed']==str(seed) and r['horizon']==h and r['target_scope']==scope and r['postprocess']==post)
                        refseed='fixed' if reference in core.DETERMINISTIC else 'cached' if reference=='NATIVE' else str(seed)
                        br=next(r for r in scores if r['model']==reference and r['seed']==refseed and r['horizon']==h and r['target_scope']==scope and r['postprocess']==post)
                        reduction_error=max(reduction_error,abs(g1.mean()-(br['mae']-ar['mae'])),abs(g2.mean()-(br['rmse']**2-ar['rmse']**2)))
                        for k,o in enumerate(eo):origin_rows.append({**tags,'origin':int(o),'mae_gain':float(g1[k].mean()),'mse_gain':float(g2[k].mean())})
    dump(a.public_output/'state_group_losses.json',loss_reports);csvout(a.public_output/'paired_origin_errors.csv',origin_rows,['seed','candidate','reference','horizon','target_scope','postprocess','origin','mae_gain','mse_gain'])
    priorpath=ROOT/'artifacts/summaries/long_history_source_ablation_v1/development_scores.csv'
    if sha(priorpath)!=c['prior_public_scores_sha256']:raise ValueError('Public anchor identity changed')
    with priorpath.open(encoding='utf-8') as f:prior=list(csv.DictReader(f))
    base_error=0.
    for row in [r for r in base_select+scores if r['model']=='RIDGE_FULL']:
        old=next(r for r in prior if r['stage']==row['stage'] and r['model']=='OD_LONG' and int(r['horizon'])==row['horizon'] and r['target_scope']==row['target_scope'] and r['postprocess']==row['postprocess'])
        base_error=max(base_error,abs(row['rmse']-float(old['rmse'])),abs(row['mae']-float(old['mae'])))
    if base_error>1e-8 or cache_error>1e-7 or reduction_error>1e-10:raise ValueError('Scoring/anchor consistency failed')
    summaries=[]
    for support,hs in [('FOUR_H',base.HORIZONS),('H3_H12_COMMON',(3,12))]:
        for kind,seed in list(predictions)+[('NATIVE','cached')]:
            if kind=='NATIVE' and support=='FOUR_H':continue
            for post in ('raw','clip_0_1'):
                for scope in ('terminal_H','path_1_to_H'):
                    rr=[r for r in scores if r['model']==kind and r['seed']==seed and r['horizon'] in hs and r['postprocess']==post and r['target_scope']==scope and r['status']=='AVAILABLE']
                    if len(rr)!=len(hs):raise ValueError('Mixed support')
                    summaries.append({'support':support,'model':kind,'seed':seed,'postprocess':post,'target_scope':scope,'macro_rmse':float(np.mean([r['rmse'] for r in rr])),'macro_mae':float(np.mean([r['mae'] for r in rr]))})
    means=[]
    for kind in core.MODELS:
        for support in ('FOUR_H','H3_H12_COMMON'):
            for post in ('raw','clip_0_1'):
                for scope in ('terminal_H','path_1_to_H'):
                    rr=[r for r in summaries if r['model']==kind and r['support']==support and r['postprocess']==post and r['target_scope']==scope];row={'model':kind,'support':support,'postprocess':post,'target_scope':scope,'seed_count':3}
                    for metric in ('rmse','mae'):
                        vv=[r['macro_'+metric] for r in rr];row.update({metric+'_mean':float(np.mean(vv)),metric+'_min':min(vv),metric+'_max':max(vv),metric+'_std_ddof1':float(np.std(vv,ddof=1))})
                    means.append(row)
    stability=[]
    for kind in core.MODELS:
        for h in base.HORIZONS:
            for post in ('raw','clip_0_1'):
                q0=predictions['BASE_FIXED','fixed'][:,h-1]
                if post=='clip_0_1':q0=np.clip(q0,0,1)
                corr=[]
                for seed in core.SEEDS:
                    q=predictions[kind,str(seed)][:,h-1]
                    if post=='clip_0_1':q=np.clip(q,0,1)
                    corr.append(q-q0)
                stability.append({'model':kind,'horizon':h,'postprocess':post,**core.relative_stability(corr)})
    mech_truth=y[mo[:,None]+np.arange(12)];mech_scores=[];mech_error=0.;mech_alignment_error=0.
    for record in mech_records:
        path=a.private_output/record['file']
        if sha(path)!=record['sha256']:raise ValueError('Mechanism prediction changed')
        saved=np.load(path,allow_pickle=False);memory=mechanism[record['model'],record['seed']]
        if not np.isfinite(saved).all():raise ValueError('Nonfinite mechanism prediction')
        for h in base.HORIZONS:
            row=scoped_scores(saved[:,:h],mech_truth[:,:h],target_scope='terminal_H',postprocess='raw');check=scoped_scores(memory[:,:h],mech_truth[:,:h],target_scope='terminal_H',postprocess='raw')
            mech_error=max(mech_error,abs(row['rmse']-check['rmse']),abs(row['mae']-check['mae']))
            qref=mechanism['BASE_FIXED','fixed'][:,h-1];yy=mech_truth[:,h-1];align=core.alignment(yy,qref,saved[:,h-1])
            baseline_mse=float(np.mean((yy-qref)**2));mech_alignment_error=max(mech_alignment_error,abs(align['mse_gain_G']-(baseline_mse-row['rmse']**2)))
            mech_scores.append({'stage':'MECH_FIXED40','model':record['model'],'seed':record['seed'],'horizon':h,'status':'AVAILABLE',**row,**align})
    if len(mech_scores)!=48 or mech_error>1e-10 or mech_alignment_error>1e-10:raise ValueError('Mechanism scope/scoring failure')
    csvout(a.public_output/'mechanism_fixed40_scores.csv',mech_scores,score_fields+['alignment_A','correction_energy_B','mse_gain_G','identity_error','zero_correction'])
    mech_stability=[]
    for kind in core.MODELS:
        for h in base.HORIZONS:
            corr=[mechanism[kind,str(seed)][:,h-1]-mechanism['BASE_FIXED','fixed'][:,h-1] for seed in core.SEEDS]
            mech_stability.append({'model':kind,'horizon':h,'postprocess':'raw','scope':'MECH_FIXED40',**core.relative_stability(corr)})
    harm_reports=[];harm_error=0.
    for row in scores:
        if row['status']!='AVAILABLE' or row['model']=='BASE_FIXED':continue
        h=row['horizon'];q=cache[h,'native'] if row['model']=='NATIVE' else predictions[row['model'],row['seed']][:,:h]
        qb=predictions['BASE_FIXED','fixed'][:,:h];yy=truth[:,:h]
        if row['target_scope']=='terminal_H':q=q[:,-1:];qb=qb[:,-1:];yy=yy[:,-1:]
        if row['postprocess']=='clip_0_1':q=np.clip(q,0,1);qb=np.clip(qb,0,1)
        delta=abs(yy-q)-abs(yy-qb);harm=float(np.maximum(delta,0).mean());benefit=float(np.maximum(-delta,0).mean())
        br=next(v for v in scores if v['model']=='BASE_FIXED' and v['horizon']==h and v['target_scope']==row['target_scope'] and v['postprocess']==row['postprocess'])
        harm_error=max(harm_error,abs(harm-benefit-(row['mae']-br['mae'])))
        harm_reports.append({k:row[k] for k in ('model','seed','horizon','target_scope','postprocess')}|{'harm':harm,'benefit':benefit,'mae_change':harm-benefit,'mean_absolute_correction':float(abs(q-qb).mean())})
    if harm_error>1e-10:raise ValueError('Harm-benefit identity failed')
    dump(a.public_output/'harm_benefit_report.json',{'rows':harm_reports,'identity_error':harm_error,'same_postprocess_reference':True})
    per_h_metrics=[]
    for kind in core.MODELS:
        for h in base.HORIZONS:
            for scope in ('terminal_H','path_1_to_H'):
                for post in ('raw','clip_0_1'):
                    rr=[v for v in scores if v['model']==kind and v['horizon']==h and v['target_scope']==scope and v['postprocess']==post]
                    row={'model':kind,'horizon':h,'target_scope':scope,'postprocess':post}
                    for metric in ('rmse','mae'):
                        vv=[v[metric] for v in rr];row.update({metric+'_mean':float(np.mean(vv)),metric+'_min':min(vv),metric+'_max':max(vv),metric+'_std_ddof1':float(np.std(vv,ddof=1))})
                    per_h_metrics.append(row)
    dump(a.public_output/'stability_summary.json',{'relative_disagreement':stability,'mechanism_relative_disagreement':mech_stability,'seed_metric_summary':means,'per_h_seed_metrics':per_h_metrics,'standard_deviation_ddof':1,'interpretation':'descriptive, no ensemble, no inferential certification'})
    dump(a.public_output/'model_summary.json',{'per_run':summaries,'seed_metric_means':means,'training_runs':runs,'uncertainty':'three preregistered seeds reused from the prior round; new scheme comparison and one previously exposed DEV window; descriptive only','primary':'FIXED_PRODUCT; raw four-H endpoint RMSE','scientific_automatic_admission':False,'sota_claim':False})
    dump(a.public_output/'verification.json',{'saved_prediction_score_error':saved_score_error,'base_anchor_error':base_error,'cache_truth_error':cache_error,'direct_loss_reduction_error':float(reduction_error),'selected_epoch_zero_aliases':sum(v['selected']['epoch']==0 for v in selection),'frozen_base_unchanged':sha(bp)==frozen_base['ridge']['sha256'] and sha(sp)==frozen_base['transform']['sha256'],'prediction_file_count':len(records),'mechanism_prediction_files':len(mech_records),'mechanism_saved_score_error':mech_error,'mechanism_alignment_error':mech_alignment_error,'linear_normal_equation_relative_error':lmeta['normal_equation_relative_error']})
    if receipt['optimizer_steps']!=13320 or len(scores)!=208 or receipt['frozen_epoch40_fit_evaluations']!=9:raise ValueError('Incomplete formal execution')
    receipt.update(status='FIXED_REFERENCE_CORRECTION_COMPLETE_REVIEW_REQUIRED',mechanism_probe_is_selection_independent=True,independent_statistical_confirmation=False,fit_rows=149325,fit_origins=543,full_base_fit_rows=240900,full_base_fit_origins=876,mechanism_score_rows=48,mechanism_prediction_files=12,select_origins=14,dev_origins=14,selection_score_rows=len(selection_rows),valid_dev_score_rows=200,missing_native_rows=8,loss_comparison_cells=len(loss_reports),paired_origin_rows=len(origin_rows),cache_decodes=4,inference_seconds_synchronized=inference_seconds,neural_training_seconds_synchronized=sum(r['training_seconds_synchronized'] for r in runs),gpu_peak_allocated_bytes=max(r['gpu_peak_allocated_bytes'] for r in runs),gpu_peak_reserved_bytes=max(r['gpu_peak_reserved_bytes'] for r in runs),scientific_automatic_admission=False,sota_claim=False)

def main():
    p=argparse.ArgumentParser()
    for n in ('config','data-root','native-root','truth-root','private-output','public-output','claim'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'));setup_cuda()
    if c['protocol_id']!='FIXED_REFERENCE_CORRECTION_V1_20260913' or c['limits']!=base.LIMITS or c['models']!=list(core.MODELS) or c['seeds']!=list(core.SEEDS):raise ValueError('Protocol mismatch')
    for name,h in c['code_sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:raise ValueError('Code changed after freeze')
    if a.private_output.exists() or a.public_output.exists():raise FileExistsError('New outputs required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();a.claim.parent.mkdir(parents=True,exist_ok=True)
    with a.claim.open('x',encoding='utf-8') as f:json.dump({'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'frozen_code_commit':head,'status':'CONSUMED_ON_START'},f)
    a.private_output.mkdir(parents=True);a.public_output.mkdir(parents=True);sys.meta_path.insert(0,NoFoundation());clock=time.perf_counter()
    receipt={'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'frozen_code_commit':head,'stage_reads':[],'base_ridge_fits':0,'neural_training_runs_started':0,'neural_training_runs_completed':0,'optimizer_steps':0,'executions':1,'device':torch.cuda.get_device_name(0),'cuda_device':'cuda:0','torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,'numpy_version':np.__version__,'cpu':platform.processor(),'cpu_threads':2,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'TF32':False,'AMP':False,'torch_compile':False,'new_foundation_inference':0,'alpha_searches':0,'old_private_model_reads':0,'automatic_next_round':False,'automation_enabled':False,'human_review_required':True}
    try:
        with threadpool_limits(limits=2):execute(a,c,receipt)
    except Exception as e:receipt.update(status='FIXED_REFERENCE_CORRECTION_BLOCKED',error_type=type(e).__name__,error=str(e));raise
    finally:
        torch.cuda.synchronize();receipt.update(total_seconds=time.perf_counter()-clock,process_peak_memory='NOT_MEASURED',gpu_memory_scope='PyTorch allocated/reserved, not total process/device memory');dump(a.public_output/'execution_receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
