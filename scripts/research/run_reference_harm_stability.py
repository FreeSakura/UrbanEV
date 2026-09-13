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
from urbanev_forecast import reference_harm as core
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
    if sum(p.numel() for p in net.parameters())!=9108:raise ValueError('Parameter count mismatch')
    opt=torch.optim.AdamW([{'params':[p for p in net.parameters() if p.ndim>=2],'weight_decay':1e-4},{'params':[p for p in net.parameters() if p.ndim<2],'weight_decay':0.}],lr=.001,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    x=torch.as_tensor(x32,dtype=torch.float32,device='cuda:0');y=torch.as_tensor(residual,dtype=torch.float32,device='cuda:0');checkpoints=[];steps=0
    def save(epoch):
        path=a.private_output/f'private_{kind}_s{seed}_e{epoch}.pt';torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},path);checkpoints.append({'epoch':epoch,'file':path.name,'sha256':sha(path)})
    save(0)
    curves.append({'model':kind,'seed':seed,'epoch':0,'lr':0.,'training_mse':float(np.mean(residual.astype(np.float64)**2)),'optimizer_steps':0,'selection_rmse':None,'harm':0.,'benefit':0.,'correction_l1':0.,'objective':float(np.mean(residual.astype(np.float64)**2)),'log_scope':'epoch0 fixed parameters'})
    for epoch in range(1,41):
        lr=core.learning_rate(epoch)
        for group in opt.param_groups:group['lr']=lr
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(x32));totals={k:0. for k in ('mse','harm','benefit','correction_l1','objective')};net.train()
        for start in range(0,len(order),4096):
            ix=torch.as_tensor(order[start:start+4096],device='cuda:0');opt.zero_grad(set_to_none=True);f=net(x[ix]);parts=core.loss_parts(y[ix],f,a.tau,kind);loss=parts['objective']
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            if torch.stack([(~torch.isfinite(p)).any() for p in net.parameters()]).any():raise ValueError('Nonfinite parameter')
            steps+=1;receipt['optimizer_steps']=receipt.get('optimizer_steps',0)+1
            for key,value in parts.items():totals[key]+=float(value.detach())*len(ix)
        curves.append({'model':kind,'seed':seed,'epoch':epoch,'lr':lr,'training_mse':totals['mse']/len(x32),'optimizer_steps':steps,'selection_rmse':None,**{k:totals[k]/len(x32) for k in ('harm','benefit','correction_l1','objective')},'log_scope':'during-traversal weighted batch means; not frozen FIT objective'})
        if epoch in core.CHECKPOINTS:save(epoch);print(json.dumps({'model':kind,'seed':seed,'epoch':epoch,'mse':totals['mse']/len(x32)}),flush=True)
    torch.cuda.synchronize()
    result={'model':kind,'seed':seed,'parameters':9108,'optimizer_steps':steps,'training_seconds_synchronized':time.perf_counter()-clock,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(0),'checkpoints':checkpoints}
    if steps!=2360:raise ValueError('Training step budget mismatch')
    del net,opt,x,y;torch.cuda.empty_cache();return result

def execute(a,c,receipt):
    reader=Reader(c,a.data_root,receipt['stage_reads']);budget=core.Budget();curves=[];runs=[]
    y,d,cap=reader.read('FIT');oo=base.origins('FIT');target=base.targets(y,oo);anchor=y[oo-1].reshape(-1)
    x=base.features(y,d,cap,oo,'LONG_OD');stats=base.fit_transform(x);z=base.transform(x,stats);del x
    budget.start_ridge();receipt['base_ridge_fits']=1;clock=time.perf_counter();beta=base.fit_ridge(z,target-anchor[:,None]);receipt['ridge_fit_seconds']=time.perf_counter()-clock
    q0=base.ridge_predict(z,anchor,beta);residual=(target-q0).astype(np.float32);scale=core.fit_scale(residual);a.tau=scale['tau_float32'];dump(a.public_output/'loss_scale.json',scale);scale_hash=sha(a.public_output/'loss_scale.json');x32=z.astype(np.float32);del z,target,q0
    bp=a.private_output/'private_base_ridge.npy';sp=a.private_output/'private_transform.npz';np.save(bp,beta);np.savez(sp,**stats)
    frozen_base={'ridge':{'file':bp.name,'sha256':sha(bp),'coefficients':int(beta.size)},'transform':{'file':sp.name,'sha256':sha(sp)},'training_only':True,'fixed_for_all_neural_runs':True}
    dump(a.public_output/'frozen_base.json',frozen_base);base_hash=sha(a.public_output/'frozen_base.json')
    for kind in core.MODELS:
        for seed in core.SEEDS:
            runs.append(train_one(kind,seed,x32,residual,a,budget,curves,receipt));receipt['neural_training_runs_completed']=len(runs)
    budget.close();del x32,residual
    curve_fields=['model','seed','epoch','lr','training_mse','optimizer_steps','selection_rmse','harm','benefit','correction_l1','objective','log_scope'];csvout(a.public_output/'training_curves.csv',curves,curve_fields)
    def stage_inputs(stage):
        if not budget.closed:raise RuntimeError('Training not closed')
        if sha(a.public_output/'loss_scale.json')!=scale_hash:raise ValueError('FIT scale changed')
        if sha(bp)!=frozen_base['ridge']['sha256'] or sha(sp)!=frozen_base['transform']['sha256']:raise ValueError('Base parameters changed')
        yy,dd,cc=reader.read(stage);o=base.origins(stage);zz=base.transform(base.features(yy,dd,cc,o,'LONG_OD'),stats);qq=base.ridge_predict(zz,yy[o-1].reshape(-1),beta)
        return yy,zz.astype(np.float32),qq,o
    y,x32,qbase,so=stage_inputs('SELECT');sy=trajectory(base.targets(y,so));base_select=metric_rows('BASE_OD','fixed',trajectory(qbase),sy,'SELECT');selection_rows=[];selection=[];inference_seconds=0.
    for run in runs:
        kind,seed=run['model'],run['seed'];choices=[]
        for cp in run['checkpoints']:
            path=a.private_output/cp['file']
            if sha(path)!=cp['sha256']:raise ValueError('Checkpoint changed')
            if cp['epoch']==0:pred=trajectory(qbase.copy())
            else:
                net=core.InteractionNet(kind,seed).cuda(0);net.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));f,_,elapsed=infer(net,x32);inference_seconds+=elapsed;pred=trajectory(qbase+f);del net
            rr=metric_rows(kind,seed,pred,sy,'SELECT');score=float(np.mean([v['rmse'] for v in rr if v['postprocess']=='raw' and v['target_scope']=='terminal_H']))
            selection_rows.extend([{'epoch':cp['epoch'],**v} for v in rr]);choices.append({**cp,'selection_rmse':score})
            next(v for v in curves if v['model']==kind and v['seed']==seed and v['epoch']==cp['epoch'])['selection_rmse']=score
        chosen=core.choose_checkpoint(choices);selection.append({'model':kind,'seed':seed,'selected':chosen,'prediction_alias':'BASE_OD' if chosen['epoch']==0 else None,'trainable_parameters':9108})
    frozen={'base_sha256':base_hash,'selected':selection,'selection_rule':'raw four-H endpoint macro RMSE, exact ties earlier epoch including0; no best-seed or model-family selection','base_frozen':True,'no_refit':True,'no_ensemble':True}
    dump(a.public_output/'frozen_selection.json',frozen);selection_hash=sha(a.public_output/'frozen_selection.json')
    score_fields=['stage','model','seed','horizon','status','target_scope','postprocess','scored_values','rmse','mae']
    csvout(a.public_output/'base_selection_scores.csv',base_select,score_fields);csvout(a.public_output/'selection_scores.csv',selection_rows,['epoch']+score_fields);csvout(a.public_output/'training_curves.csv',curves,curve_fields)
    y,x32,qbase,eo=stage_inputs('PREDICT');groups=core.state_groups(y[eo-1],y[eo-1]-y[eo-25]);predictions={('BASE_OD','fixed'):trajectory(qbase)};activity=[]
    for item in selection:
        kind,seed=item['model'],item['seed'];path=a.private_output/item['selected']['file']
        if sha(path)!=item['selected']['sha256']:raise ValueError('Selected checkpoint changed')
        if item['selected']['epoch']==0:f=np.zeros_like(qbase);cross=np.zeros_like(qbase) if kind=='PRODUCT_POSREG' else None
        else:
            net=core.InteractionNet(kind,seed).cuda(0);net.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));f,cross,elapsed=infer(net,x32,kind=='PRODUCT_POSREG');inference_seconds+=elapsed;del net
        predictions[kind,str(seed)]=trajectory(qbase+f)
        if cross is not None:
            for h in base.HORIZONS:activity.append({'seed':seed,'horizon':h,'interaction_mean_square':float(np.mean(cross[:,h-1]**2)),'total_extra_mean_square':float(np.mean(f[:,h-1]**2)),'interpretation':'module activity, not causal importance or necessity'})
    records=[]
    for (kind,seed),pred in predictions.items():
        for post,value in [('raw',pred),('clip_0_1',np.clip(pred,0,1))]:
            path=a.private_output/f'private_DEV_{kind}_{seed}_{post}.npy';np.save(path,value);records.append({'model':kind,'seed':seed,'postprocess':post,'file':path.name,'sha256':sha(path)})
    dump(a.public_output/'prediction_freeze.json',{'selection_sha256':selection_hash,'predictions':records,'rolling_past_observations_available':True,'final_target_prefix_not_yet_decoded':True});dump(a.public_output/'interaction_activity.json',activity)
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
        main=predictions['PRODUCT_POSREG',str(seed)]
        for reference in ('BASE_OD','PRODUCT_MSE','PRODUCT_MIX','PRODUCT_L1','CONCAT_MSE','CONCAT_POSREG','NATIVE'):
            for h in ((3,12) if reference=='NATIVE' else base.HORIZONS):
                pred_ref=cache[h,'native'] if reference=='NATIVE' else predictions[reference,'fixed' if reference=='BASE_OD' else str(seed)]
                for scope in ('terminal_H','path_1_to_H'):
                    yy=truth[:,:h];qr=pred_ref[:,:h];qm=main[:,:h]
                    if scope=='terminal_H':yy=yy[:,-1:];qr=qr[:,-1:];qm=qm[:,-1:]
                    for post in ('raw','clip_0_1'):
                        rb=np.clip(qr,0,1) if post=='clip_0_1' else qr;mt=np.clip(qm,0,1) if post=='clip_0_1' else qm
                        report,g1,g2=core.grouped_losses(yy,rb,mt,groups);tags={'seed':seed,'candidate':'PRODUCT_POSREG','reference':reference,'horizon':h,'target_scope':scope,'postprocess':post};loss_reports.append({**tags,**report})
                        ar=next(r for r in scores if r['model']=='PRODUCT_POSREG' and r['seed']==str(seed) and r['horizon']==h and r['target_scope']==scope and r['postprocess']==post)
                        refseed='fixed' if reference=='BASE_OD' else 'cached' if reference=='NATIVE' else str(seed)
                        br=next(r for r in scores if r['model']==reference and r['seed']==refseed and r['horizon']==h and r['target_scope']==scope and r['postprocess']==post)
                        reduction_error=max(reduction_error,abs(g1.mean()-(br['mae']-ar['mae'])),abs(g2.mean()-(br['rmse']**2-ar['rmse']**2)))
                        for k,o in enumerate(eo):origin_rows.append({**tags,'origin':int(o),'mae_gain':float(g1[k].mean()),'mse_gain':float(g2[k].mean())})
    dump(a.public_output/'state_group_losses.json',loss_reports);csvout(a.public_output/'paired_origin_errors.csv',origin_rows,['seed','candidate','reference','horizon','target_scope','postprocess','origin','mae_gain','mse_gain'])
    priorpath=ROOT/'artifacts/summaries/long_history_source_ablation_v1/development_scores.csv'
    if sha(priorpath)!=c['prior_public_scores_sha256']:raise ValueError('Public anchor identity changed')
    with priorpath.open(encoding='utf-8') as f:prior=list(csv.DictReader(f))
    base_error=0.
    for row in base_select+[r for r in scores if r['model']=='BASE_OD']:
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
                q0=predictions['BASE_OD','fixed'][:,h-1]
                if post=='clip_0_1':q0=np.clip(q0,0,1)
                corr=[]
                for seed in core.SEEDS:
                    q=predictions[kind,str(seed)][:,h-1]
                    if post=='clip_0_1':q=np.clip(q,0,1)
                    corr.append(q-q0)
                stability.append({'model':kind,'horizon':h,'postprocess':post,**core.relative_stability(corr)})
    factorial=[]
    for seed in core.SEEDS:
        for h in base.HORIZONS:
            for scope in ('terminal_H','path_1_to_H'):
                for post in ('raw','clip_0_1'):
                    def score(kind,key):return next(r[key] for r in scores if r['model']==kind and r['seed']==str(seed) and r['horizon']==h and r['target_scope']==scope and r['postprocess']==post)
                    row={'seed':seed,'horizon':h,'target_scope':scope,'postprocess':post}
                    for metric in ('rmse','mae'):row['interaction_'+metric]=(score('PRODUCT_MSE',metric)-score('PRODUCT_POSREG',metric))-(score('CONCAT_MSE',metric)-score('CONCAT_POSREG',metric))
                    factorial.append(row)
    harm_reports=[];harm_error=0.
    for row in scores:
        if row['status']!='AVAILABLE' or row['model']=='BASE_OD':continue
        h=row['horizon'];q=cache[h,'native'] if row['model']=='NATIVE' else predictions[row['model'],row['seed']][:,:h]
        qb=predictions['BASE_OD','fixed'][:,:h];yy=truth[:,:h]
        if row['target_scope']=='terminal_H':q=q[:,-1:];qb=qb[:,-1:];yy=yy[:,-1:]
        if row['postprocess']=='clip_0_1':q=np.clip(q,0,1);qb=np.clip(qb,0,1)
        delta=abs(yy-q)-abs(yy-qb);harm=float(np.maximum(delta,0).mean());benefit=float(np.maximum(-delta,0).mean())
        br=next(v for v in scores if v['model']=='BASE_OD' and v['horizon']==h and v['target_scope']==row['target_scope'] and v['postprocess']==row['postprocess'])
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
    dump(a.public_output/'stability_summary.json',{'relative_disagreement':stability,'paired_factorial':factorial,'seed_metric_summary':means,'per_h_seed_metrics':per_h_metrics,'standard_deviation_ddof':1,'interpretation':'descriptive, no ensemble, no inferential certification'})
    dump(a.public_output/'model_summary.json',{'per_run':summaries,'seed_metric_means':means,'training_runs':runs,'uncertainty':'three preregistered new seeds and one previously exposed DEV window; descriptive only','primary':'PRODUCT_POSREG; raw four-H endpoint RMSE','scientific_automatic_admission':False,'sota_claim':False})
    dump(a.public_output/'verification.json',{'saved_prediction_score_error':saved_score_error,'base_anchor_error':base_error,'cache_truth_error':cache_error,'direct_loss_reduction_error':float(reduction_error),'selected_epoch_zero_aliases':sum(v['selected']['epoch']==0 for v in selection),'frozen_base_unchanged':sha(bp)==frozen_base['ridge']['sha256'] and sha(sp)==frozen_base['transform']['sha256'],'prediction_file_count':len(records)})
    if receipt['optimizer_steps']!=42480 or len(scores)!=320:raise ValueError('Incomplete formal execution')
    receipt.update(status='REFERENCE_HARM_STABILITY_COMPLETE_REVIEW_REQUIRED',fit_rows=240900,fit_origins=876,select_origins=14,dev_origins=14,selection_score_rows=len(selection_rows),valid_dev_score_rows=312,missing_native_rows=8,loss_comparison_cells=len(loss_reports),paired_origin_rows=len(origin_rows),cache_decodes=4,inference_seconds_synchronized=inference_seconds,neural_training_seconds_synchronized=sum(r['training_seconds_synchronized'] for r in runs),gpu_peak_allocated_bytes=max(r['gpu_peak_allocated_bytes'] for r in runs),gpu_peak_reserved_bytes=max(r['gpu_peak_reserved_bytes'] for r in runs),scientific_automatic_admission=False,sota_claim=False)

def main():
    p=argparse.ArgumentParser()
    for n in ('config','data-root','native-root','truth-root','private-output','public-output','claim'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'));setup_cuda()
    if c['protocol_id']!='REFERENCE_HARM_STABILITY_V1_20260913' or c['limits']!=base.LIMITS or c['models']!=list(core.MODELS) or c['seeds']!=list(core.SEEDS) or c['kappa']!=core.KAPPA:raise ValueError('Protocol mismatch')
    for name,h in c['code_sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:raise ValueError('Code changed after freeze')
    if a.private_output.exists() or a.public_output.exists():raise FileExistsError('New outputs required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();a.claim.parent.mkdir(parents=True,exist_ok=True)
    with a.claim.open('x',encoding='utf-8') as f:json.dump({'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'frozen_code_commit':head,'status':'CONSUMED_ON_START'},f)
    a.private_output.mkdir(parents=True);a.public_output.mkdir(parents=True);sys.meta_path.insert(0,NoFoundation());clock=time.perf_counter()
    receipt={'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'frozen_code_commit':head,'stage_reads':[],'base_ridge_fits':0,'neural_training_runs_started':0,'neural_training_runs_completed':0,'optimizer_steps':0,'executions':1,'device':torch.cuda.get_device_name(0),'cuda_device':'cuda:0','torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,'numpy_version':np.__version__,'cpu':platform.processor(),'cpu_threads':2,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'TF32':False,'AMP':False,'torch_compile':False,'new_foundation_inference':0,'alpha_searches':0,'old_private_model_reads':0,'automatic_next_round':False,'automation_enabled':False,'human_review_required':True}
    try:
        with threadpool_limits(limits=2):execute(a,c,receipt)
    except Exception as e:receipt.update(status='REFERENCE_HARM_STABILITY_BLOCKED',error_type=type(e).__name__,error=str(e));raise
    finally:
        torch.cuda.synchronize();receipt.update(total_seconds=time.perf_counter()-clock,process_peak_memory='NOT_MEASURED',gpu_memory_scope='PyTorch allocated/reserved, not total process/device memory');dump(a.public_output/'execution_receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
