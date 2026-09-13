"""Locked historical replay and a matched development comparison; not a six-fold test."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,csv,hashlib,json,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
from run_long_history_source_ablation import Reader,dump,csvout,sha
from run_conditional_od_interaction import setup_cuda,NoFoundation
from urbanev_forecast import short_state as base,history_source as hs,comparison_core as core
from urbanev_forecast.benchmark_metrics import scoped_scores

FIELDS=['cohort','model','seed','information_track','horizon','status','target_scope','postprocess','scored_values','rmse','mae']

def track(kind):return 'GLOBAL_O_168' if kind in ('TIMEXER_GLOBAL_O','NATIVE') else 'LOCAL_O_168' if kind=='RIDGE_O' else 'LOCAL_OD_168'
def trajectory(a,n):return a.reshape(n,275,12).transpose(0,2,1)
def metric_rows(cohort,kind,seed,q,y,horizons=(3,6,9,12),information=None):
    return [{'cohort':cohort,'model':kind,'seed':str(seed),'information_track':information or track(kind),'horizon':h,'status':'AVAILABLE',**scoped_scores(q[:,:h],y[:,:h],target_scope=sc,postprocess=po)} for h in horizons for sc in ('terminal_H','path_1_to_H') for po in ('raw','clip_0_1')]

def forward(net,x,kind):
    if kind.startswith('TIMEXER'):return net(x)
    return net(x.reshape(-1,341)).reshape(x.shape[0],275,12).transpose(1,2)

def predict(net,x,kind):
    outputs=[];net.eval();torch.cuda.synchronize();started=time.perf_counter()
    with torch.no_grad():
        for start in range(0,len(x),2):
            xx=torch.as_tensor(x[start:start+2],dtype=torch.float32,device='cuda:0');q=forward(net,xx,kind)
            if not torch.isfinite(q).all():raise ValueError('Nonfinite model output')
            outputs.append(q.cpu().numpy().astype(np.float64))
    torch.cuda.synchronize();return np.concatenate(outputs),time.perf_counter()-started

def build_net(kind,seed,author_class):
    return (core.TimeXerAdapter(kind,seed,author_class) if kind.startswith('TIMEXER') else core.InteractionNet(kind,seed)).cuda(0)

def train_one(kind,seed,x,y,a,budget,author_class,curves,receipt):
    budget.start_neural(kind,seed);receipt['neural_runs_started']=len(budget.neural);random.seed(seed);np.random.seed(seed)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter();net=build_net(kind,seed,author_class)
    opt=torch.optim.AdamW([{'params':[p for p in net.parameters() if p.ndim>=2],'weight_decay':1e-4},{'params':[p for p in net.parameters() if p.ndim<2],'weight_decay':0.}],lr=.001,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    xgpu=torch.as_tensor(x,dtype=torch.float32,device='cuda:0');ygpu=torch.as_tensor(y,dtype=torch.float32,device='cuda:0');checkpoints=[];steps=0
    def save(epoch):
        p=a.private_output/f'private_{kind}_{seed}_e{epoch}.pt';torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},p);checkpoints.append({'epoch':epoch,'file':p.name,'sha256':sha(p)})
    save(0)
    for epoch in range(1,41):
        net.train();lr=core.learning_rate(epoch)
        for g in opt.param_groups:g['lr']=lr
        total=0.
        for indices in core.effective_batches(len(x),seed,epoch):
            opt.zero_grad(set_to_none=True)
            for micro,weight in core.microbatches(indices):
                q=forward(net,xgpu[micro],kind);loss=((q-ygpu[micro])**2).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                (loss*weight).backward();total+=float(loss.detach())*len(micro)
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1;receipt['optimizer_steps']+=1
            if torch.stack([(~torch.isfinite(p)).any() for p in net.parameters()]).any():raise ValueError('Nonfinite parameter')
        curves.append({'model':kind,'seed':seed,'epoch':epoch,'lr':lr,'training_mse':total/len(x),'optimizer_steps':steps,'scope':'raw-unit during-traversal mean over origins,regions,12outputs'})
        if epoch in core.CHECKPOINTS:save(epoch);print(json.dumps({'model':kind,'seed':seed,'epoch':epoch,'mse':total/len(x)}),flush=True)
    torch.cuda.synchronize()
    if steps!=2200:raise ValueError('Step budget mismatch')
    result={'model':kind,'seed':seed,'parameters':sum(p.numel() for p in net.parameters()),'steps':steps,'seconds':time.perf_counter()-started,'gpu_peak_allocated':torch.cuda.max_memory_allocated(),'gpu_peak_reserved':torch.cuda.max_memory_reserved(),'checkpoints':checkpoints}
    del net,opt,xgpu,ygpu;torch.cuda.empty_cache();return result

def execute(a,c,receipt):
    csvout(a.public_output/'candidate_coverage.csv',c['candidate_coverage'],list(c['candidate_coverage'][0]))
    csvout(a.public_output/'baseline_sources.csv',c['baseline_sources'],list(c['baseline_sources'][0]))
    dump(a.public_output/'comparison_manifest.json',{'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'author_source':c['author_source'],'archive_whitelist':c['archive_whitelist'],'archive_contracts':c['archive_contracts'],'same_training_contract_for_new_core_only':True,'formal_six_fold_in_this_execution':False})
    reader=Reader(c,a.data_root,receipt['stage_reads']);budget=core.Budget();author_class=core.load_author_core(a.author_root,c['author_source']['files_sha256_lf'])
    y,d,cap=reader.read('FIT');oo=base.origins('FIT');x=base.features(y,d,cap,oo,'LONG_OD');stats=base.fit_transform(x);z=base.transform(x,stats);anchor=y[oo-1].reshape(-1);target=base.targets(y,oo)
    statpath=a.private_output/'private_fit_stats.npz';np.savez(statpath,**stats);stat_hash=sha(statpath);g,b=hs.sufficient_statistics(z,target-anchor[:,None]);ridges={};metadata={}
    masks={'RIDGE_O':list(range(168))+list(range(336,341)),'RIDGE_OD':list(range(341))}
    for name,mask in masks.items():
        budget.ridge(name);receipt['ridge_fits']=len(budget.ridges);started=time.perf_counter();beta,check=hs.solve_mask(g,b,mask);ridges[name]=beta
        p=a.private_output/f'private_{name}.npy';np.save(p,beta);metadata[name]={'file':p.name,'sha256':sha(p),'fit_seconds':time.perf_counter()-started,**check}
    qbase=hs.predict(z,anchor,masks['RIDGE_OD'],ridges['RIDGE_OD']);raw32=x.astype(np.float32).reshape(876,275,341);z32=z.astype(np.float32).reshape(876,275,341)
    target32=trajectory(target.astype(np.float32),876);residual32=trajectory((target-qbase).astype(np.float32),876);del x,z,g,b,target,qbase
    dump(a.public_output/'frozen_ridges.json',{'models':metadata,'fit_stats_sha256':stat_hash,'masks':masks,'fit_origins':[169,1045],'shared_available_training_pool':True})
    runs=[];curves=[]
    for kind in core.MODELS:
        for seed in core.SEEDS:
            is_tf=kind.startswith('TIMEXER');runs.append(train_one(kind,seed,raw32 if is_tf else z32,target32 if is_tf else residual32,a,budget,author_class,curves,receipt));receipt['neural_runs_completed']=len(runs)
    budget.close();del raw32,z32,target32,residual32
    csvout(a.public_output/'training_curves.csv',curves,['model','seed','epoch','lr','training_mse','optimizer_steps','scope'])
    def stage(stage):
        if not budget.closed or sha(statpath)!=stat_hash:raise ValueError('Training/statistics not frozen')
        for value in metadata.values():
            if sha(a.private_output/value['file'])!=value['sha256']:raise ValueError('Ridge changed')
        yy,dd,cc=reader.read(stage);orig=base.origins(stage);raw=base.features(yy,dd,cc,orig,'LONG_OD');zz=base.transform(raw,stats);p=yy[orig-1].reshape(-1)
        det={name:trajectory(hs.predict(zz,p,masks[name],ridges[name]),14) for name in masks}
        return yy,raw.astype(np.float32).reshape(14,275,341),zz.astype(np.float32).reshape(14,275,341),det,orig
    y,raw,z,det,orig=stage('SELECT');truth=trajectory(base.targets(y,orig),14);selection=[];selection_rows=[];inference_seconds=0.
    for run in runs:
        choices=[];kind=run['model'];seed=run['seed']
        for cp in run['checkpoints']:
            p=a.private_output/cp['file']
            if sha(p)!=cp['sha256']:raise ValueError('Checkpoint changed')
            net=build_net(kind,seed,author_class);net.load_state_dict(torch.load(p,map_location='cuda:0',weights_only=True));q,seconds=predict(net,raw if kind.startswith('TIMEXER') else z,kind);inference_seconds+=seconds;del net
            if not kind.startswith('TIMEXER'):q=det['RIDGE_OD']+q
            rr=metric_rows('NEW_CORE_SELECT',kind,seed,q,truth);score=float(np.mean([v['rmse'] for v in rr if v['target_scope']=='terminal_H' and v['postprocess']=='raw']));selection_rows.extend([{'epoch':cp['epoch'],**v} for v in rr]);choices.append({**cp,'selection_rmse':score})
        chosen=core.choose_checkpoint(choices);selection.append({'model':kind,'seed':seed,'selected':chosen,'alias':'RIDGE_OD' if chosen['epoch']==0 and not kind.startswith('TIMEXER') else None,'untrained_initialization_selected':chosen['epoch']==0 and kind.startswith('TIMEXER')})
    dump(a.public_output/'frozen_selection.json',{'models':selection,'no_best_seed':True,'no_ensemble':True,'selection_rule':'raw four-H endpoint macroRMSE, exact ties earliest epoch'});frozen_selection_hash=sha(a.public_output/'frozen_selection.json')
    csvout(a.public_output/'selection_scores.csv',selection_rows,['epoch']+FIELDS)
    csvout(a.public_output/'ridge_selection_scores.csv',[row for name,q in det.items() for row in metric_rows('NEW_CORE_SELECT',name,'fixed',q,truth)],FIELDS)
    y,raw,z,det,orig=stage('PREDICT');preds={(name,'fixed'):q for name,q in det.items()}
    for item in selection:
        kind,seed=item['model'],item['seed'];cp=item['selected'];p=a.private_output/cp['file']
        if sha(p)!=cp['sha256']:raise ValueError('Locked checkpoint changed')
        net=build_net(kind,seed,author_class);net.load_state_dict(torch.load(p,map_location='cuda:0',weights_only=True));q,seconds=predict(net,raw if kind.startswith('TIMEXER') else z,kind);inference_seconds+=seconds;del net
        preds[kind,str(seed)]=q if kind.startswith('TIMEXER') else det['RIDGE_OD']+q
    records=[]
    for (kind,seed),q in preds.items():
        p=a.private_output/f'private_DEV_{kind}_{seed}_raw.npy';np.save(p,q);records.append({'model':kind,'seed':seed,'file':p.name,'sha256':sha(p)})
    dump(a.public_output/'prediction_freeze.json',{'selection_sha256':frozen_selection_hash,'predictions':records,'evaluation_losses_not_computed':True,'each_origin_uses_only_past_inputs':True})
    y,d,cap=reader.read('SCORE');truth=y[orig[:,None]+np.arange(12)];scores=[];save_error=0.
    for record in records:
        p=a.private_output/record['file']
        if sha(p)!=record['sha256']:raise ValueError('Prediction file changed')
        q=np.load(p,allow_pickle=False).astype(np.float64);rr=metric_rows('NEW_MATCHED_CORE',record['model'],record['seed'],q,truth);scores.extend(rr)
        original=metric_rows('NEW_MATCHED_CORE',record['model'],record['seed'],preds[record['model'],record['seed']],truth)
        save_error=max(save_error,max(abs(v[k]-w[k]) for v,w in zip(rr,original) for k in ('rmse','mae')))
    caches={};truth_error=0.
    for entry in c['cache_files']:
        h=entry['horizon'];system=entry['system'];name=f'private_truth_1392_h{h}.npy' if system=='truth' else f'private_1392_h{h}_native_a1.npy'
        if h not in (3,12) or system not in ('truth','native') or entry['cut']!=1392 or entry['file']!=name:raise ValueError('Cache whitelist violation')
        root=a.truth_root if system=='truth' else a.native_root;p=root/name
        if sha(p)!=entry['sha256']:raise ValueError('Cache identity mismatch')
        q=np.load(p,allow_pickle=False).astype(np.float64)
        if q.shape!=(14,h,275) or not np.isfinite(q).all():raise ValueError('Cache shape mismatch')
        caches[h,system]=q;receipt['cache_decodes']+=1
    for h in (3,12):truth_error=max(truth_error,float(abs(caches[h,'truth']-truth[:,:h]).max()));scores.extend(metric_rows('CACHED_NATIVE_REFERENCE','NATIVE','cached',caches[h,'native'],truth,(h,)))
    for h in (6,9):
        for sc in ('terminal_H','path_1_to_H'):
            for po in ('raw','clip_0_1'):scores.append({'cohort':'CACHED_NATIVE_REFERENCE','model':'NATIVE','seed':'cached','information_track':'GLOBAL_O_168','horizon':h,'status':'NOT_AVAILABLE','target_scope':sc,'postprocess':po,'scored_values':0,'rmse':None,'mae':None})
    csvout(a.public_output/'matched_core_scores.csv',scores,FIELDS)
    with (ROOT/'artifacts/summaries/long_history_source_ablation_v1/development_scores.csv').open(encoding='utf-8') as f:old_anchor=list(csv.DictReader(f))
    anchor_error=0.
    for row in scores:
        if row['model']!='RIDGE_OD':continue
        old=next(v for v in old_anchor if v['stage']=='DEV_EVAL' and v['model']=='OD_LONG' and int(v['horizon'])==row['horizon'] and v['target_scope']==row['target_scope'] and v['postprocess']==row['postprocess'])
        anchor_error=max(anchor_error,*(abs(row[k]-float(old[k])) for k in ('rmse','mae')))
    if anchor_error>1e-8:raise ValueError('Common ridge anchor mismatch')
    roots=json.loads(a.archive_roots.read_text());archive=[];failures=[];dedup={};physical_decodes=0
    for entry in c['archive_whitelist']:
        role=entry['root_role'];p=(Path(roots[role])/entry['file']).resolve()
        if p.parent!=Path(roots[role]).resolve():raise ValueError('Archive path escaped role root')
        if not p.is_file() or sha(p)!=entry['sha256']:
            failures.append({'cohort':role,'file':entry['file'],'status':'MISSING_ARTIFACT' if not p.is_file() else 'IDENTITY_MISMATCH'});continue
        if entry['sha256'] not in dedup:
            q=np.load(p,allow_pickle=False).astype(np.float64)
            if list(q.shape)!=entry['shape'] or not np.isfinite(q).all():raise ValueError('Archive shape/value mismatch')
            dedup[entry['sha256']]=q;physical_decodes+=1
        q=dedup[entry['sha256']]
        for name in entry['logical_models']:
            rr=metric_rows(role,name,entry['seed'],q,truth,information='HISTORICAL_CONTRACT_SEE_MANIFEST');archive.extend([{**row,'prediction_sha256':entry['sha256']} for row in rr])
    csvout(a.public_output/'archive_common_scores.csv',archive,FIELDS+['prediction_sha256'])
    historical_score_error=0.;historical_checked=0
    for cohort in sorted(set(row['cohort'] for row in archive)):
        with (ROOT/'artifacts/summaries'/cohort/'development_scores.csv').open(encoding='utf-8-sig') as f:published=list(csv.DictReader(f))
        lookup={((v.get('model') or v.get('system')),str(v.get('seed') or 'fixed'),int(v['horizon']),v['target_scope'],v['postprocess']):v for v in published if v.get('stage','DEV_EVAL')=='DEV_EVAL' and v.get('status','AVAILABLE')=='AVAILABLE'}
        for row in [v for v in archive if v['cohort']==cohort]:
            old=lookup.get((row['model'],row['seed'],row['horizon'],row['target_scope'],row['postprocess']))
            if old:
                historical_checked+=1;historical_score_error=max(historical_score_error,*(abs(row[k]-float(old[k])) for k in ('rmse','mae')))
    if historical_score_error>1e-10:raise ValueError('Historical published score mismatch')
    summary=[]
    for kind,seed in dict.fromkeys((r['model'],r['seed']) for r in scores):
        for support,hs_ in [('FOUR_H',(3,6,9,12)),('H3_H12',(3,12))]:
            for po in ('raw','clip_0_1'):
                rr=[r for r in scores if r['model']==kind and r['seed']==seed and r['horizon'] in hs_ and r['postprocess']==po and r['target_scope']=='terminal_H' and r['status']=='AVAILABLE']
                if len(rr)==len(hs_):summary.append({'model':kind,'seed':seed,'support':support,'postprocess':po,'information_track':track(kind),'macro_rmse':float(np.mean([r['rmse'] for r in rr])),'macro_mae':float(np.mean([r['mae'] for r in rr]))})
    dump(a.public_output/'model_summary.json',{'per_instance':summary,'training_runs':runs,'no_ensemble':True,'three_seeds_descriptive_only':True})
    if save_error>1e-10 or truth_error>1e-7 or sha(a.public_output/'frozen_selection.json')!=frozen_selection_hash:raise ValueError('Score or freeze consistency failed')
    if receipt['optimizer_steps']!=33000 or len(selection_rows)!=1200 or len(scores)!=288:raise ValueError('Core comparison incomplete')
    actual_sources=[{**row,'this_execution':'COMPLETED_AUTHOR_CORE_ADAPTED_DEVELOPMENT' if row['method'].startswith('TimeXer LOCAL') else row['this_execution']} for row in c['baseline_sources']]
    csvout(a.public_output/'baseline_sources.csv',actual_sources,list(actual_sources[0]))
    actual_coverage=[{**row,'common_origin_status':'LOCKED_REPLAY_COMPLETED' if row['common_origin_status']=='REGISTERED_FOR_LOCKED_REPLAY' and row['id'] not in {v['cohort'] for v in failures} else row['common_origin_status']} for row in c['candidate_coverage']]
    csvout(a.public_output/'candidate_coverage.csv',actual_coverage,list(actual_coverage[0]))
    dump(a.public_output/'verification.json',{'saved_prediction_score_error':save_error,'native_truth_error':truth_error,'ridge_od_anchor_error':anchor_error,'historical_published_score_error':historical_score_error,'historical_published_rows_checked':historical_checked,'historical_failures':failures,'historical_raw_files_registered':len(c['archive_whitelist']),'historical_unique_arrays_decoded':physical_decodes,'new_raw_prediction_files':len(records)})
    dump(a.public_output/'coverage_and_limits.json',{'historical_registered':len(c['archive_whitelist']),'historical_missing_or_mismatch':failures,'core_models_planned':list(core.MODELS),'core_models_completed':list(core.MODELS),'formal_six_fold_complete':False,'latest_all_baselines_reproduced':False,'new_foundation_calls':0,'scope':'DEVELOPMENT_ONLY_NO_FORMAL_SIX_FOLD','same_training_protocol_applies_to_new_core_only':True,'protected_data_opened':False})
    receipt.update(status='COMPARISON_PARTIAL_WITH_EXPLICIT_GAPS_REVIEW_REQUIRED',core_complete=True,formal_six_fold_complete=False,core_valid_score_rows=280,core_missing_score_rows=8,historical_score_rows=len(archive),historical_unique_arrays_decoded=physical_decodes,inference_seconds=inference_seconds,sota_claim=False)

def main():
    p=argparse.ArgumentParser()
    for key in ('config','data-root','author-root','archive-roots','native-root','truth-root','private-output','public-output','claim'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text());setup_cuda()
    if c['protocol_id']!='COMPREHENSIVE_DEVELOPMENT_COMPARISON_V1_20260913' or c['models']!=list(core.MODELS) or c['seeds']!=list(core.SEEDS) or len(c['archive_whitelist'])>128:raise ValueError('Protocol mismatch')
    for name,expected in c['code_sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:raise ValueError('Frozen source changed')
    if a.private_output.exists() or a.public_output.exists():raise FileExistsError('Fresh outputs required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();a.claim.parent.mkdir(parents=True,exist_ok=True)
    with a.claim.open('x',encoding='utf-8') as f:json.dump({'protocol_id':c['protocol_id'],'status':'CONSUMED_ON_START','config_sha256':sha(a.config),'frozen_code_commit':head},f)
    a.private_output.mkdir(parents=True);a.public_output.mkdir(parents=True);sys.meta_path.insert(0,NoFoundation());started=time.perf_counter()
    receipt={'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'frozen_code_commit':head,'stage_reads':[],'ridge_fits':0,'neural_runs_started':0,'neural_runs_completed':0,'optimizer_steps':0,'cache_decodes':0,'foundation_inference':0,'automatic_next_round':False,'automation_enabled':False,'human_review_required':True,'device':torch.cuda.get_device_name(0),'torch_version':str(torch.__version__),'cpu_threads':2}
    try:
        with threadpool_limits(limits=2):execute(a,c,receipt)
    except Exception as e:receipt.update(status='COMPARISON_BLOCKED',error_type=type(e).__name__,error=str(e));raise
    finally:receipt['total_seconds']=time.perf_counter()-started;dump(a.public_output/'execution_receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
