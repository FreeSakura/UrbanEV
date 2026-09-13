"""Thirteen generalized-ridge fits with fixed selection and six evaluation identities."""
import argparse,csv,hashlib,json,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from run_long_history_source_ablation import Reader,Forbidden,dump,csvout,sha,trajectory
from urbanev_forecast import short_state as base
from urbanev_forecast import history_source as history
from urbanev_forecast import lag_regularization as core
from urbanev_forecast.benchmark_metrics import scoped_scores

def metric_rows(pred,truth,meta):
    return [{**meta,'horizon':h,**scoped_scores(pred[:,:h],truth[:,:h],target_scope=scope,postprocess=post)} for h in base.HORIZONS for post in ('raw','clip_0_1') for scope in ('terminal_H','path_1_to_H')]

def execute(a,c,receipt):
    reader=Reader(c,a.data_root,receipt['stage_reads']);budget=core.Budget()
    y,d,cap=reader.read('FIT');oo=base.origins('FIT');p=y[oo-1].reshape(-1);res=base.targets(y,oo)-p[:,None]
    x=base.features(y,d,cap,oo,'LONG_OD');stats=base.fit_transform(x);z=base.transform(x,stats);del x
    g,b=history.sufficient_statistics(z,res);r2=np.mean(res**2,axis=0);del z,res
    stpath=a.private_output/'private_transform.npz';np.savez(stpath,**stats)
    active,dpos,l,qs,manifest=core.penalties(stats['inactive']);dump(a.public_output/'penalty_manifest.json',manifest)
    all_models={};meta={};diagnostics=[]
    plan=[('BASE_OD','BASE',0.,np.zeros_like(qs[core.FAMILIES[0]]))]+[(core.fit_id(f,gamma),f,gamma,qs[f]) for f in core.FAMILIES for gamma in core.GAMMAS[1:]]
    for id,family,gamma,q in plan:
        budget.start(id);receipt['fits_started']=len(budget.ids);clock=time.perf_counter();beta,details=core.solve(g,b,active,q,gamma,r2,l,dpos)
        path=a.private_output/f'private_{id}.npy';np.save(path,beta);all_models[id]=beta
        meta[id]={'family':family,'gamma':gamma,'file':path.name,'sha256':sha(path),'fit_seconds':time.perf_counter()-clock,'edf':details['edf'],'normal_equation_relative_residual':details['normal_equation_relative_residual'],'coefficient_count':details['coefficient_count'],'training_objective_total':details['training_objective_total']}
        for output in details['per_output']:diagnostics.append({'fit_id':id,'family':family,'gamma':gamma,'edf':details['edf'],**output})
        receipt['fits_completed']=len(all_models)
    budget.close();del g,b
    dump(a.public_output/'frozen_models.json',{'models':meta,'transform':{'file':stpath.name,'sha256':sha(stpath)},'training_closed':True,'unique_fits':13});models_hash=sha(a.public_output/'frozen_models.json')
    csvout(a.public_output/'regularization_diagnostics.csv',diagnostics,['fit_id','family','gamma','edf','future_hour','data_mse','base_penalty','actual_Q_penalty','extra_penalty','training_objective','D_second_difference_norm','D_coefficient_norm'])
    def predict_ids(y,d,cap,stage,ids):
        if not budget.closed or sha(a.public_output/'frozen_models.json')!=models_hash:raise RuntimeError('Models are not frozen')
        o=base.origins(stage);z=base.transform(base.features(y,d,cap,o,'LONG_OD'),stats);anchor=y[o-1].reshape(-1);predictions={}
        for id in dict.fromkeys(ids):
            if sha(a.private_output/meta[id]['file'])!=meta[id]['sha256']:raise ValueError('Weight identity changed')
            beta=all_models[id];pred=trajectory(anchor[:,None]+beta[0]+z@beta[1:])
            if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
            predictions[id]=pred
        return predictions,o
    y,d,cap=reader.read('SELECT');sp,so=predict_ids(y,d,cap,'SELECT',all_models);sy=y[so[:,None]+np.arange(12)]
    selection_rows=[]
    for id,pred in sp.items():selection_rows.extend(metric_rows(pred,sy,{'fit_id':id,'family':meta[id]['family'],'gamma':meta[id]['gamma']}))
    if len(selection_rows)!=208:raise ValueError('Incomplete unique-parameter selection scores')
    scores={id:float(np.mean([r['rmse'] for r in selection_rows if r['fit_id']==id and r['postprocess']=='raw' and r['target_scope']=='terminal_H'])) for id in all_models}
    chosen,mapping=core.select(scores)
    csvout(a.public_output/'selection_scores.csv',selection_rows,['fit_id','family','gamma','horizon','target_scope','postprocess','scored_values','rmse','mae'])
    frozen={'gamma_by_family':chosen,'system_to_fit_id':mapping,'selection_rmse_by_fit':scores,'shared_zero_fit':'BASE_OD','selection_rule':'raw four-H endpoint macro RMSE, exact ties smaller gamma; no MAE or effect-size filter','models_sha256':models_hash,'same_gamma_scrambled':chosen['TIME_SMOOTH_D'],'unique_dev_parameter_sets':len(set(mapping.values())),'no_refit':True}
    dump(a.public_output/'frozen_selection.json',frozen);selection_hash=sha(a.public_output/'frozen_selection.json')
    y,d,cap=reader.read('PREDICT');ep,eo=predict_ids(y,d,cap,'PREDICT',mapping.values());records=[]
    for id,pred in ep.items():
        for post,array in [('raw',pred),('clip_0_1',np.clip(pred,0,1))]:
            path=a.private_output/f'private_DEV_{id}_{post}.npy';np.save(path,array);records.append({'fit_id':id,'postprocess':post,'file':path.name,'sha256':sha(path)})
    dump(a.public_output/'prediction_freeze.json',{'selection_sha256':selection_hash,'system_to_fit_id':mapping,'predictions':records,'before_last_target_extension':True,'rolling_past_eval_observations_available':True})
    for rec in records:
        if sha(a.private_output/rec['file'])!=rec['sha256']:raise ValueError('Frozen prediction changed')
    y,d,cap=reader.read('SCORE');truth=y[eo[:,None]+np.arange(12)];np.save(a.private_output/'private_DEV_truth.npy',truth)
    if sha(a.public_output/'frozen_selection.json')!=selection_hash:raise ValueError('Selection changed after evaluation labels')
    dev_unique={id:metric_rows(pred,truth,{}) for id,pred in ep.items()};rows=[]
    for system,id in mapping.items():
        aliases=[s for s,v in mapping.items() if v==id]
        for row in dev_unique[id]:rows.append({'system':system,'fit_id':id,'alias_systems':'|'.join(aliases),**row})
    csvout(a.public_output/'development_scores.csv',rows,['system','fit_id','alias_systems','horizon','target_scope','postprocess','scored_values','rmse','mae'])
    reports=[];origin_rows=[];max_reduction=0.;max_identity=0.
    for reference in ('BASE_OD','GLOBAL_RIDGE_SELECTED','D_NORM_SELECTED','SCRAMBLED_SELECTED','SCRAMBLED_MATCHED'):
        for h in base.HORIZONS:
            for scope in ('terminal_H','path_1_to_H'):
                yy=truth[:,:h];baseline=ep['BASE_OD'][:,:h];ref=ep[mapping[reference]][:,:h];cand=ep[mapping['TIME_SELECTED']][:,:h]
                if scope=='terminal_H':yy=yy[:,-1:];baseline=baseline[:,-1:];ref=ref[:,-1:];cand=cand[:,-1:]
                for post in ('raw','clip_0_1'):
                    qb=np.clip(ref,0,1) if post=='clip_0_1' else ref;qt=np.clip(cand,0,1) if post=='clip_0_1' else cand
                    report,g1,g2=core.grouped_loss(yy,baseline,qb,qt);meta_row={'candidate':'TIME_SELECTED','reference':reference,'horizon':h,'target_scope':scope,'postprocess':post}
                    reports.append({**meta_row,**report});max_identity=max(max_identity,report['identity_error'],report['group_sum_error'])
                    tr=next(r for r in rows if r['system']=='TIME_SELECTED' and r['horizon']==h and r['target_scope']==scope and r['postprocess']==post)
                    br=next(r for r in rows if r['system']==reference and r['horizon']==h and r['target_scope']==scope and r['postprocess']==post)
                    max_reduction=max(max_reduction,abs(g2.mean()-(br['rmse']**2-tr['rmse']**2)),abs(g1.mean()-(br['mae']-tr['mae'])))
                    for j,o in enumerate(eo):origin_rows.append({**meta_row,'origin':int(o),'mae_gain':float(g1[j].mean()),'mse_gain':float(g2[j].mean())})
    dump(a.public_output/'paired_loss_report.json',reports);csvout(a.public_output/'paired_origin_errors.csv',origin_rows,['candidate','reference','horizon','target_scope','postprocess','origin','mae_gain','mse_gain'])
    priorpath=ROOT/'artifacts/summaries/long_history_source_ablation_v1/development_scores.csv';prior_summary=ROOT/'artifacts/summaries/long_history_source_ablation_v1/summary.json'
    if sha(priorpath)!=c['prior_scores_sha256'] or sha(prior_summary)!=c['prior_summary_sha256']:raise ValueError('Public anchor identity mismatch')
    with priorpath.open(encoding='utf-8') as f:prior=list(csv.DictReader(f))
    previous=json.loads(prior_summary.read_text())['rows'];anchor_error=0.
    for stage,rr in [('SELECT',[r for r in selection_rows if r['fit_id']=='BASE_OD']),('DEV_EVAL',dev_unique['BASE_OD'])]:
        for row in rr:
            old=next(r for r in prior if r['stage']==stage and r['model']=='OD_LONG' and int(r['horizon'])==row['horizon'] and r['target_scope']==row['target_scope'] and r['postprocess']==row['postprocess'])
            anchor_error=max(anchor_error,abs(row['rmse']-float(old['rmse'])),abs(row['mae']-float(old['mae'])))
        for scope in ('terminal_H','path_1_to_H'):
            for post in ('raw','clip_0_1'):
                selected_rows=[r for r in rr if r['target_scope']==scope and r['postprocess']==post];old=next(r for r in previous if r['stage']==stage and r['model']=='OD_LONG' and r['target_scope']==scope and r['postprocess']==post)
                anchor_error=max(anchor_error,abs(np.mean([r['rmse'] for r in selected_rows])-old['macro_rmse']),abs(np.mean([r['mae'] for r in selected_rows])-old['macro_mae']))
    if anchor_error>1e-8 or max_reduction>1e-10 or max_identity>1e-10:raise ValueError('Anchor or loss identity verification failed')
    summary=[]
    for stage in ('SELECT','DEV_EVAL'):
        for system,id in mapping.items():
            rr=[r for r in selection_rows if r['fit_id']==id] if stage=='SELECT' else dev_unique[id]
            for post in ('raw','clip_0_1'):
                for scope in ('terminal_H','path_1_to_H'):
                    subset=[r for r in rr if r['postprocess']==post and r['target_scope']==scope];summary.append({'stage':stage,'system':system,'fit_id':id,'postprocess':post,'target_scope':scope,'macro_rmse':float(np.mean([r['rmse'] for r in subset])),'macro_mae':float(np.mean([r['mae'] for r in subset]))})
    dump(a.public_output/'summary.json',{'rows':summary,'gamma_by_family':chosen,'primary_method':'TIME_SELECTED','primary_metric':'raw four-H endpoint macro RMSE','scientific_automatic_admission':False,'sota_claim':False,'uncertainty':'exposed development windows, 1056 used for gamma selection; no independent confirmation or p-values'})
    dump(a.public_output/'verification.json',{'anchor_max_error':float(anchor_error),'direct_loss_reduction_max_error':float(max_reduction),'loss_identity_or_bin_sum_max_error':max_identity,'normal_equation_max_relative_residual':max(m['normal_equation_relative_residual'] for m in meta.values()),'unique_selection_parameter_sets':len(all_models),'dev_unique_parameter_sets':len(ep),'dev_logical_identities':len(mapping),'old_private_arrays_read':0})
    receipt.update(status='DURATION_LAG_REGULARIZATION_COMPLETE_REVIEW_REQUIRED',fit_origins=876,fit_rows=240900,select_origins=14,dev_origins=14,selection_score_rows=len(selection_rows),dev_logical_score_rows=len(rows),dev_unique_score_rows=sum(len(v) for v in dev_unique.values()),unique_dev_parameter_sets=len(ep),loss_contrast_cells=len(reports),paired_origin_rows=len(origin_rows),scientific_automatic_admission=False,sota_claim=False)

def main():
    p=argparse.ArgumentParser()
    for n in ('config','data-root','private-output','public-output','claim'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'))
    if c['protocol_id']!='DURATION_LAG_REGULARIZATION_V1_20260913' or c['limits']!=base.LIMITS or c['gammas']!=list(core.GAMMAS):raise ValueError('Frozen protocol mismatch')
    for name,h in c['code_sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:raise ValueError('Code changed after freeze')
    if a.private_output.exists() or a.public_output.exists():raise FileExistsError('New outputs required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();a.claim.parent.mkdir(parents=True,exist_ok=True)
    with a.claim.open('x',encoding='utf-8') as f:json.dump({'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'frozen_code_commit':head,'status':'CONSUMED_ON_START'},f)
    a.private_output.mkdir(parents=True);a.public_output.mkdir(parents=True);sys.meta_path.insert(0,Forbidden());clock=time.perf_counter()
    receipt={'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'frozen_code_commit':head,'stage_reads':[],'fits_started':0,'fits_completed':0,'executions':1,'device':'CPU','threads':2,'dtype':'float64','numpy_version':np.__version__,'cpu':platform.processor(),'neural_training_runs':0,'new_foundation_inference':0,'alpha_searches':0,'gamma_grid_search':'four fixed candidates per family, shared gamma0, SELECT only','old_private_array_reads':0,'automation_enabled':False,'automatic_next_round':False,'human_review_required':True}
    try:
        with threadpool_limits(limits=2):execute(a,c,receipt)
    except Exception as e:receipt.update(status='DURATION_LAG_REGULARIZATION_BLOCKED',error_type=type(e).__name__,error=str(e));raise
    finally:
        receipt.update(elapsed_seconds=time.perf_counter()-clock,peak_memory='NOT_MEASURED');dump(a.public_output/'execution_receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
