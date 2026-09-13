"""Eight fixed deterministic history ablations, stage-bounded real data access."""
import argparse,csv,hashlib,importlib.abc,io,json,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast import short_state as base
from urbanev_forecast import history_source as hs
from urbanev_forecast.bounded_predictor_csv import prefix_bytes,parse_prefix
from urbanev_forecast.comparability_bridge import zone_capacities
from urbanev_forecast.benchmark_metrics import scoped_scores

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def csvout(p,rows,fields):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def trajectory(a):return a.reshape(14,275,12).transpose(0,2,1)

class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','chronos','timesfm','transformers') or fullname=='urbanev_forecast.foundation':raise RuntimeError('Neural/foundation calls forbidden')

class Reader:
    def __init__(self,c,root,ledger):self.c=c;self.root=root;self.ledger=ledger;self.position=0;self.cap=None;self.y=None;self.d=None
    def read(self,stage):
        if stage!=('FIT','SELECT','PREDICT','SCORE')[self.position]:raise RuntimeError('Read stage out of order')
        for name,n in base.LIMITS[stage].items():
            blob=prefix_bytes(self.root,name,n,n,set(base.LIMITS[stage]));expected=self.c['prefixes'][stage][name]['sha256']
            if digest(blob)!=expected:raise ValueError('Frozen prefix mismatch')
            if self.cap is None:
                self.cols=next(csv.reader([blob.splitlines()[0].decode('utf-8-sig')]))[1:]
                if len(self.cols)!=275 or digest('\n'.join(self.cols).encode())!=self.c['column_order_sha256']:raise ValueError('Column identity mismatch')
                p=(self.root/'inf.csv').resolve()
                if p.parent!=self.root.resolve() or sha(p)!=self.c['static_info_sha256']:raise ValueError('Static identity mismatch')
                self.cap=zone_capacities(pd.read_csv(p,usecols=['station_id','TAZID','charge_count'],dtype={'TAZID':str}),self.cols)
                self.ledger.append({'stage':stage,'file':'inf.csv','columns':['station_id','TAZID','charge_count'],'rule':'groupby TAZID sum'})
            raw,record=parse_prefix(blob,n,self.cols,expected,nonnegative=True)
            if name=='occupancy.csv':self.y=(raw.astype(np.float32)/self.cap.astype(np.float32)[None,:]).astype(np.float64);values=self.y
            else:self.d=raw/self.cap[None,:];values=self.d
            if ((values<0)|(values>1)).any():raise ValueError('Invalid normalized rate; no repair')
            self.ledger.append({'stage':stage,'file':name,**record})
        self.position+=1
        return self.y,self.d,self.cap

class FitBudget:
    def __init__(self):self.names=[];self.closed=False
    def start(self,name):
        if self.closed or name not in hs.MASKS or name in self.names:raise RuntimeError('Fit whitelist/closure violation')
        self.names.append(name)
    def close(self):
        if set(self.names)!=set(hs.MASKS):raise RuntimeError('Required fits missing')
        self.closed=True

def execute(a,c,receipt):
    reader=Reader(c,a.data_root,receipt['stage_reads']);budget=FitBudget()
    y,d,cap=reader.read('FIT');oo=base.origins('FIT');anchor=y[oo-1].reshape(-1);target=base.targets(y,oo)
    x=base.features(y,d,cap,oo,'LONG_OD');stats=base.fit_transform(x);z=base.transform(x,stats);del x
    statpath=a.private_output/'private_mother_transform.npz';np.savez(statpath,**stats)
    g,b=hs.sufficient_statistics(z,target-anchor[:,None]);del z
    models={};model_meta={};masks={}
    for name,mask in hs.MASKS.items():
        budget.start(name);receipt['ridge_fits_started']=len(budget.names);clock=time.perf_counter()
        beta,check=hs.solve_mask(g,b,mask);models[name]=beta
        path=a.private_output/f'private_model_{name}.npy';np.save(path,beta)
        model_meta[name]={'file':path.name,'sha256':sha(path),'fit_seconds':time.perf_counter()-clock,**check}
        masks[name]={'mother_feature_indices':mask,'nominal_columns':len(mask),'unique_column_identities':len(set(mask)),'zeroed_columns':int(stats['inactive'][mask].sum()),'effective_df':check['effective_df'],'coefficient_count':check['coefficient_count'],'anchor_reconstruction':name in ('S','OD_LONG')}
        receipt['ridge_fits_completed']=len(models)
    budget.close();del g,b,target
    frozen={'models':model_meta,'transform':{'file':statpath.name,'sha256':sha(statpath)},'all_eight_frozen':True,'selection_action':'NONE_FIXED_ABLATIONS','lambda':.01,'normalization':'1/N summed over twelve outputs; unpenalized intercept','config_sha256':sha(a.config)}
    dump(a.public_output/'frozen_models.json',frozen);frozen_hash=sha(a.public_output/'frozen_models.json');dump(a.public_output/'feature_masks.json',masks)
    rows=[];paired=[];contributions=[];factorials=[];prediction_records=[];max_reduction_error=0.;max_identity_error=0.
    def predict_stage(y,d,cap,stage):
        if not budget.closed:raise RuntimeError('Models must be frozen before later predictions')
        if sha(statpath)!=frozen['transform']['sha256'] or sha(a.public_output/'frozen_models.json')!=frozen_hash:raise ValueError('Frozen transform/models changed')
        origins=base.origins(stage);zz=base.transform(base.features(y,d,cap,origins,'LONG_OD'),stats);p=y[origins-1].reshape(-1);predictions={}
        for name,mask in hs.MASKS.items():
            path=a.private_output/model_meta[name]['file']
            if sha(path)!=model_meta[name]['sha256']:raise ValueError('Weight identity changed')
            raw=trajectory(hs.predict(zz,p,mask,models[name]))
            if not np.isfinite(raw).all():raise ValueError('Nonfinite prediction')
            predictions[name,'raw']=raw;predictions[name,'clip_0_1']=np.clip(raw,0,1)
        for (name,post),pred in predictions.items():
            path=a.private_output/f'private_{stage}_{name}_{post}.npy';np.save(path,pred)
            prediction_records.append({'stage':stage,'model':name,'postprocess':post,'file':path.name,'sha256':sha(path)})
        return predictions,origins
    def score_stage(predictions,truth,origins,stage):
        nonlocal max_reduction_error,max_identity_error
        stage_rows=[]
        for (name,post),pred in predictions.items():
            for h in base.HORIZONS:
                for scope in ('terminal_H','path_1_to_H'):
                    score=scoped_scores(pred[:,:h],truth[:,:h],target_scope=scope,postprocess=post)
                    row={'stage':stage,'model':name,'horizon':h,**score};stage_rows.append(row);rows.append(row)
        for post in ('raw','clip_0_1'):
            mse={name:float(np.mean([v['rmse']**2 for v in stage_rows if v['model']==name and v['postprocess']==post and v['target_scope']=='terminal_H'])) for name in hs.MASKS}
            attribution=hs.factorial(mse);max_identity_error=max(max_identity_error,attribution['path_identity_max_error'])
            factorials.append({'stage':stage,'postprocess':post,'quantity':'equal mean of four endpoint MSEs, not square of macro RMSE','mse':mse,'gains':attribution})
            for candidate,reference in hs.CONTRASTS:
                for h in base.HORIZONS:
                    for scope in ('terminal_H','path_1_to_H'):
                        yy=truth[:,:h];qb=predictions[reference,post][:,:h];qc=predictions[candidate,post][:,:h]
                        if scope=='terminal_H':yy=yy[:,-1:];qb=qb[:,-1:];qc=qc[:,-1:]
                        report,g1,g2=hs.loss_attribution(yy,qb,qc);max_identity_error=max(max_identity_error,report['loss_identity_error'])
                        meta={'stage':stage,'candidate':candidate,'reference':reference,'horizon':h,'target_scope':scope,'postprocess':post}
                        contributions.append({**meta,**report})
                        for k,o in enumerate(origins):paired.append({**meta,'origin':int(o),'mse_gain':float(g2[k].mean()),'mae_gain':float(g1[k].mean())})
                        cr=next(v for v in stage_rows if v['model']==candidate and v['horizon']==h and v['target_scope']==scope and v['postprocess']==post)
                        br=next(v for v in stage_rows if v['model']==reference and v['horizon']==h and v['target_scope']==scope and v['postprocess']==post)
                        max_reduction_error=max(max_reduction_error,abs((br['rmse']**2-cr['rmse']**2)-report['mse']['net_mean_gain']),abs((br['mae']-cr['mae'])-report['mae']['net_mean_gain']))
    y,d,cap=reader.read('SELECT');sp,so=predict_stage(y,d,cap,'SELECT');score_stage(sp,y[so[:,None]+np.arange(12)],so,'SELECT')
    # No selection or refit after inspecting SELECT; every model proceeds.
    y,d,cap=reader.read('PREDICT');ep,eo=predict_stage(y,d,cap,'PREDICT')
    dump(a.public_output/'prediction_freeze.json',{'frozen_models_sha256':frozen_hash,'predictions':prediction_records,'dev_predictions_before_final_score_prefix':16,'rolling_history_available':True})
    for rec in prediction_records:
        if sha(a.private_output/rec['file'])!=rec['sha256']:raise ValueError('Saved prediction changed')
    y,d,cap=reader.read('SCORE');truth=y[eo[:,None]+np.arange(12)];np.save(a.private_output/'private_dev_truth.npy',truth);score_stage(ep,truth,eo,'DEV_EVAL')
    if sha(a.public_output/'frozen_models.json')!=frozen_hash:raise ValueError('Models changed after scores')
    prior_path=ROOT/'artifacts/summaries/short_state_relaxation_v1/development_scores.csv'
    if sha(prior_path)!=c['prior_public_scores_sha256']:raise ValueError('Public anchor score identity changed')
    with prior_path.open(encoding='utf-8') as f:prior=list(csv.DictReader(f))
    anchor_error=0.
    for row in rows:
        if row['stage']!='DEV_EVAL' or row['model'] not in ('S','OD_LONG'):continue
        oldname='RIDGE_SHORT_OD' if row['model']=='S' else 'RIDGE_LONG_OD'
        old=next(v for v in prior if v['model']==oldname and v['postprocess']==row['postprocess'] and int(v['horizon'])==row['horizon'] and v['target_scope']==row['target_scope'])
        anchor_error=max(anchor_error,abs(float(old['rmse'])-row['rmse']),abs(float(old['mae'])-row['mae']))
    if anchor_error>1e-8 or max_reduction_error>1e-10 or max_identity_error>1e-10:raise ValueError('Anchor or metric identity verification failed')
    if len(rows)!=256 or len(models)!=8:raise ValueError('Incomplete fixed comparison')
    csvout(a.public_output/'development_scores.csv',rows,['stage','model','horizon','target_scope','postprocess','scored_values','rmse','mae'])
    csvout(a.public_output/'paired_origin_errors.csv',paired,['stage','candidate','reference','horizon','target_scope','postprocess','origin','mse_gain','mae_gain'])
    dump(a.public_output/'factorial_attribution.json',{'results':factorials,'interpretation':'Positive empirical MSE gain means improvement in these fitted models; not ideal information value or causal attribution'})
    dump(a.public_output/'loss_contributions.json',contributions)
    summary=[]
    for stage in ('SELECT','DEV_EVAL'):
        for name in hs.MASKS:
            for post in ('raw','clip_0_1'):
                for scope in ('terminal_H','path_1_to_H'):
                    rr=[v for v in rows if (v['stage'],v['model'],v['postprocess'],v['target_scope'])==(stage,name,post,scope)]
                    summary.append({'stage':stage,'model':name,'postprocess':post,'target_scope':scope,'macro_rmse':float(np.mean([v['rmse'] for v in rr])),'macro_mae':float(np.mean([v['mae'] for v in rr]))})
    dump(a.public_output/'summary.json',{'rows':summary,'primary':'OD_LONG vs O_LONG; raw four-H endpoint macro RMSE','selection_action':'NONE_FIXED_ABLATIONS','uncertainty':'Descriptive exposed-development results; no independent confirmation or significance claim','scientific_automatic_admission':False,'sota_claim':False})
    dump(a.public_output/'verification.json',{'old_anchor_score_max_error':anchor_error,'loss_reduction_max_error':max_reduction_error,'factorial_or_loss_identity_max_error':max_identity_error,'normal_equation_max_relative_residual':max(v['normal_equation_relative_residual'] for v in model_meta.values()),'old_private_weights_or_predictions_read':0,'native_arrays_read':0,'deterministic_refits_are_not_independent_replications':True})
    receipt.update(status='LONG_HISTORY_ABLATION_COMPLETE_REVIEW_REQUIRED',training_closed=True,fit_origins=876,training_rows=240900,select_origins=14,dev_origins=14,score_rows=len(rows),loss_contrast_rows=len(contributions),paired_origin_rows=len(paired),max_target_index=1559,max_duration_index=1546,scientific_automatic_admission=False,sota_claim=False)

def main():
    p=argparse.ArgumentParser()
    for n in ('config','data-root','private-output','public-output','claim'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'))
    if c['protocol_id']!='LONG_HISTORY_SOURCE_ABLATION_V1_20260913' or c['masks']!=hs.MASKS or c['limits']!=base.LIMITS:raise ValueError('Protocol mismatch')
    for name,h in c['code_sha256_lf'].items():
        if digest((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))!=h:raise ValueError('Code changed after freeze')
    if a.private_output.exists() or a.public_output.exists():raise FileExistsError('New outputs required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();a.claim.parent.mkdir(parents=True,exist_ok=True)
    with a.claim.open('x',encoding='utf-8') as f:json.dump({'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'frozen_code_commit':head,'status':'CONSUMED_ON_START'},f)
    a.private_output.mkdir(parents=True);a.public_output.mkdir(parents=True);sys.meta_path.insert(0,Forbidden())
    clock=time.perf_counter();r={'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'frozen_code_commit':head,'stage_reads':[],'ridge_fits_started':0,'ridge_fits_completed':0,'executions':1,'device':'CPU','threads':2,'dtype':'float64','numpy_version':np.__version__,'cpu':platform.processor(),'selection_action':'NONE_FIXED_ABLATIONS','neural_training_runs':0,'new_foundation_inference':0,'alpha_searches':0,'hyperparameter_searches':0,'old_private_array_reads':0,'automatic_next_round':False,'automation_enabled':False,'human_review_required':True}
    try:
        with threadpool_limits(limits=2):execute(a,c,r)
    except Exception as e:r.update(status='LONG_HISTORY_ABLATION_BLOCKED',error_type=type(e).__name__,error=str(e));raise
    finally:
        r.update(elapsed_seconds=time.perf_counter()-clock,peak_memory='NOT_MEASURED');dump(a.public_output/'execution_receipt.json',r);print(json.dumps(r),flush=True)

if __name__=='__main__':main()
