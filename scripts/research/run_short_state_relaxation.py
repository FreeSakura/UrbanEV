"""Frozen short-state experiment: 8 neural runs, 2 ridge fits, no foundation calls."""
import argparse,csv,hashlib,importlib.abc,io,json,platform,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast import short_state as core
from urbanev_forecast.short_state_torch import StateNet
from urbanev_forecast.bounded_predictor_csv import prefix_bytes,parse_prefix
from urbanev_forecast.comparability_bridge import zone_capacities,baseline_predictions
from urbanev_forecast.benchmark_metrics import scoped_scores

def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def csvout(p,rows,fields):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def trajectory(a):return a.reshape(14,275,12).transpose(0,2,1)
def rep(name):return 'LONG_OD' if 'LONG' in name else 'SHORT_O' if name=='RELAX_SHORT_O' else 'SHORT_OD'

class NoFoundation(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('chronos','timesfm','transformers') or fullname=='urbanev_forecast.foundation':raise RuntimeError('Foundation inference forbidden')

class Inputs:
    def __init__(self,c,root):self.c=c;self.root=root;self.ledger=[];self.position=0;self.cap=None;self.y=None;self.d=None
    def read(self,stage):
        if stage!=('FIT','SELECT','PREDICT','SCORE')[self.position]:raise RuntimeError('Stage order violation')
        for name,n in core.LIMITS[stage].items():
            b=prefix_bytes(self.root,name,n,n,set(core.LIMITS[stage]));expected=self.c['prefixes'][stage][name]['sha256']
            if digest(b)!=expected:raise ValueError('Prefix hash mismatch')
            if self.cap is None:
                self.cols=next(csv.reader([b.splitlines()[0].decode('utf-8-sig')]))[1:]
                if len(self.cols)!=275 or digest('\n'.join(self.cols).encode())!=self.c['column_order_sha256']:raise ValueError('Region order mismatch')
                p=(self.root/'inf.csv').resolve()
                if p.parent!=self.root.resolve() or sha(p)!=self.c['static_info_sha256']:raise ValueError('Static source mismatch')
                self.cap=zone_capacities(pd.read_csv(p,usecols=['station_id','TAZID','charge_count'],dtype={'TAZID':str}),self.cols)
                self.ledger.append({'stage':stage,'file':'inf.csv','columns':['station_id','TAZID','charge_count'],'capacity':'TAZID sum'})
            a,r=parse_prefix(b,n,self.cols,expected,nonnegative=True)
            if name=='occupancy.csv':self.y=(a.astype(np.float32)/self.cap.astype(np.float32)[None,:]).astype(np.float64);v=self.y
            else:self.d=a/self.cap[None,:];v=self.d
            if ((v<0)|(v>1)).any():raise ValueError('Illegal source rate; no repair')
            self.ledger.append({'stage':stage,'file':name,**r})
        self.position+=1
        return self.y,self.d,self.cap

def predict(net,x,p):
    values=[];net.eval()
    with torch.no_grad():
        for start in range(0,len(x),4096):
            v=net(torch.as_tensor(x[start:start+4096],dtype=torch.float32),torch.as_tensor(p[start:start+4096],dtype=torch.float32))
            if not torch.isfinite(v).all():raise ValueError('Nonfinite prediction')
            values.append(v.numpy().astype(np.float64))
    return np.concatenate(values)

def metric_rows(name,seed,pred,y,post,stage='DEV_EVAL',hs=core.HORIZONS):
    return [{'stage':stage,'model':name,'seed':str(seed),'horizon':h,'status':'AVAILABLE',**scoped_scores(pred[:,:h],y[:,:h],target_scope=s,postprocess=post)} for h in hs for s in ('terminal_H','path_1_to_H')]

def train(name,seed,x,p,y,a,budget,curves,receipt):
    budget.start('neural');receipt['neural_training_runs']=budget.neural;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    net=StateNet(x.shape[1],name.startswith('RELAX'),seed);count=sum(v.numel() for v in net.parameters())
    expected=1410 if name.startswith('RELAX') else 12396 if 'LONG' in name else 1740
    if count!=expected:raise ValueError('Parameter count mismatch')
    opt=torch.optim.AdamW([{'params':[v for v in net.parameters() if v.ndim>=2],'weight_decay':1e-4},{'params':[v for v in net.parameters() if v.ndim<2],'weight_decay':0.}],lr=.001,betas=(.9,.999),eps=1e-8,foreach=False)
    xx=torch.as_tensor(x,dtype=torch.float32);pp=torch.as_tensor(p,dtype=torch.float32);yy=torch.as_tensor(y,dtype=torch.float32)
    steps=0;checkpoints=[];started=time.perf_counter()
    for epoch in range(1,21):
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(x));total=0.;net.train()
        for start in range(0,len(order),4096):
            ix=torch.from_numpy(order[start:start+4096]);opt.zero_grad(set_to_none=True);out=net(xx[ix],pp[ix]);loss=((out-yy[ix])**2).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1;receipt['optimizer_steps']=receipt.get('optimizer_steps',0)+1;total+=float(loss.detach())*len(ix)
            if any(not torch.isfinite(v).all() for v in net.parameters()):raise ValueError('Nonfinite parameter')
        curves.append({'model':name,'seed':seed,'epoch':epoch,'training_mse':total/len(x),'optimizer_steps':steps,'selection_rmse':None})
        if epoch in (5,10,15,20):
            pth=a.private_output/f'private_{name}_s{seed}_e{epoch}.pt';torch.save(net.state_dict(),pth);checkpoints.append({'epoch':epoch,'file':pth.name,'sha256':sha(pth)})
            print(json.dumps({'model':name,'seed':seed,'epoch':epoch,'training_mse':total/len(x)}),flush=True)
    if steps!=1180:raise ValueError('Optimization budget mismatch')
    return {'model':name,'seed':seed,'parameters':count,'steps':steps,'elapsed_seconds':time.perf_counter()-started,'checkpoints':checkpoints}

def execute(a,c,receipt):
    inp=Inputs(c,a.data_root);receipt['stage_reads']=inp.ledger;budget=core.FitBudget();curves=[];runs=[];stats={};ridges={}
    y,d,cap=inp.read('FIT');oo=core.origins('FIT');yy=core.targets(y,oo);anchor=y[oo-1].reshape(-1)
    if (len(oo),len(yy))!=(876,240900):raise ValueError('Fit population mismatch')
    for representation,names in [('SHORT_OD',core.NEURAL[:3]),('LONG_OD',core.NEURAL[3:])]:
        x=core.features(y,d,cap,oo,representation);st=core.fit_transform(x);z=core.transform(x,st);del x;stats[representation]=st
        np.savez(a.private_output/f'private_transform_{representation}.npz',**st)
        name='RIDGE_'+representation;budget.start('ridge');receipt['deterministic_ridge_fits']=budget.ridge;clock=time.perf_counter();beta=core.fit_ridge(z,yy-anchor[:,None]);pth=a.private_output/f'private_{name}.npy';np.save(pth,beta)
        if not np.isfinite(beta).all():raise ValueError('Nonfinite ridge')
        ridges[name]={'file':pth.name,'sha256':sha(pth),'coefficients':int(beta.size),'representation':representation,'elapsed_seconds':time.perf_counter()-clock}
        x32=z.astype(np.float32);del z
        for name in names:
            use=x32
            if name=='RELAX_SHORT_O':
                use=x32.copy();use[:,2]=0.;mask={k:v.copy() for k,v in st.items()};mask['mean'][2]=0.;mask['std'][2]=0.;mask['inactive'][2]=True;stats['SHORT_O']=mask
                np.savez(a.private_output/'private_transform_SHORT_O.npz',**mask)
            for seed in core.SEEDS:runs.append(train(name,seed,use,anchor,yy,a,budget,curves,receipt))
            receipt.update(neural_training_runs=budget.neural,deterministic_ridge_fits=budget.ridge)
        del x32,use
    budget.close()
    transform_meta={k:{'file':f'private_transform_{k}.npz','sha256':sha(a.private_output/f'private_transform_{k}.npz'),'width':len(v['mean']),'inactive_columns':int(v['inactive'].sum())} for k,v in stats.items()}
    curve_fields=['model','seed','epoch','training_mse','optimizer_steps','selection_rmse'];csvout(a.public_output/'training_curves.csv',curves,curve_fields)
    # Offline checkpoint selection: all twenty-epoch trajectories are already closed.
    y,d,cap=inp.read('SELECT');so=core.origins('SELECT');sy=trajectory(core.targets(y,so));sp=y[so-1].reshape(-1);selected=[]
    for meta in runs:
        name,seed=meta['model'],meta['seed'];representation=rep(name);x=core.transform(core.features(y,d,cap,so,representation),stats[representation]).astype(np.float32);choices=[]
        for cp in meta['checkpoints']:
            pth=a.private_output/cp['file']
            if sha(pth)!=cp['sha256']:raise ValueError('Checkpoint changed')
            net=StateNet(x.shape[1],name.startswith('RELAX'),seed);net.load_state_dict(torch.load(pth,map_location='cpu',weights_only=True));pred=trajectory(predict(net,x,sp));rows=metric_rows(name,seed,pred,sy,'raw','SELECT')
            score=float(np.mean([r['rmse'] for r in rows if r['target_scope']=='terminal_H']));choices.append({**cp,'selection_rmse':score,'scores':rows})
            next(v for v in curves if v['model']==name and v['seed']==seed and v['epoch']==cp['epoch'])['selection_rmse']=score
        selected.append({'model':name,'seed':seed,'representation':representation,'parameters':meta['parameters'],'selected':core.checkpoint_choice(choices)})
    selection={'selected_neural':selected,'ridge':ridges,'transforms':transform_meta,'training_closed':True,'selection_rule':'minimum SELECT mean four endpoint RMSEs; exact ties earlier epoch; no model or seed selection','neural_ensemble':False,'config_sha256':sha(a.config)}
    dump(a.public_output/'frozen_selection.json',selection);selection_hash=sha(a.public_output/'frozen_selection.json');csvout(a.public_output/'training_curves.csv',curves,curve_fields)
    if any(sha(a.private_output/v['file'])!=v['sha256'] for v in transform_meta.values()):raise ValueError('Transform changed')
    y,d,cap=inp.read('PREDICT');eo=core.origins('PREDICT');ep=y[eo-1].reshape(-1);predictions={}
    for s in selected:
        pth=a.private_output/s['selected']['file']
        if sha(pth)!=s['selected']['sha256']:raise ValueError('Selected checkpoint changed')
        representation=s['representation'];x=core.transform(core.features(y,d,cap,eo,representation),stats[representation]).astype(np.float32)
        net=StateNet(x.shape[1],s['model'].startswith('RELAX'),s['seed']);net.load_state_dict(torch.load(pth,map_location='cpu',weights_only=True));predictions[s['model'],str(s['seed']),'raw']=trajectory(predict(net,x,ep))
    for name,meta in ridges.items():
        pth=a.private_output/meta['file']
        if sha(pth)!=meta['sha256']:raise ValueError('Ridge changed')
        x=core.transform(core.features(y,d,cap,eo,meta['representation']),stats[meta['representation']]);pred=trajectory(core.ridge_predict(x,ep,np.load(pth,allow_pickle=False)))
        predictions[name,'deterministic','raw']=pred;predictions[name,'deterministic','clip_0_1']=np.clip(pred,0,1)
    for name,pred in baseline_predictions(y,eo,12).items():predictions[name,'deterministic','raw']=pred
    frozen=[]
    for key,pred in predictions.items():
        pth=a.private_output/('private_prediction_'+'_'.join(key)+'.npy');np.save(pth,pred);frozen.append({'model':key[0],'seed':key[1],'postprocess':key[2],'file':pth.name,'sha256':sha(pth)})
    dump(a.public_output/'prediction_freeze.json',{'selection_sha256':selection_hash,'predictions':frozen,'final_score_prefix_not_yet_decoded':True,'rolling_earlier_evaluation_observations_available':True})
    y,d,cap=inp.read('SCORE');truth=y[eo[:,None]+np.arange(12)];np.save(a.private_output/'private_truth_H12.npy',truth)
    if sha(a.public_output/'frozen_selection.json')!=selection_hash:raise ValueError('Selection changed after labels')
    rows=[];paired=[];reduction_error=0.
    def record(name,seed,pred,post,hs):
        nonlocal reduction_error
        rr=metric_rows(name,seed,pred,truth,post,hs=hs);rows.extend(rr)
        for row in rr:
            h=row['horizon'];p=pred[:,:h];t=truth[:,:h]
            if post=='clip_0_1':p=np.clip(p,0,1)
            err=p-t
            if row['target_scope']=='terminal_H':err=err[:,-1:]
            em=np.sum(err**2,axis=(1,2))/(err.shape[1]*err.shape[2]);ea=np.sum(abs(err),axis=(1,2))/(err.shape[1]*err.shape[2])
            reduction_error=max(reduction_error,abs(float(np.sqrt(em.mean()))-row['rmse']),abs(float(ea.mean())-row['mae']))
            for k,o in enumerate(eo):paired.append({'model':name,'seed':str(seed),'postprocess':post,'horizon':h,'target_scope':row['target_scope'],'origin':int(o),'mse':float(em[k]),'mae':float(ea[k])})
    for entry in frozen:
        pth=a.private_output/entry['file']
        if sha(pth)!=entry['sha256']:raise ValueError('Prediction changed')
        record(entry['model'],entry['seed'],np.load(pth,allow_pickle=False),entry['postprocess'],core.HORIZONS)
    cache={};lineage=0.
    for entry in c['cache_files']:
        if entry['cut']!=1392 or entry['horizon'] not in (3,12) or entry['system'] not in ('native','truth'):raise ValueError('Cache not allowed')
        base=a.native_root if entry['system']=='native' else a.truth_root;pth=(base/entry['file']).resolve()
        if pth.parent!=base.resolve() or sha(pth)!=entry['sha256']:raise ValueError('Cache identity failed')
        arr=np.load(pth,allow_pickle=False).astype(np.float64)
        if arr.shape!=(14,entry['horizon'],275) or not np.isfinite(arr).all():raise ValueError('Cache values failed')
        cache[entry['horizon'],entry['system']]=arr;inp.ledger.append({'stage':'SCORE','file':entry['file'],'kind':'cache_decode','sha256':entry['sha256']})
    for h in (3,12):
        lineage=max(lineage,float(np.max(abs(cache[h,'truth']-truth[:,:h]))))
        for name,post in [('native_raw','raw'),('native_clip','clip_0_1')]:record(name,'cached',cache[h,'native'],post,(h,))
    for name,post in [('native_raw','raw'),('native_clip','clip_0_1')]:
        for h in (6,9):
            for s in ('terminal_H','path_1_to_H'):rows.append({'stage':'DEV_EVAL','model':name,'seed':'cached','horizon':h,'status':'NOT_AVAILABLE','target_scope':s,'postprocess':post,'scored_values':0,'rmse':None,'mae':None})
    with (ROOT/'artifacts/summaries/comparability_bridge_v2/comparison_table.csv').open(encoding='utf-8') as f:prior=list(csv.DictReader(f))
    reference_error=0.
    for row in rows:
        if row['model'] not in ('last','day','week','native_raw','native_clip') or row['status']!='AVAILABLE':continue
        old=next(v for v in prior if v['system']==row['model'] and v['cut']=='1392' and int(v['horizon'])==row['horizon'] and v['target_scope']==row['target_scope'])
        reference_error=max(reference_error,abs(row['rmse']-float(old['rmse'])),abs(row['mae']-float(old['mae'])))
    if lineage>1e-7 or reduction_error>1e-12 or reference_error>1e-12:raise ValueError('Metric or lineage mismatch')
    csvout(a.public_output/'development_scores.csv',rows,['stage','model','seed','horizon','status','target_scope','postprocess','scored_values','rmse','mae'])
    csvout(a.public_output/'paired_origin_errors.csv',paired,['model','seed','postprocess','horizon','target_scope','origin','mse','mae'])
    summary=[]
    for support,hs in [('FOUR_H',core.HORIZONS),('H3_H12_COMMON',(3,12))]:
        keys=sorted({(r['model'],r['seed'],r['postprocess']) for r in rows if r['status']=='AVAILABLE'})
        for name,seed,post in keys:
            if name.startswith('native') and len(hs)==4:continue
            for s in ('terminal_H','path_1_to_H'):
                rr=[r for r in rows if (r['model'],r['seed'],r['postprocess'],r['target_scope'])==(name,seed,post,s) and r['horizon'] in hs and r['status']=='AVAILABLE']
                if len(rr)!=len(hs):raise ValueError('Mixed score support')
                summary.append({'support':support,'model':name,'seed':seed,'postprocess':post,'target_scope':s,'macro_rmse':float(np.mean([r['rmse'] for r in rr])),'macro_mae':float(np.mean([r['mae'] for r in rr]))})
    means=[]
    for name in core.NEURAL:
        for support in ('FOUR_H','H3_H12_COMMON'):
            for s in ('terminal_H','path_1_to_H'):
                rr=[r for r in summary if r['model']==name and r['support']==support and r['target_scope']==s]
                item={'model':name,'support':support,'target_scope':s,'seed_count':2}
                for metric in ('rmse','mae'):
                    vv=[r['macro_'+metric] for r in rr];item.update({metric+'_mean':float(np.mean(vv)),metric+'_min':min(vv),metric+'_max':max(vv)})
                means.append(item)
    dump(a.public_output/'model_comparison_summary.json',{'status':'SHORT_STATE_EXPERIMENT_COMPLETE_REVIEW_REQUIRED','per_run':summary,'neural_seed_metric_means':means,'runs':runs,'ridge':ridges,'checks':{'cached_truth_error':lineage,'bridge_reference_error':reference_error,'direct_origin_reduction_error':reduction_error},'primary_candidate':'RELAX_SHORT_OD','core_contrasts':['RELAX_SHORT_OD vs RELAX_SHORT_O','RELAX_SHORT_OD vs DIRECT_SHORT_OD and RIDGE_SHORT_OD raw/clip','SHORT vs LONG within DIRECT and RIDGE'],'scientific_automatic_admission':False,'sota_claim':False,'uncertainty':'Two seeds and fourteen exposed origins: descriptive means/ranges only; no significance or unseen-test claim'})
    receipt.update(status='SHORT_STATE_EXPERIMENT_COMPLETE_REVIEW_REQUIRED',stage_reads=inp.ledger,cache_decodes=4,fit_origins=876,fit_rows=240900,select_origins=14,eval_origins=14,optimizer_steps=9440,development_score_rows=len(rows),paired_origin_rows=len(paired),scientific_automatic_admission=False,sota_claim=False)

def main():
    p=argparse.ArgumentParser()
    for n in ('config','data-root','native-root','truth-root','private-output','public-output','claim'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'))
    if c['protocol_id']!='SHORT_STATE_RELAXATION_V1_20260913' or c['limits']!=core.LIMITS or c['seeds']!=list(core.SEEDS):raise ValueError('Protocol mismatch')
    for name,h in c['code_sha256_lf'].items():
        if digest((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))!=h:raise ValueError('Code changed after freeze')
    if a.private_output.exists() or a.public_output.exists():raise FileExistsError('New outputs required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();a.claim.parent.mkdir(parents=True,exist_ok=True)
    with a.claim.open('x',encoding='utf-8') as f:json.dump({'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'code_commit':head,'status':'CONSUMED_ON_START'},f)
    a.private_output.mkdir(parents=True);a.public_output.mkdir(parents=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);sys.meta_path.insert(0,NoFoundation())
    started=time.perf_counter();receipt={'protocol_id':c['protocol_id'],'config_sha256':sha(a.config),'frozen_code_commit':head,'device':'CPU','threads':2,'torch_version':torch.__version__,'numpy_version':np.__version__,'cpu':platform.processor(),'neural_training_runs':0,'deterministic_ridge_fits':0,'executions':1,'new_foundation_inference':0,'alpha_searches':0,'old_candidate_reselection':False,'automatic_next_round':False,'automation_enabled':False,'human_review_required':True}
    try:
        with threadpool_limits(limits=2):execute(a,c,receipt)
    except Exception as e:receipt.update(status='SHORT_STATE_EXPERIMENT_BLOCKED',error_type=type(e).__name__,error=str(e));raise
    finally:
        receipt.update(elapsed_seconds=time.perf_counter()-started,peak_memory='NOT_MEASURED');dump(a.public_output/'execution_receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
