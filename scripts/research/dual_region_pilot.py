"""Registered K32 partition/swap pilot; all supervised selection is in training."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast.data import load_rate_prefix,window_starts,scores
from urbanev_forecast.foundation import FoundationBackend,file_hash,LEVELS

def affinity(x):
    z=x-x.mean(axis=0);norm=np.sqrt((z*z).sum(axis=0))
    result=np.abs(z.T@z/np.maximum(norm[:,None]*norm[None,:],1e-12))
    np.fill_diagonal(result,0);return result

def residual_affinity(values):
    t=np.arange(168,576);lags=np.stack([values[t-lag] for lag in (1,2,24,48,168)],axis=-1)
    calendar=np.stack([np.sin(t*2*np.pi/24),np.cos(t*2*np.pi/24),np.sin(t*2*np.pi/168),np.cos(t*2*np.pi/168),np.ones(len(t))],axis=-1)
    x=np.concatenate([lags,np.broadcast_to(calendar[:,None,:],(len(t),275,5))],axis=-1)
    residual=[]
    for cut,end in [(288,384),(384,480),(480,576)]:
        train=t<cut;valid=(t>=cut)&(t<end);out=np.empty((valid.sum(),275))
        for c in range(275):
            a=x[train,c];reg=np.eye(10)*.001;reg[-1,-1]=0
            beta=np.linalg.solve(a.T@a/len(a)+reg,a.T@values[t[train],c]/len(a))
            out[:,c]=values[t[valid],c]-x[valid,c]@beta
        residual.append(out)
    return affinity(np.concatenate(residual))

def partition(matrix):
    remaining=list(range(275));groups=[]
    while remaining:
        seed=remaining[int(np.argmax(matrix[np.ix_(remaining,remaining)].sum(axis=1)))];group=[seed];remaining.remove(seed)
        while remaining and len(group)<32:
            chosen=remaining[int(np.argmax(matrix[np.ix_(remaining,group)].mean(axis=1)))];group.append(chosen);remaining.remove(chosen)
        groups.append(sorted(group))
    return groups

def validate_groups(groups):
    if sorted(c for group in groups for c in group)!=list(range(275)) or any(not 1<=len(g)<=32 for g in groups):
        raise ValueError('Invalid disjoint partition')

class GroupedChronos:
    def __init__(self,path):
        self.base=FoundationBackend('chronos2',path,'cuda');self.base.model.eval();self.calls=0;self.channels=0
    def predict(self,x,groups,h):
        # Separate tasks share this origin, with explicit independent group IDs.
        q,p=self.base.pipeline.predict_quantiles(inputs=[x[:,g].T.copy() for g in groups],prediction_length=h,
            quantile_levels=LEVELS,batch_size=sum(map(len,groups)),context_length=168,cross_learning=False)
        self.calls+=1;self.channels+=sum(map(len,groups))
        if len(p)!=len(groups):raise ValueError('Task grouping changed')
        result=np.concatenate([v.cpu().numpy().T for v in p],axis=1)
        if result.shape!=(h,sum(map(len,groups))) or not np.isfinite(result).all():raise ValueError('Invalid prediction')
        return result

def pair_loss(backend,values,groups):
    indices=sum(groups,[]);loss=0.
    for h in (3,12):
        for origin in (192,288,384,480):
            pred=backend.predict(values[origin-168:origin],groups,h)
            truth=values[origin:origin+h,indices]
            loss+=float(np.sum((pred.astype(float)-truth)**2))/(4*2*h*275)
    return loss

def search(backend,values,start,weights,guided):
    groups=copy.deepcopy(start);rng=np.random.default_rng(42);tried=set();log=[]
    calls=backend.calls;channels=backend.channels;began=time.perf_counter()
    # Keep the 19-node tail fixed: every proposal uses exactly two groups of 32.
    active=[c for g in groups if len(g)==32 for c in g]
    for step in range(24):
        owner={c:i for i,g in enumerate(groups) for c in g}
        if guided:
            totals=np.stack([weights[:,g].sum(axis=1) for g in groups],axis=1)
            best=(-float('inf'),None)
            for a in active:
                for b in active:
                    if a>=b or owner[a]==owner[b] or (a,b) in tried:continue
                    ga,gb=owner[a],owner[b]
                    gain=totals[b,ga]+totals[a,gb]-totals[a,ga]-totals[b,gb]-2*weights[a,b]
                    if gain>best[0]:best=(gain,(a,b))
            a,b=best[1]
        else:
            while True:
                a,b=sorted(rng.choice(active,2,replace=False).tolist())
                if owner[a]!=owner[b] and (a,b) not in tried:break
        tried.add((a,b));ga,gb=owner[a],owner[b]
        old=[groups[ga],groups[gb]];new=[sorted([c for c in groups[ga] if c!=a]+[b]),sorted([c for c in groups[gb] if c!=b]+[a])]
        before=pair_loss(backend,values,old);after=pair_loss(backend,values,new)
        accept=after<before
        if accept:groups[ga],groups[gb]=new
        log.append({'proposal':step+1,'training_joint_mse_before':before,'training_joint_mse_after':after,'accepted':accept})
        print(json.dumps({'search':'guided' if guided else 'random','proposal':step+1,'accepted':accept}),flush=True)
    validate_groups(groups)
    return groups,{'proposals':log,'backend_calls':backend.calls-calls,'channel_tasks':backend.channels-channels,'seconds':time.perf_counter()-began}

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--info',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--reference-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use fresh output')
    values,source=load_rate_prefix(a.csv,648);a.output.mkdir(parents=True)
    backend=GroupedChronos(a.model_dir);residual=residual_affinity(values[:576]);raw=affinity(values[288:576])
    cols=pd.read_csv(a.csv,nrows=0).columns[1:].astype(str).tolist()
    info=pd.read_csv(a.info,dtype={'TAZID':str}).groupby('TAZID')[['longitude','latitude']].mean().reindex(cols)
    if not np.isfinite(info.to_numpy()).all():raise ValueError('Missing geographic metadata')
    order=np.lexsort((info.latitude.to_numpy(),info.longitude.to_numpy()))
    split=lambda v:[sorted(v[i:i+32].tolist()) for i in range(0,275,32)]
    groups={'contiguous32':split(np.arange(275)),'random32':split(np.random.default_rng(42).permutation(275)),
        'geographic32':split(order),'raw_correlation32':partition(raw),'residual32':partition(residual)}
    searches={}
    for name,guided in [('random_swap32',False),('utility_swap32',True)]:
        groups[name],searches[name]=search(backend,values,groups['residual32'],residual,guided)
    assert searches['random_swap32']['backend_calls']==searches['utility_swap32']['backend_calls']==384
    assert searches['random_swap32']['channel_tasks']==searches['utility_swap32']['channel_tasks']==384*64
    (a.output/'private_partitions.json').write_text(json.dumps(groups),encoding='utf-8')
    receipt=json.loads((a.reference_cache/'receipt.json').read_text());cache=json.loads((a.reference_cache/'cache_summary.json').read_text())
    if receipt['source']!=source:raise ValueError('Reference data differs')
    for key in ['weight_sha256','api_source_sha256','config_sha256','dtype']:
        if receipt['backend'][key]!=backend.base.metadata[key]:raise ValueError('Reference backend differs')
    metrics={name:[] for name in ['full275',*groups]};timings={}
    for h in (3,12):
        origins=window_starts(648,168,h,576,648);truth=values[origins[:,None]+np.arange(h)]
        folder=a.reference_cache/f'h{h}';rec=next(x for x in cache['horizons'] if x['horizon']==h)
        if file_hash(folder/'native.npy')!=rec['native_sha256']:raise ValueError('Reference hash differs')
        old_origins=np.load(folder/'origins.npy');ref=np.load(folder/'native.npy')[old_origins>=576]
        if not np.array_equal(old_origins[old_origins>=576],origins):raise ValueError('Reference origins differ')
        metrics['full275'].append({'horizon':h,'raw':scores(ref,truth),'clipped':scores(np.clip(ref,0,1),truth)})
        for name,partition_groups in groups.items():
            validate_groups(partition_groups);pred=np.empty_like(truth);indices=sum(partition_groups,[]);start=time.perf_counter();calls=backend.calls
            for i,origin in enumerate(origins):pred[i][:,indices]=backend.predict(values[origin-168:origin],partition_groups,h)
            np.save(a.output/f'private_prediction_{name}_h{h}.npy',pred)
            metrics[name].append({'horizon':h,'raw':scores(pred,truth),'clipped':scores(np.clip(pred,0,1),truth)})
            timings[f'{name}_h{h}']={'seconds':time.perf_counter()-start,'calls':backend.calls-calls}
            print(json.dumps({'system':name,'horizon':h,'rmse':metrics[name][-1]['clipped']['rmse']}),flush=True)
    macro={k:{m:float(np.mean([c['clipped'][m] for c in v])) for m in ['rmse','mae']} for k,v in metrics.items()}
    controls=[k for k in groups if k!='utility_swap32'];best=min(controls,key=lambda k:macro[k]['rmse']);candidate=macro['utility_swap32']
    gain=100*(1-candidate['rmse']/macro[best]['rmse']);degradation=[100*(x['clipped']['rmse']/y['clipped']['rmse']-1) for x,y in zip(metrics['utility_swap32'],metrics[best])]
    passed=gain>=1 and candidate['mae']<=macro[best]['mae'] and max(degradation)<=1
    report={'protocol_id':'DUAL_DIRECTION_PILOT_V1_20260910','route':'A','source':source,'backend':backend.base.metadata,'metrics':metrics,'macro':macro,
        'searches':searches,'evaluation_timing':timings,'timing_is_not_controlled_latency_benchmark':True,
        'partition_sha256':hashlib.sha256((a.output/'private_partitions.json').read_bytes()).hexdigest(),
        'gate':{'decision':'GO_PILOT_ONLY' if passed else 'NO_GO','best_matched_control':best,'rmse_gain_percent':gain,'mae_non_degradation':candidate['mae']<=macro[best]['mae'],'cell_degradation_percent':degradation,
        'rmse_gain_vs_full275_percent':100*(1-candidate['rmse']/macro['full275']['rmse'])},
        'test_accessed':False,'sota_claim':False,'scope':'single backbone, fold1 exposed development, K32 only',
        'script_sha256':file_hash(Path(__file__))}
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report['gate']),flush=True)
if __name__=='__main__':main()
