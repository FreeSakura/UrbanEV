"""Duration-information and soft observation-operator pilot, CPU only.

Both interval conventions are sensitivities, not identified physical truth.
"""
import argparse,hashlib,io,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast.data import load_rate_prefix,scores
from urbanev_forecast.foundation import file_hash

SYSTEMS=('occupancy_only','duration_concat','free_left','operator_left','free_right','operator_right')

class ObservationModel(nn.Module):
    def __init__(self,horizon,operator=False):
        super().__init__();self.horizon=horizon;self.operator=operator
        self.network=nn.Sequential(nn.Linear(336,32),nn.GELU(),nn.Linear(32,horizon*4*2))
    def forward(self,x):
        raw=self.network(x).reshape(-1,self.horizon,4,2)
        occupied=torch.sigmoid(raw[...,0])
        active=torch.sigmoid(raw[...,1])
        if self.operator:active=occupied*active
        return occupied[:,:,-1],active.mean(dim=-1),occupied,active

def examples(o,d,origins,h):
    x=np.stack([np.concatenate((o[s-168:s].T,d[s-169:s-1].T),axis=1) for s in origins]).reshape(-1,336)
    target=np.stack([o[s:s+h].T for s in origins]).reshape(-1,h)
    left=np.stack([d[s-1:s+h-1].T for s in origins]).reshape(-1,h)
    right=np.stack([d[s:s+h].T for s in origins]).reshape(-1,h)
    return tuple(torch.from_numpy(np.asarray(a,np.float32)) for a in (x,target,left,right))

def train_one(train,val,h,system,seed,out):
    torch.manual_seed(seed);np.random.seed(seed)
    model=ObservationModel(h,system.startswith('operator'));optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    x,y,dl,dr=train;vx,vy,vdl,vdr=val
    if system=='occupancy_only':x=x.clone();vx=vx.clone();x[:,168:]=0;vx[:,168:]=0
    duration=dl if system.endswith('left') else dr
    auxiliary=0. if system in ('occupancy_only','duration_concat') else .1
    loader=DataLoader(TensorDataset(x,y,duration),batch_size=4096,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');stale=0;history=[];state=None;best_epoch=0
    for epoch in range(1,21):
        model.train();losses=[]
        for bx,by,bd in loader:
            optimizer.zero_grad(set_to_none=True);po,pa,op,ap=model(bx)
            smooth=(op.flatten(1)[:,1:]-op.flatten(1)[:,:-1]).square().mean()+(ap.flatten(1)[:,1:]-ap.flatten(1)[:,:-1]).square().mean()
            loss=(po-by).square().mean()+auxiliary*(pa-bd).square().mean()+.001*smooth
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():prediction=torch.cat([model(b)[0] for b in vx.split(4096)]).numpy()
        metric=scores(prediction,vy.numpy())['rmse'];history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),'validation_rmse':metric})
        if metric<best:
            best=metric;stale=0;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=4:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():prediction=torch.cat([model(b)[0] for b in vx.split(4096)]).numpy()
    np.save(out/f'private_prediction_{system}_h{h}_s{seed}.npy',prediction)
    torch.save({'state_dict':state,'horizon':h,'system':system,'seed':seed},out/f'private_model_{system}_h{h}_s{seed}.pt')
    return {'system':system,'horizon':h,'seed':seed,'best_epoch':best_epoch,'epochs_executed':len(history),'parameters':sum(p.numel() for p in model.parameters()),
            'training_loss_history':history,'scores':scores(prediction,vy.numpy()),'prediction_sha256':file_hash(out/f'private_prediction_{system}_h{h}_s{seed}.npy')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--duration',type=Path,required=True);p.add_argument('--info',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use fresh output')
    a.output.mkdir(parents=True);torch.set_num_threads(2);started=time.perf_counter()
    o,source=load_rate_prefix(a.csv,648)
    with a.duration.open('rb') as handle:payload=b''.join(handle.readline() for _ in range(649))
    frame=pd.read_csv(io.BytesIO(payload),index_col=0,parse_dates=True)
    columns=pd.read_csv(a.csv,nrows=0).columns[1:].astype(str).tolist()
    if frame.shape!=(648,275) or list(frame.columns.astype(str))!=columns or not frame.index.equals(pd.date_range('2022-09-01',periods=648,freq='h')):
        raise ValueError('Duration/date/column mismatch')
    capacity=pd.read_csv(a.info,dtype={'TAZID':str}).groupby('TAZID')['charge_count'].sum().reindex(columns).to_numpy(np.float32)
    if not np.isfinite(capacity).all() or (capacity<=0).any():raise ValueError('Invalid capacity')
    d=frame.to_numpy(np.float32)/capacity[None,:]
    qualification={'finite':bool(np.isfinite(d).all()),'minimum':float(np.nanmin(d)),'maximum':float(np.nanmax(d)),
        'out_of_bounds':int(((d<0)|(d>1)).sum()),'duration_prefix_sha256':hashlib.sha256(payload).hexdigest(),
        'interval_endpoint_identified':False,'real_time_release_availability_identified':False,
        'usage':'conservative lagged released data; two interval alignment sensitivities, not physical identification'}
    if not qualification['finite'] or qualification['out_of_bounds']:
        (a.output/'result.json').write_text(json.dumps({'route':'B','decision':'BLOCKED_DATA_QUALIFICATION','qualification':qualification},indent=2),encoding='utf-8');return
    records=[];counts={}
    for h in (3,12):
        # One extra hour makes the lagged-duration history entirely observed.
        tr=np.arange(169,576-h+1,3);va=np.arange(576,648-h+1)
        train=examples(o,d,tr,h);val=examples(o,d,va,h)
        counts[str(h)]={'training_windows':len(tr),'validation_windows':len(va),'training_node_samples':len(train[0]),'validation_node_samples':len(val[0])}
        for seed in (42,43,44):
            for system in SYSTEMS:
                record=train_one(train,val,h,system,seed,a.output);records.append(record)
                print(json.dumps({k:record[k] for k in ('system','horizon','seed','best_epoch','scores')}),flush=True)
    macro={}
    for system in SYSTEMS:
        per_seed={str(seed):{m:float(np.mean([r['scores'][m] for r in records if r['system']==system and r['seed']==seed])) for m in ('rmse','mae')} for seed in (42,43,44)}
        macro[system]={'per_seed':per_seed,**{m:float(np.mean([v[m] for v in per_seed.values()])) for m in ('rmse','mae')},
            **{m+'_seed_sd':float(np.std([v[m] for v in per_seed.values()],ddof=1)) for m in ('rmse','mae')}}
    gain=lambda candidate,reference:100*(1-macro[candidate]['rmse']/macro[reference]['rmse'])
    information={'rmse_gain_percent':gain('duration_concat','occupancy_only'),'mae_non_degradation':macro['duration_concat']['mae']<=macro['occupancy_only']['mae']}
    information['decision']='GO_PILOT_ONLY' if information['rmse_gain_percent']>=1 and information['mae_non_degradation'] else 'NO_GO'
    mechanisms={}
    for alignment in ('left','right'):
        candidate='operator_'+alignment;controls=['duration_concat','free_'+alignment];best=min(controls,key=lambda k:macro[k]['rmse'])
        g=gain(candidate,best);mae=macro[candidate]['mae']<=macro[best]['mae']
        mechanisms[alignment]={'best_same_input_control':best,'rmse_gain_percent':g,'mae_non_degradation':mae,'decision':'GO_PILOT_ONLY' if g>=1 and mae else 'NO_GO'}
    report={'protocol_id':'DUAL_DIRECTION_PILOT_V1_20260910','route':'B','source':source,'qualification':qualification,'counts':counts,'records':records,'macro':macro,
        'information_gate':information,'mechanism_gates':mechanisms,'physical_claim_status':'NOT_QUALIFIED: endpoint and live publication semantics unresolved',
        'elapsed_seconds':time.perf_counter()-started,'device':'cpu,2 threads','test_accessed':False,'sota_claim':False,
        'scope':'shared small MLP, fold1 exposed development, two alignment assumptions, no recovered latent-state truth',
        'script_sha256':file_hash(Path(__file__))}
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'information':information,'mechanisms':mechanisms}),flush=True)
if __name__=='__main__':main()
