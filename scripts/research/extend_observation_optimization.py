"""Post-hoc common-budget optimization diagnosis; original v1 remains unchanged."""
import argparse,hashlib,io,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dual_observation_pilot import ObservationModel,examples,SYSTEMS,load_rate_prefix,scores

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--duration',type=Path,required=True);p.add_argument('--info',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh output required')
    if not torch.cuda.is_available():raise RuntimeError('GPU required for this recorded followup')
    a.output.mkdir(parents=True);torch.set_num_threads(2);started=time.perf_counter()
    o,source=load_rate_prefix(a.csv,648)
    with a.duration.open('rb') as f:raw=b''.join(f.readline() for _ in range(649))
    frame=pd.read_csv(io.BytesIO(raw),index_col=0,parse_dates=True);cols=pd.read_csv(a.csv,nrows=0).columns[1:].astype(str).tolist()
    if frame.shape!=(648,275) or list(frame.columns.astype(str))!=cols or not frame.index.equals(pd.date_range('2022-09-01',periods=648,freq='h')):raise ValueError('Duration identity mismatch')
    cap=pd.read_csv(a.info,dtype={'TAZID':str}).groupby('TAZID')['charge_count'].sum().reindex(cols).to_numpy(np.float32)
    d=frame.to_numpy(np.float32)/cap[None,:]
    if not np.isfinite(d).all() or ((d<0)|(d>1)).any():raise ValueError('Invalid duration')
    original=json.loads((a.initial/'result.json').read_text())
    if original['source']!=source or original['qualification']['duration_prefix_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Original source differs')
    records=[]
    for h in (3,12):
        tr=np.arange(169,576-h+1,3);va=np.arange(576,648-h+1)
        train=tuple(v.cuda() for v in examples(o,d,tr,h));val=tuple(v.cuda() for v in examples(o,d,va,h))
        for seed in (42,43,44):
            for system in SYSTEMS:
                torch.manual_seed(seed+1000);torch.cuda.manual_seed_all(seed+1000)
                model=ObservationModel(h,system.startswith('operator')).cuda()
                path=a.initial/f'private_model_{system}_h{h}_s{seed}.pt'
                initial=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(initial['state_dict'])
                x,y,dl,dr=train;vx,vy,_,_=val
                if system=='occupancy_only':x=x.clone();vx=vx.clone();x[:,168:]=0;vx[:,168:]=0
                duration=dl if system.endswith('left') else dr
                weight=0. if system in ('occupancy_only','duration_concat') else .1
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                def predict():
                    model.eval()
                    with torch.no_grad():return torch.cat([model(b)[0] for b in vx.split(4096)]).cpu().numpy()
                prediction=predict();truth=vy.cpu().numpy();best=scores(prediction,truth)['rmse'];best_epoch=0;stale=0
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};history=[]
                for epoch in range(1,81):
                    model.train();permutation=torch.randperm(len(x),device='cuda');losses=[]
                    for index in permutation.split(4096):
                        optimizer.zero_grad(set_to_none=True);po,pa,op,ap=model(x[index])
                        smooth=(op.flatten(1)[:,1:]-op.flatten(1)[:,:-1]).square().mean()+(ap.flatten(1)[:,1:]-ap.flatten(1)[:,:-1]).square().mean()
                        loss=(po-y[index]).square().mean()+weight*(pa-duration[index]).square().mean()+.001*smooth
                        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
                    prediction=predict();rmse=scores(prediction,truth)['rmse'];history.append({'additional_epoch':epoch,'training_loss':float(np.mean(losses)),'validation_rmse':rmse})
                    if rmse<best:
                        best=rmse;best_epoch=epoch;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    else:stale+=1
                    if stale>=10:break
                model.load_state_dict(state);prediction=predict()
                np.save(a.output/f'private_prediction_{system}_h{h}_s{seed}.npy',prediction)
                torch.save(state,a.output/f'private_model_{system}_h{h}_s{seed}.pt')
                row={'system':system,'horizon':h,'seed':seed,'additional_epochs':len(history),'best_additional_epoch':best_epoch,'history':history,'scores':scores(prediction,truth),
                    'initial_checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'prediction_sha256':hashlib.sha256((a.output/f'private_prediction_{system}_h{h}_s{seed}.npy').read_bytes()).hexdigest()}
                records.append(row);print(json.dumps({k:row[k] for k in ['system','horizon','seed','additional_epochs','best_additional_epoch','scores']}),flush=True)
    macro={system:{m:float(np.mean([r['scores'][m] for r in records if r['system']==system])) for m in ('rmse','mae')} for system in SYSTEMS}
    report={'id':'OBSERVATION_OPTIMIZATION_FOLLOWUP_V1_1','source':source,'status':'POST_HOC_DIAGNOSTIC_NOT_INDEPENDENT_CONFIRMATION','original_v1_gate_unchanged':True,
        'optimizer_state_reset':True,'additional_epochs_cap':80,'patience':10,'all_original_systems_seeds_horizons_included':True,
        'changed_execution':'GPU vectorized minibatches, new shuffle seed=original+1000; architecture/loss/learning-rate/data unchanged',
        'records':records,'macro':macro,'test_accessed':False,'sota_claim':False,'elapsed_seconds':time.perf_counter()-started,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(macro,indent=2),flush=True)
if __name__=='__main__':main()
