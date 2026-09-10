"""Post-hoc optimization sanity: persistence and fixed ridge, not new gate results."""
import argparse,hashlib,importlib.util,io,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast.data import load_rate_prefix,scores

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--duration',type=Path,required=True);p.add_argument('--info',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh output required')
    o,source=load_rate_prefix(a.csv,648)
    with a.duration.open('rb') as f:payload=b''.join(f.readline() for _ in range(649))
    frame=pd.read_csv(io.BytesIO(payload),index_col=0);cols=pd.read_csv(a.csv,nrows=0).columns[1:].astype(str)
    cap=pd.read_csv(a.info,dtype={'TAZID':str}).groupby('TAZID')['charge_count'].sum().reindex(cols).to_numpy(float)
    if list(frame.columns.astype(str))!=list(cols):raise ValueError('Column identity mismatch')
    d=frame.to_numpy(float)/cap[None,:];rows=[]
    with threadpool_limits(limits=2):
        for h in (3,12):
            tr=np.arange(169,576-h+1,3);va=np.arange(576,648-h+1)
            target=np.stack([o[s:s+h].T for s in va]).reshape(-1,h)
            last=np.concatenate([o[s-1] for s in va])[:,None];last=np.repeat(last,h,axis=1)
            daily=np.stack([o[s-24:s-24+h].T for s in va]).reshape(-1,h)
            result={'horizon':h,'last_value':scores(last,target),'daily_naive':scores(daily,target)}
            for aux in (False,True):
                def design(origins):
                    histories=np.stack([o[s-168:s].T for s in origins]).reshape(-1,168).astype(float)
                    anchor=histories[:,-1:].copy();features=histories-anchor
                    if aux:
                        duration=np.stack([d[s-169:s-1].T for s in origins]).reshape(-1,168)
                        features=np.concatenate([features,duration],axis=1)
                    return features,anchor
                x,anchor=design(tr);vx,van=design(va);y=np.stack([o[s:s+h].T for s in tr]).reshape(-1,h)-anchor
                mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
                x=(x-mean)/scale;vx=(vx-mean)/scale
                x=np.column_stack([x,np.ones(len(x))]);vx=np.column_stack([vx,np.ones(len(vx))])
                penalty=np.eye(x.shape[1])*.01;penalty[-1,-1]=0
                beta=np.linalg.solve(x.T@x/len(x)+penalty,x.T@y/len(x))
                pred=vx@beta+van
                result['ridge_with_duration' if aux else 'ridge_occupancy']={'raw':scores(pred,target),'clipped':scores(np.clip(pred,0,1),target)}
            rows.append(result)
    report={'status':'POST_HOC_OPTIMIZATION_DIAGNOSTIC','not_registered_gate_result':True,'source':source,'fixed_ridge':.01,'training_stride':3,'rows':rows,'test_accessed':False,'sota_claim':False,
        'purpose':'Check whether original neural controls are credible against elementary baselines; do not replace the frozen neural pilot with these results',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
