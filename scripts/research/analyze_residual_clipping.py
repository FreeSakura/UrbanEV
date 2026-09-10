"""Diagnostic of the derivative of clipped MAE; never tunes alpha."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast.data import load_rate_prefix
from urbanev_forecast.foundation import file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('New output required')
    r=json.loads((a.run/'result.json').read_text());o,source=load_rate_prefix(a.csv,1747)
    if r['stage']!='screen' or r['source']!=source:raise ValueError('Wrong stage or data')
    rows=[]
    for cell in r['cells']:
        def pred(system,alpha):
            row=next(v for v in cell['records'] if v['system']==system and v['alpha']==alpha);path=a.run/row['prediction_file']
            if file_hash(path)!=row['prediction_sha256']:raise ValueError('Changed predictions')
            return np.load(path).astype(float)
        p0=pred('native',0);delta=pred('orthogonal_duration',1)-p0
        origins=np.arange(cell['validation_first_origin'],cell['validation_last_origin']+1,12)
        y=o[origins[:,None]+np.arange(cell['horizon'])].astype(float);e=y-np.clip(p0,0,1)
        qprime=np.where((p0>0)&(p0<1),delta,0.)
        qprime=np.where(p0==0,np.maximum(delta,0),qprime)
        qprime=np.where(p0==1,np.minimum(delta,0),qprime)
        slope=float(np.mean(np.where(e==0,np.abs(qprime),-qprime*np.sign(e))))
        eps=1e-7
        finite=float((np.mean(np.abs(y-np.clip(p0+eps*delta,0,1)))-np.mean(np.abs(e)))/eps)
        rows.append({'cut':cell['cut'],'horizon':cell['horizon'],'clipped_mae_right_derivative':slope,
                     'finite_difference_epsilon':eps,'finite_difference':finite,
                     'derivative_absolute_difference':abs(slope-finite),'raw_prediction_outside_unit_fraction':float(np.mean((p0<0)|(p0>1)))})
    out={'status':'POST_HOC_DERIVATIVE_DIAGNOSTIC','rows':rows,'macro_clipped_mae_right_derivative':float(np.mean([v['clipped_mae_right_derivative'] for v in rows])),
         'no_new_alpha_selection':True,'registered_gate_unchanged':True,'script_sha256':file_hash(Path(__file__))}
    a.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))
if __name__=='__main__':main()
