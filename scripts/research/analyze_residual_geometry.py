"""Post-fit identity diagnostics only; never changes model selection or gates."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast.data import load_rate_prefix
from urbanev_forecast.foundation import file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('New diagnostic output required')
    report=json.loads((a.run/'result.json').read_text())
    if report['stage']!='screen':raise ValueError('Screen diagnostics only')
    o,source=load_rate_prefix(a.csv,1747)
    if source!=report['source']:raise ValueError('Source mismatch')
    rows=[]
    for cell in report['cells']:
        h=cell['horizon'];origins=np.arange(cell['validation_first_origin'],cell['validation_last_origin']+1,12)
        y=o[origins[:,None]+np.arange(h)].astype(float)
        def prediction(system,alpha):
            rec=next(r for r in cell['records'] if r['system']==system and r['alpha']==alpha)
            path=a.run/rec['prediction_file']
            if file_hash(path)!=rec['prediction_sha256']:raise ValueError('Prediction changed')
            return np.load(path).astype(float)
        base=prediction('native',0);error=y-base
        for system in ('bias','occupancy','duplicate','raw_duration','orthogonal_duration','permuted_orthogonal'):
            delta=prediction(system,1)-base
            aa=float(np.mean(error*delta));bb=float(np.mean(delta**2))
            direct=float(np.mean(error**2)-np.mean((error-delta)**2))
            derivative=float(np.mean(np.where(error==0,np.abs(delta),-delta*np.sign(error))))
            rows.append({'cut':cell['cut'],'horizon':h,'system':system,'error_direction_moment_a':aa,'direction_second_moment_b':bb,
                         'raw_mse_gain_at_alpha1':direct,'identity_absolute_error':abs(direct-(2*aa-bb)),
                         'empirical_raw_mse_alpha_star':aa/bb if bb>0 else None,'raw_mae_right_derivative_at_zero':derivative,
                         'diagnostic_not_an_alpha_recommendation':True})
    out={'status':'POST_FIT_IDENTITY_DIAGNOSTIC','source':source,'rows':rows,'selection_and_gates_unchanged':True,
         'script_sha256':file_hash(Path(__file__)),'scope':'raw losses on already used rolling holdouts; not clipped metric derivatives or fresh confirmation'}
    a.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'maximum_identity_error':max(r['identity_absolute_error'] for r in rows),'rows':len(rows)}))


if __name__=='__main__':main()
