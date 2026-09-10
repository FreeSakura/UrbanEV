"""Target-free training-history input-order diagnostic, not accuracy evaluation."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--repository',type=Path,required=True);parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--csv',type=Path,required=True);parser.add_argument('--backend',required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
sys.path.insert(0,str(a.repository/'src'))
from urbanev_forecast.foundation import FoundationBackend
from urbanev_forecast.data import load_rate_prefix
if a.output.exists():raise FileExistsError('fresh output required')
origins=[168,336,480];seeds=[11,22,33]
values,source=load_rate_prefix(a.csv,max(origins))
b=FoundationBackend(a.backend,a.model_dir,'cuda');rows=[]
for origin in origins:
 x=values[origin-168:origin];_,reference=b.predict(x,12)
 for seed in seeds:
  permutation=np.random.default_rng(seed).permutation(275);inverse=np.argsort(permutation)
  _,permuted=b.predict(x[:,permutation],12);restored=permuted[:,inverse]
  difference=restored.astype(float)-reference.astype(float)
  rows.append({'origin':origin,'seed':seed,'prediction_change_rms':float(np.sqrt(np.mean(difference**2))),'prediction_change_max_abs':float(np.max(np.abs(difference)))})
r={'scope':'training-history-only input-order diagnostic','not_forecast_accuracy':True,'future_label_array_constructed':False,'future_labels_used_for_metrics':False,'history_rows_loaded':480,'history':168,'channels':275,'horizon':12,'origins':origins,'seeds':seeds,'backend':b.metadata,'source':source,'rows':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps({'backend':a.backend,'mean_prediction_change_rms':float(np.mean([r['prediction_change_rms'] for r in rows])),'max_change':max(r['prediction_change_max_abs'] for r in rows)}),flush=True)
