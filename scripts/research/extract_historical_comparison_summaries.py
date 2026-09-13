"""Extract original reported legacy metrics without reopening targets or predictions."""
from pathlib import Path
import json,csv,hashlib
r=Path(__file__).resolve().parents[2];out=r/'artifacts/summaries/comprehensive_development_comparison_v1';records=[]
for cohort,relative in [('legacy_fixed_fusion','artifacts/summaries/fixed_fusion/FIXED_FUSION_SUMMARY.json'),('legacy_router','artifacts/summaries/router/ROUTER_V1_SUMMARY.json')]:
 p=r/relative;j=json.loads(p.read_text());digest=hashlib.sha256(p.read_bytes()).hexdigest()
 for v in j['methods']:
  records.append({'cohort':cohort,'method':v['method'],'cells':v.get('cells',24),'rmse':v['RMSE'],'mae':v.get('MAE'),'result_role':'ORACLE_DIAGNOSTIC_NOT_DEPLOYABLE' if 'oracle' in v['method'].lower() else 'HISTORICAL_REPORTED','output_convention':'original artifact convention, not newly rescored','source':relative,'source_sha256':digest})
p=r/'artifacts/summaries/chronos2/CHRONOS2_QUALIFICATION_SUMMARY.json';j=json.loads(p.read_text());records.append({'cohort':'legacy_chronos_qualification','method':'Chronos2_native_clipped','cells':j['cells'],'rmse':j['Chronos2_clipped_macro']['RMSE'],'mae':j['Chronos2_clipped_macro']['MAE'],'result_role':'HISTORICAL_REPORTED','output_convention':'clipped native, original qualification protocol','source':str(p.relative_to(r)).replace('\\','/'),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
p=r/'artifacts/summaries/distillation/D1_SUMMARY.json';j=json.loads(p.read_text())
cell_path=r/'artifacts/summaries/distillation/D1_CELLS.csv'
with cell_path.open(encoding='utf-8-sig') as f:cells=list(csv.DictReader(f))
for method,value in j['macro_RMSE'].items():
 rr=[v for v in cells if v['branch']==method];assert len(rr)==24
 assert abs(sum(float(v['RMSE']) for v in rr)/24-value)<1e-12
 records.append({'cohort':'legacy_distillation','method':method,'cells':24,'rmse':value,'mae':sum(float(v['MAE']) for v in rr)/24,'result_role':'HISTORICAL_REPORTED','output_convention':'D1 original24cell aggregate, no new target scoring','source':str(cell_path.relative_to(r)).replace(chr(92),'/'),'source_sha256':hashlib.sha256(cell_path.read_bytes()).hexdigest()})
with (out/'historical_six_fold_reported.csv').open('w',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
print('historical source records',len(records),'no new inference or target scoring')
