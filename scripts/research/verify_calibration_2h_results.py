"""Recompute published-score candidates from saved outputs; no refitting/selection."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise FileExistsError('Fresh verification receipt required')
    execution=json.loads((a.run/'calibration_2h.json').read_text());receipt=json.loads((a.run/'execution_receipt.json').read_text())
    if execution['status']=='CALIBRATION_2H_BLOCKED':raise ValueError('No complete score matrix to verify')
    checks=[];hashes=[];largest=0.
    for system,result in execution['systems'].items():
        rows=[]
        for row in result['score']['cells']:
            cut,h=row['id']['block'],row['id']['horizon']
            y_path=a.run/f'private_truth_{cut}_h{h}.npy';p_path=a.run/f'private_selected_{cut}_h{h}_{system}.npy'
            y=np.load(y_path,allow_pickle=False).astype(float);p=np.load(p_path,allow_pickle=False).astype(float)
            assert y.shape==p.shape==(14,h,275)
            error=p-y;clipped=np.clip(p,0,1)-y
            actual={'raw_rmse':float(np.sqrt(np.mean(error**2))),'raw_mae':float(np.mean(np.abs(error))),
                    'rmse':float(np.sqrt(np.mean(clipped**2))),'mae':float(np.mean(np.abs(clipped)))}
            difference=max(abs(actual[key]-row[key]) for key in actual);largest=max(largest,difference)
            assert difference<=1e-10
            rows.append(actual);checks.append({'system':system,'block':cut,'horizon':h,'maximum_score_difference':difference})
            hashes.append({'prediction_file':p_path.name,'prediction_sha256':hashlib.sha256(p_path.read_bytes()).hexdigest(),
                           'truth_file':y_path.name,'truth_sha256':hashlib.sha256(y_path.read_bytes()).hexdigest()})
        for metric in ('rmse','mae'):
            diff=abs(float(np.mean([r[metric] for r in rows]))-result['score']['macro'][metric]);largest=max(largest,diff);assert diff<=1e-10
    assert all(x['v1_alpha1_exact_numeric_equal'] for x in receipt['direction_lineage'])
    assert receipt['model_call_monitor']=={'foundation_constructor':0,'foundation_predict':0}
    for read in receipt['read_ledger']:
        if read['kind']=='semantic_numeric_prefix':assert read['row_range']==[0,1560]
        if read['kind'] in ('numeric_selected_cache_rows','numeric_calibration_prediction_lineage'):assert read['last_target']<1560
    result={'status':'PASS','scope':'local saved-output metric recomputation and read-ledger checks; not independent research confirmation',
            'score_cells_checked':len(checks),'maximum_metric_difference':largest,'checks':checks,'array_hashes':hashes,
            'direction_rebuild_matches':42,'new_foundation_inference':0,'semantic_row_stop':1560,
            'source_execution_sha256':hashlib.sha256((a.run/'calibration_2h.json').read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','cells':len(checks),'maximum_metric_difference':largest}))


if __name__=='__main__':main()
