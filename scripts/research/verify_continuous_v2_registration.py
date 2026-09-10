"""Metadata-only V2 registration check. Never loads labels or scores predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def validate_registration(c):
    if c['registered_horizons'] != [3,6,9,12] or c['primary_candidate']!='orthogonal_duration':
        raise ValueError('Unexpected registered horizons or primary candidate')
    if c['duration_latest_index']!='origin-2':
        raise ValueError('Registered duration delay changed')
    counts=[]
    for window,cut in zip(c['calibration_windows'],[720,1056,1392],strict=True):
        if window['fit_prefix']!=[0,cut] or window['evaluation_range']!=[cut,cut+168]:
            raise ValueError('Calibration boundaries changed')
        if window['fit_origins']!=list(range(192,cut,12)) or window['evaluation_origins']!=list(range(cut,cut+168,12)):
            raise ValueError('Calibration origin list changed')
        counts.append(len(window['fit_origins']))
    tail=c['development_tail']
    if tail['fit_prefix']!=[0,1560] or tail['evaluation_range']!=[1560,1740]:
        raise ValueError('Tail boundaries changed')
    if tail['fit_origins']!=list(range(192,1560,12)) or tail['evaluation_origins']!=list(range(1560,1740,12)):
        raise ValueError('Tail must use common15 origins; extra1740 is prohibited')
    flattened=[]
    for block,start in zip(tail['diagnostic_blocks'],[1560,1620,1680],strict=True):
        if block['range']!=[start,start+60] or block['origins']!=list(range(start,start+60,12)):
            raise ValueError('Tail diagnostic block changed')
        flattened.extend(block['origins'])
    if flattened!=tail['evaluation_origins']:
        raise ValueError('Blocks must cover the tail origins once')
    if c['closed_ranges']['third_fold_validation']!=[1747,1966] or c['closed_ranges']['third_fold_test']!=[1966,2184]:
        raise ValueError('Closed validation/test boundaries changed')
    checks=[]
    for window in c['calibration_windows']+[tail]:
        for h in c['registered_horizons']:
            fit=window['fit_origins'];evaluation=window['evaluation_origins']
            if any(s<192 or s+h>window['fit_prefix'][1] for s in fit):
                raise ValueError('Training target crosses fitting boundary')
            if any(s<window['evaluation_range'][0] or s+h>window['evaluation_range'][1] for s in evaluation):
                raise ValueError('Evaluation target crosses registered boundary')
            if min(fit)-169<0:
                raise ValueError('Input lag boundary invalid')
            checks.append({'fit_stop':window['fit_prefix'][1],'horizon':h,'fit_origins':len(fit),
                           'fit_last_target':fit[-1]+h-1,'evaluation_origins':len(evaluation),
                           'evaluation_last_target':evaluation[-1]+h-1})
    return {'calibration_fit_counts':counts,'tail_fit_count':len(tail['fit_origins']),
            'tail_origins_per_horizon':len(tail['evaluation_origins']),'target_boundary_checks':checks}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cache-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new output receipt')
    c=json.loads(a.config.read_text());checks=validate_registration(c)
    old=json.loads((a.cache_run/'result.json').read_text());coverage=[]
    if old['protocol']!='RESIDUAL_INFORMATION_V1_20260910' or old['source']['rows']!=1747 or old['information_gate'] is not False:
        raise ValueError('Wrong source cache protocol or state')
    prior_scored_ranges=[{'cut':cell['cut'],'horizon':cell['horizon'],
                          'first_origin':cell['validation_first_origin'],
                          'last_origin':cell['validation_last_origin'],
                          'last_target':cell['validation_last_origin']+cell['horizon']-1,
                          'last_fitting_target':cell['train_last_target']} for cell in old['cells']]
    if any(row['last_target']>=1560 or row['last_fitting_target']>=1560 for row in prior_scored_ranges):
        raise ValueError('Tail overlaps a previous recorded target-fitting or scoring range')
    required=set(c['development_tail']['fit_origins']+c['development_tail']['evaluation_origins'])
    for window in c['calibration_windows']:
        required.update(window['fit_origins']);required.update(window['evaluation_origins'])
    for h in c['registered_horizons']:
        records=[r for r in old['cache'] if r['horizon']==h]
        if not records:
            if h in (3,12):raise ValueError('Required H3/H12 source cache is missing')
            coverage.append({'horizon':h,'status':'NOT_IN_VERIFIED_V1_CACHE',
                             'action':'native inference deferred until two-horizon information gate passes; no H12 truncation'})
            continue
        rec=records[0];origins_path=a.cache_run/f'private_origins_h{h}.npy';prediction_path=a.cache_run/f'private_base_h{h}.npy'
        if digest(origins_path)!=rec['origins_sha256'] or digest(prediction_path)!=rec['prediction_sha256']:
            raise ValueError('Cache file hash changed')
        origins=np.load(origins_path,allow_pickle=False)
        predictions=np.load(prediction_path,mmap_mode='r',allow_pickle=False)
        if origins.ndim!=1 or not np.issubdtype(origins.dtype,np.integer) or len(np.unique(origins))!=len(origins):
            raise ValueError('Invalid origin inventory')
        if predictions.shape!=(len(origins),h,275):raise ValueError('Cache shape mismatch')
        missing=sorted(required-set(origins.tolist()))
        if missing:raise ValueError(f'Missing registered origins for H{h}: {missing}')
        coverage.append({'horizon':h,'status':'PASS','shape':list(predictions.shape),
                         'origins_sha256':rec['origins_sha256'],'prediction_sha256':rec['prediction_sha256'],
                         'required_unique_origins':len(required),'missing_origins':[],
                         'excluded_tail_origins_present_in_cache':[int(s) for s in origins if s>=1560 and s not in c['development_tail']['evaluation_origins']]})
    result={'protocol':c['protocol_id'],'status':'REGISTRATION_AND_INDEX_CHECK_PASS_SOLVER_REVIEW_PENDING',
            'config_sha256':digest(a.config),'verifier_sha256':digest(Path(__file__)),
            'source_result_sha256':digest(a.cache_run/'result.json'),'source_prefix_metadata':old['source'],
            **checks,'cache_coverage':coverage,'tail_exposure':c['exposure'],
            'prior_recorded_ranges':prior_scored_ranges,'tail_overlaps_prior_recorded_fitting_or_scoring_targets':False,
            'actions_performed':['configuration validation','origin/index checks','opaque cache byte hashes','array header shape inspection'],
            'raw_label_files_opened':False,'prediction_values_scored':False,'new_inference':False,
            'continuous_search':False,'tail_scoring':False,'third_fold_validation_or_test_opened':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'fit_counts':checks['calibration_fit_counts']+[checks['tail_fit_count']],
                      'cache_coverage':[{k:v for k,v in row.items() if k in ['horizon','status','missing_origins']} for row in coverage]}))


if __name__=='__main__':main()
