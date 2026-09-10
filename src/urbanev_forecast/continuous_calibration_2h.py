"""Stage-specific admission and pure-array two-horizon orchestration."""
import hashlib
import json
import math

AUTH_SHA='a9677c4559a7c336a315b659a4d9ddea371d67ee2efead97bf16f1d4590fe58b'
CONFIG_SHA='6c905dc4c4f7a98376d13277f6dba4ca13da027ffe9de31e7ff8f774345cf9ca'
CONFIG_CANONICAL='d840dca653ed26806604508a88e7ebb1d76f60b008cdd083676ecf57b46047b9'
ACCEPTED_COMMIT='e7deb7754efea619e540dccdc810b5c3ebdd7e38'
VALID_SOLVER={'OK','ZERO_DIRECTION','ZERO_ONLY_FEASIBLE','OPTIMAL_ALPHA_ZERO'}


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def authorize_stage(registration,solver_config,authorization,stage,manifest,observed,prior_statuses=None):
    from .continuous_step import validate_solver_registration
    blocked=lambda reason:{'status':'NOT_AUTHORIZED','reason':reason,'real_data_loader_called':False}
    if authorization is None or manifest is None or observed is None:return blocked('Explicit stage authorization and identities required')
    if canonical(authorization)!=AUTH_SHA:return blocked('Authorization record changed')
    if stage!='calibrate_h3_h12' or manifest.get('stage')!=stage:return blocked('Wrong stage')
    if not validate_solver_registration(registration,solver_config):return blocked('Registration changed')
    if canonical(solver_config)!=CONFIG_CANONICAL or observed.get('solver_config_sha256')!=CONFIG_SHA:return blocked('Solver configuration identity mismatch')
    if authorization['accepted_solver_commit']!=ACCEPTED_COMMIT or observed.get('accepted_solver_commit')!=ACCEPTED_COMMIT:return blocked('Accepted code commit mismatch')
    if observed.get('code_identity')!=authorization['accepted_code_identity']:return blocked('Accepted solver or fitting code changed')
    if manifest.get('authorization_id')!=authorization['authorization_id'] or manifest.get('maximum_semantic_row_stop')!=1560:return blocked('Wrong scope or authorization id')
    if manifest.get('horizons')!=[3,12] or manifest.get('new_foundation_inference')!=0 or manifest.get('automatic_stage_advance') is not False:return blocked('Unauthorized horizon, inference or advancement')
    if manifest.get('systems')!=registration['systems']:return blocked('Required controls changed')
    expected=[]
    for h in (3,12):
        for cut in (720,1056,1392):
            expected.append({'cut':cut,'horizon':h,'fit_origins':list(range(192,cut,12)),
                             'calibration_origins':list(range(cut,cut+168,12)),
                             'fit_target_stop':cut,'calibration_target_stop':cut+168})
    if manifest.get('cells')!=expected:return blocked('Origin or target boundary changed')
    if any(s+c['horizon']>c['fit_target_stop'] for c in expected for s in c['fit_origins']):return blocked('Training target crosses boundary')
    if any(s+c['horizon']>min(c['calibration_target_stop'],1560) for c in expected for s in c['calibration_origins']):return blocked('Calibration target crosses boundary')
    if prior_statuses is not None:
        if set(prior_statuses)!=set(manifest['systems']):return blocked('Required control status missing')
        if any(v in ('INPUT_BLOCKED','NUMERICAL_BLOCKED','STAGE_BLOCKED') for v in prior_statuses.values()):return blocked('Required control is blocked')
    return {'status':'AUTHORIZED','stage':stage,'accepted_solver_commit':ACCEPTED_COMMIT,
            'maximum_semantic_row_stop':1560,'new_foundation_inference':0,'automatic_stage_advance':False,
            'scientific_information_gate':'NOT_RUN','max_executions':1}


def guarded_load(registration,solver_config,authorization,stage,manifest,observed,loader,prior_statuses=None):
    admission=authorize_stage(registration,solver_config,authorization,stage,manifest,observed,prior_statuses)
    if admission['status']!='AUTHORIZED':return {'admission':admission,'payload':None}
    return {'admission':admission,'payload':loader()}


def execute_systems(cells_by_system,solver_config,registration,on_result=None,solver=None,scorer=None):
    from .continuous_step import solve_continuous_alpha,score_fixed_alpha,evaluate_registered_gates
    solver=solver or solve_continuous_alpha;scorer=scorer or score_fixed_alpha
    outputs={};scores={};alphas={}
    for system in registration['systems']:
        if system not in cells_by_system:
            return {'status':'CALIBRATION_2H_BLOCKED','blocking_system':system,'reason':'Missing required system','systems':outputs,'gate':'NOT_RUN'}
        if system=='native':
            score=scorer(cells_by_system[system],0.)
            result={'status':'FIXED_NATIVE','alpha':0.,'score':score}
        else:result=solver(cells_by_system[system],solver_config)
        outputs[system]=result
        if on_result:on_result(system,result)
        outer_ok=result.get('status') in VALID_SOLVER or (system=='native' and result.get('status')=='FIXED_NATIVE')
        if not outer_ok or result.get('score',{}).get('status')!='OK':
            return {'status':'CALIBRATION_2H_BLOCKED','blocking_system':system,
                    'reason':result.get('reason','Invalid required score'),'systems':outputs,'gate':'NOT_RUN'}
        scores[system]=result['score'];alphas[system]=result['alpha']
    gates=evaluate_registered_gates(scores,alphas,registration)
    if gates.get('status')!='OK':
        return {'status':'CALIBRATION_2H_BLOCKED','reason':gates.get('reason','Gate input invalid'),'systems':outputs,'gate':gates}
    return {'status':'CALIBRATION_2H_PASS_REVIEW_REQUIRED' if gates['information_gate'] else 'CALIBRATION_2H_INFORMATION_NO_GO',
            'systems':outputs,'scores':scores,'selected_alphas':alphas,'gate':gates,'automatic_stage_advance':False}


def explain_gates(execution,registration):
    """Report evidence for the unchanged gate implementation; assert agreement."""
    if execution['status']=='CALIBRATION_2H_BLOCKED':
        return {'stage_status':execution['status'],'information_gate':'NOT_RUN','structure_gate':'NOT_RUN',
                'reason':execution.get('reason'),'blocking_system':execution.get('blocking_system'),
                'system_statuses':{s:execution['systems'].get(s,{}).get('status','NOT_RUN') for s in registration['systems']}}
    scores=execution['scores'];gate=execution['gate'];reference=gate['Cstar']
    c=scores['orthogonal_duration'];r=scores[reference];native=scores['native'];p=scores['permuted_orthogonal'];raw=scores['raw_duration']
    cells=[];blocks={}
    for index,row in enumerate(c['cells']):
        cr,rr,nr,ar=row['rmse'],r['cells'][index]['rmse'],native['cells'][index]['rmse'],raw['cells'][index]['rmse']
        cells.append({'id':row['id'],'candidate_rmse':cr,'Cstar_rmse':rr,'native_rmse':nr,'raw_duration_rmse':ar,
                      'candidate_le_1p01_native':cr<=1.01*nr,'candidate_le_1p01_Cstar':cr<=1.01*rr,
                      'candidate_le_1p01_raw_duration':cr<=1.01*ar})
        blocks.setdefault(row['id']['block'],[]).append(index)
    block_rows=[{'block':block,'candidate_rmse_mean':math.fsum(c['cells'][i]['rmse'] for i in ix)/len(ix),
                 'Cstar_rmse_mean':math.fsum(r['cells'][i]['rmse'] for i in ix)/len(ix),
                 'candidate_better':math.fsum(c['cells'][i]['rmse'] for i in ix)<math.fsum(r['cells'][i]['rmse'] for i in ix)} for block,ix in blocks.items()]
    criteria={'alpha_positive':execution['selected_alphas']['orthogonal_duration']>0,
        'macro_rmse_gain_at_least_1pct_vs_Cstar':r['macro']['rmse']>0 and c['macro']['rmse']<=.99*r['macro']['rmse'],
        'macro_mae_not_above_native':c['macro']['mae']<=native['macro']['mae'],
        'macro_mae_not_above_Cstar':c['macro']['mae']<=r['macro']['mae'],
        'all_cells_safe_vs_native':all(v['candidate_le_1p01_native'] for v in cells),
        'all_cells_safe_vs_Cstar':all(v['candidate_le_1p01_Cstar'] for v in cells),
        'at_least_two_positive_blocks':sum(v['candidate_better'] for v in block_rows)>=2,
        'rmse_strictly_better_than_permuted':c['macro']['rmse']<p['macro']['rmse'],
        'mae_not_above_permuted':c['macro']['mae']<=p['macro']['mae']}
    structure={'macro_rmse_gain_at_least_1pct_vs_raw':raw['macro']['rmse']>0 and c['macro']['rmse']<=.99*raw['macro']['rmse'],
        'mae_not_above_raw':c['macro']['mae']<=raw['macro']['mae'],
        'all_cells_safe_vs_raw':all(v['candidate_le_1p01_raw_duration'] for v in cells)}
    if all(criteria.values())!=gate['information_gate'] or all(structure.values())!=gate['structure_gate']:
        raise ValueError('Explanation differs from accepted scientific gate')
    return {'stage_status':execution['status'],'information_gate':gate['information_gate'],'structure_gate':gate['structure_gate'],
            'Cstar':reference,'criteria':criteria,'structure_criteria':structure,'cells':cells,'blocks':block_rows,
            'macro':{s:score['macro'] for s,score in scores.items()},'selected_alphas':execution['selected_alphas'],
            'failed_information_criteria':[k for k,v in criteria.items() if not v],
            'automatic_stage_advance':False}
