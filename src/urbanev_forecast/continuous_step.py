"""Pure-array continuous-step solver. No data/model/file loading and no gate calls.

Cells contain an id and p/delta/y arrays. ids may include stage/block/horizon.
All numerical blocks are explicit; no numerical tolerance enlarges MAE budgets.
"""
import hashlib
import json
import math
import numpy as np
from .continuous_exact import solve_exact

REGISTRATION_SHA = '05da76dfc3c91fb2e9e4804960ab62d29adc0f7bbd4acf724ac654b1efb860e9'
NUMERICS = dict(alpha_xtol=1e-12, objective_search_atol=1e-12,
                objective_tie_atol=1e-12, max_bisection_iterations=80,
                score_verification_atol=1e-10, boundary_repair_max_alpha_change=1e-10,
                boundary_repair_max_objective_increase=1e-10, max_boundary_repairs=1,
                native_cell_rmse_multiplier=1.01, mae_scientific_degradation_budget=0.0)


class InputBlocked(ValueError):
    pass


class NumericalBlocked(ArithmeticError):
    pass


def _prepare(cells):
    if not isinstance(cells,(list,tuple)) or not cells:
        raise InputBlocked('Nonempty cells required')
    prepared=[];seen=set()
    for c in cells:
        if not isinstance(c,dict) or not {'id','p','delta','y'} <= set(c):
            raise InputBlocked('Cell requires id,p,delta,y')
        key=json.dumps(c['id'],sort_keys=True)
        if key in seen:raise InputBlocked('Duplicate cell id')
        seen.add(key)
        if any(np.iscomplexobj(c[name]) for name in ('p','delta','y')):raise InputBlocked('Real arrays required')
        values=[np.asarray(c[name],dtype=np.float64) for name in ('p','delta','y')]
        if values[0].size==0 or any(v.shape!=values[0].shape for v in values):
            raise InputBlocked('Empty cell or mismatched shape')
        if any(not np.isfinite(v).all() for v in values):raise InputBlocked('Nonfinite input')
        if np.any((values[2]<0)|(values[2]>1)):raise InputBlocked('Labels outside[0,1]')
        prepared.append({'id':c['id'],'shape':values[0].shape,
                         **dict(zip(('p','delta','y'),[v.ravel(order='C') for v in values]))})
    return prepared


def _score_prepared(cells,alpha):
    rows=[]
    for cell in cells:
        raw=cell['p']+alpha*cell['delta']
        if not np.isfinite(raw).all():raise NumericalBlocked('Prediction arithmetic overflow')
        q=np.clip(raw,0,1);e=q-cell['y'];n=len(e)
        mse=math.fsum(float(v)*float(v) for v in e)/n
        mae=math.fsum(abs(float(v)) for v in e)/n
        eraw=raw-cell['y']
        rows.append({'id':cell['id'],'n':n,'rmse':math.sqrt(mse),'mae':mae,
                     'data_fingerprint':hashlib.sha256(cell['p'].tobytes()+cell['y'].tobytes()).hexdigest(),
                     'mse':mse,'raw_rmse':math.sqrt(math.fsum(float(v)*float(v) for v in eraw)/n),
                     'raw_mae':math.fsum(abs(float(v)) for v in eraw)/n})
    if any(not math.isfinite(row[k]) for row in rows for k in ('rmse','mae','raw_rmse','raw_mae')):
        raise NumericalBlocked('Nonfinite scores')
    stage={row['id'].get('stage','calibration') if isinstance(row['id'],dict) else 'calibration' for row in rows}
    if len(stage)!=1:raise InputBlocked('Mixed evaluation stages')
    if stage=={'tail'}:
        groups={}
        for row in rows:
            if 'horizon' not in row['id']:raise InputBlocked('Tail requires horizon id')
            groups.setdefault(row['id']['horizon'],[]).append(row)
        aggregate=[{'rmse':math.sqrt(math.fsum(r['mse']*r['n'] for r in group)/sum(r['n'] for r in group)),
                    'mae':math.fsum(r['mae']*r['n'] for r in group)/sum(r['n'] for r in group)} for group in groups.values()]
    else:aggregate=rows
    return {'status':'OK','alpha':float(alpha),'cells':rows,
            'macro':{key:math.fsum(row[key] for row in aggregate)/len(aggregate) for key in ('rmse','mae')},
            'aggregation':'per-horizon pooled then equal horizons' if stage=={'tail'} else 'equal cells'}


def score_fixed_alpha(cells,alpha):
    """Independent direct scorer using raw p + alpha*delta before clipping."""
    try:
        if not math.isfinite(float(alpha)) or not 0<=alpha<=1:raise InputBlocked('Invalid alpha')
        with np.errstate(over='raise',invalid='raise'):
            return _score_prepared(_prepare(cells),float(alpha))
    except (InputBlocked,ValueError,TypeError,KeyError) as e:
        return {'status':'INPUT_BLOCKED','reason':str(e)}
    except (ArithmeticError,OverflowError,FloatingPointError) as e:
        return {'status':'NUMERICAL_BLOCKED','reason':str(e)}


def solve_continuous_alpha(cells,solver_config):
    """Exact feasible-set classification with enclosed numerical optimization."""
    try:
        if any(solver_config.get(k)!=v for k,v in NUMERICS.items()):
            raise InputBlocked('Numerical contract changed')
        if solver_config.get('repair_spec_id')!='CONTINUOUS_STEP_V2_NUMERICAL_REPAIR_20260910' or solver_config.get('root_refinement_bits')!=[64,128,256,512]:
            raise InputBlocked('Reviewed exact numerical repair configuration required')
        prepared=_prepare(cells)
        if any(isinstance(c['id'],dict) and c['id'].get('stage')=='tail' for c in prepared):
            raise InputBlocked('Tail selection prohibited')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            result=solve_exact(prepared,solver_config,_score_prepared)
        result['stats']['segments']=result['stats']['segments_total']
        return result
    except (InputBlocked,ValueError,TypeError,KeyError) as e:
        return {'status':'INPUT_BLOCKED','reason':str(e),'stats':{}}
    except (ArithmeticError,OverflowError,FloatingPointError) as e:
        return {'status':'NUMERICAL_BLOCKED','reason':str(e),'stats':{}}


def validate_solver_registration(registration,solver_config):
    actual=hashlib.sha256(json.dumps(registration,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return actual==REGISTRATION_SHA==solver_config.get('registration_canonical_sha256')


def authorize_real_calibration(registration,solver_config):
    """Read-only admission check; no runtime path to private data is implemented."""
    if not validate_solver_registration(registration,solver_config):return {'status':'INPUT_BLOCKED','reason':'Registration changed'}
    return {'status':'NOT_AUTHORIZED','reason':'Synthetic implementation only; explicit research code acceptance pending'}


def evaluate_registered_gates(system_scores,selected_alphas,registration):
    """Score-only post-selection comparison; never invokes the solver."""
    try:
        if hashlib.sha256(json.dumps(registration,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=REGISTRATION_SHA:
            raise InputBlocked('Registration changed')
        systems=registration['systems']
        if set(system_scores)!=set(systems) or set(selected_alphas)!=set(systems):raise InputBlocked('Missing required system')
        if any(system_scores[s].get('status')!='OK' for s in systems):
            return {'status':'STAGE_BLOCKED','reason':'A required system is numerically/input blocked or missing; none may be dropped'}
        native=system_scores['native'];ids=[json.dumps(r['id'],sort_keys=True) for r in native['cells']]
        for s in systems:
            score=system_scores[s]
            if [json.dumps(r['id'],sort_keys=True) for r in score['cells']]!=ids or [r['n'] for r in score['cells']]!=[r['n'] for r in native['cells']]:
                raise InputBlocked('Cell identities or support differ')
            if [r['data_fingerprint'] for r in score['cells']]!=[r['data_fingerprint'] for r in native['cells']]:raise InputBlocked('Native arrays or targets differ')
            if any(not math.isfinite(float(score['macro'][k])) or score['macro'][k]<0 for k in ('rmse','mae')) or any(row['n']<=0 or any(not math.isfinite(float(row[k])) or row[k]<0 for k in ('rmse','mae','mse')) for row in score['cells']):raise InputBlocked('Invalid score values')
            if not math.isfinite(float(selected_alphas[s])) or not 0<=selected_alphas[s]<=1:raise InputBlocked('Invalid selected alpha')
            if score['alpha']!=selected_alphas[s]:raise InputBlocked('Score and frozen alpha differ')
        if selected_alphas['native']!=0:raise InputBlocked('Native alpha must be zero')
        blocks={};horizons=set();stages=set()
        for i,row in enumerate(native['cells']):
            ident=row['id']
            if not isinstance(ident,dict) or not {'stage','block','horizon'}<=set(ident):raise InputBlocked('Gate cell id requires stage/block/horizon')
            blocks.setdefault(ident['block'],[]).append(i);horizons.add(ident['horizon']);stages.add(ident['stage'])
        if len(blocks)!=3 or len(stages)!=1 or horizons not in ({3,12},{3,6,9,12}):raise InputBlocked('Wrong stage dimensions')
        if any({native['cells'][i]['id']['horizon'] for i in ix}!=horizons or len(ix)!=len(horizons) for ix in blocks.values()):raise InputBlocked('Incomplete block/horizon grid')
        if stages=={'tail'} and horizons!={3,6,9,12}:raise InputBlocked('Tail requires four horizons')
        if stages not in ({'tail'},{'calibration'}):raise InputBlocked('Unknown stage')
        references=registration['non_duration_controls']
        reference=min(references,key=lambda s:system_scores[s]['macro']['rmse'])
        candidate=system_scores['orthogonal_duration'];control=system_scores[reference];perm=system_scores['permuted_orthogonal'];raw=system_scores['raw_duration']
        def gain(c,r):return r['macro']['rmse']>0 and c['macro']['rmse']<=.99*r['macro']['rmse']
        def cell_safe(c,r):return all(a['rmse']<=1.01*b['rmse'] for a,b in zip(c['cells'],r['cells']))
        positive=sum(math.fsum(candidate['cells'][i]['rmse'] for i in ix)<math.fsum(control['cells'][i]['rmse'] for i in ix) for ix in blocks.values())
        info=selected_alphas['orthogonal_duration']>0 and gain(candidate,control) and candidate['macro']['mae']<=native['macro']['mae'] and candidate['macro']['mae']<=control['macro']['mae'] and cell_safe(candidate,native) and cell_safe(candidate,control) and positive>=2 and candidate['macro']['rmse']<perm['macro']['rmse'] and candidate['macro']['mae']<=perm['macro']['mae']
        structure=gain(candidate,raw) and candidate['macro']['mae']<=raw['macro']['mae'] and cell_safe(candidate,raw)
        return {'status':'OK','information_gate':bool(info),'structure_gate':bool(structure),'Cstar':reference,
                'positive_blocks':int(positive),'third_fold_validation_admitted':False,'alpha_reselection_performed':False}
    except (InputBlocked,ValueError,TypeError,KeyError) as e:
        return {'status':'INPUT_BLOCKED','reason':str(e)}
