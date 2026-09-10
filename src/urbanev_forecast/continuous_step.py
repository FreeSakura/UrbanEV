"""Pure-array continuous-step solver. No data/model/file loading and no gate calls.

Cells contain an id and p/delta/y arrays. ids may include stage/block/horizon.
All numerical blocks are explicit; no numerical tolerance enlarges MAE budgets.
"""
from array import array
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import sys
import numpy as np

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


class _Sum:
    """Neumaier compensated state; product roundoff is still independently checked."""
    def __init__(self,value):self.hi=float(value);self.lo=0.
    @property
    def value(self):return self.hi+self.lo
    def add(self,x):
        t=self.hi+x
        self.lo += (self.hi-t)+x if abs(self.hi)>=abs(x) else (x-t)+self.hi
        self.hi=t
    def zero(self):self.hi=0.;self.lo=0.


@dataclass
class _Segment:
    left:float
    right:float
    W:tuple
    G:tuple
    U:tuple
    B:tuple
    slopes:tuple


def _exact_event(p,d,y,sample,kind):
    bound=0. if kind==0 else (1. if kind==2 else float(y[sample]))
    return (Fraction.from_float(bound)-Fraction.from_float(float(p[sample])))/Fraction.from_float(float(d[sample]))


def _events(cells,stats):
    p=np.concatenate([c['p'] for c in cells]);d=np.concatenate([c['delta'] for c in cells]);y=np.concatenate([c['y'] for c in cells])
    owner=np.concatenate([np.full(len(c['p']),j,dtype=np.int32) for j,c in enumerate(cells)])
    times=[];samples=[];kinds=[];ix=np.flatnonzero(d!=0)
    for kind,bound in [(0,np.zeros(len(ix))),(1,y[ix]),(2,np.ones(len(ix)))]:
        with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='raise'):
            t=(bound-p[ix])/d[ix]
        for pos in np.flatnonzero((t==0)|(t==1)):
            exact=_exact_event(p,d,y,int(ix[pos]),kind)
            if 0<exact<1 and exact!=Fraction.from_float(float(t[pos])):
                raise NumericalBlocked('Interior event collapses onto domain endpoint')
        good=np.isfinite(t)&(t>0)&(t<1)
        times.append(t[good]);samples.append(ix[good]);kinds.append(np.full(good.sum(),kind,dtype=np.int8))
    ts=np.concatenate(times);ss=np.concatenate(samples);kk=np.concatenate(kinds)
    order=np.argsort(ts,kind='stable');ts,ss,kk=ts[order],ss[order],kk[order]
    stats['events']=len(ts);stats['full_sample_scans']+=3
    # Exact-rational comparisons are used only to detect float event collisions.
    for j in range(1,len(ts)):
        if ts[j]-ts[j-1] <= 4*math.ulp(float(ts[j])):
            a=_exact_event(p,d,y,int(ss[j-1]),int(kk[j-1]));b=_exact_event(p,d,y,int(ss[j]),int(kk[j]))
            if (ts[j]==ts[j-1] and a!=b) or (ts[j]>ts[j-1] and a>=b):
                raise NumericalBlocked('Different event locations cannot be reliably ordered')
    return p,d,y,owner,ts,ss,kk


def _segments(cells,events,stats):
    p,d,y,owner,ts,ss,kk=events
    q=np.clip(p,0,1)
    active=((p>0)&(p<1))|((p==0)&(d>0))|((p==1)&(d<0))
    v=np.where(active,d,0.);residual=q-y
    slope=np.where(residual>0,v,np.where(residual<0,-v,np.abs(v)))
    if np.any((v!=0)&(v*v==0)):raise NumericalBlocked('Nonzero direction second moment underflows')
    states=[];counts=[];n=[]
    offset=0
    for cell in cells:
        size=len(cell['p']);a=slice(offset,offset+size);n.append(size)
        terms=[math.fsum(float(z)*float(z) for z in residual[a])/size,
               math.fsum(float(r)*float(w) for r,w in zip(residual[a],v[a]))/size,
               math.fsum(float(z)*float(z) for z in v[a])/size,
               math.fsum(abs(float(z)) for z in residual[a])/size,
               math.fsum(float(z) for z in slope[a])/size]
        states.append([_Sum(z) for z in terms]);counts.append(int(np.count_nonzero(v[a])));offset+=size
    stats['full_sample_scans']+=1
    left=0.;index=0
    while True:
        right=float(ts[index]) if index<len(ts) else 1.
        values=[tuple(row[k].value for row in states) for k in range(5)]
        if any(not math.isfinite(z) for row in values for z in row):raise NumericalBlocked('Nonfinite scan state')
        if any(z<0 for z in values[0]+values[2]):raise NumericalBlocked('Negative scan MSE or direction energy')
        yield _Segment(left,right,*values)
        if right==1:break
        distance=right-left
        for state in states:
            w,g,u,b,m=[x.value for x in state]
            state[0].add(2*g*distance);state[0].add(u*distance*distance)
            state[1].add(u*distance);state[3].add(m*distance)
        stop=index+1
        while stop<len(ts) and ts[stop]==right:stop+=1
        unique={}
        for at in range(index,stop):unique.setdefault(int(ss[at]),set()).add(int(kk[at]))
        changes={}
        for sample,kinds in unique.items():
            cell=int(owner[sample]);dv=float(d[sample]);oldv=float(v[sample]);oldm=float(slope[sample])
            if 0 in kinds:
                qe=0.;newv=dv if dv>0 else 0.
            elif 2 in kinds:
                qe=1.;newv=dv if dv<0 else 0.
            else:qe=float(y[sample]);newv=oldv
            err=qe-float(y[sample]);newm=newv if err>0 else (-newv if err<0 else abs(newv))
            changes.setdefault(cell,[[],[],[]])
            changes[cell][0].append(err*(newv-oldv))
            changes[cell][1].append(newv*newv-oldv*oldv)
            changes[cell][2].append(newm-oldm)
            counts[cell]+=int(newv!=0)-int(oldv!=0)
            v[sample]=newv;slope[sample]=newm
        for cell,(gs,us,ms) in changes.items():
            states[cell][1].add(math.fsum(gs)/n[cell]);states[cell][2].add(math.fsum(us)/n[cell]);states[cell][4].add(math.fsum(ms)/n[cell])
            if counts[cell]==0:
                # This zero is proved by absence of any moving sample, not clipping a negative state.
                states[cell][1].zero();states[cell][2].zero();states[cell][4].zero()
        index=stop;left=right


def _linear_interval(b,c,lo,hi):
    if b==0:return (lo,hi) if c<=0 else None
    root=-c/b
    if not math.isfinite(root):raise NumericalBlocked('Nonfinite linear root')
    if b>0:hi=min(hi,root)
    else:lo=max(lo,root)
    return (lo,hi) if lo<=hi else None


def _quadratic_interval(a,b,c,lo,hi):
    if a<0:raise NumericalBlocked('Nonconvex quadratic scan state')
    if a==0:return _linear_interval(b,c,lo,hi)
    bb=b*b;ac=4*a*c;disc=math.fsum([bb,-ac])
    if not all(math.isfinite(z) for z in (bb,ac,disc)):raise NumericalBlocked('Quadratic overflow')
    bound=32*sys.float_info.epsilon*(abs(bb)+abs(ac))
    if abs(disc)<=bound:
        exact=Fraction.from_float(b)**2-4*Fraction.from_float(a)*Fraction.from_float(c)
        if exact!=0:raise NumericalBlocked('Discriminant sign or multiplicity unresolved at float64 precision')
        disc=0.
    if disc<0:return None
    root=math.sqrt(disc)
    z=-.5*(b+math.copysign(root,b))
    if z==0:r1=r2=-b/(2*a)
    else:r1,r2=sorted((z/a,c/z))
    lo,hi=max(lo,r1),min(hi,r2)
    return (lo,hi) if lo<=hi else None


def _feasible_segment(seg,native):
    lo,hi=0.,seg.right-seg.left
    meanB=math.fsum(seg.B)/len(seg.B);meanm=math.fsum(seg.slopes)/len(seg.slopes)
    interval=_linear_interval(meanm,meanB-native['macro']['mae'],lo,hi)
    if interval is None:return None
    lo,hi=interval
    for w,g,u,row in zip(seg.W,seg.G,seg.U,native['cells']):
        cap=(1.01*row['rmse'])**2
        interval=_quadratic_interval(u,2*g,w-cap,lo,hi)
        if interval is None:return None
        lo,hi=interval
    return lo,hi


def _scan_score(seg,t):
    rmses=[];maes=[]
    for w,g,u,b,m in zip(seg.W,seg.G,seg.U,seg.B,seg.slopes):
        mse=math.fsum([w,2*g*t,u*t*t]);mae=math.fsum([b,m*t])
        if mse<0 or mae<0:raise NumericalBlocked('Negative polynomial loss')
        rmses.append(math.sqrt(mse));maes.append(mae)
    return {'cell_rmse':rmses,'cell_mae':maes,'rmse':math.fsum(rmses)/len(rmses),'mae':math.fsum(maes)/len(maes)}


def _derivative(seg,t,side):
    terms=[]
    for w,g,u in zip(seg.W,seg.G,seg.U):
        mse=math.fsum([w,2*g*t,u*t*t]);numerator=math.fsum([g,u*t])
        if mse<0:raise NumericalBlocked('Negative derivative MSE')
        if mse==0:
            if numerator!=0:raise NumericalBlocked('Zero norm with contradictory moment')
            terms.append(side*math.sqrt(u) if u else 0.)
        else:terms.append(numerator/math.sqrt(mse))
    return math.fsum(terms)/len(terms)


def _minimum(seg,lo,hi,config,stats):
    if lo==hi:return lo
    if _derivative(seg,lo,1)>=0:return lo
    if _derivative(seg,hi,-1)<0:return hi
    left,right=lo,hi;lip=math.fsum(math.sqrt(u) for u in seg.U)/len(seg.U)
    stats['interior_minimizations']+=1
    for _ in range(config['max_bisection_iterations']):
        if right-left<=config['alpha_xtol'] and lip*(right-left)<=config['objective_search_atol']:
            return right
        mid=left+(right-left)/2
        if mid==left or mid==right:raise NumericalBlocked('Bisection has no representable interior')
        if _derivative(seg,mid,1)>=0:right=mid
        else:left=mid
        stats['bisection_iterations']+=1
    raise NumericalBlocked('Bisection exhausted its registered budget')


def _direct_feasible(score,native):
    return score['macro']['mae']<=native['macro']['mae'] and all(r['rmse']<=1.01*n['rmse'] for r,n in zip(score['cells'],native['cells']))


def solve_continuous_alpha(cells,solver_config):
    """Select alpha without system names, C*, other scores, or gate thresholds."""
    stats={'events':0,'segments':0,'candidates':0,'interior_minimizations':0,'bisection_iterations':0,
           'full_sample_scans':0,'boundary_repairs':0,'replay_segments':0}
    try:
        if any(solver_config.get(k)!=v for k,v in NUMERICS.items()):raise InputBlocked('Numerical contract changed')
        prepared=_prepare(cells)
        if any(isinstance(c['id'],dict) and c['id'].get('stage')=='tail' for c in prepared):raise InputBlocked('Tail selection prohibited')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            native=_score_prepared(prepared,0.);stats['full_sample_scans']+=1
            if all(np.all(c['delta']==0) for c in prepared):
                return {'status':'ZERO_DIRECTION','alpha':0.,'positive_feasible_exists':True,'feasible_components':[[0.,1.]],'score':native,'stats':stats}
            events=_events(prepared,stats);candidates=array('d');components=[]
            for seg in _segments(prepared,events,stats):
                stats['segments']+=1;interval=_feasible_segment(seg,native)
                if interval is None:continue
                lo,hi=interval;aa,bb=seg.left+lo,seg.left+hi
                if components and aa<=components[-1][1]:components[-1][1]=max(components[-1][1],bb)
                else:components.append([aa,bb])
                best=_minimum(seg,lo,hi,solver_config,stats)
                for t in (lo,hi,best):
                    objective=_scan_score(seg,t)['rmse']
                    candidates.extend((seg.left+t,objective,aa,bb))
            if not components:raise NumericalBlocked('Empty feasibility contradicts native alpha0')
            if not any(a<=0<=b for a,b in components):raise NumericalBlocked('Native point disappeared from feasibility')
            candidates.extend((0.,native['macro']['rmse'],0.,0.))
            table=np.frombuffer(candidates,dtype=np.float64).reshape(-1,4);stats['candidates']=len(table)
            rmin=float(np.min(table[:,1]));possible=np.flatnonzero(table[:,1]<=rmin+solver_config['objective_tie_atol'])
            selected=int(possible[np.argmin(table[possible,0])]);alpha,objective,lo,hi=map(float,table[selected])
            scan=None
            for seg in _segments(prepared,events,stats):
                stats['replay_segments']+=1
                if seg.left<=alpha<=seg.right:
                    scan=_scan_score(seg,alpha-seg.left);break
            if scan is None:raise NumericalBlocked('Selected point absent from replay')
            direct=_score_prepared(prepared,alpha);stats['full_sample_scans']+=1
            differences=[abs(scan['rmse']-direct['macro']['rmse']),abs(scan['mae']-direct['macro']['mae'])]
            differences += [abs(x-r['rmse']) for x,r in zip(scan['cell_rmse'],direct['cells'])]
            differences += [abs(x-r['mae']) for x,r in zip(scan['cell_mae'],direct['cells'])]
            discrepancy=max(differences)
            if discrepancy>solver_config['score_verification_atol']:raise NumericalBlocked('Scan/direct discrepancy exceeds contract')
            original_alpha=alpha
            if not _direct_feasible(direct,native):
                violation=max([direct['macro']['mae']-native['macro']['mae']]+[r['rmse']-1.01*n['rmse'] for r,n in zip(direct['cells'],native['cells'])])
                if violation>solver_config['score_verification_atol']:raise NumericalBlocked('Selected point infeasible beyond verification tolerance')
                middle=lo+(hi-lo)/2;distance=min(solver_config['boundary_repair_max_alpha_change'],(hi-lo)/4,abs(middle-alpha))
                repaired=alpha+math.copysign(distance,middle-alpha)
                if distance<=0 or repaired==alpha or not lo<repaired<hi:raise NumericalBlocked('No representable repair inside same feasible interval')
                fixed=_score_prepared(prepared,repaired);stats['full_sample_scans']+=1;stats['boundary_repairs']=1
                if not _direct_feasible(fixed,native) or fixed['macro']['rmse']-direct['macro']['rmse']>solver_config['boundary_repair_max_objective_increase']:
                    raise NumericalBlocked('Single registered boundary repair failed')
                alpha,direct=repaired,fixed
            positive=any(b>0 for a,b in components)
            status='OK' if alpha>0 else ('OPTIMAL_ALPHA_ZERO' if positive else 'ZERO_ONLY_FEASIBLE')
            return {'status':status,'alpha':alpha,'original_selected_alpha':original_alpha,'positive_feasible_exists':positive,
                    'feasible_components':components,'score':direct,'scan_direct_max_difference':discrepancy,
                    'finite_candidate_minimum':rmin,'stats':stats,
                    'claim':'registered-domain numerical all-segment search with listed verification; not an exact global certificate'}
    except (InputBlocked,ValueError,TypeError,KeyError) as e:
        return {'status':'INPUT_BLOCKED','reason':str(e),'stats':stats}
    except (NumericalBlocked,ArithmeticError,FloatingPointError,OverflowError) as e:
        return {'status':'NUMERICAL_BLOCKED','reason':str(e),'stats':stats}


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
