"""Incremental exact feasibility with enclosed numerical objective optimization."""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import json
import math
import time
import numpy as np
from .continuous_boundaries import (BITS,Boundary,BoundaryUnresolved,compare,sign,
    intersect_polynomial,sqrt_bounds,polynomial_bounds,divide_interval,add_interval,
    midpoint_inside,note_bits)

ZERO=F(0);ONE=F(1)


@dataclass
class Segment:
    index:int
    left:Boundary
    right:Boundary
    moment_sums:tuple
    sizes:tuple
    mae:tuple
    native_mse:tuple
    native_mae:F

    def mse(self,j):
        return tuple(v/self.sizes[j] for v in self.moment_sums[j][:3])

    def constraints(self):
        yield (ZERO,self.mae[0],self.mae[1]-self.native_mae)
        for j,m0 in enumerate(self.native_mse):
            a,b,c=self.mse(j)
            yield a,b,c-F(10201,10000)*m0


def _representation(p,d,y,alpha):
    value=p+d*alpha
    if value<0 or (value==0 and d<=0):u,v=ZERO,ZERO
    elif value>1 or (value==1 and d>=0):u,v=ONE,ZERO
    else:u,v=p,d
    residual=u+v*alpha-y
    sigma=(1 if residual>0 else -1) if residual else (1 if v>0 else (-1 if v<0 else 0))
    return u,v,sigma


def _contribution(rep,y):
    u,v,sigma=rep;r=u-y
    return (v*v,2*r*v,r*r,sigma*v,sigma*r)


class Sweep:
    def __init__(self,cells,stats):
        self.stats=stats;self.sizes=tuple(len(c['p']) for c in cells)
        self.samples=[];self.events=[];self.initial=[];self.initial_reps=[]
        for j,cell in enumerate(cells):
            state=[ZERO]*5
            for pp,dd,yy in zip(cell['p'],cell['delta'],cell['y']):
                p,d,y=(F.from_float(float(v)) for v in (pp,dd,yy))
                sample=len(self.samples);self.samples.append((p,d,y,j))
                rep=_representation(p,d,y,ZERO);self.initial_reps.append(rep)
                contribution=_contribution(rep,y)
                state=[old+add for old,add in zip(state,contribution)]
                note_bits(stats,p,d,y,*contribution)
                if d:
                    for target in (ZERO,y,ONE):
                        event=(target-p)/d
                        if 0<event<1:self.events.append((event,sample))
            self.initial.append(tuple(state));note_bits(stats,*state)
        self.events.sort()
        self.native_mse=tuple(state[2]/n for state,n in zip(self.initial,self.sizes))
        self.native_mae=sum((state[4]/n for state,n in zip(self.initial,self.sizes)),ZERO)/len(cells)
        stats['events']=len(self.events);stats['exact_initialization_passes']=1

    def segments(self):
        state=[list(row) for row in self.initial];reps=list(self.initial_reps)
        c=len(state);mae1=sum((row[3]/n for row,n in zip(state,self.sizes)),ZERO)/c
        mae0=self.native_mae;index=0;left=ZERO;number=0
        while True:
            right=self.events[index][0] if index<len(self.events) else ONE
            yield Segment(number,Boundary.of(left),Boundary.of(right),tuple(tuple(row) for row in state),
                          self.sizes,(mae1,mae0),self.native_mse,self.native_mae)
            if right==1:break
            stop=index
            changed=set()
            while stop<len(self.events) and self.events[stop][0]==right:
                changed.add(self.events[stop][1]);stop+=1
            old_states={};old_mae=(mae1,mae0)
            for sample in changed:
                p,d,y,j=self.samples[sample]
                if j not in old_states:old_states[j]=tuple(state[j])
                old=_contribution(reps[sample],y)
                rep=_representation(p,d,y,right);new=_contribution(rep,y);reps[sample]=rep
                difference=tuple(b-a for a,b in zip(old,new))
                state[j]=[a+delta for a,delta in zip(state[j],difference)]
                mae1+=difference[3]/self.sizes[j]/c;mae0+=difference[4]/self.sizes[j]/c
                note_bits(self.stats,*state[j],mae1,mae0,right)
            for j,old in old_states.items():
                new=state[j]
                before=(old[0]*right+old[1])*right+old[2]
                after=(new[0]*right+new[1])*right+new[2]
                if before!=after or old[3]*right+old[4]!=new[3]*right+new[4]:
                    self.stats['event_continuity_failures']+=1
                    raise BoundaryUnresolved('EVENT_LOSS_DISCONTINUITY')
            if old_mae[0]*right+old_mae[1]!=mae1*right+mae0:
                self.stats['event_continuity_failures']+=1
                raise BoundaryUnresolved('MACRO_MAE_EVENT_DISCONTINUITY')
            self.stats['event_groups_processed']+=1
            left=right;index=stop;number+=1


def _check_point(segment,point,stats):
    stats['boundary_nodes_checked']+=1
    for position,polynomial in enumerate(segment.constraints()):
        value=sign(polynomial,point,stats)
        if value>0:return False,position
    return True,None


def _feasible(segment,stats):
    interval=(segment.left,segment.right)
    for position,polynomial in enumerate(segment.constraints()):
        interval=intersect_polynomial(interval,polynomial,stats)
        if interval is None:return None,position
    for point in interval:
        okay,_=_check_point(segment,point,stats)
        if not okay:raise BoundaryUnresolved('RETAINED_BOUNDARY_NOT_FEASIBLE')
    return interval,None


def _norm_bounds(poly,point,bits,stats):
    exact_sign=sign(poly,point,stats)
    if exact_sign<0:raise BoundaryUnresolved('NEGATIVE_EXACT_MSE')
    if exact_sign==0:return ZERO,ZERO
    lower,upper=polynomial_bounds(poly,point,bits)
    if upper<=0:raise BoundaryUnresolved('POSITIVE_MSE_ENCLOSURE_CONTRADICTION')
    # Zero is a certified lower bound only because exact positivity was proved.
    lower=lower if lower>0 else ZERO
    return sqrt_bounds(lower,bits)[0],sqrt_bounds(upper,bits)[1]


def _objective_bounds(segment,point,bits,stats):
    intervals=[_norm_bounds(segment.mse(j),point,bits,stats) for j in range(len(segment.sizes))]
    return sum((a for a,b in intervals),ZERO)/len(intervals),sum((b for a,b in intervals),ZERO)/len(intervals)


def _derivative_sign(segment,point,side,stats):
    for bits in BITS:
        total=(ZERO,ZERO);retry=False
        for j in range(len(segment.sizes)):
            a,b,c=segment.mse(j);sgn=sign((a,b,c),point,stats)
            if sgn<0:raise BoundaryUnresolved('NEGATIVE_EXACT_DERIVATIVE_MSE')
            if sgn==0:
                if sign((ZERO,a,b/2),point,stats)!=0:raise BoundaryUnresolved('ZERO_NORM_NONZERO_MOMENT')
                low,high=sqrt_bounds(a,bits)
                term=(low,high) if side>0 else (-high,-low)
            else:
                numerator=polynomial_bounds((ZERO,a,b/2),point,bits)
                if numerator==(ZERO,ZERO):term=(ZERO,ZERO)
                else:
                    norm=_norm_bounds((a,b,c),point,bits,stats)
                    if norm[0]<=0:retry=True;break
                    term=divide_interval(numerator,norm)
            total=add_interval(total,term)
        stats['max_objective_refinement_bits']=max(stats['max_objective_refinement_bits'],bits)
        if retry:continue
        if total[0]>0:return 1
        if total[1]<0:return -1
        if total==(ZERO,ZERO):return 0
    raise BoundaryUnresolved('DERIVATIVE_SIGN_UNRESOLVED')


def _stop_search(segment,lo,hi,config,stats):
    for bits in BITS:
        left,_=lo.bounds(bits);_,right=hi.bounds(bits);width=right-left
        lipschitz=sum((sqrt_bounds(segment.mse(j)[0],bits)[1] for j in range(len(segment.sizes))),ZERO)/len(segment.sizes)
        if width<=F(str(config['alpha_xtol'])) and width*lipschitz<=F(str(config['objective_search_atol'])):
            return True
    return False


def _minimum(segment,interval,config,stats):
    lo,hi=interval
    if compare(lo,hi,stats)==0:return lo
    if _stop_search(segment,lo,hi,config,stats):return lo
    if _derivative_sign(segment,lo,1,stats)>=0:return lo
    if _derivative_sign(segment,hi,-1,stats)<0:return hi
    stats['interior_minimizations']+=1
    for _ in range(config['max_bisection_iterations']):
        if _stop_search(segment,lo,hi,config,stats):return hi
        middle=midpoint_inside(lo,hi,stats)
        if _derivative_sign(segment,middle,1,stats)>=0:hi=middle
        else:lo=middle
        stats['bisection_iterations']+=1
    raise BoundaryUnresolved('BISECTION_BUDGET_EXHAUSTED')


def _merge_component(components,interval,stats):
    if components and compare(interval[0],components[-1][1],stats)<=0:
        if compare(interval[1],components[-1][1],stats)>0:components[-1]=(components[-1][0],interval[1])
    else:components.append(interval)


def _within(point,interval,stats):
    return compare(point,interval[0],stats)>=0 and compare(point,interval[1],stats)<=0


def _direct_feasible(score,native):
    return score['macro']['mae']<=native['macro']['mae'] and all(a['rmse']<=1.01*b['rmse'] for a,b in zip(score['cells'],native['cells']))


def solve_exact(prepared,config,score_callback):
    stats={name:0 for name in ('segments_total','segments_classified','certified_empty_segments','feasible_intervals',
        'feasible_singletons','unresolved_segments','boundary_nodes_checked','boundary_comparisons_unresolved',
        'event_continuity_failures','event_groups_processed','max_root_refinement_bits','max_objective_refinement_bits',
        'max_exact_integer_bits','full_array_direct_checks','boundary_repairs','interior_minimizations','bisection_iterations',
        'replay_segments','events','exact_initialization_passes')}
    components=[];trace=[];ledger=hashlib.sha256();candidates=[];direct_cache={}
    def direct(alpha):
        if alpha not in direct_cache:
            if stats['full_array_direct_checks']>=8:raise BoundaryUnresolved('FULL_ARRAY_CHECK_BUDGET_EXCEEDED')
            direct_cache[alpha]=score_callback(prepared,alpha);stats['full_array_direct_checks']+=1
        return direct_cache[alpha]
    try:
        native=direct(0.);direct(1.)
        sweep=Sweep(prepared,stats)
        if all(d==0 for p,d,y,j in sweep.samples):
            return {'status':'ZERO_DIRECTION','alpha':0.,'positive_feasible_exists':True,
                'feasible_components':[[0.,1.]],'exact_feasible_components':[[Boundary.of(0).encode(),Boundary.of(1).encode()]],
                'score':native,'stats':stats}
        for segment in sweep.segments():
            stats['segments_total']+=1
            try:
                left_ok,_=_check_point(segment,segment.left,stats);right_ok,_=_check_point(segment,segment.right,stats)
                interval,empty_reason=_feasible(segment,stats)
                if interval is None:
                    category='CERTIFIED_EMPTY';stats['certified_empty_segments']+=1
                else:
                    singleton=compare(interval[0],interval[1],stats)==0
                    category='CERTIFIED_FEASIBLE_SINGLETON' if singleton else 'CERTIFIED_FEASIBLE_INTERVAL'
                    stats['feasible_singletons' if singleton else 'feasible_intervals']+=1
                    _merge_component(components,interval,stats)
                    best=_minimum(segment,interval,config,stats)
                    for point in (*interval,best):
                        obj=_objective_bounds(segment,point,64,stats)
                        candidates.append({'point':point,'objective':obj,'interval':interval,'segment':segment.index})
                stats['segments_classified']+=1
                entry={'segment':segment.index,'left':segment.left.encode(),'right':segment.right.encode(),
                       'classification':category,'left_feasible':left_ok,'right_feasible':right_ok,
                       'empty_constraint_index':empty_reason,
                       'feasible':None if interval is None else [p.encode() for p in interval]}
                if len(trace)<256:trace.append(entry)
                ledger.update((json.dumps(entry,sort_keys=True,separators=(',',':'))+'\n').encode())
            except BoundaryUnresolved:
                stats['unresolved_segments']+=1
                raise
        if not components or not any(_within(Boundary.of(0),interval,stats) for interval in components):
            raise BoundaryUnresolved('NATIVE_ZERO_NOT_IN_EXACT_FEASIBLE_SET')
        if stats['segments_classified']!=stats['segments_total'] or stats['unresolved_segments']:
            raise BoundaryUnresolved('SEGMENT_COMPLETENESS_UNRESOLVED')
        # Decide the fixed finite-candidate tie set only after every segment.
        tolerance=F(str(config['objective_tie_atol']));eligible=None
        for bits in BITS:
            min_lower=min(c['objective'][0] for c in candidates);min_upper=min(c['objective'][1] for c in candidates)
            definite=[i for i,c in enumerate(candidates) if c['objective'][1]<=min_lower+tolerance]
            ambiguous=[i for i,c in enumerate(candidates) if c['objective'][0]<=min_upper+tolerance and i not in definite]
            if not ambiguous:
                eligible=definite;break
            if bits==512:break
            next_bits=BITS[BITS.index(bits)+1];needed={candidates[i]['segment'] for i in ambiguous}
            # Refinement replays exact event updates, not full-array rescoring.
            groups={}
            for i,c in enumerate(candidates):
                if c['segment'] in needed:groups.setdefault(c['segment'],[]).append(i)
            for segment in sweep.segments():
                stats['replay_segments']+=1
                if segment.index in groups:
                    for i in groups[segment.index]:candidates[i]['objective']=_objective_bounds(segment,candidates[i]['point'],next_bits,stats)
        if not eligible:raise BoundaryUnresolved('FINITE_CANDIDATE_TIE_COMPARISON_UNRESOLVED')
        chosen=eligible[0]
        for i in eligible[1:]:
            if compare(candidates[i]['point'],candidates[chosen]['point'],stats)<0:chosen=i
        candidate=candidates[chosen];point=candidate['point'];interval=candidate['interval']
        singleton=compare(interval[0],interval[1],stats)==0
        alpha=point.approximate();represented=Boundary.of(F.from_float(alpha))
        if singleton and compare(point,represented,stats)!=0:
            raise BoundaryUnresolved('UNREPRESENTABLE_SELECTED_SINGLETON')
        selected_segment=None
        for segment in sweep.segments():
            stats['replay_segments']+=1
            if segment.index==candidate['segment']:selected_segment=segment;break
        if selected_segment is None:raise BoundaryUnresolved('SELECTED_SEGMENT_NOT_FOUND')
        exact_float_ok=_within(represented,interval,stats) and _check_point(selected_segment,represented,stats)[0]
        score=direct(alpha);original_alpha=alpha
        original_score=score
        if not exact_float_ok or not _direct_feasible(score,native):
            if singleton:raise BoundaryUnresolved('SELECTED_SINGLETON_DIRECT_CHECK_FAILED')
            objective_approx=float(sum(candidate['objective'])/2)
            if abs(score['macro']['rmse']-objective_approx)>config['score_verification_atol']:
                raise BoundaryUnresolved('PRE_REPAIR_SCORE_DISCREPANCY')
            low,high=interval[0].approximate(),interval[1].approximate();middle=low+(high-low)/2
            distance=min(config['boundary_repair_max_alpha_change'],(high-low)/4,abs(middle-alpha))
            repaired=alpha+math.copysign(distance,middle-alpha)
            rpoint=Boundary.of(F.from_float(repaired))
            if distance<=0 or repaired==alpha or not _within(rpoint,interval,stats) or compare(rpoint,interval[0],stats)==0 or compare(rpoint,interval[1],stats)==0:
                raise BoundaryUnresolved('NO_REPRESENTABLE_SINGLE_REPAIR_INSIDE_COMPONENT')
            stats['boundary_repairs']=1
            fixed=direct(repaired)
            if not _check_point(selected_segment,rpoint,stats)[0] or not _direct_feasible(fixed,native) or fixed['macro']['rmse']-original_score['macro']['rmse']>config['boundary_repair_max_objective_increase']:
                raise BoundaryUnresolved('REGISTERED_SINGLE_REPAIR_FAILED')
            alpha= repaired;represented=rpoint;score=fixed
        # Verify exact aggregate evaluation against fixed-order float64 scoring.
        norms=[_norm_bounds(selected_segment.mse(j),represented,128,stats) for j in range(len(prepared))]
        mse_mae=selected_segment.mae[0]*represented.rational+selected_segment.mae[1]
        errors=[abs(float(mse_mae)-score['macro']['mae'])]
        errors += [abs(float((a+b)/2)-r['rmse']) for (a,b),r in zip(norms,score['cells'])]
        errors.append(abs(float(sum((a+b for a,b in norms),ZERO)/(2*len(norms)))-score['macro']['rmse']))
        for sums,n,row in zip(selected_segment.moment_sums,selected_segment.sizes,score['cells']):
            errors.append(abs(float((sums[3]*represented.rational+sums[4])/n)-row['mae']))
        discrepancy=max(errors)
        if discrepancy>config['score_verification_atol']:raise BoundaryUnresolved('EXACT_SCAN_DIRECT_DISCREPANCY')
        positive=any(compare(hi,Boundary.of(0),stats)>0 for lo,hi in components)
        status='OK' if alpha>0 else ('OPTIMAL_ALPHA_ZERO' if positive else 'ZERO_ONLY_FEASIBLE')
        return {'status':status,'alpha':alpha,'original_selected_alpha':original_alpha,'selected_exact_point':point.encode(),
            'positive_feasible_exists':positive,'feasible_components':[[lo.approximate(),hi.approximate()] for lo,hi in components],
            'exact_feasible_components':[[lo.encode(),hi.encode()] for lo,hi in components],
            'score':score,'scan_direct_max_difference':discrepancy,'stats':stats,
            'segment_trace_prefix':trace,'segment_ledger_sha256':ledger.hexdigest(),
            'segment_trace_truncated':stats['segments_total']>len(trace),
            'claim':'complete registered-domain exact feasibility classification and enclosed numerical candidate optimization; not an exact objective optimum certificate'}
    except (BoundaryUnresolved,ArithmeticError,OverflowError,FloatingPointError) as e:
        return {'status':'NUMERICAL_BLOCKED','reason':str(e),'stats':stats,
            'exact_feasible_components':[[lo.encode(),hi.encode()] for lo,hi in components],
            'segment_trace_prefix':trace,'segment_ledger_sha256':ledger.hexdigest(),
            'selected_exact_point':point.encode() if 'point' in locals() else None}
