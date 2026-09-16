"""AP1 comparison hierarchy, origin panels and near-zero sign enclosures."""
from itertools import combinations
import time
import numpy as np
from .persistent_events import paired_bounds,label_bounds,event_labels,automaton,njit,component_sign_certificate
from .event_relaxations import PairwiseLP


@njit(cache=True)
def integer_extrema(o,c,k,edges,event):
    big=np.int64(2**61);n=len(edges)
    low=np.full(n,big,np.int64);high=np.full(n,-big,np.int64);low[0]=high[0]=0
    for t in range(len(o)):
        nl=np.full(n,big,np.int64);nh=np.full(n,-big,np.int64);w=c[t-k+1] if t>=k-1 else 0
        for s in range(n):
            if low[s]==big:continue
            for bit in range(2):
                if o[t]==-1 or o[t]==bit:
                    d=edges[s,bit];v=w if event[d] else 0
                    nl[d]=min(nl[d],low[s]+v);nh[d]=max(nh[d],high[s]+v)
        low,high=nl,nh
    return low.min(),high.max()


def sign(lower,upper,tol=1e-10):
    if lower>tol:return 'B'
    if upper< -tol:return 'A'
    if lower==0 and upper==0:return 'TIE'
    if lower< -tol and upper>tol:return 'OPPOSITE_POSSIBLE'
    return 'BOUNDARY'


def origin_blocks(start,stop,k,length):
    """Yield [origin_start, origin_stop) and support_stop; no missing tail labels."""
    for a in range(start,stop-k+1,length):
        b=min(a+length,stop-k+1)
        yield a,b,b+k-1


def score_panel(o,predictions,k,ell,metadata,truth=None,imputed_bits=None):
    lp=PairwiseLP(o,k,ell);rows=[];known=lp.lo==lp.hi
    for name_a,name_b in combinations(predictions,2):
        a=np.asarray(predictions[name_a]);b=np.asarray(predictions[name_b]);n=len(a);c=2*(b-a)
        clock=time.perf_counter();joint=paired_bounds(o,a,b,k,ell);joint_seconds=time.perf_counter()-clock
        il,iu=joint['independent_lower_sum']/n,joint['independent_upper_sum']/n
        jl,ju=joint['lower_sum']/n,joint['upper_sum']/n
        gl=max(0.,jl-il);gu=max(0.,iu-ju)
        numeric='FLOAT_OUTWARD';pad=8*np.finfo(float).eps*len(o)+2e-14
        jlo,jhi=jl-pad,ju+pad
        if np.array_equal(a,b):jlo=jhi=0.;numeric='IDENTICAL_PREDICTIONS'
        elif min(abs(jl),abs(ju))<1e-8:
            scale=10**11;ci=np.rint(c*scale).astype(np.int64)
            constant=np.rint((a*a-b*b)*scale).astype(np.int64).sum()
            if known.all():low=high=int(np.dot(ci,lp.lo.astype(np.int64)))
            else:
                edges,event,_=automaton(k,ell);low,high=integer_extrema(np.asarray(o,np.int8),ci,k,edges,event)
            error=1/scale+2e-14
            jlo=(int(constant)+int(low))/scale/n-error;jhi=(int(constant)+int(high))/scale/n+error
            numeric='INTEGER_GRID_OUTWARD'
        pair=lp.bounds(a,b)
        pl,pu=pair['lower'],pair['upper']
        if pl is not None:
            pl=max(il-2e-14,pl-2e-14);pu=min(iu+2e-14,pu+2e-14)
            if pl>jl+1e-8 or pu<ju-1e-8:raise AssertionError('Pairwise outer bound excluded the exact bound')
        ind_sign=sign(il-2e-14,iu+2e-14) if not np.array_equal(a,b) else 'TIE'
        lp_sign=sign(pl,pu) if pl is not None else 'NOT_SOLVED'
        joint_sign=sign(jlo,jhi)
        cert=component_sign_certificate(o,c,k,ell)
        category=('NO_AMBIGUOUS_LABELS' if joint['ambiguous_windows']==0 else
                  'INDEPENDENT_ALREADY_DIRECTED' if ind_sign in ('A','B') else
                  'JOINT_NEW_STRICT' if joint_sign in ('A','B') else
                  'JOINT_OPPOSITE_POSSIBLE' if joint_sign=='OPPOSITE_POSSIBLE' else 'TIE_OR_NONSTRICT_BOUNDARY')
        row={**metadata,'K':k,'L':ell,'model_A':name_a,'model_B':name_b,'N':n,
             'support_records':len(o),'missing_records':int(np.sum(o==-1)),
             'ambiguous_windows':joint['ambiguous_windows'],'independent_lower':il,'independent_upper':iu,
             'pairwise_lower':pl,'pairwise_upper':pu,'joint_lower':jl,'joint_upper':ju,
             'joint_enclosure_lower':jlo,'joint_enclosure_upper':jhi,'numeric_method':numeric,
             'Gamma_minus':gl*n,'Gamma_plus':gu*n,'W_ind':iu-il,
             'ambiguous_weight':float(np.abs(c[lp.ambiguous]).sum()/n),
             'lower_zero_margin':-il,'upper_zero_margin':iu,
             'mixed_components':cert['mixed_sign_components'],'implications':lp.edge_count,
             'lp_variables':lp.variable_count,'lp_constraints':0 if lp.matrix is None else lp.matrix.shape[0],
             'pair_preprocess_seconds':lp.preprocess_seconds,'pair_solve_seconds':pair['seconds'],
             'pair_status':pair['status'],'joint_seconds':joint_seconds,
             'independent_direction':ind_sign,'pairwise_direction':lp_sign,'joint_direction':joint_sign,
             'independent_undirected':ind_sign not in ('A','B'),
             'pairwise_new':ind_sign not in ('A','B') and lp_sign in ('A','B'),
             'joint_new':ind_sign not in ('A','B') and joint_sign in ('A','B'),
             'higher_order_new':lp_sign not in ('A','B','NOT_SOLVED') and joint_sign in ('A','B'),
             'category':category,'truth_delta':'','truth_contained':'','locf_delta':'','locf_wrong':'','identified_delta':'','identified_wrong':''}
        if abs(row['ambiguous_weight']-row['W_ind'])>1e-9:raise AssertionError('Independent-width identity failed')
        if truth is not None:
            y=event_labels(truth,k,ell);delta=float(np.mean((a-y)**2-(b-y)**2))
            row['truth_delta']=delta;row['truth_contained']=jlo-1e-10<=delta<=jhi+1e-10
            if not row['truth_contained']:raise AssertionError('Complete artificial truth outside bounds')
            if imputed_bits is not None:
                yi=event_labels(imputed_bits,k,ell);score=float(np.mean((a-yi)**2-(b-yi)**2))
                row['locf_delta']=score;row['locf_wrong']=score*delta<0 and min(abs(score),abs(delta))>1e-10
            if known.any():
                score=float(np.mean((a[known]-lp.lo[known])**2-(b[known]-lp.lo[known])**2))
                row['identified_delta']=score;row['identified_wrong']=score*delta<0 and min(abs(score),abs(delta))>1e-10
        rows.append(row)
    models=list(predictions);candidate={}
    for method in ('independent','pairwise','joint'):
        dominated=set();winners={model:set() for model in models}
        for row in rows:
            direction=row[method+'_direction']
            if direction in ('A','B'):
                winner=row['model_'+direction];loser=row['model_B' if direction=='A' else 'model_A']
                dominated.add(loser);winners[winner].add(loser)
        candidate[method+'_undominated']=';'.join(sorted(set(models)-dominated))
        candidate[method+'_best']=';'.join(sorted(model for model,beaten in winners.items() if len(beaten)==len(models)-1))
    candidate.update(metadata);candidate.update(K=k,L=ell,N=len(next(iter(predictions.values()))))
    candidate['pairwise_set_changed']=candidate['pairwise_undominated']!=candidate['independent_undominated']
    candidate['joint_set_changed']=candidate['joint_undominated']!=candidate['independent_undominated']
    candidate['higher_order_set_changed']=candidate['joint_undominated']!=candidate['pairwise_undominated']
    return rows,candidate
