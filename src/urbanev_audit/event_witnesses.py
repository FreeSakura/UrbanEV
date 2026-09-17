"""Replayable completion witnesses and bounded structural certificates.

The path solver is ordinary weighted-automaton traceback. Structural checks
use all completions of small shared-unknown components, never sampled closure.
"""
from itertools import product
import numpy as np
from scipy.optimize import linprog
from .persistent_events import automaton, event_labels, label_bounds, njit, _feasible
from .event_relaxations import PairwiseLP


@njit(cache=True)
def _advance(o, c, k, edges, event, start, stop, initial, required, trace):
    cost=initial.copy(); count=len(edges)
    parents=np.full((stop-start if trace else 0,count),-1,np.int32)
    for t in range(start,stop):
        nxt=np.full(count,np.inf)
        w=c[t-k+1] if t>=k-1 else 0.
        req=required[t-k+1] if t>=k-1 else -1
        for s in range(count):
            if not np.isfinite(cost[s]):continue
            for bit in range(2):
                d=edges[s,bit]
                if (o[t]==-1 or o[t]==bit) and (req==-1 or req==event[d]):
                    value=cost[s]+(w if event[d] else 0.)
                    if value<nxt[d]:
                        nxt[d]=value
                        if trace:parents[t-start,d]=2*s+bit
        cost=nxt
    return cost,parents


def completion_witness(observed,coefficients,k,ell,side='lower',required=None,block=2048):
    """Checkpointed traceback: O((T/block+block)*states) auxiliary memory."""
    o=np.asarray(observed,np.int8);c=np.asarray(coefficients,float)
    if side=='upper':c=-c
    edges,event,_=automaton(k,ell)
    req=np.full(len(c),-1,np.int8) if required is None else np.asarray(required,np.int8)
    current=np.full(len(edges),np.inf);current[0]=0.
    starts=list(range(0,len(o),block));checkpoints=[]
    for start in starts:
        checkpoints.append(current)
        current,_=_advance(o,c,k,edges,event,start,min(start+block,len(o)),current,req,False)
    state=int(np.argmin(current))
    if not np.isfinite(current[state]):return None
    bits=np.empty(len(o),np.int8)
    for start,initial in reversed(list(zip(starts,checkpoints))):
        stop=min(start+block,len(o))
        _,parents=_advance(o,c,k,edges,event,start,stop,initial,req,True)
        for t in range(stop-1,start-1,-1):
            encoded=int(parents[t-start,state]);bits[t]=encoded%2;state=encoded//2
    if not np.array_equal(bits[o!=-1],o[o!=-1]):raise AssertionError('Witness changed observations')
    labels=event_labels(bits,k,ell)
    if not np.array_equal(labels[req!=-1],req[req!=-1]):raise AssertionError('Witness violates label requirements')
    return bits


def exact_brier_difference(a,b,y):
    """Exact rational sign for the stored binary64 predictions; no tolerance."""
    ar=[float(v).as_integer_ratio() for v in a];br=[float(v).as_integer_ratio() for v in b]
    exponent=max(d.bit_length()-1 for _,d in ar+br);scale=1<<exponent
    total=0
    for (an,ad),(bn,bd),label in zip(ar,br,y):
        av=an*(scale//ad);bv=bn*(scale//bd)
        total+=(av-bv)*(av+bv-2*int(label)*scale)
    return {'delta':total/(scale*scale*len(y)), 'sign':(total>0)-(total<0),
            'sum_numerator':str(total),'sum_denominator_power2':2*exponent}


def pairwise_pattern(lp,coefficients,side):
    """A binary optimum of the represented order LP, checked for implications.

    A non-realizable optimum is only a structural counterexample, not proof
    of a strict objective gap: another optimum can still be realizable.
    """
    c=np.asarray(coefficients,float)*(1 if side=='lower' else -1)
    w=c[lp.ambiguous];pattern=lp.lo.copy();chosen=(w<0).astype(np.int8)
    if np.all(w<=0) and np.any(w<0):chosen[:]=1
    if lp.matrix is not None and not (np.all(w>=0) or np.all(w<=0)):
        objective=np.zeros(lp.variable_count);objective[:len(lp.lp_indices)]=w[lp.lp_indices]
        result=linprog(objective,A_ub=lp.matrix,b_ub=np.zeros(lp.matrix.shape[0]),bounds=(0,1),method='highs')
        if not result.success:raise RuntimeError(result.message)
        chosen[lp.lp_indices]=(result.x[:len(lp.lp_indices)]>=.5).astype(np.int8)
        if abs(float(objective@result.x)-float(w[lp.lp_indices]@chosen[lp.lp_indices]))>1e-7:
            raise ArithmeticError('Thresholded LP solution not numerically optimal')
    pattern[lp.ambiguous]=chosen
    zero=lp.ambiguous[chosen==0]
    for q in np.flatnonzero(chosen):
        j=np.searchsorted(zero,lp.first[q])
        if j<len(zero) and zero[j]<=lp.last[q]:raise AssertionError('Invalid binary implication pattern')
    return pattern


def dependency_components(o,k,ell):
    lo,hi=label_bounds(o,k,ell);active=np.flatnonzero(lo!=hi);missing=np.flatnonzero(o==-1)
    groups=[];last=-1
    for s in active:
        first=int(np.searchsorted(missing,s));end=int(np.searchsorted(missing,s+k))
        if not groups or first>=last:groups.append([])
        groups[-1].append(int(s));last=max(last,end)
    return [np.asarray(g,dtype=int) for g in groups]


def structural_certificate(observed,k,ell,max_unknowns=8,max_labels=512):
    """Exhaustive closure, simple sufficient cases, or explicit unresolved."""
    o=np.asarray(observed,np.int8);records=[]
    for indices in dependency_components(o,k,ell):
        start=int(indices[0]);end=int(indices[-1])+k;local=o[start:end]
        missing=np.flatnonzero(local==-1);m=len(indices);u=len(missing)
        record={'first_origin':start,'last_origin':int(indices[-1]),'labels':m,'unknowns':u,
                'status':'UNRESOLVED_BUDGET','patterns':None,'closure_operation':'','pattern_a':'','pattern_b':'','unreachable_pattern':''}
        if u==1 or m<=2:
            record['status']='SUFFICIENT_SMALL_COMPONENT'
        elif u<=max_unknowns and m<=max_labels:
            patterns={}
            for assignment in product((0,1),repeat=u):
                bits=local.copy();bits[missing]=assignment
                labels=event_labels(bits,k,ell)[indices-start]
                code=sum(int(bit)<<i for i,bit in enumerate(labels))
                patterns.setdefault(code,''.join(map(str,assignment)))
            record.update(status='EXACT_CLOSED',patterns=len(patterns))
            codes=sorted(patterns)
            for a in codes:
                for b in codes:
                    for op,c in [('AND',a&b),('OR',a|b)]:
                        if c not in patterns:
                            record.update(status='HIGHER_STRUCTURE',closure_operation=op,
                              pattern_a=hex(a),pattern_b=hex(b),unreachable_pattern=hex(c),
                              assignment_a=patterns[a],assignment_b=patterns[b])
                            break
                    if record['status']=='HIGHER_STRUCTURE':break
                if record['status']=='HIGHER_STRUCTURE':break
        records.append(record)
    status=('HIGHER_STRUCTURE' if any(r['status']=='HIGHER_STRUCTURE' for r in records) else
            'UNRESOLVED_BUDGET' if any(r['status']=='UNRESOLVED_BUDGET' for r in records) else 'STRUCTURALLY_EXACT')
    return status,records


def pattern_feasible(o,pattern,k,ell):
    edges,event,_=automaton(k,ell)
    return bool(_feasible(np.asarray(o,np.int8),np.asarray(pattern,np.int8),k,edges,event))
