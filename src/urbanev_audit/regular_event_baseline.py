"""Generic regular-language compilation and weighted DAG path baseline.

No trailing-run/last-hit state is used: Thompson-style NFA, determinization,
DFA concatenation/union, and Hopcroft minimization operate on language graphs.
R_d = R_0 union (R_{d-1} Sigma), R_0 = Sigma* 1^L.
"""
from collections import deque
from functools import lru_cache
import time
import numpy as np
from .persistent_events import njit

_PROFILE={'determinize_seconds':0.,'minimize_seconds':0.,'union_product_seconds':0.}


def canonical(edges,accept,start=0):
    order=[start];ids={start:0};rows=[]
    for state in order:
        row=[]
        for dest in edges[state]:
            dest=int(dest)
            if dest not in ids:ids[dest]=len(order);order.append(dest)
            row.append(ids[dest])
        rows.append(row)
    return np.asarray(rows,np.int64),np.asarray([accept[s] for s in order],np.bool_)


def minimize(edges,accept):
    clock=time.perf_counter()
    n=len(edges);parts=[set(np.flatnonzero(~accept)),set(np.flatnonzero(accept))];parts=[x for x in parts if x]
    block=np.empty(n,np.int64)
    for i,part in enumerate(parts):
        for x in part:block[x]=i
    predecessors=[[[] for _ in range(n)] for _ in range(2)]
    for s in range(n):
        for bit in range(2):predecessors[bit][edges[s,bit]].append(s)
    queue=deque(range(len(parts)));pending=set(range(len(parts)))
    while queue:
        a=queue.popleft();pending.discard(a);splitter=parts[a].copy()
        for bit in range(2):
            touched={}
            for dest in splitter:
                for s in predecessors[bit][dest]:touched.setdefault(int(block[s]),set()).add(s)
            for b,inside in touched.items():
                if len(inside)==len(parts[b]):continue
                outside=parts[b]-inside;parts[b]=inside;new=len(parts);parts.append(outside)
                for s in outside:block[s]=new
                if b in pending:queue.append(new);pending.add(new)
                else:
                    add=b if len(inside)<=len(outside) else new
                    if add not in pending:queue.append(add);pending.add(add)
    reduced=np.empty((len(parts),2),np.int64);final=np.empty(len(parts),np.bool_)
    for i,part in enumerate(parts):
        s=next(iter(part));reduced[i]=block[edges[s]];final[i]=accept[s]
    result=canonical(reduced,final,int(block[0]));_PROFILE['minimize_seconds']+=time.perf_counter()-clock
    return result


def determinize(transitions,epsilon,final,start=0,max_states=100000,deadline=None):
    clock=time.perf_counter()
    def closure(states):
        states=set(states);stack=list(states)
        while stack:
            for q in epsilon.get(stack.pop(),()):
                if q not in states:states.add(q);stack.append(q)
        return frozenset(states)
    initial=closure([start]);states=[initial];ids={initial:0};edges=[];accept=[]
    for subset in states:
        if len(states)>max_states or (deadline is not None and time.perf_counter()>deadline):raise TimeoutError('Declared DFA construction budget reached')
        accept.append(bool(subset & final));row=[]
        for bit in (0,1):
            dest=closure(q for s in subset for q in transitions.get((s,bit),()))
            if dest not in ids:ids[dest]=len(states);states.append(dest)
            row.append(ids[dest])
        edges.append(row)
    result=np.asarray(edges,np.int64),np.asarray(accept,np.bool_);_PROFILE['determinize_seconds']+=time.perf_counter()-clock
    return result


def append_sigma(dfa,deadline):
    edges,accept=dfa;n=len(edges)
    transitions={(s,b):[int(edges[s,b])] for s in range(n) for b in (0,1)}
    epsilon={int(s):[n] for s in np.flatnonzero(accept)}
    transitions[n,0]=transitions[n,1]=[n+1]
    return minimize(*determinize(transitions,epsilon,{n+1},deadline=deadline))


def union(left,right,deadline):
    clock=time.perf_counter()
    a,af=left;b,bf=right;states=[(0,0)];ids={(0,0):0};edges=[];accept=[]
    for x,y in states:
        if len(states)>100000 or time.perf_counter()>deadline:raise TimeoutError('Declared DFA union budget reached')
        accept.append(af[x] or bf[y]);row=[]
        for bit in (0,1):
            dest=(int(a[x,bit]),int(b[y,bit]))
            if dest not in ids:ids[dest]=len(states);states.append(dest)
            row.append(ids[dest])
        edges.append(row)
    _PROFILE['union_product_seconds']+=time.perf_counter()-clock
    return minimize(np.asarray(edges,np.int64),np.asarray(accept,np.bool_))


@lru_cache(maxsize=64)
def compile_event(window,run_length,timeout=60.):
    start=time.perf_counter();deadline=start+timeout
    for key in _PROFILE:_PROFILE[key]=0.
    # NFA for Sigma* followed by a literal string of L ones.
    transitions={(0,0):[0],(0,1):[0,1]}
    for i in range(1,run_length):transitions[i,1]=[i+1]
    base=minimize(*determinize(transitions,{}, {run_length},deadline=deadline));dfa=base
    for _ in range(window-run_length):dfa=union(base,append_sigma(dfa,deadline),deadline)
    return dfa[0],dfa[1],{'construction_seconds':time.perf_counter()-start,'states':len(dfa[0]),**_PROFILE,
                         'method':'Generic NFA determinization, concatenation/union and Hopcroft minimization; incremental regular expression evaluation'}


@njit(cache=True)
def generic_edge_paths(observed,coefficients,window,edges,accept):
    n=len(edges);src=np.arange(2*n)//2;symbols=np.arange(2*n)%2;dst=edges.reshape(-1)
    low=np.full(n,np.inf);high=np.full(n,-np.inf);low[0]=high[0]=0.
    for t in range(len(observed)):
        nl=np.full(n,np.inf);nh=np.full(n,-np.inf);w=coefficients[t-window+1] if t>=window-1 else 0.
        for e in range(len(src)):
            if observed[t]==-1 or observed[t]==symbols[e]:
                s=src[e];d=dst[e];cost=w if accept[d] else 0.
                nl[d]=min(nl[d],low[s]+cost);nh[d]=max(nh[d],high[s]+cost)
        low,high=nl,nh
    # All terminal states are legal: acceptance is an emission, not a constraint.
    return low.min(),high.max()
