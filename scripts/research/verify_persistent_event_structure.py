"""Exhaustive constructed-case verification of the persistent-event proofs."""
import argparse
from itertools import product,combinations
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_audit.event_cover_geometry import EventCoverGeometry
from urbanev_audit.persistent_events import event_labels,automaton,_feasible


def direct_labels(z,k,ell):
    return tuple(int(any(all(z[r:r+ell]) for r in range(s,s+k-ell+1))) for s in range(len(z)-k+1))


def patterns(o,k,ell):
    ix=np.flatnonzero(o==-1);result=set()
    for bits in product((0,1),repeat=len(ix)):
        z=o.copy();z[ix]=bits;result.add(direct_labels(z,k,ell))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-t',type=int,default=6)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    count=0;covers=0;inversions=0;longcases=0
    for t in range(1,a.max_t+1):
        for seq in product((-1,0,1),repeat=t):
            o=np.array(seq,np.int8)
            for k in range(1,t+1):
                for ell in range(1,k+1):
                    g=EventCoverGeometry(o,k,ell);ys=patterns(o,k,ell);count+=1
                    for j in g.ambiguous:
                        others=[int(i) for i in g.ambiguous if i!=j]
                        for size in range(len(others)+1):
                            for C in combinations(others,size):
                                exact=all(not y[j] or any(y[i] for i in C) for y in ys)
                                assert g.cover_valid(j,C)==exact;covers+=1
                                if exact:
                                    left=[i for i in C if i<j];right=[i for i in C if i>j]
                                    near=([max(left)] if left else [])+([min(right)] if right else [])
                                    assert g.cover_valid(j,near)
                    if k<2*ell-1:continue
                    longcases+=1
                    pairs=[(i,j) for i in g.ambiguous for j in g.ambiguous if all(y[i]<=y[j] for y in ys)]
                    pairset=set()
                    for y in product((0,1),repeat=g.n):
                        bounded=all(g.lo<=y) and all(y<=g.hi)
                        if bounded and all(y[i]<=y[j] for i,j in pairs):pairset.add(y)
                        z=g.realize(y);inversions+=1
                        assert (z is not None)==(y in ys)
                        if z is not None:
                            assert np.array_equal(z[o!=-1],o[o!=-1])
                            assert direct_labels(z,k,ell)==y
                    assert (g.diagnose()['status']=='PAIRWISE_EXACT')==(ys==pairset)
                    assert all(tuple(max(x,y) for x,y in zip(u,v)) in ys for u in ys for v in ys)
        print(json.dumps({'T':t,'cases':count,'covers':covers}),flush=True)
    family=[]
    for ell in range(2,13):
        for k in range(ell,2*ell-1):
            t=k+2;m=k-ell+1;o=np.full(t,-1,np.int8)
            z1=np.zeros(t,np.int8);z1[:ell]=1;z2=np.zeros(t,np.int8);z2[m+1:m+1+ell]=1
            assert direct_labels(z1,k,ell)==(1,0,0) and direct_labels(z2,k,ell)==(0,0,1)
            edges,event,_=automaton(k,ell)
            assert not _feasible(o,np.array([1,0,1],np.int8),k,edges,event)
            assert EventCoverGeometry(o,k,ell).binary_cover_feasible([1,0,1])
            family.append({'K':k,'L':ell,'T':t,'feasible_a':[1,0,0],'feasible_b':[0,0,1],'infeasible_OR':[1,0,1]})
    g=EventCoverGeometry(np.full(10,-1,np.int8),6,3);ys=patterns(g.o,6,3);c=np.array([1,-1,1,-1,1])
    lp=g.cover_lp(c);minimum=min(np.dot(c,y) for y in ys)
    assert minimum==0 and abs(lp.fun+.5)<1e-12
    pa=np.array([0,.75,0,.75,0]);pb=np.array([.5,.25,.5,.25,.5])
    base=float(np.sum(pa*pa-pb*pb));assert base==.25 and np.array_equal(2*(pb-pa),c)
    counter={'K':6,'L':3,'T':10,'N':5,'observed':g.o.tolist(),'c':c.tolist(),'pattern_count':len(ys),
        'patterns':[list(y) for y in sorted(ys)],'LP_pattern':lp.x.tolist(),'LP_min':float(lp.fun),'integer_min':int(minimum),
        'prediction_A':pa.tolist(),'prediction_B':pb.tolist(),'risk_exact_lower':float((base+minimum)/5),'risk_cover_LP_lower':float((base+lp.fun)/5)}
    report={'status':'PASS','max_T':a.max_t,'ternary_observation_K_L_cases':count,'long_regime_cases':longcases,
        'arbitrary_subset_cover_tests':covers,'full_binary_label_inversion_tests':inversions,'sharpness_family_cases':len(family),
        'LP_gap_verified':True,'wall_seconds':time.perf_counter()-start,'verification_role':'Constructed finite checks accompanying proofs; not new real-data experiments'}
    for name,value in [('verification.json',report),('short_window_counterfamily.json',family),('cover_LP_counterexample.json',counter)]:
        (a.output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
