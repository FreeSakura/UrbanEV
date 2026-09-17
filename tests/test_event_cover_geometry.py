from itertools import product,combinations
import numpy as np
from urbanev_audit.event_cover_geometry import EventCoverGeometry
from urbanev_audit.persistent_events import event_labels


def all_labels(o,k,ell):
    missing=np.flatnonzero(o==-1);labels=set()
    for values in product((0,1),repeat=len(missing)):
        z=o.copy();z[missing]=values
        y=tuple(int(any(all(z[r:r+ell]) for r in range(s,s+k-ell+1))) for s in range(len(o)-k+1))
        labels.add(y)
    return labels


def test_complete_small_geometry_and_sufficiency():
    for t in range(1,6):
        for seq in product((-1,0,1),repeat=t):
            o=np.array(seq,np.int8)
            for k in range(1,t+1):
                for ell in range(1,k+1):
                    g=EventCoverGeometry(o,k,ell);patterns=all_labels(o,k,ell)
                    for j in g.ambiguous:
                        for size in (1,2):
                            for cover in combinations([i for i in range(g.n) if i!=j],size):
                                valid=all(not y[j] or any(y[i] for i in cover) for y in patterns)
                                assert g.cover_valid(j,cover)==valid
                    pairs=[(i,j) for i in g.ambiguous for j in g.ambiguous if all(y[i]<=y[j] for y in patterns)]
                    pairset={y for y in product((0,1),repeat=g.n) if all(g.lo<=y) and all(y<=g.hi) and all(y[i]<=y[j] for i,j in pairs)}
                    result=g.diagnose()
                    if k>=2*ell-1:
                        assert (result['status']=='PAIRWISE_EXACT')==(pairset==patterns)
                        assert all(g.binary_cover_feasible(y)==(y in patterns) for y in product((0,1),repeat=g.n))
                        assert all(tuple(np.maximum(a,b)) in patterns for a in patterns for b in patterns)
                        for y in patterns:
                            z=g.realize(y)
                            assert z is not None and np.array_equal(z[o!=-1],o[o!=-1])
                            assert tuple(event_labels(z,k,ell))==y
                    if result['status']=='PAIRWISE_INSUFFICIENT':assert pairset!=patterns


def test_sharp_short_window_counterfamily():
    for ell in range(2,8):
        for k in range(ell,2*ell-1):
            g=EventCoverGeometry(np.full(k+2,-1,np.int8),k,ell)
            patterns=all_labels(g.o,k,ell)
            assert (1,0,0) in patterns and (0,0,1) in patterns
            assert (1,0,1) not in patterns
            assert g.binary_cover_feasible([1,0,1])


def test_observed_one_extensions_are_needed():
    o=np.array([-1,1,1,0,-1,1],np.int8)
    g=EventCoverGeometry(o,3,2)
    # First label is known true; cover claims cannot ignore fixed observations.
    assert g.lo[0]==1
    assert g.cover_valid(0,[2]) is False


def test_all_covers_binary_complete_but_LP_not_integral():
    g=EventCoverGeometry(np.full(10,-1,np.int8),6,3)
    patterns=all_labels(g.o,6,3);c=np.array([1,-1,1,-1,1])
    assert min(np.dot(c,y) for y in patterns)==0
    result=g.cover_lp(c)
    assert abs(result.fun+.5)<1e-12
    assert np.allclose(result.x,[.5,1,.5,1,.5])


def test_random_larger_observed_patterns():
    rng=np.random.default_rng(104)
    for _ in range(120):
        ell=int(rng.integers(1,5));k=int(rng.integers(2*ell-1,2*ell+4));n=int(rng.integers(3,8))
        o=rng.choice([-1,0,1],size=k+n-1)
        if np.sum(o==-1)>8:continue
        g=EventCoverGeometry(o,k,ell);patterns=all_labels(o,k,ell)
        assert all(tuple(np.maximum(a,b)) in patterns for a in patterns for b in patterns)
        for y in product((0,1),repeat=n):
            z=g.realize(y)
            assert (z is not None)==(y in patterns)
            if z is not None:assert tuple(event_labels(z,k,ell))==y
