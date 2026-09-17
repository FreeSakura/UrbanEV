from itertools import product
from fractions import Fraction
import numpy as np
from urbanev_audit.event_witnesses import (completion_witness, exact_brier_difference,
    pairwise_pattern, structural_certificate, pattern_feasible)
from urbanev_audit.event_relaxations import PairwiseLP
from urbanev_audit.persistent_events import event_labels


def test_traceback_and_closure_against_all_small_completions():
    rng=np.random.default_rng(20260917)
    for t in range(1,6):
        for obs in product((-1,0,1),repeat=t):
            o=np.array(obs,np.int8);missing=np.flatnonzero(o==-1)
            for k in range(1,t+1):
                for ell in range(1,k+1):
                    c=rng.normal(size=t-k+1);ys=[]
                    for z in product((0,1),repeat=len(missing)):
                        bits=o.copy();bits[missing]=z;ys.append(event_labels(bits,k,ell))
                    expected=np.array([c@y for y in ys]);patterns={tuple(y) for y in ys}
                    for side,target in [('lower',expected.min()),('upper',expected.max())]:
                        bits=completion_witness(o,c,k,ell,side,block=2)
                        assert np.isclose(c@event_labels(bits,k,ell),target,atol=1e-12)
                    closed=all(tuple(np.minimum(a,b)) in patterns and tuple(np.maximum(a,b)) in patterns for a in ys for b in ys)
                    status,_=structural_certificate(o,k,ell)
                    assert (status=='STRUCTURALLY_EXACT')==closed


def test_pairwise_integrality_and_feasible_ties():
    rng=np.random.default_rng(23)
    for _ in range(70):
        o=rng.choice([-1,0,1],size=8);k=4;ell=2;c=rng.integers(-2,3,size=5).astype(float)
        lp=PairwiseLP(o,k,ell)
        for side in ['lower','upper']:
            pattern=pairwise_pattern(lp,c,side)
            feasible=pattern_feasible(o,pattern,k,ell)
            trace=completion_witness(o,np.zeros(len(c)),k,ell,required=pattern,block=3)
            assert (trace is not None)==feasible
            if feasible:assert np.array_equal(event_labels(trace,k,ell),pattern)


def test_exact_brier_sign_handles_cancellation_and_ties():
    a=np.array([.5,np.nextafter(.5,1),0,1]);b=np.array([.5,.5,1,0]);y=np.array([1,0,1,1])
    expected=sum((Fraction(float(x))-int(t))**2-(Fraction(float(z))-int(t))**2 for x,z,t in zip(a,b,y))
    out=exact_brier_difference(a,b,y)
    assert Fraction(int(out['sum_numerator']),1<<out['sum_denominator_power2'])==expected
    assert out['sign']==1
    assert exact_brier_difference(a,a,y)['sign']==0
