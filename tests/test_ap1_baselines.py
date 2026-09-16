from itertools import product
import numpy as np
import pytest
from scipy.optimize import linprog
from urbanev_audit.persistent_events import automaton, event_labels, paired_bounds, _extrema
from urbanev_audit.event_relaxations import PairwiseLP
from urbanev_audit.regular_event_baseline import compile_event,canonical,generic_edge_paths
from urbanev_audit.event_comparison_ap1 import origin_blocks,score_panel


def test_all_pairwise_relations_match_enumerated_completions():
    rng=np.random.default_rng(123)
    for _ in range(120):
        t=8;k=int(rng.integers(1,7));ell=int(rng.integers(1,k+1));o=rng.choice([-1,0,1],t)
        missing=np.flatnonzero(o==-1);labels=[]
        for bits in product((0,1),repeat=len(missing)):
            z=o.copy();z[missing]=bits;labels.append(event_labels(z,k,ell))
        labels=np.asarray(labels);lp=PairwiseLP(o,k,ell)
        for q,i in enumerate(lp.ambiguous):
            for j in lp.ambiguous:
                truth=not np.any((labels[:,i]==1)&(labels[:,j]==0))
                assert truth==(lp.first[q]<=j<=lp.last[q])
        a,b=rng.random((2,t-k+1));pair=lp.bounds(a,b);joint=paired_bounds(o,a,b,k,ell)
        assert pair['lower']<=joint['lower_sum']/len(a)+1e-9
        assert pair['upper']>=joint['upper_sum']/len(a)-1e-9
        if len(lp.ambiguous):
            constraints=[];w=2*(b-a);m=len(lp.ambiguous)
            for x,i in enumerate(lp.ambiguous):
                for y,j in enumerate(lp.ambiguous):
                    if i!=j and not np.any((labels[:,i]==1)&(labels[:,j]==0)):
                        row=np.zeros(m);row[x]=1;row[y]=-1;constraints.append(row)
            matrix=np.array(constraints) if constraints else None
            rhs=np.zeros(len(constraints)) if constraints else None
            base=float(np.sum(a*a-b*b+w*lp.lo))
            lower=linprog(w[lp.ambiguous],A_ub=matrix,b_ub=rhs,bounds=(0,1),method='highs')
            upper=linprog(-w[lp.ambiguous],A_ub=matrix,b_ub=rhs,bounds=(0,1),method='highs')
            assert pair['lower']==pytest.approx((base+lower.fun)/len(a),abs=1e-9)
            assert pair['upper']==pytest.approx((base-upper.fun)/len(a),abs=1e-9)


def test_three_window_higher_order_counterexample():
    o=[-1]*4;a=[0.,1.,0.];b=[.5]*3
    pair=PairwiseLP(o,2,1).bounds(a,b);joint=paired_bounds(o,a,b,2,1)
    assert [pair['lower'],pair['upper']]==pytest.approx([-.25,.75])
    assert [joint['lower_sum']/3,joint['upper_sum']/3]==pytest.approx([1/12,.75])


def test_generic_compiler_without_manual_state_is_isomorphic():
    for k in [1,2,4,8,16,24]:
        for ell in {1,min(3,k),k}:
            edges,accept,_=compile_event(k,ell)
            manual,event,_=automaton(k,ell);manual,event=canonical(manual,event)
            expected=1+ell+sum(min(ell,d) for d in range(1,k-ell+1))
            assert len(edges)==expected
            np.testing.assert_array_equal(edges,manual);np.testing.assert_array_equal(accept,event)


def test_weighted_path_uses_every_terminal_state():
    rng=np.random.default_rng(99)
    for k,ell in [(1,1),(8,3),(24,3),(24,24)]:
        o=rng.choice([-1,0,1],50);c=rng.normal(size=51-k)
        edges,accept,_=compile_event(k,ell)
        generic=generic_edge_paths(o,c,k,edges,accept)
        manual,event,_=automaton(k,ell)
        expected=_extrema(o,c,k,manual,event)
        assert generic==pytest.approx(expected,abs=1e-11)


def test_origin_blocks_keep_full_support_and_overlap_across_boundaries():
    blocks=list(origin_blocks(100,500,24,168))
    assert sum(b-a for a,b,e in blocks)==377
    assert all(e-a==(b-a)+23 for a,b,e in blocks)
    assert blocks[0][2]-blocks[1][0]==23
    assert blocks[-1][2]==500


def test_near_zero_is_not_declared_a_strict_win():
    o=np.array([0,1,0,1,0]);pred={'a':np.full(4,.5),'b':np.full(4,.5+1e-13),'c':np.full(4,.6)}
    rows,_=score_panel(o,pred,2,1,{})
    assert rows[0]['numeric_method']=='INTEGER_GRID_OUTWARD'
    assert rows[0]['joint_direction'] not in ('A','B')
