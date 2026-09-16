"""Independent completion enumeration and the original V3 DP as oracles."""
from itertools import product

import numpy as np
import pytest

from urbanev_audit.persistent_events import automaton, event_labels, paired_bounds, weighted_bounds, component_sign_certificate, endpoint_certificate
from urbanev_audit.paired_events import joint_brier_bounds


def brute(observed, coefficients, window, run_length):
    missing = np.flatnonzero(np.asarray(observed) == -1)
    values = []
    for fill in product((0, 1), repeat=len(missing)):
        z = np.asarray(observed).copy()
        z[missing] = fill
        labels = [any(all(z[t:t+run_length]) for t in range(s, s+window-run_length+1))
                  for s in range(len(z)-window+1)]
        values.append(float(np.dot(labels, coefficients)))
    return min(values), max(values)


def test_random_signed_weights_against_enumeration():
    rng = np.random.default_rng(20260916)
    for _ in range(80):
        t = int(rng.integers(1, 11))
        k = int(rng.integers(1, t+1))
        ell = int(rng.integers(1, k+1))
        observed = rng.choice([-1, 0, 1], t)
        c = rng.uniform(-2, 2, t-k+1)
        actual = weighted_bounds(observed, c, k, ell)
        expected = brute(observed, c, k, ell)
        assert (actual["lower_sum"], actual["upper_sum"]) == pytest.approx(expected, abs=1e-12)
        assert actual["reachable_states"] <= ell*(k-ell+2)


def test_compressed_matches_original_v3_suffix_dp():
    rng = np.random.default_rng(91)
    for k in range(1, 13):
        for ell in {1, k, max(1,k//2)}:
            o = rng.choice([-1,0,1], 35)
            a, b = rng.random((2,len(o)-k+1))
            actual = paired_bounds(o,a,b,k,ell)
            original = joint_brier_bounds(o,a,b,k,ell)
            for key in ("lower_sum","upper_sum","independent_lower_sum","independent_upper_sum"):
                assert actual[key] == pytest.approx(original[key],abs=1e-11)


def test_shared_missing_bit_can_change_winner_certification():
    # D1=z-.35 from (.10,.60); D2=.55-z from (.80,.30).
    result = paired_bounds([0,-1,0],[.10,.80],[.60,.30],2,1)
    assert result["lower_sum"]/2 == pytest.approx(.10)
    assert result["upper_sum"]/2 == pytest.approx(.10)
    assert result["independent_lower_sum"]/2 == pytest.approx(-.40)
    assert result["independent_upper_sum"]/2 == pytest.approx(.60)


def test_same_sign_coefficients_need_no_joint_tightening():
    rng = np.random.default_rng(8)
    for k, ell in [(1,1),(6,1),(6,6),(24,3),(72,5)]:
        o = rng.choice([-1,0,1], 100)
        for sign in [-1,1]:
            r = weighted_bounds(o,sign*rng.random(101-k),k,ell)
            assert r['lower_sum'] == pytest.approx(r['independent_lower_sum'])
            assert r['upper_sum'] == pytest.approx(r['independent_upper_sum'])


def test_events_expire_and_long_runs_keep_refreshing():
    bits = np.array([1,1,1,1,0,0,0,0,0,1,1,1])
    actual = event_labels(bits,5,3)
    expected = [any(np.all(bits[j:j+3]) for j in range(s,s+3)) for s in range(8)]
    assert np.array_equal(actual,expected)
    assert len(automaton(24,3)[2]) <= 69
    assert len(automaton(180,5)[2]) <= 885


def test_component_condition_is_stronger_than_global_same_sign():
    o=[-1,0,0,0,-1];c=np.array([1.,0.,0.,-1.])
    certificate=component_sign_certificate(o,c,2,1)
    assert certificate['components']==2 and certificate['independent_bounds_provably_exact']
    result=weighted_bounds(o,c,2,1)
    assert result['lower_sum']==result['independent_lower_sum']
    assert result['upper_sum']==result['independent_upper_sum']


def test_component_certificate_against_random_exact_dp():
    rng=np.random.default_rng(55)
    for _ in range(200):
        o=rng.choice([-1,0,1],25);k=int(rng.integers(1,10));ell=int(rng.integers(1,k+1))
        c=rng.normal(size=26-k)
        if component_sign_certificate(o,c,k,ell)['independent_bounds_provably_exact']:
            r=weighted_bounds(o,c,k,ell)
            assert r['lower_sum']==pytest.approx(r['independent_lower_sum'])
            assert r['upper_sum']==pytest.approx(r['independent_upper_sum'])


def test_unattainable_independent_endpoints_have_an_explicit_conflict():
    for side in ('lower','upper'):
        result=endpoint_certificate([0,-1,0],[1.,-1.],2,1,side,explain=True)
        assert not result['independent_endpoint_attainable']
        assert len(result['conflict_core'])==2
        assert {x['required_event'] for x in result['conflict_core']}=={0,1}


def test_endpoint_feasibility_iff_independent_bound_is_attained():
    rng=np.random.default_rng(77)
    for _ in range(100):
        o=rng.choice([-1,0,1],12);k=int(rng.integers(1,9));ell=int(rng.integers(1,k+1))
        c=rng.integers(-4,5,13-k).astype(float)
        r=weighted_bounds(o,c,k,ell)
        for side in ('lower','upper'):
            result=endpoint_certificate(o,c,k,ell,side)
            assert result['independent_endpoint_attainable']==(r[side+'_sum']==r['independent_'+side+'_sum'])
