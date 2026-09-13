"""Tiny exhaustive path checks independent of the research parameter systems."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('coarse_memory_runner', Path(__file__).resolve().parents[1] / 'scripts/research/run_coarse_memory_identifiability.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

P = np.array([[.5,.3,.2],[.1,.6,.3],[.2,.2,.6]])
PI = np.array([.2,.5,.3])


def test_polynomial_counts_destination_activity_and_endpoint():
    expected = np.zeros((4,3,3))
    for path in itertools.product(range(3), repeat=4):
        mass = np.prod([P[path[t],path[t+1]] for t in range(3)])
        count = sum(state == 1 for state in path[1:])
        expected[count,path[0],path[-1]] += mass
    np.testing.assert_allclose(module.coefficients(P,3),expected,atol=1e-14,rtol=0)


def test_delayed_count_filter_matches_exhaustive_paths_without_K3():
    groups={}
    for path in itertools.product(range(3),repeat=7):
        mass = PI[path[0]]*np.prod([P[path[t],path[t+1]] for t in range(6)])
        obs=tuple(int(path[t] != 0) for t in [0,2,4,6])
        counts=(sum(x==1 for x in path[1:3]),sum(x==1 for x in path[3:5]))
        key=obs+counts
        groups.setdefault(key,np.zeros(3))[path[-1]] += mass
    for history, joint in groups.items():
        mass,posterior=module.forward_filter(P,PI,history,2)
        np.testing.assert_allclose(mass,joint.sum(),atol=1e-14,rtol=0)
        np.testing.assert_allclose(posterior,joint/joint.sum(),atol=1e-14,rtol=0)
