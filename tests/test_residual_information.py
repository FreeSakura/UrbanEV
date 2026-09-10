import importlib.util
import sys
from pathlib import Path
import numpy as np


def module():
    folder=Path(__file__).resolve().parents[1]/'scripts/research'
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('residual_information',folder/'residual_information.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_ols_innovation_removes_linear_span_without_future_refit():
    m=module();rng=np.random.default_rng(21)
    x=np.column_stack([rng.normal(size=(100,4)),np.ones(100)])
    vx=np.column_stack([rng.normal(size=(20,4)),np.ones(20)])
    a=rng.normal(size=(5,11));d=x@a;vd=vx@a
    z,vz,cross=m.residual_features(x,vx,d,vd)
    assert cross<1e-10
    assert np.max(np.abs(z))==0 and np.max(np.abs(vz))==0
    # Training transformations must not change when only validation is changed.
    noise=rng.normal(size=d.shape)
    z1,_,_=m.residual_features(x,vx,d+noise,vd)
    z2,_,_=m.residual_features(x,vx,d+noise,vd+100)
    np.testing.assert_array_equal(z1,z2)


def test_residual_features_have_registered_delays():
    m=module();o=np.arange(400*275).reshape(400,275).astype(float);d=o+1e6
    base=np.zeros((1,3,275));x,z=m.features(o,d,[300],base)
    assert x.shape==(275,20) and z.shape==(1,275,11)
    np.testing.assert_array_equal(x[:,0],o[299])
    np.testing.assert_array_equal(z[0,:,0],d[298])
    np.testing.assert_array_equal(z[0,:,8],d[131])
    np.testing.assert_array_equal(z[0,:,9],d[275:299].mean(0))


def test_risk_identity_and_ridge_perturbation_are_exact():
    rng=np.random.default_rng(21);z=rng.normal(size=(200,5));e=rng.normal(size=200)
    sigma=z.T@z/len(z);c=z.T@e/len(z);lam=.01
    w=np.linalg.solve(sigma+lam*np.eye(5),c);u=rng.normal(size=5)*.02
    gain=lambda v:np.mean(e**2)-np.mean((e-z@v)**2)
    np.testing.assert_allclose(gain(w),2*w@c-w@sigma@w,atol=1e-14)
    np.testing.assert_allclose(gain(w+u),gain(w)+2*lam*u@w-u@sigma@u,atol=1e-14)
