import numpy as np
import pytest
from urbanev_forecast import short_state as s

def test_exact_forward_boundaries():
    fit=s.origins('FIT');ev=s.origins('SCORE')
    assert len(fit)==876 and fit[-1]+11==1055 and fit[-1]-2==1042
    assert ev[-1]+11==1559 and ev[-1]-2==1546
    assert fit[-1]+11<s.origins('SELECT')[0]
    assert s.origins('SELECT')[-1]+11<ev[0]

def test_inputs_ascend_and_ignore_unavailable_values():
    y=np.arange(600,dtype=float).reshape(300,2)/600;d=y/2;cap=np.array([2,5]);oo=np.array([192])
    x=s.features(y,d,cap,oo,'LONG_OD');o=s.features(y,d,cap,oo,'SHORT_O')
    yy=y.copy();dd=d.copy();yy[192:]=999;dd[191:]=999
    np.testing.assert_array_equal(s.features(yy,dd,cap,oo,'LONG_OD'),x)
    np.testing.assert_array_equal(x[0,:168],y[24:192,0]);np.testing.assert_array_equal(x[0,168:336],d[23:191,0])
    np.testing.assert_array_equal(s.features(y,d*999,cap,oo,'SHORT_O'),o)
    np.testing.assert_array_equal(s.features(y,d,cap,oo,'SHORT_OD')[:,:2],o[:,:2])
    assert x.shape==(2,341) and o.shape==(2,8)

def test_transform_fit_does_not_use_select_labels():
    x=np.array([[1.,0,3],[2,0,4],[3,0,5]])
    st=s.fit_transform(x);before={k:v.copy() for k,v in st.items()}
    s.transform(x+100,st)
    for key in st:np.testing.assert_array_equal(st[key],before[key])
    assert np.all(s.transform(x+100,st)[:,1]==0)
    # Fitting is a function of training inputs/labels only; later labels are not arguments.
    z=s.transform(x,st);target=np.array([[1.,2],[2,3],[4,5]])
    beta=s.fit_ridge(z,target);phi=np.column_stack([np.ones(len(z)),z]);pen=np.diag([0,.01,.01,.01])
    np.testing.assert_allclose((phi.T@phi/3+pen)@beta,phi.T@target/3,atol=1e-12)

def test_selection_ignores_mae_and_evaluation_and_freezes_budget():
    rows=[{'epoch':10,'selection_rmse':.11,'mae':0.},{'epoch':5,'selection_rmse':.10,'mae':99.}]
    assert s.checkpoint_choice(rows)['epoch']==5
    assert s.checkpoint_choice([{'epoch':10,'selection_rmse':1},{'epoch':5,'selection_rmse':1}])['epoch']==5
    budget=s.FitBudget()
    for _ in range(8):budget.start('neural')
    for _ in range(2):budget.start('ridge')
    budget.close()
    with pytest.raises(RuntimeError):budget.start('neural')

def test_calendar_special_values():
    sin,cos=s.phase(np.array([0,6,12,18,24]),24)
    np.testing.assert_array_equal(sin,[0,1,0,-1,0]);np.testing.assert_array_equal(cos,[1,0,-1,0,1])
