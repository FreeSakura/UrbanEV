import numpy as np
import pytest
from urbanev_forecast.benchmark_metrics import scoped_scores


def test_scoping_changes_target_and_ranking():
    y=np.zeros((2,3,2)); a=y.copy();a[:,:2,:]=.6;b=y+.3
    terminal=scoped_scores(a,y,target_scope='terminal_H',postprocess='raw')
    path=scoped_scores(a,y,target_scope='path_1_to_H',postprocess='raw')
    assert terminal['rmse']==0 and terminal['scored_values']==4
    assert path['scored_values']==12
    assert path['rmse']>scoped_scores(b,y,target_scope='path_1_to_H',postprocess='raw')['rmse']


def test_clipping_is_named_and_never_mutates_inputs():
    p=np.array([[[-1.],[2.]]]);y=np.array([[[0.],[1.]]]);original=p.copy()
    assert scoped_scores(p,y,target_scope='path_1_to_H',postprocess='clip_0_1')['rmse']==0
    assert scoped_scores(p,y,target_scope='path_1_to_H',postprocess='raw')['rmse']==1
    np.testing.assert_array_equal(original,p)


@pytest.mark.parametrize('p,y',[(np.zeros((3,2)),np.zeros((3,2))),(np.zeros((1,0,2)),np.zeros((1,0,2))),(np.array([[[np.nan]]]),np.zeros((1,1,1)))])
def test_invalid_score_population_rejected(p,y):
    with pytest.raises(ValueError):scoped_scores(p,y,target_scope='terminal_H',postprocess='raw')
