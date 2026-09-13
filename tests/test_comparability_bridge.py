import numpy as np
import pandas as pd
from urbanev_forecast.comparability_bridge import build_inputs,baseline_predictions,zone_capacities


def test_station_capacities_are_summed_in_region_order():
    info=pd.DataFrame({'TAZID':['a','b','a'],'charge_count':[2,7,3]})
    np.testing.assert_array_equal(zone_capacities(info,['b','a']),[7,5])


def test_future_changes_cannot_change_inputs_or_baselines():
    y=np.arange(250*2,dtype=float).reshape(250,2)/500;d=y/2;origin=np.array([192])
    original=build_inputs(y,d,origin);p=baseline_predictions(y,origin,12)
    changed_y=y.copy();changed_d=d.copy();changed_y[192:]=999;changed_d[191:]=999
    updated=build_inputs(changed_y,changed_d,origin)
    for k in original:np.testing.assert_array_equal(updated[k],original[k])
    for k,v in baseline_predictions(changed_y,origin,12).items():np.testing.assert_array_equal(v,p[k])
    np.testing.assert_array_equal(original['long_duration'][0],d[23:191])
    np.testing.assert_array_equal(original['short'][0,:,2],d[190])


def test_baseline_offsets_are_actual_future_hour_offsets():
    y=np.arange(250).reshape(-1,1);p=baseline_predictions(y,np.array([192]),3)
    np.testing.assert_array_equal(p['last'][0,:,0],[191,191,191])
    np.testing.assert_array_equal(p['day'][0,:,0],[168,169,170])
    np.testing.assert_array_equal(p['week'][0,:,0],[24,25,26])
