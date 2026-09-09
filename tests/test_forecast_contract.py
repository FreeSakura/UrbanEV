import numpy as np
import pytest
from urbanev_forecast.data import fold_bounds, scores, window_starts


def test_all_targets_stay_in_their_partition():
    for fold in range(1,7):
        train,val,end = fold_bounds(fold)
        for horizon in (3,6,9,12):
            for start,stop in ((0,train),(train,val),(val,end)):
                origins=window_starts(end,168,horizon,start,stop)
                assert origins.min()>=max(168,start)
                assert origins.max()+horizon-1==stop-1
                assert origins.min()-168>=0
    assert fold_bounds(6)[-1] == 4344
    with pytest.raises(ValueError):fold_bounds(0)


def test_score_uses_all_steps_and_rejects_invalid_predictions():
    target=np.zeros((2,3,2));prediction=np.zeros_like(target);prediction[0,0,0]=1
    result=scores(prediction,target)
    assert result["rmse"]==pytest.approx(np.sqrt(1/12))
    assert result["mae"]==pytest.approx(1/12)
    prediction[1,0,0]=np.nan
    with pytest.raises(ValueError):scores(prediction,target)


def test_preparation_and_loading_do_not_parse_later_targets(tmp_path):
    import pandas as pd
    from urbanev_forecast.prepare import prepare_rates
    from urbanev_forecast.data import load_rate_prefix
    columns=[str(i) for i in range(275)]
    source=tmp_path/"source";source.mkdir()
    frame=pd.DataFrame(np.ones((30,275)),index=pd.date_range("2022-09-01",periods=30,freq="h"),columns=columns)
    occupancy=source/"occupancy.csv";frame.to_csv(occupancy,index_label="date")
    with occupancy.open("a") as handle:handle.write("invalid later target row\n")
    pd.DataFrame({"TAZID":columns,"charge_count":np.full(275,2)}).to_csv(source/"inf.csv",index=False)
    output=tmp_path/"rates.csv"
    prepare_rates(source,30,output)
    with output.open("a") as handle:handle.write("invalid later target row\n")
    values,receipt=load_rate_prefix(output,30)
    assert values.shape==(30,275) and np.all(values==.5)
    assert receipt["rows"]==30
