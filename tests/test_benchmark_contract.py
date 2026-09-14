import numpy as np
import pytest

from urbanev_forecast.benchmark_contract import (
    HORIZONS, cell_metadata, fold_plan, macro24, origin_digest, origins, paired_cell, score_arrays,
    declared_rates, select_checkpoint, training_partitions, validate_training_spec,
    validation_rmse, upstream_percentage_mirror,
)


def test_calendar_boundaries_and_horizon_specific_counts():
    expected = [(576, 648, 720), (1171, 1318, 1464), (1747, 1966, 2184),
                (2342, 2636, 2928), (2937, 3305, 3672), (3475, 3910, 4344)]
    total = 0
    for f, (tr, va, te) in enumerate(expected, 1):
        p = fold_plan(f)
        assert (p.train_stop, p.validation_stop, p.test_stop) == (tr, va, te)
        for h in HORIZONS:
            o = origins(f, "test", 168, h)
            assert (o[0], o[-1], len(o)) == (va, te-h, te-va-h+1)
            total += len(o)
            # Train windows cannot train on a validation target.
            assert origins(f, "train", 168, h)[-1]+h == tr
            assert origins(f, "validation", 168, h)[-1]+h == va
    assert total == 5960
    # Joint-H12 support would omit valid earlier-H origin windows.
    assert len(origins(1, "test", 168, 3))-len(origins(1, "test", 168, 12)) == 9


@pytest.mark.parametrize("fold", [True, 0, 7, 1.5, "1"])
def test_invalid_fold_rejected(fold):
    with pytest.raises(ValueError):
        fold_plan(fold)


def test_endpoint_path_and_clipping_do_not_silently_mix():
    p = np.array([[[100.], [2.]], [[100.], [-1.]]])
    y = np.zeros_like(p)
    end = score_arrays(p, y, scope="terminal", postprocess="raw")
    path = score_arrays(p, y, scope="path", postprocess="raw")
    clipped = score_arrays(p, y, scope="terminal", postprocess="clip_0_1")
    assert end["rmse"] == pytest.approx(np.sqrt(2.5))
    assert end["mae"] == 1.5
    assert path["mse"] == 5001.25
    assert clipped["mse"] == .5
    np.testing.assert_array_equal(p, [[[100.], [2.]], [[100.], [-1.]]])
    with pytest.raises(ValueError):
        score_arrays(p[:, 0], y[:, 0], scope="terminal", postprocess="raw")


def test_macro_is_mean_cell_rmse_not_pooled_mse_and_rejects_gaps():
    cells = {(f, h): dict(rmse=float(f), mae=float(f)/2, scope="terminal", postprocess="raw",
                         split="test", history=168, number=f, horizon=h, contract="MATCHED_TERMINAL_168")
             for f in range(1, 7) for h in HORIZONS}
    m = macro24(cells)
    assert m["rmse"] == 3.5 and m["mae"] == 1.75
    assert m["contract"] == "MATCHED_TERMINAL_168"
    assert m["rmse"] != pytest.approx(np.sqrt(np.mean(np.arange(1, 7)**2)))
    cells[(1, 3)]["scope"] = "path"
    with pytest.raises(ValueError, match="mixed scope"):
        macro24(cells)
    del cells[(1, 3)]
    with pytest.raises(ValueError, match="exactly"):
        macro24(cells)


def test_origin_manifest_rejects_missing_shifted_or_reordered_predictions():
    o = origins(1, "test", 168, 3)
    y = np.zeros((len(o), 3, 275))
    kwargs = dict(fold=1, horizon=3, history=168, split="test", scope="terminal", postprocess="raw")
    assert paired_cell(y, y, o, **kwargs)["rmse"] == 0
    assert origin_digest(o) == cell_metadata(1, 3, 168)["origin_sha256"]
    for bad in [o+1, o[::-1], o[:-1], o.astype(float)]:
        with pytest.raises(ValueError):
            paired_cell(y, y, bad, **kwargs)
    with pytest.raises(ValueError):
        paired_cell(y[:, :, :-1], y[:, :, :-1], o, **kwargs)


def test_nonfinite_errors_fail_instead_of_becoming_missing_scores():
    y = np.zeros((2, 3, 4))
    y[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        score_arrays(y, y, scope="terminal", postprocess="raw")


def test_three_contracts_keep_their_different_windows():
    assert [fold_plan(f, "UPSTREAM_CLASSICAL").validation_stop for f in range(1, 7)] == [648,1317,1965,2634,3304,3909]
    assert origins(1, "train", 168, 3, contract="MATCHED_TERMINAL_168")[0] == 169
    assert origins(1, "train", 168, 3)[0] == 168
    assert origins(1, "test", 12, 3, contract="UPSTREAM_CLASSICAL")[0] == 660
    assert origins(1, "test", 12, 3, contract="UPSTREAM_CLASSICAL")[-1] == 716
    assert cell_metadata(1, 3, 168, contract="UPSTREAM_CLASSICAL")["count"] == 0
    with pytest.raises(ValueError):
        origins(1, "train", 12, 3, contract="MATCHED_TERMINAL_168")


def test_declared_units_and_duplicate_capacity_policy():
    common = dict(region_ids=["B", "A"], capacity_level="station", capacity_aggregation="sum_by_TAZID")
    counts = np.array([[5, 3]], np.float32)
    records = [("A", 2), ("B", 10), ("A", 4)]
    rates = declared_rates(counts, raw_unit="busy_count", capacity_records=records, **common)
    np.testing.assert_array_equal(rates, [[.5,.5]])
    np.testing.assert_array_equal(declared_rates(rates, raw_unit="occupancy_rate", capacity_records=records, **common), rates)
    with pytest.raises(ValueError, match="duplicate"):
        declared_rates(counts, ["B","A"], raw_unit="busy_count", capacity_records=records,
                       capacity_level="region", capacity_aggregation="already_aggregated")
    with pytest.raises(ValueError, match="raw_unit"):
        declared_rates(rates, raw_unit="guess", **common)


def test_validation_tail_weighting_can_change_selected_checkpoint():
    # A has 32 exact elements and 1 error=10; B has all elements error=2.
    a, b = [(0.,32),(100.,1)], [(128.,32),(4.,1)]
    assert validation_rmse(a) < validation_rmse(b)
    assert np.mean([0,100]) > np.mean([4,4])  # wrong batch-mean rule reverses choice
    chosen = select_checkpoint({10:a, 5:a, 1:b}, split="validation", scope="terminal", postprocess="raw")
    assert chosen["epoch"] == 5
    with pytest.raises(ValueError):
        select_checkpoint({1:a}, split="test", scope="terminal", postprocess="raw")
    with pytest.raises(ValueError, match="counts differ"):
        select_checkpoint({1:a,2:[(0,1)]}, split="validation", scope="terminal", postprocess="raw")


def test_endpoint_and_path_rankings_can_reverse():
    y=np.zeros((1,3,1)); a=np.array([[[10.],[10.],[0.]]]); b=np.ones_like(y)
    assert score_arrays(a,y,scope="terminal",postprocess="raw")["rmse"] < score_arrays(b,y,scope="terminal",postprocess="raw")["rmse"]
    assert score_arrays(a,y,scope="path",postprocess="raw")["rmse"] > score_arrays(b,y,scope="path",postprocess="raw")["rmse"]


def test_training_preparation_has_no_test_factory_call_or_implicit_budget():
    calls=[]
    def factory(split):
        calls.append(split)
        assert split != "test"
        return split
    assert set(training_partitions(factory)) == {"train", "validation"}
    assert calls == ["train", "validation"]
    with pytest.raises(ValueError, match="explicit"):
        validate_training_spec({})
    base=dict(epochs=2,seeds=[1,2,3],search_candidates=["config_a"],training_objective="raw_path_mse",prediction_mode="horizon_specific")
    assert validate_training_spec(base)["prediction_mode"] != validate_training_spec({**base,"prediction_mode":"joint_12"})["prediction_mode"]


def test_past_input_and_target_indices_never_overlap():
    for f in range(1,7):
        for h in HORIZONS:
            for split in ("train","validation","test"):
                o=origins(f,split,168,h,contract="MATCHED_TERMINAL_168")
                for t in (o[0],o[-1]):
                    history=np.arange(t-168,t); duration=np.arange(t-169,t-1); labels=np.arange(t,t+h)
                    assert history[-1] < labels[0] and duration[-1] == labels[0]-2
                    assert not np.intersect1d(history,labels).size


def test_upstream_percentage_order_preserved_and_zero_denominator_exposed():
    y=np.array([0.,.01,.02,-.01]); p=np.array([-.5,-.5,-.5,-.5])
    # y becomes [.02,.03,.04,.03]; only originally-zero y modifies prediction.
    expected=np.mean(np.abs(np.array([.52,-.5,-.5,-.5])-np.array([.02,.03,.04,.03]))/np.array([.02,.03,.04,.03]))
    assert upstream_percentage_mirror(p,y)["mape"] == pytest.approx(expected)
    assert np.isnan(upstream_percentage_mirror(np.ones(3),np.ones(3))["rae"])
    assert np.isinf(upstream_percentage_mirror(np.zeros(3),np.ones(3))["rae"])


@pytest.fixture
def complete_cells():
    return {(f,h):dict(number=f,horizon=h,rmse=.1,mae=.05,scope="terminal",
                       postprocess="raw",split="test",history=168,contract="UPSTREAM_TRANSFORMER")
            for f in range(1,7) for h in HORIZONS}


def test_macro_rejects_one_different_contract_in_complete_table(complete_cells):
    complete_cells[(1,3)]["contract"]="UPSTREAM_CLASSICAL"
    with pytest.raises(ValueError,match="mixed contract"):
        macro24(complete_cells)


def test_macro_rejects_missing_contract_identity(complete_cells):
    del complete_cells[(1,3)]["contract"]
    with pytest.raises(ValueError,match="contract identity"):
        macro24(complete_cells)


def test_macro_rejects_row_and_key_mismatch(complete_cells):
    complete_cells[(1,3)]["number"]=2
    with pytest.raises(ValueError,match="cell key"):
        macro24(complete_cells)
    complete_cells[(1,3)]["number"]=1
    complete_cells[(1,3)]["horizon"]=6
    with pytest.raises(ValueError,match="cell key"):
        macro24(complete_cells)
