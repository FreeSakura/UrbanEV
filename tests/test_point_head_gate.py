import importlib.util
from pathlib import Path


def evaluator():
    path=Path(__file__).resolve().parents[1]/"scripts/research/evaluate_point_heads.py"
    spec=importlib.util.spec_from_file_location("point_head_evaluator",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_gate_requires_rmse_gain_and_mae_non_degradation():
    module=evaluator()
    cells=[]
    for horizon in (3,12):
        systems={name:{"clipped":{"rmse":.1,"mae":.05}} for name in module.CONTROLS}
        systems["simplex_head"]={"clipped":{"rmse":.098,"mae":.049}}
        cells.append({"horizon":horizon,"systems":systems})
    assert module.gate(cells)["decision"]=="GO"
    cells[1]["systems"]["simplex_head"]["clipped"]["mae"]=.052
    assert module.gate(cells)["decision"]=="NO_GO"


def test_gate_does_not_replace_a_control_with_a_per_cell_oracle():
    module=evaluator();cells=[]
    for index,h in enumerate((3,12)):
        systems={name:{"clipped":{"rmse":.2,"mae":.1}} for name in module.CONTROLS}
        systems["native_q50"]["clipped"]["rmse"]=(.05,.15)[index]
        systems["quantile_ridge"]["clipped"]["rmse"]=(.12,.06)[index]
        systems["simplex_head"]={"clipped":{"rmse":.1,"mae":.1}}
        cells.append({"horizon":h,"systems":systems})
    result=module.gate(cells)
    assert result["best_control_family"]=="quantile_ridge"
    assert result["decision"]=="NO_GO"
