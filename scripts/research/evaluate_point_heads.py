"""Evaluate the frozen B2 point systems; write aggregates, keep parameters private."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from urbanev_forecast.data import load_rate_prefix, scores, window_starts
from urbanev_forecast.foundation import file_hash
from urbanev_forecast.quantile_head import quantile_design, fit_quantile_head

GRID=(.001,.01,.1)
CONTROLS=("native_q50","postprocessed_q50","bounded_midpoint","native_bias","native_affine","midpoint_affine","quantile_ridge")


def score_pair(prediction,target):
    return {"raw":scores(prediction,target),"clipped":scores(np.clip(prediction,0,1),target)}


def fit_affine(prediction,target):
    design=np.column_stack((prediction,np.ones(len(prediction))))
    return np.linalg.lstsq(design,target,rcond=None)[0]


def gate(cells):
    macro={name:{metric:float(np.mean([cell["systems"][name]["clipped"][metric] for cell in cells]))
        for metric in ("rmse","mae")} for name in (*CONTROLS,"simplex_head")}
    best=min(CONTROLS,key=lambda name:macro[name]["rmse"])
    gain=100*(1-macro["simplex_head"]["rmse"]/macro[best]["rmse"])
    degradations=[100*(cell["systems"]["simplex_head"]["clipped"]["rmse"]/cell["systems"][best]["clipped"]["rmse"]-1) for cell in cells]
    mae_ok=macro["simplex_head"]["mae"]<=macro[best]["mae"]
    return {"macro":macro,"best_control_family":best,"rmse_gain_percent":gain,
            "macro_mae_non_degradation":bool(mae_ok),"cell_rmse_degradation_percent":degradations,
            "decision":"GO" if gain>=1 and mae_ok and max(degradations)<=1 else "NO_GO",
            "criterion":">=1% macro RMSE gain, no macro MAE degradation, <=1% degradation in either cell",
            "independent_confirmation":False,"sota_claim":False}


def evaluate_horizon(q,p,origins,values,horizon):
    expected=np.r_[window_starts(648,168,horizon,0,576),window_starts(648,168,horizon,576,648)]
    if not np.array_equal(origins,expected):raise ValueError("Cache does not match all registered origins")
    if q.shape!=(len(origins),horizon,275,9) or p.shape!=q.shape[:-1]:raise ValueError("Wrong cache shape")
    if not np.isfinite(q).all() or not np.isfinite(p).all() or not np.allclose(q[...,4],p,atol=1e-7,rtol=1e-6):
        raise ValueError("Invalid or non-native cache values")
    ntrain=int((origins<576).sum())
    y=values[origins[:,None]+np.arange(horizon)[None,:]].astype(np.float64)
    ytrain,yval=y[:ntrain].ravel(),y[ntrain:].ravel()
    qtrain=np.asarray(q[:ntrain]).reshape(-1,9);qval=np.asarray(q[ntrain:]).reshape(-1,9)
    ptrain=np.asarray(p[:ntrain],np.float64).ravel();pval=np.asarray(p[ntrain:],np.float64).ravel()
    levels=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
    xt,prior,bt=quantile_design(qtrain,levels);xv,_,bv=quantile_design(qval,levels)
    weights={}
    bias=float((ytrain-ptrain).mean());weights["native_bias"]=np.asarray([bias])
    affine=fit_affine(ptrain,ytrain);weights["native_affine"]=affine
    mid_affine=fit_affine(bt["midpoint"],ytrain);weights["midpoint_affine"]=mid_affine
    predictions={"native_q50":pval,"postprocessed_q50":xv[:,5],"bounded_midpoint":bv["midpoint"],
        "native_bias":pval+bias,"native_affine":affine[0]*pval+affine[1],
        "midpoint_affine":mid_affine[0]*bv["midpoint"]+mid_affine[1]}
    gram=xt.T@xt/len(xt);linear=xt.T@ytrain/len(xt)
    candidates={"quantile_ridge":[],"simplex_head":[]}
    for strength in GRID:
        ridge_weights=np.linalg.solve(gram+strength*np.eye(xt.shape[1]),linear+strength*prior)
        ridge_prediction=xv@ridge_weights
        candidates["quantile_ridge"].append({"ridge":strength,"scores":score_pair(ridge_prediction,yval)})
        weights[f"ridge_{strength}"]=ridge_weights
        head=fit_quantile_head(qtrain,levels,ytrain,ridge=strength,max_steps=100000,tolerance=1e-8)
        if not head["converged"]:raise RuntimeError(f"Simplex solver did not converge for ridge {strength}")
        candidates["simplex_head"].append({"ridge":strength,"iterations":head["iterations"],
            "projected_gradient_residual":head["projected_gradient_residual"],"converged":True,
            "scores":score_pair(xv@head["weights"],yval)})
        weights[f"simplex_{strength}"]=head["weights"]
    systems={name:score_pair(prediction,yval) for name,prediction in predictions.items()}
    selected={}
    for family,items in candidates.items():
        chosen=min(items,key=lambda item:item["scores"]["clipped"]["rmse"])
        selected[family]=chosen["ridge"];systems[family]=chosen["scores"]
    row={"horizon":horizon,"training_windows":ntrain,"validation_windows":len(origins)-ntrain,
        "training_label_cells":len(ytrain),"validation_label_cells":len(yval),"systems":systems,
        "regularization_candidates":candidates,"selected_regularization":selected,
        "repair":{"training_fraction":bt["repaired_fraction"],"validation_fraction":bv["repaired_fraction"],
          "validation_q50_changed_fraction":float(np.mean(xv[:,5]!=pval)),
          "validation_out_of_bounds_fraction":float(np.mean(np.any((qval<0)|(qval>1),axis=1))),
          "validation_crossed_fraction":float(np.mean(np.any(np.diff(qval,axis=1)<0,axis=1)))},
        "prediction_mean_width":float(np.mean(bv["upper"]-bv["lower"])),
        "parameter_sha256":{name:hashlib.sha256(np.asarray(value,dtype='<f8').tobytes()).hexdigest() for name,value in weights.items()}}
    return row,weights


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--csv",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("Use a fresh output directory")
    started=time.perf_counter();values,source=load_rate_prefix(args.csv,648)
    receipt=json.loads((args.cache/"receipt.json").read_text())
    summary=json.loads((args.cache/"cache_summary.json").read_text())
    if summary.get("status")!="COMPLETE" or source!=receipt["source"]:raise ValueError("Cache or source identity mismatch")
    args.output.mkdir(parents=True);cells=[]
    for record in summary["horizons"]:
        h=record["horizon"];folder=args.cache/f"h{h}"
        for file,key in (("quantiles.npy","quantiles_sha256"),("native.npy","native_sha256"),("origins.npy","origins_sha256")):
            if file_hash(folder/file)!=record[key]:raise ValueError("Cached prediction hash changed")
        q=np.load(folder/"quantiles.npy",mmap_mode="r");p=np.load(folder/"native.npy",mmap_mode="r");origins=np.load(folder/"origins.npy")
        row,parameters=evaluate_horizon(q,p,origins,values,h);cells.append(row)
        np.savez(args.output/f"private_head_parameters_h{h}.npz",**parameters)
        print(json.dumps({"horizon":h,"selected_regularization":row["selected_regularization"],
            "validation_rmse":{k:v["clipped"]["rmse"] for k,v in row["systems"].items()}}),flush=True)
    if [cell["horizon"] for cell in cells]!=[3,12]:raise ValueError("Both registered horizons are required")
    report={"protocol_id":receipt["protocol_id"],"backend":receipt["backend"],"source":source,
        "cache_summary_sha256":file_hash(args.cache/"cache_summary.json"),"cells":cells,"gate":gate(cells),
        "evaluation_code_sha256":file_hash(Path(__file__)),"head_code_sha256":file_hash(ROOT/"src/urbanev_forecast/quantile_head.py"),
        "elapsed_seconds":time.perf_counter()-started,"test_accessed":False,"scope":"fold1 development selection evidence only",
        "historical_failed_protocols_reclassified":False,"learned_parameters_published":False}
    (args.output/"evaluation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(report["gate"],indent=2),flush=True)


if __name__=="__main__":main()
