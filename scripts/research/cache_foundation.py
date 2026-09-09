"""Generate PRIVATE train/validation quantile caches after a passing profile."""
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
from urbanev_forecast.data import load_rate_prefix, window_starts
from urbanev_forecast.foundation import FoundationBackend, file_hash, validate_prediction


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def prefix_hash(values, count):
    return hashlib.sha256(np.ascontiguousarray(values[:count]).tobytes()).hexdigest()


def fill_cache(backend, values, origins, horizon, directory, resume=False):
    """No label array is ever passed to the backend; flush before advancing progress."""
    directory.mkdir(exist_ok=True)
    progress_path=directory/"progress.json"
    q_shape=(len(origins),horizon,values.shape[1],9)
    p_shape=q_shape[:-1]
    if resume and progress_path.exists():
        progress=json.loads(progress_path.read_text())
        if progress["origins_sha256"]!=hashlib.sha256(origins.tobytes()).hexdigest():
            raise ValueError("Resume origin identity changed")
        q=np.lib.format.open_memmap(directory/"quantiles.npy",mode="r+")
        p=np.lib.format.open_memmap(directory/"native.npy",mode="r+")
        if q.shape!=q_shape or p.shape!=p_shape or q.dtype!=np.float32 or p.dtype!=np.float32:
            raise ValueError("Resume cache shape or dtype changed")
        completed=progress["completed"]
        if not 0<=completed<=len(origins):raise ValueError("Invalid resume progress")
        if not np.array_equal(np.load(directory/"origins.npy",allow_pickle=False),origins):
            raise ValueError("Stored origin array changed")
        if (progress.get("completed_quantiles_sha256")!=prefix_hash(q,completed)
            or progress.get("completed_native_sha256")!=prefix_hash(p,completed)):
            raise ValueError("Completed-prefix integrity check failed; legacy unsealed progress cannot resume")
    else:
        if progress_path.exists() or (directory/"quantiles.npy").exists():
            raise FileExistsError("Explicit resume is required for an existing cache")
        q=np.lib.format.open_memmap(directory/"quantiles.npy",mode="w+",dtype=np.float32,shape=q_shape)
        p=np.lib.format.open_memmap(directory/"native.npy",mode="w+",dtype=np.float32,shape=p_shape)
        q[:]=np.nan;p[:]=np.nan;q.flush();p.flush()
        np.save(directory/"origins.npy",origins)
        completed=0
    started=time.perf_counter(); initial=completed
    for index in range(completed,len(origins)):
        origin=int(origins[index])
        forecast,native=backend.predict(values[origin-168:origin].copy(),horizon)
        forecast,native=validate_prediction(forecast,native,horizon,values.shape[1])
        q[index]=forecast;p[index]=native
        if (index+1)%20==0 or index+1==len(origins):
            q.flush();p.flush()
            write_json(progress_path,{"completed":index+1,"total":len(origins),
                "origins_sha256":hashlib.sha256(origins.tobytes()).hexdigest(),
                "completed_quantiles_sha256":prefix_hash(q,index+1),
                "completed_native_sha256":prefix_hash(p,index+1)})
            print(json.dumps({"horizon":horizon,"completed":index+1,"total":len(origins),
                              "seconds_this_session":round(time.perf_counter()-started,2)}),flush=True)
    if not np.isfinite(q).all() or not np.isfinite(p).all():raise ValueError("Incomplete prediction cache")
    return {"horizon":horizon,"windows":len(origins),"shape":list(q.shape),
        "new_windows_this_session":len(origins)-initial,"seconds_this_session":time.perf_counter()-started,
        "quantiles_sha256":file_hash(directory/"quantiles.npy"),"native_sha256":file_hash(directory/"native.npy"),
        "origins_sha256":file_hash(directory/"origins.npy")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=("chronos2","timesfm3"),required=True)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--csv",type=Path,required=True)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    profile=json.loads(args.profile.read_text())
    if profile.get("status")!="PASS" or not profile.get("cache_authorized_by_feasibility"):
        raise ValueError("A successful full-variate profile is required")
    if args.output.exists() and not args.resume:raise FileExistsError("Use a fresh cache directory or explicit resume")
    values,source=load_rate_prefix(args.csv,648)
    backend=FoundationBackend(args.backend,args.model_dir,args.device)
    for field in ("backend","weight_sha256","config_sha256","api_source_sha256","package_version","dtype","device"):
        if backend.metadata[field]!=profile["backend"][field]:raise ValueError(f"Profile/backend drift: {field}")
    protocol_path=ROOT/"configs/research/FM_CALIBRATION_FOLD1_V1.json"
    receipt={"protocol_id":"FM_CALIBRATION_FOLD1_V1_20260909","protocol_sha256":file_hash(protocol_path),
        "backend":backend.metadata,"source":source,"training_end":576,"validation_end":648,
        "test_accessed":False,"layout":"origin,H_step,zone,quantile","profile_sha256":file_hash(args.profile),
        "runner_sha256":file_hash(Path(__file__)),"adapter_sha256":file_hash(ROOT/"src/urbanev_forecast/foundation.py"),
        "data_code_sha256":file_hash(ROOT/"src/urbanev_forecast/data.py")}
    args.output.mkdir(parents=True,exist_ok=True)
    receipt_path=args.output/"receipt.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text())!=receipt:raise ValueError("Resume receipt changed")
    else:write_json(receipt_path,receipt)
    results=[]
    for horizon in (3,12):
        train=window_starts(len(values),168,horizon,0,576)
        val=window_starts(len(values),168,horizon,576,648)
        origins=np.r_[train,val]
        result=fill_cache(backend,values,origins,horizon,args.output/f"h{horizon}",args.resume)
        result.update({"training_windows":len(train),"validation_windows":len(val)})
        results.append(result)
    write_json(args.output/"cache_summary.json",{"status":"COMPLETE","protocol_id":receipt["protocol_id"],
        "backend":args.backend,"source":source,"horizons":results,"test_accessed":False,"contains_targets":False})
    print(json.dumps({"status":"COMPLETE","backend":args.backend}),flush=True)


if __name__=="__main__":main()
