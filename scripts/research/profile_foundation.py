"""Profile one TRAINING origin before authorizing a development prediction cache."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from urbanev_forecast.data import load_rate_prefix
from urbanev_forecast.foundation import FoundationBackend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=("chronos2","timesfm3"),required=True)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--csv",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("Use a fresh profile directory")
    args.output.mkdir(parents=True)
    import torch
    values,source=load_rate_prefix(args.csv,168)
    started=time.perf_counter()
    try:
        backend=FoundationBackend(args.backend,args.model_dir,args.device)
        load_seconds=time.perf_counter()-started
        seen=[]
        def hook(module,arguments,keywords):
            shapes={str(k):list(v.shape) for k,v in keywords.items() if hasattr(v,"shape")}
            shapes.update({f"arg_{i}":list(v.shape) for i,v in enumerate(arguments) if hasattr(v,"shape")})
            # Group labels are non-data structural metadata, not target values.
            if "group_ids" in keywords:shapes["unique_group_ids"]=int(keywords["group_ids"].unique().numel())
            if len(seen)<10:seen.append(shapes)
        if backend.model.training:raise ValueError("A profiling backend must be in evaluation mode")
        observed_module=backend.model.transformer_stack if args.backend=="timesfm3" else backend.model
        handle=observed_module.register_forward_pre_hook(hook,with_kwargs=True)
        measurements=[]
        for horizon in (3,12):
            if args.device=="cuda":torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            start=time.perf_counter();q,p=backend.predict(values,horizon)
            if args.device=="cuda":torch.cuda.synchronize()
            measurements.append({"horizon":horizon,"seconds":time.perf_counter()-start,"shape":list(q.shape),
                "peak_allocated_bytes":torch.cuda.max_memory_allocated() if args.device=="cuda" else None,
                "peak_reserved_bytes":torch.cuda.max_memory_reserved() if args.device=="cuda" else None,
                "prediction_sha256":hashlib.sha256(q.tobytes()).hexdigest()})
        handle.remove()
        if args.backend=="chronos2":
            joint_verified=bool(seen) and all(s.get("context")==[275,168] and s.get("unique_group_ids")==1 for s in seen)
        else:
            joint_verified=bool(seen) and all(s.get("arg_0",[])[:2]==[1,275] for s in seen)
        if not joint_verified:raise ValueError("Observed backend calls did not verify one complete 275-variable group")
        report={"status":"PASS","scope":"single training origin, no target labels loaded", "origin_index":168,
            "source":source,"backend":backend.metadata,"model_load_seconds":load_seconds,
            "measurements":measurements,"forward_input_shapes":seen,
            "observed_module":"transformer_stack" if args.backend=="timesfm3" else "model_forward",
            "model_training":False,"joint_group_verified":True,"test_accessed":False,
            "cache_authorized_by_feasibility":True,"sota_claim":False}
    except Exception as exc:
        report={"status":"FAIL","backend":args.backend,"error_type":type(exc).__name__,
            "scope":"single training origin","test_accessed":False,"cache_authorized_by_feasibility":False}
        (args.output/"profile.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        raise
    (args.output/"profile.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
