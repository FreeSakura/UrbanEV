"""Local-only foundation backends with one complete multivariate task per call.

Install upstream dependencies separately and supply pinned local weights. This
module never downloads weights or uses the TimesFM evaluator's variate chunking.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import numpy as np

LEVELS = [.1, .2, .3, .4, .5, .6, .7, .8, .9]


def validate_prediction(quantiles, native, horizon, channels):
    q, p = np.asarray(quantiles), np.asarray(native)
    if q.shape != (horizon, channels, len(LEVELS)) or p.shape != (horizon, channels):
        raise ValueError("Expected complete [H, channels, 9] quantiles and [H, channels] point")
    if not np.isfinite(q).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite foundation prediction")
    if not np.allclose(q[...,4], p, atol=1e-7, rtol=1e-6):
        raise ValueError("Raw point is not the requested native 0.5 quantile")
    return q.astype(np.float32), p.astype(np.float32)


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


class FoundationBackend:
    def __init__(self, name, model_dir, device="cuda"):
        import torch
        self.name, self.device = name, device
        local=Path(model_dir)
        if not local.is_dir() or not (local/"model.safetensors").is_file():
            raise ValueError("Supply a pinned local model directory with model.safetensors")
        torch.set_num_threads(2)
        if name == "chronos2":
            from chronos import Chronos2Pipeline
            self.pipeline=Chronos2Pipeline.from_pretrained(str(local),device_map=device,
                torch_dtype=torch.float32,local_files_only=True)
            if not set(LEVELS).issubset(self.pipeline.quantiles):
                raise ValueError("The frozen nine-level grid must be native; interpolation is not permitted")
            api_file=inspect.getfile(Chronos2Pipeline)
            version=importlib.metadata.version("chronos-forecasting")
            self.model=self.pipeline.model
        elif name == "timesfm3":
            from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster
            self.pipeline=TimesFM3Forecaster.from_pretrained(str(local),device=device,
                per_core_batch_size=1,local_files_only=True)
            if list(self.pipeline.config.quantiles) != LEVELS:
                raise ValueError("Unexpected TimesFM quantile grid")
            api_file=inspect.getfile(TimesFM3Forecaster)
            version=importlib.metadata.version("timesfm")
            self.model=self.pipeline.model
        else:raise ValueError("Unknown foundation backend")
        self.metadata={"backend":name,"package_version":version,"api_source_sha256":file_hash(api_file),
            "weight_sha256":file_hash(local/"model.safetensors"),"config_sha256":file_hash(local/"config.json"),
            "torch_version":str(torch.__version__),"numpy_version":np.__version__,"cuda_version":torch.version.cuda,
            "dtype":"float32","device":device,"quantile_levels":LEVELS,
            "cross_origin_learning":False,"variate_chunking":False,"native_point":"raw Q0.5 before sort or clip",
            "timesfm_class":"TimesFM3Forecaster, not the chunking evaluator" if name=="timesfm3" else None}

    def predict(self, context, horizon):
        x=np.asarray(context,dtype=np.float32)
        if x.ndim!=2 or not np.isfinite(x).all() or x.shape[0]!=168 or x.shape[1]!=275 or horizon not in (3,12):
            raise ValueError("This protocol requires one 168x275 origin and H3 or H12")
        if self.name == "chronos2":
            qs,ps=self.pipeline.predict_quantiles(inputs=[x.T.copy()],prediction_length=horizon,
                quantile_levels=LEVELS,batch_size=275,context_length=168,cross_learning=False)
            if len(qs)!=1 or len(ps)!=1:raise ValueError("Backend split the multivariate task")
            q=qs[0].cpu().numpy().transpose(1,0,2);p=ps[0].cpu().numpy().T
        else:
            out=self.pipeline.predict(x.T.copy(),horizon=horizon,return_quantiles=True,
                use_symmetric_averaging=False,make_positive=False,sort_quantiles=False,
                use_znorm=False,padding_mode="none")
            q=out.quantiles.transpose(1,0,2);p=out.forecast.T
        return validate_prediction(q,p,horizon,275)
