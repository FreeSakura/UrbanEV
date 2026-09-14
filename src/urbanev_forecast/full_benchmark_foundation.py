"""Pinned local native point and Q0.5 for four independently requested horizons."""
from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path
import os

import numpy as np

from .foundation import LEVELS, file_hash


class FullFoundationBackend:
    def __init__(self,name,model_dir,device='cuda'):
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.name=name;self.device=device;root=Path(model_dir)
        if not root.is_dir() or not (root/'model.safetensors').is_file():
            raise ValueError('Pinned local model directory required')
        if name=='chronos2':
            from chronos import Chronos2Pipeline
            self.pipeline=Chronos2Pipeline.from_pretrained(str(root),device_map=device,torch_dtype=torch.float32,local_files_only=True)
            cls=Chronos2Pipeline;version=importlib.metadata.version('chronos-forecasting')
            if not set(LEVELS).issubset(self.pipeline.quantiles):raise ValueError('Unexpected native quantile grid')
        elif name=='timesfm3':
            from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster
            self.pipeline=TimesFM3Forecaster.from_pretrained(str(root),device=device,per_core_batch_size=1,local_files_only=True)
            cls=TimesFM3Forecaster;version=importlib.metadata.version('timesfm')
            if list(self.pipeline.config.quantiles)!=LEVELS:raise ValueError('Unexpected TimesFM grid')
        else:raise ValueError('Unknown foundation family')
        api=Path(inspect.getfile(cls))
        self.metadata={'name':name,'package_version':version,'api_filename':api.name,'api_sha256':file_hash(api),
                       'weights':{p.name:file_hash(p) for p in sorted(root.glob('*.safetensors'))},
                       'config_sha256':file_hash(root/'config.json'),'torch':str(torch.__version__),
                       'dtype':'float32','device':device,'horizons':[3,6,9,12],'history':168,'channels':275,
                       'native_point':'API native point, no conditional-mean assertion',
                       'q05':'requested native 0.5 quantile; saved separately even if aliased',
                       'local_files_only':True,'cross_origin_learning':False,'variate_chunking':False}

    def predict(self,context,horizon):
        x=np.asarray(context,dtype=np.float32)
        if x.shape!=(168,275) or horizon not in (3,6,9,12) or not np.isfinite(x).all():
            raise ValueError('One complete origin and registered horizon required')
        if self.name=='chronos2':
            quantiles,points=self.pipeline.predict_quantiles(inputs=[x.T.copy()],prediction_length=horizon,
                    quantile_levels=LEVELS,batch_size=275,context_length=168,cross_learning=False)
            if len(quantiles)!=1 or len(points)!=1:raise ValueError('Origin task split')
            q=quantiles[0].cpu().numpy().transpose(1,0,2);point=points[0].cpu().numpy().T
        else:
            output=self.pipeline.predict(x.T.copy(),horizon=horizon,return_quantiles=True,
                    use_symmetric_averaging=False,make_positive=False,sort_quantiles=False,use_znorm=False,padding_mode='none')
            q=output.quantiles.transpose(1,0,2);point=output.forecast.T
        if q.shape!=(horizon,275,9) or point.shape!=(horizon,275):raise ValueError('Incomplete native outputs')
        if not np.isfinite(q).all() or not np.isfinite(point).all():raise ValueError('Nonfinite native outputs')
        return point.astype(np.float32),q[:,:,4].astype(np.float32)
