"""Forward residual supervision with explicit reference-prediction context."""
import numpy as np
import torch
from torch import nn
from . import short_state as base
from .conditional_od import InteractionNet as OriginalNet, learning_rate, choose_checkpoint, state_groups, grouped_losses
from .reference_harm import relative_stability

MODELS=('P_IS_X','P_FWD_X','P_IS_BC','P_FWD_BC','C_IS_BC','C_FWD_BC')
SEEDS=(20260915,20260916,20260917)
CHECKPOINTS=(0,5,10,20,40)
BLOCKS=(('B1',457,468,455,480,661),('B2',649,660,647,672,853),('B3',841,852,839,864,1045))

def meta_origins():return np.concatenate([np.arange(b[4],b[5]) for b in BLOCKS])
def parameter_count(scheme):return 9396 if scheme.endswith('_BC') else 9108

class InteractionNet(OriginalNet):
    def __init__(self,scheme,seed):
        if scheme not in MODELS:raise ValueError('Unknown scheme')
        super().__init__('OD_PRODUCT' if scheme.startswith('P_') else 'OD_CONCAT_MLP',seed)
        self.scheme=scheme
        self.context=nn.Linear(12,24,bias=False) if scheme.endswith('_BC') else None
        if self.context is not None:nn.init.zeros_(self.context.weight)
    def forward(self,x):
        if x.shape[1]!=353:raise ValueError('Expected 341 history + 12 base context columns')
        z=x[:,:341];extra=self.context(x[:,341:]) if self.context is not None else 0.
        if self.kind=='OD_CONCAT_MLP':return self.head(torch.tanh(self.second(torch.tanh(self.first(z)+extra))))
        u=torch.tanh(self.first(torch.cat([z[:,:168],z[:,336:]],dim=1))+extra)
        v=torch.tanh(self.second(z[:,168:336]))
        return self.head(torch.cat([u,v,u*v],dim=1))

def fit_model(y,d,cap,origins):
    oo=np.asarray(origins,dtype=int)
    if len(y)!=int(oo[-1])+12 or len(d)!=int(oo[-1])-1:raise ValueError('Only exact allowed local prefixes may reach fit_model')
    x=base.features(y,d,cap,oo,'LONG_OD');stats=base.fit_transform(x);z=base.transform(x,stats)
    target=base.targets(y,oo)-y[oo-1].reshape(-1,1)
    beta=base.fit_ridge(z,target)
    return {'stats':stats,'beta':beta,'train_origins':oo.copy(),'max_training_label':int(oo[-1]+11)}

def predict_model(model,y,d,cap,origins):
    x=base.features(y,d,cap,origins,'LONG_OD');z=base.transform(x,model['stats'])
    return base.ridge_predict(z,y[np.asarray(origins)-1].reshape(-1),model['beta'])

def fit_scale(residual_is):
    if residual_is.dtype!=np.float64 or not np.isfinite(residual_is).all():raise ValueError('Scale requires finite raw float64 IS residuals')
    rms=float(np.sqrt(np.mean(residual_is**2)))
    return {'rms_float64':rms,'scale_float64':max(rms,1e-6),'floor_applied':rms<1e-6,'source':'BF in-sample residual on common 543 meta origins, before float32 cast'}

def augment(z,qb,anchor,scale):
    if scale<=0 or not np.isfinite(scale):raise ValueError('Invalid frozen scale')
    return np.concatenate([z, (qb-anchor[:,None])/scale],axis=1).astype(np.float32)

class Budget:
    def __init__(self):self.ridge=[];self.neural=[];self.closed=False
    def start_ridge(self,name):
        if self.closed or name not in ('B1','B2','B3','BF') or name in self.ridge or name!=('B1','B2','B3','BF')[len(self.ridge)]:raise RuntimeError('Base fit order/budget violation')
        self.ridge.append(name)
    def start_neural(self,name,seed):
        if self.closed or len(self.ridge)!=4 or name not in MODELS or seed not in SEEDS or (name,seed) in self.neural:raise RuntimeError('Neural budget violation')
        self.neural.append((name,seed))
    def close(self):
        if len(self.ridge)!=4 or len(self.neural)!=18:raise RuntimeError('Required fits missing')
        self.closed=True
