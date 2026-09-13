"""Reference-relative positive harm is a training penalty, not a safety guarantee."""
import numpy as np
import torch
from .conditional_od import InteractionNet as OriginalNet, learning_rate, choose_checkpoint, state_groups, grouped_losses

MODELS=('PRODUCT_MSE','PRODUCT_POSREG','PRODUCT_MIX','PRODUCT_L1','CONCAT_MSE','CONCAT_POSREG')
SEEDS=(20260915,20260916,20260917)
CHECKPOINTS=(0,5,10,20,40)
KAPPA=.5

class InteractionNet(OriginalNet):
    def __init__(self,scheme,seed):
        if scheme not in MODELS:raise ValueError('Unknown training scheme')
        super().__init__('OD_PRODUCT' if scheme.startswith('PRODUCT_') else 'OD_CONCAT_MLP',seed)

def fit_scale(e32):
    if e32.dtype!=np.float32 or not np.isfinite(e32).all():raise ValueError('Expected finite float32 FIT residual')
    rms=float(np.sqrt(np.mean(e32.astype(np.float64)**2)))
    return {'rms_float64':rms,'scale_float64':max(rms,1e-6),'floor_applied':rms<1e-6,'kappa':KAPPA,'tau_float32':float(np.float32(KAPPA*max(rms,1e-6))),'source':'FIT in-sample ridge residual cast to float32 then RMS float64'}

def loss_parts(e,f,tau,scheme):
    if scheme not in MODELS:raise ValueError('Unknown training scheme')
    err=e-f;delta=err.abs()-e.abs()
    parts={'mse':err.square().mean(),'harm':torch.relu(delta).mean(),'benefit':torch.relu(-delta).mean(),'correction_l1':f.abs().mean()}
    penalty={'MSE':lambda:parts['mse']*0,'POSREG':lambda:parts['harm'],'MIX':lambda:delta.mean(),'L1':lambda:parts['correction_l1']}[scheme.split('_',1)[1]]()
    parts['objective']=parts['mse']+tau*penalty
    return parts

def relative_stability(corrections):
    a=np.asarray(corrections,dtype=np.float64)
    if a.shape[0]!=3 or not np.isfinite(a).all():raise ValueError('Exactly three finite seed arrays required')
    total=float(np.mean(a*a));variance=float(sum(np.mean((a[i]-a[j])**2) for i in range(3) for j in range(i+1,3))/9)
    return {'T':total,'V':variance,'eta':variance/total if total>0 else None,'all_zero':total==0,'seed_count':3,'interpretation':'relative disagreement; no ensemble score or accuracy guarantee'}

class Budget:
    def __init__(self):self.ridge=0;self.neural=[];self.closed=False
    def start_ridge(self):
        if self.closed or self.ridge:raise RuntimeError('Only one base ridge fit')
        self.ridge=1
    def start_neural(self,name,seed):
        if self.closed or name not in MODELS or seed not in SEEDS or (name,seed) in self.neural:raise RuntimeError('Neural budget violation')
        self.neural.append((name,seed))
    def close(self):
        if self.ridge!=1 or len(self.neural)!=18:raise RuntimeError('Required fits missing')
        self.closed=True
