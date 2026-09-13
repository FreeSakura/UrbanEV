"""Equal-parameter residual models for observable O-D interactions, not causal states."""
import math
import numpy as np
import torch
from torch import nn
MODELS=('OD_PRODUCT','OD_SEPARABLE','OD_CONCAT_MLP','OO_PRODUCT')
SEEDS=(20260913,20260914)
CHECKPOINTS=(0,5,10,20,40)

class InteractionNet(nn.Module):
    def __init__(self,kind,seed):
        super().__init__()
        if kind not in MODELS:raise ValueError('Unknown model')
        self.kind=kind
        self.first=nn.Linear(341 if kind=='OD_CONCAT_MLP' else 173,24)
        self.second=nn.Linear(24 if kind=='OD_CONCAT_MLP' else 168,24)
        self.head=nn.Linear(24 if kind=='OD_CONCAT_MLP' else 72,12)
        for layer,subseed in [(self.first,seed+11),(self.second,seed+13)]:
            torch.manual_seed(subseed);nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def encoded(self,x):
        u=torch.tanh(self.first(torch.cat([x[:,:168],x[:,336:]],dim=1)))
        v=torch.tanh(self.second(x[:,:168] if self.kind=='OO_PRODUCT' else x[:,168:336]))
        return u,v
    def forward(self,x):
        if self.kind=='OD_CONCAT_MLP':return self.head(torch.tanh(self.second(torch.tanh(self.first(x)))))
        u,v=self.encoded(x);third=(u.square()+v.square())/2 if self.kind=='OD_SEPARABLE' else u*v
        return self.head(torch.cat([u,v,third],dim=1))
    def interaction_component(self,x):
        if self.kind!='OD_PRODUCT':raise ValueError('Only the product model has the registered interaction diagnostic')
        u,v=self.encoded(x)
        return (u*v)@self.head.weight[:,48:].T

def learning_rate(epoch):
    if not 1<=epoch<=40:raise ValueError('Epoch out of budget')
    return 1e-4+(1e-3-1e-4)/2*(1+math.cos(math.pi*(epoch-1)/39))

def choose_checkpoint(rows):return min(rows,key=lambda r:(r['selection_rmse'],r['epoch']))

def state_groups(p,change24):return (p>=.5).astype(np.int64)*2+(change24>=0).astype(np.int64)

def grouped_losses(y,reference,candidate,groups):
    eb=y-reference;et=y-candidate;delta=candidate-reference;g1=abs(eb)-abs(et);g2=eb**2-et**2
    mask_ids=np.broadcast_to(groups[:,None,:],y.shape);rows=[]
    for index,label in enumerate(('low_falling','low_nonfalling','high_falling','high_nonfalling')):
        mask=mask_ids==index;r={'group':label,'sample_count':int(mask.sum())}
        for name,g in [('mae',g1[mask]),('mse',g2[mask])]:r[name]={'improved_count':int((g>0).sum()),'worsened_count':int((g<0).sum()),'equal_count':int((g==0).sum()),'positive_gain_sum':float(g[g>0].sum()),'negative_gain_sum':float(g[g<0].sum()),'net_sum':float(g.sum()),'mean_gain':float(g.mean()) if g.size else None}
        rows.append(r)
    err=max(float(np.max(abs(g2-(2*eb*delta-delta**2)))),abs(sum(r['mae']['net_sum'] for r in rows)-float(g1.sum())),abs(sum(r['mse']['net_sum'] for r in rows)-float(g2.sum())))
    if err>1e-10:raise ValueError('Grouped loss identity failed')
    return {'sample_count':y.size,'group_rule':'prediction-time p<0.5 / >=0.5 crossed with y[o-1]-y[o-25]<0 / >=0; diagnostic only','groups':rows,'identity_error':err,'mae_mean_gain':float(g1.mean()),'mse_mean_gain':float(g2.mean())},g1,g2

class Budget:
    def __init__(self):self.ridge=0;self.neural=[];self.closed=False
    def start_ridge(self):
        if self.closed or self.ridge:raise RuntimeError('Only one base ridge fit')
        self.ridge=1
    def start_neural(self,name,seed):
        if self.closed or name not in MODELS or seed not in SEEDS or (name,seed) in self.neural:raise RuntimeError('Neural budget violation')
        self.neural.append((name,seed))
    def close(self):
        if self.ridge!=1 or len(self.neural)!=8:raise RuntimeError('Required fits missing')
        self.closed=True
