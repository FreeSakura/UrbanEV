"""Prefix-bounded six-fold inputs and streaming ridge statistics.

No full flattened training design is retained in memory. Old 1560-row experiment
readers remain unchanged; this module requires its own full-benchmark identity.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_contract import origins, fold_plan
from .bounded_predictor_csv import parse_prefix
from .comparability_bridge import zone_capacities
from .short_state import features, transform


def read_exact_prefix(root, name, rows):
    if name not in ('occupancy.csv','duration.csv') or type(rows) is not int or not 1<=rows<=4344:
        raise ValueError('Unregistered full-benchmark prefix')
    root=Path(root).resolve();p=(root/name).resolve()
    if p.parent!=root:
        raise ValueError('Data path escaped root')
    with p.open('rb',buffering=0) as f:
        lines=[f.readline() for _ in range(rows+1)]
    if any(not line for line in lines):
        raise ValueError('Incomplete prefix')
    return b''.join(lines)


class FoldReader:
    """Only train/validation constructors exist before the explicit predict stage."""
    def __init__(self, root, identity, fold, horizon, ledger):
        self.root=Path(root);self.identity=identity;self.fold=fold;self.horizon=horizon
        self.ledger=ledger;self.columns=identity['columns'];self.cap=None
        self.training_closed=False

    def close_training(self):
        self.training_closed=True

    def _capacity(self):
        if self.cap is None:
            p=(self.root/'inf.csv').resolve()
            if p.parent!=self.root.resolve() or hashlib.sha256(p.read_bytes()).hexdigest()!=self.identity['static_sha256']:
                raise ValueError('Static identity mismatch')
            frame=pd.read_csv(p,usecols=['station_id','TAZID','charge_count'],dtype={'TAZID':str})
            if frame['station_id'].duplicated().any() or not np.isfinite(frame['charge_count']).all() or (frame['charge_count']<=0).any():
                raise ValueError('Invalid station capacity identity')
            self.cap=zone_capacities(frame,self.columns)
        return self.cap

    def read(self, stage):
        bounds=fold_plan(self.fold,'MATCHED_TERMINAL_168')
        if stage=='train': stop=bounds.train_stop
        elif stage=='validation': stop=bounds.validation_stop
        elif stage=='predict':
            if not self.training_closed:raise RuntimeError('Freeze training before prediction')
            stop=bounds.test_stop-self.horizon
        else:raise ValueError('Reader supports train, validation and predict only; scoring is separate')
        if self.training_closed and stage in ('train','validation'):
            raise RuntimeError('Training/selection access closed')
        cap=self._capacity();arrays={}
        for name,rows in [('occupancy.csv',stop),('duration.csv',stop-1 if stage=='predict' else stop)]:
            payload=read_exact_prefix(self.root,name,rows)
            expected=self.identity['prefixes'][name][str(rows)]
            raw,receipt=parse_prefix(payload,rows,self.columns,expected,nonnegative=True)
            if name=='occupancy.csv':
                a=(raw.astype(np.float32)/cap.astype(np.float32)[None,:]).astype(np.float64)
            else:a=raw/cap[None,:]
            if np.any(a<0) or np.any(a>1):raise ValueError('Invalid declared normalized input; no silent repair')
            arrays[name]=a
            self.ledger.append({'fold':self.fold,'horizon':self.horizon,'stage':stage,'file':name,**receipt})
        return arrays['occupancy.csv'],arrays['duration.csv'],cap


def raw_chunks(y,d,cap,oo,size=16):
    for start in range(0,len(oo),size):
        o=oo[start:start+size]
        yield o,features(y,d,cap,o,'LONG_OD')


def streaming_transform(y,d,cap,oo):
    """Parallel-variance merge avoids E[X²]-E[X]² cancellation."""
    count=0;mean=np.zeros(341);m2=np.zeros(341)
    for _,x in raw_chunks(y,d,cap,oo):
        n=len(x);mu=x.mean(axis=0);block_m2=np.square(x-mu).sum(axis=0)
        delta=mu-mean;total=count+n
        m2+=block_m2+delta*delta*count*n/total
        mean+=delta*n/total;count=total
    if not count:raise ValueError('Empty training design')
    std=np.sqrt(m2/count);inactive=std<=1e-10*np.maximum(1.,abs(mean))
    return dict(mean=mean,std=std,inactive=inactive)


def streaming_statistics(y,d,cap,oo,horizon,stats):
    gram=np.zeros((342,342));rhs=np.zeros((342,horizon));count=0
    for o,raw in raw_chunks(y,d,cap,oo):
        z=transform(raw,stats);x=np.column_stack((np.ones(len(z)),z))
        target=y[o[:,None]+np.arange(horizon)].transpose(0,2,1).reshape(-1,horizon)
        anchor=y[o-1].reshape(-1)
        residual=target-anchor[:,None]
        gram+=x.T@x;rhs+=x.T@residual;count+=len(x)
    return gram/count,rhs/count


def ridge_solution(gram,rhs,mask,regularization):
    if regularization<=0 or not np.isfinite(regularization):raise ValueError('Positive ridge lambda required')
    ix=np.r_[0,np.asarray(mask)+1];a=gram[np.ix_(ix,ix)].copy();b=rhs[ix]
    a[1:,1:]+=np.eye(len(mask))*regularization
    beta=np.linalg.solve(a,b)
    denominator=np.linalg.norm(a,np.inf)*np.linalg.norm(beta,np.inf)+np.linalg.norm(b,np.inf)
    error=np.linalg.norm(a@beta-b,np.inf)/max(denominator,np.finfo(float).tiny)
    if not np.isfinite(beta).all() or not np.isfinite(error) or error>1e-10:
        raise ValueError('Ridge normal equations failed')
    return beta,float(error)


def predict_ridge(y,d,cap,oo,horizon,stats,mask,beta):
    predictions=[]
    for o,raw in raw_chunks(y,d,cap,oo):
        z=transform(raw,stats)
        q=y[o-1].reshape(-1,1)+beta[0]+z[:,mask]@beta[1:]
        predictions.append(q.reshape(len(o),len(cap),horizon).transpose(0,2,1))
    return np.concatenate(predictions)


def target_windows(y,oo,horizon):
    if oo[0]<0 or oo[-1]+horizon>len(y):raise ValueError('Target outside declared prefix')
    return y[oo[:,None]+np.arange(horizon)]


class TensorWindows:
    """Construct just one origin batch on device; no multi-GB design cache."""
    def __init__(self,y,d,cap,stats,device='cuda'):
        import torch
        from .short_state import phase
        self.torch=torch;self.device=device
        self.y=torch.as_tensor(y,dtype=torch.float64,device=device)
        self.d=torch.as_tensor(d,dtype=torch.float64,device=device)
        self.cap=torch.as_tensor(np.log1p(cap),dtype=torch.float64,device=device)
        self.mean=torch.as_tensor(stats['mean'],dtype=torch.float64,device=device)
        self.std=torch.as_tensor(np.where(stats['inactive'],1.,stats['std']),dtype=torch.float64,device=device)
        self.inactive=torch.as_tensor(stats['inactive'],device=device)
        o=np.arange(len(y)+1);s24,c24=phase(o,24);s168,c168=phase(o,168)
        self.calendar=torch.as_tensor(np.stack((s24,c24,s168,c168),axis=1),dtype=torch.float64,device=device)
        self.lags=torch.arange(168,device=device)

    def inputs(self,oo,standardized=False):
        t=self.torch;o=t.as_tensor(oo,dtype=t.long,device=self.device)
        if o.ndim!=1 or len(o)==0 or int(o.min())<169 or int(o.max())>len(self.y) or int(o.max())-1>len(self.d):
            raise ValueError('History outside device prefix')
        yp=self.y[o[:,None]-168+self.lags].transpose(1,2)
        dp=self.d[o[:,None]-169+self.lags].transpose(1,2)
        calendar=self.calendar[o][:,None,:].expand(-1,len(self.cap),-1)
        capacity=self.cap[None,:,None].expand(len(o),-1,-1)
        raw=t.cat((yp,dp,calendar,capacity),dim=2)
        if not standardized:return raw.float()
        z=(raw-self.mean)/self.std
        z[:,:,self.inactive]=0
        return z.float()
