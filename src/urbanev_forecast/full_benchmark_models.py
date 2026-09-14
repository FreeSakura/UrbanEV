"""Explicit H-specific author adapters for the full matched UrbanEV benchmark."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

_SOURCE_PATHS = set()


def load_verified_author(root, lock):
    """Verify the complete registered dependency closure, then isolate its namespace."""
    root = Path(root).resolve()
    for rel, expected in lock['files'].items():
        p = (root / rel).resolve()
        if root not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Author source identity mismatch: {rel}')
    for name in list(sys.modules):
        if name in ('layers', 'utils') or name.startswith(('layers.', 'utils.')):
            del sys.modules[name]
    for old in _SOURCE_PATHS:
        while old in sys.path:
            sys.path.remove(old)
    _SOURCE_PATHS.add(str(root))
    sys.path.insert(0, str(root))
    entry = root / lock['entry']
    payload = entry.read_bytes()
    if hashlib.sha256(payload).hexdigest() != lock['files'][lock['entry']]:
        raise ValueError('Author entry changed before import')
    module = ModuleType('_author_' + lock['revision'])
    module.__file__ = str(entry)
    exec(compile(payload, str(entry), 'exec'), module.__dict__)
    return module.Model


class AuthorForecaster(nn.Module):
    """All adapters return [origin,H,zone]; only declared past inputs are passed."""
    def __init__(self, kind, horizon, author_class, config, channels=275):
        super().__init__()
        self.kind, self.horizon, self.channels = kind, horizon, channels
        if horizon not in (3,6,9,12):
            raise ValueError('Expected registered H3/H6/H9/H12')
        cfg = dict(config)
        cfg.update(seq_len=168, pred_len=horizon, enc_in=2 if kind=='TIMEXER_LOCAL_OD' else channels)
        self.model = author_class(SimpleNamespace(**cfg))

    def forward(self, raw):
        if raw.ndim != 3 or raw.shape[1:] != (self.channels,341):
            raise ValueError('Expected [origins,zones,341] O/D/context features')
        batch = raw.shape[0]
        if self.kind == 'TIMEXER_LOCAL_OD':
            x = torch.stack((raw[:,:,168:336],raw[:,:,:168]),dim=-1).reshape(batch*self.channels,168,2)
            marker=raw[:,:,336:].reshape(batch*self.channels,5)[:,None,:].expand(-1,168,-1)
            output=self.model(x,marker,None,None)
            if output.shape != (batch*self.channels,self.horizon,1):
                raise ValueError('Unexpected local TimeXer output')
            return output.reshape(batch,self.channels,self.horizon).transpose(1,2)
        x=raw[:,:,:168].transpose(1,2)
        if self.kind in ('DLINEAR','PATCHTST'):
            output=self.model(x)
        elif self.kind in ('ITRANSFORMER','TIMEXER_GLOBAL_O'):
            output=self.model(x,None,None,None)
        else:
            raise ValueError('Unknown author adapter')
        if output.shape != (batch,self.horizon,self.channels):
            raise ValueError('Unexpected author output')
        return output


class ResidualForecaster(nn.Module):
    """Previous project residual families adapted only to H-specific output size."""
    def __init__(self, kind, horizon, seed):
        super().__init__()
        if kind not in ('OD_PRODUCT','OD_SEPARABLE','OD_CONCAT_MLP') or horizon not in (3,6,9,12):
            raise ValueError('Unknown residual family or horizon')
        self.kind, self.horizon = kind, horizon
        concat=kind=='OD_CONCAT_MLP'
        self.first=nn.Linear(341 if concat else 173,24)
        self.second=nn.Linear(24 if concat else 168,24)
        self.head=nn.Linear(24 if concat else 72,horizon)
        for layer, subseed in ((self.first,seed+11),(self.second,seed+13)):
            torch.manual_seed(subseed)
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, standardized):
        n,c,k=standardized.shape
        x=standardized.reshape(n*c,k)
        if self.kind=='OD_CONCAT_MLP':
            q=self.head(torch.tanh(self.second(torch.tanh(self.first(x)))))
        else:
            u=torch.tanh(self.first(torch.cat((x[:,:168],x[:,336:]),dim=1)))
            v=torch.tanh(self.second(x[:,168:336]))
            third=(u.square()+v.square())/2 if self.kind=='OD_SEPARABLE' else u*v
            q=self.head(torch.cat((u,v,third),dim=1))
        return q.reshape(n,c,self.horizon).transpose(1,2)
