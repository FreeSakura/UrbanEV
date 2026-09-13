"""Explicit origin-batch and author-core TimeXer contracts for matched comparisons."""
import hashlib,importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .conditional_od import InteractionNet,learning_rate,choose_checkpoint

MODELS=('OD_PRODUCT','OD_SEPARABLE','OD_CONCAT_MLP','TIMEXER_LOCAL_OD','TIMEXER_GLOBAL_O')
SEEDS=(20260915,20260916,20260917)
CHECKPOINTS=(0,5,10,20,40)

def load_author_core(source_root,expected_hashes):
    root=Path(source_root).resolve()
    for rel,expected in expected_hashes.items():
        p=(root/rel).resolve()
        if root not in p.parents or hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:raise ValueError('Author source identity mismatch')
    # Author imports top-level layers/utils. Refuse unexpected namespace collisions.
    local=Path(__file__).resolve().parents[2]/'models/timexer'
    for name,module in list(sys.modules.items()):
        if name in ('layers','utils') or name.startswith(('layers.','utils.')):
            file=getattr(module,'__file__',None)
            if file and not any(parent in Path(file).resolve().parents for parent in (root,local.resolve())):raise RuntimeError('Unexpected author dependency namespace collision')
            del sys.modules[name]
    sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location('_verified_author_timexer',root/'models/TimeXer.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Model

def timexer_inputs(raw,mode):
    if raw.ndim!=3 or raw.shape[-1]!=341:raise ValueError('Expected [origins, zones,341] raw history/context')
    b,n,_=raw.shape
    if mode=='TIMEXER_LOCAL_OD':
        x=torch.stack([raw[:,:,168:336],raw[:,:,:168]],dim=-1).reshape(b*n,168,2)
        marker=raw[:,:,336:].reshape(b*n,5).unsqueeze(1).expand(-1,168,-1)
        return x,marker
    if mode=='TIMEXER_GLOBAL_O':return raw[:,:,:168].transpose(1,2),None
    raise ValueError('Unknown TimeXer information track')

class TimeXerAdapter(nn.Module):
    def __init__(self,mode,seed,author_class,channels=275):
        super().__init__();self.mode=mode;self.channels=channels;torch.manual_seed(seed)
        cfg=SimpleNamespace(task_name='long_term_forecast',features='MS' if mode=='TIMEXER_LOCAL_OD' else 'M',enc_in=2 if mode=='TIMEXER_LOCAL_OD' else channels,seq_len=168,pred_len=12,patch_len=12,d_model=64,n_heads=4,e_layers=2,d_ff=128,dropout=.1,factor=3,activation='gelu',use_norm=True,embed='timeF',freq='h')
        if mode not in ('TIMEXER_LOCAL_OD','TIMEXER_GLOBAL_O'):raise ValueError('Unknown TimeXer track')
        self.model=author_class(cfg)
    def forward(self,raw):
        b,n,_=raw.shape
        if n!=self.channels:raise ValueError('Region count changed')
        x,marker=timexer_inputs(raw,self.mode);q=self.model(x,marker,None,None)
        expected=(b*n,12,1) if self.mode=='TIMEXER_LOCAL_OD' else (b,12,n)
        if tuple(q.shape)!=expected:raise ValueError('Unexpected author output shape')
        return q.reshape(b,n,12).transpose(1,2) if self.mode=='TIMEXER_LOCAL_OD' else q

def effective_batches(origins,seed,epoch):
    order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(origins)
    return [order[start:start+16] for start in range(0,len(order),16)]

def microbatches(indices):
    if len(indices)==0:raise ValueError('Empty effective batch')
    return [(indices[i:i+2],len(indices[i:i+2])/len(indices)) for i in range(0,len(indices),2)]

class Budget:
    def __init__(self):self.ridges=[];self.neural=[];self.closed=False
    def ridge(self,name):
        if self.closed or name not in ('RIDGE_O','RIDGE_OD') or name in self.ridges:raise RuntimeError('Ridge budget violation')
        self.ridges.append(name)
    def start_neural(self,name,seed):
        if self.closed or len(self.ridges)!=2 or name not in MODELS or seed not in SEEDS or (name,seed) in self.neural:raise RuntimeError('Neural budget violation')
        self.neural.append((name,seed))
    def close(self):
        if len(self.neural)!=15 or len(self.ridges)!=2:raise RuntimeError('Required core methods missing')
        self.closed=True
