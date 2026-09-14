import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def script(name):
    p=Path(__file__).resolve().parents[1]/'scripts/research'/name
    spec=importlib.util.spec_from_file_location(name.replace('.','_'),p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_claim_is_consumed_once_and_config_changes_are_rejected(tmp_path):
    from urbanev_forecast.full_benchmark_guard import consume_or_validate_claim
    cfg=tmp_path/'config.json';cfg.write_text('{}')
    private=tmp_path/'run'
    with pytest.raises(RuntimeError):consume_or_validate_claim(private,cfg,'score')
    first=consume_or_validate_claim(private,cfg,'train')
    assert consume_or_validate_claim(private,cfg,'train')==first
    cfg.write_text('{"changed":true}')
    with pytest.raises(ValueError):consume_or_validate_claim(private,cfg,'predict')


def test_partial_prediction_set_cannot_be_scored(tmp_path):
    mod=script('score_full_benchmark.py')
    c={'ridge_models':[],'neural_models':[],'foundation_models':{},'folds':list(range(1,7)),
       'horizons':[3,6,9,12],'training':{'seeds':[20260921,20260922]}}
    with pytest.raises(ValueError,match='Complete unique'):
        mod.verify_prediction_set(c,[],tmp_path)
    with pytest.raises(FileNotFoundError):mod.verify_training_budget(c,tmp_path)


def test_paired_bootstrap_is_reproducible_and_zero_for_identical_forecasts():
    mod=script('score_full_benchmark.py')
    from urbanev_forecast.benchmark_contract import origins
    c={'folds':list(range(1,7)),'horizons':[3,6,9,12],
       'bootstrap':{'replicates':30,'block_length':24,'seed':20260914,'references':['RIDGE_OD_CTX','CHRONOS2_POINT']}}
    losses={}
    for f in c['folds']:
        for h in c['horizons']:
            n=len(origins(f,'test',168,h,contract='MATCHED_TERMINAL_168'))
            e=np.linspace(.01,.2,n)
            a=np.column_stack((e**2*275,e*275,np.full(n,275)))
            for model in ('RIDGE_OD_CTX','CHRONOS2_POINT','CANDIDATE'):
                losses[model,'fixed',f,h]=a.copy()
    first=mod.uncertainty(c,losses)
    assert first==mod.uncertainty(c,losses)
    assert all(r['candidate_minus_reference']==r['percentile_low']==r['percentile_high']==0 for r in first)


def test_foundation_keeps_native_point_distinct_from_q05():
    torch=pytest.importorskip('torch')
    from urbanev_forecast.full_benchmark_foundation import FullFoundationBackend
    class Fake:
        def predict_quantiles(self,**kwargs):
            assert len(kwargs['inputs'])==1 and kwargs['inputs'][0].shape==(275,168)
            assert kwargs['cross_learning'] is False
            h=kwargs['prediction_length']
            return [torch.zeros(275,h,9)],[torch.ones(275,h)]
    model=FullFoundationBackend.__new__(FullFoundationBackend);model.name='chronos2';model.pipeline=Fake()
    for h in (3,6,9,12):
        point,q05=model.predict(np.zeros((168,275)),h)
        assert point.shape==q05.shape==(h,275) and np.all(point==1) and np.all(q05==0)


def test_gpu_epoch_resume_matches_uninterrupted_optimizer_and_dropout(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('Local CUDA workflow integration; CPU CI has other contract checks')
    mod=script('run_full_benchmark.py');mod.setup()
    y=np.broadcast_to(np.linspace(.1,.5,648)[:,None],(648,275)).copy();d=y*.7;cap=np.ones(275)*8
    class Reader:
        def __init__(self,*args):pass
        def read(self,stage):
            assert stage in ('train','validation')
            n=576 if stage=='train' else 648
            return y[:n],d[:n],cap
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.dropout=torch.nn.Dropout(.3);self.linear=torch.nn.Linear(1,12)
        def forward(self,x):return self.linear(self.dropout(x[:,:,167,None])).transpose(1,2)
    def make(*args):torch.manual_seed(args[2]);return Toy().cuda()
    monkeypatch.setattr(mod,'FoldReader',Reader);monkeypatch.setattr(mod,'make_model',make)
    c={'training':{'epochs':2,'checkpoints':[0,1,2],'microbatch':8,'batch_origins':16,'weight_decay':1e-4,
                   'gradient_clip':1.,'minimum_lr_fraction':.1},'data_identity':{}}
    config=tmp_path/'synthetic_config.json';config.write_text(json.dumps(c))
    spec={'name':'SYNTHETIC_TOY','kind':'author','objective':'MSE'}
    def run(directory):
        directory.mkdir(exist_ok=True)
        stats=directory/'stats.npz';np.savez(stats,mean=np.zeros(341),std=np.ones(341),inactive=np.zeros(341,dtype=bool))
        a=SimpleNamespace(private=directory,config=config,data=tmp_path/'never_read')
        return mod.train_one(a,c,spec,1,12,77,.001,{'stats':mod.record(stats,directory)}, {}, {})
    direct=run(tmp_path/'direct')
    original_dump=mod.dump
    def interrupt(path,value):
        original_dump(path,value)
        if path.name.endswith('_resume.json') and value['epoch']==1:raise InterruptedError('controlled epoch boundary')
    monkeypatch.setattr(mod,'dump',interrupt)
    with pytest.raises(InterruptedError):run(tmp_path/'resumed')
    monkeypatch.setattr(mod,'dump',original_dump)
    resumed=run(tmp_path/'resumed')
    assert direct['optimizer_steps']==resumed['optimizer_steps']==50
    assert direct['validations']==resumed['validations']
    direct_state=torch.load(tmp_path/'direct'/direct['best_weights']['file'],weights_only=True)
    resume_state=torch.load(tmp_path/'resumed'/resumed['best_weights']['file'],weights_only=True)
    for name in direct_state:torch.testing.assert_close(direct_state[name],resume_state[name],rtol=0,atol=0)
