"""Synthetic-only comparison preparation. Accepts source roots, never real data.

Requires exact upstream source files listed in the public contract. Does not
download code, load model checkpoints, or perform optimizer steps.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from urbanev_forecast import benchmark_contract as b
from urbanev_forecast.author_dlinear import DLinearAdapter, load_author_model

ROOT = Path(__file__).resolve().parents[2]


def module_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urbanev-source', required=True, type=Path)
    parser.add_argument('--dlinear-source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config_bytes = (ROOT/'configs/research/EVALUATION_CONTRACT_AUTHOR_BASELINE_PREP_V1.json').read_bytes()
    config = json.loads(config_bytes)
    assert config['real_execution_authorized'] is False
    assert config['real_training_spec'] is None
    assert not args.output.exists(), 'Use a new output directory; preserve old verification receipts'
    for name, directory in [('urbanev', args.urbanev_source), ('dlinear', args.dlinear_source)]:
        for rel, expected in config['source_lock'][name]['files'].items():
            assert hashlib.sha256((directory/rel).read_bytes()).hexdigest() == expected, rel

    # Exact author imports in this standalone process; no previous project utils.
    transformer = args.urbanev_source/'code-transformer'
    if any(key == 'utils' or key.startswith('utils.') for key in sys.modules):
        raise RuntimeError('Run in a fresh process to avoid upstream utility import collisions')
    sys.path.insert(0, str(transformer))
    loader = module_at(transformer/'data_provider/data_loader.py', '_upstream_loader').Dataset_Custom
    metric = module_at(transformer/'utils/metrics.py', '_upstream_metric').metric
    # Exact AST function bodies, avoiding unrelated model imports in classical utils.
    tree = ast.parse((args.urbanev_source/'code/utils.py').read_text(encoding='utf-8'))
    extracted = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                and n.name in ('split_cv', 'create_rnn_data')], type_ignores=[])
    classical = {'np':np, 'StandardScaler':StandardScaler}
    exec(compile(extracted, 'pinned_classical_functions', 'exec'), classical)
    dates = pd.date_range('2022-09-01', periods=4344, freq='h')
    timeline = np.arange(4344, dtype=np.float64)
    # OT starts first; upstream moves it last. Values encode time and channel.
    frame = pd.DataFrame({'date':dates, 'OT':timeline+20000, 'one':timeline, 'two':timeline+10000})
    ordered = frame[['one','two','OT']].to_numpy()
    transformer_cases = classical_cases = 0
    probes = 0
    with tempfile.TemporaryDirectory(prefix='urbanev_synthetic_') as tmp:
        frame.to_csv(Path(tmp)/'synthetic.csv', index=False)
        for f in range(1,7):
            parts = classical['split_cv'](SimpleNamespace(fold=f,total_fold=6,pred_type='region',feat='occ'), dates, ordered)
            for L in (12,168):
                for H in b.HORIZONS:
                    for index, split in enumerate(('train','validation','test')):
                        ds = loader(SimpleNamespace(fold=f), tmp, flag='val' if split=='validation' else split,
                                    size=[L,L,H], features='M', data_path='synthetic.csv', target='OT', timeenc=1)
                        o = b.origins(f,split,L,H)
                        assert len(ds) == len(o)
                        for k in sorted({0,len(o)//2,len(o)-1}):
                            x, y, _, _ = ds[k]
                            np.testing.assert_array_equal(x, ordered[o[k]-L:o[k]])
                            np.testing.assert_array_equal(y[-H:], ordered[o[k]:o[k]+H])
                            probes += 1
                        transformer_cases += 1
                        cx, cy = classical['create_rnn_data'](parts[index], L, H)
                        co = b.origins(f,split,L,H,contract='UPSTREAM_CLASSICAL')
                        assert len(cx) == len(cy) == len(co)
                        if len(co):
                            np.testing.assert_array_equal(cy, ordered[co+H-1])
                            np.testing.assert_array_equal(cx[0], ordered[co[0]-L:co[0]])
                            np.testing.assert_array_equal(cx[-1], ordered[co[-1]-L:co[-1]])
                        classical_cases += 1

    rng = np.random.default_rng(20260914)
    metric_errors=[]
    for H in b.HORIZONS:
        p=rng.normal(.1,.2,(7,H,3)); y=rng.uniform(.0,.5,(7,H,3))
        y[0,-1,:]=[0.,.01,.02]; y[1,-1,0]=-.01
        with contextlib.redirect_stdout(io.StringIO()):
            original=metric(p,y,SimpleNamespace())
        current=b.score_arrays(p,y,scope='terminal',postprocess='raw')
        diagnostic=b.upstream_percentage_mirror(p[:,-1,:],y[:,-1,:])
        reconstructed=[current['mse'],current['rmse'],diagnostic['mape'],diagnostic['rae'],current['mae']]
        error=float(np.max(np.abs(np.asarray(original)-reconstructed)))
        assert error < 1e-12
        metric_errors.append(error)

    torch.set_num_threads(2)
    torch.manual_seed(20260914)
    author=load_author_model(args.dlinear_source)
    cases=[]
    for L in (12,168):
        for H in b.HORIZONS:
            for C in (1,275):
                adapter=DLinearAdapter(args.dlinear_source,history=L,horizon=H,channels=C).double()
                direct=author(SimpleNamespace(seq_len=L,pred_len=H,enc_in=C,individual=False)).double()
                direct.load_state_dict(adapter.model.state_dict())
                x=torch.randn(2,L,C,dtype=torch.float64)
                actual=adapter(x); expected=direct(x)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                actual.square().mean().backward(); expected.square().mean().backward()
                for pa,pd_ in zip(adapter.parameters(),direct.parameters()):
                    assert torch.isfinite(pa.grad).all()
                    torch.testing.assert_close(pa.grad,pd_.grad,rtol=0,atol=0)
                permutation=torch.arange(C-1,-1,-1)
                with torch.no_grad():
                    torch.testing.assert_close(adapter(x[:,:,permutation]),actual[:,:,permutation],rtol=1e-12,atol=1e-12)
                    changed=x.clone(); changed[:,:,0]+=5
                    if C>1:
                        torch.testing.assert_close(adapter(changed)[:,:,1:],actual[:,:,1:],rtol=0,atol=0)
                    # Changing future labels leaves the explicitly past-only input identical.
                    full=torch.cat((x,torch.zeros(2,H,C,dtype=x.dtype)),dim=1)
                    changed_future=full.clone(); changed_future[:,L:,:]=999
                    torch.testing.assert_close(adapter(full[:,:L]),adapter(changed_future[:,:L]),rtol=0,atol=0)
                for kind in ('constant','step','impulse'):
                    v=np.ones(L) if kind=='constant' else np.zeros(L)
                    if kind=='step': v[L//2:]=1
                    if kind=='impulse': v[0]=1
                    trend=np.convolve(np.pad(v,(12,12),mode='edge'),np.ones(25)/25,mode='valid')
                    t=torch.tensor(v,dtype=torch.float64)[None,:,None]
                    seasonal, moving=adapter.model.decompsition(t)
                    np.testing.assert_allclose(moving.detach().numpy().ravel(),trend,rtol=0,atol=1e-14)
                    torch.testing.assert_close(seasonal+moving,t,rtol=0,atol=1e-14)
                count=sum(p.numel() for p in adapter.parameters())
                assert count==2*H*(L+1)
                cases.append(dict(history=L,horizon=H,channels=C,parameters=count,output_max_error=0.,gradient_max_error=0.))

    rows=[]
    for identity in b.CONTRACTS:
        histories=(168,) if identity=='MATCHED_TERMINAL_168' else (12,168)
        for L in histories:
            for f in range(1,7):
                for H in b.HORIZONS:
                    for split in ('train','validation','test'):
                        rows.append(b.cell_metadata(f,H,L,split,identity))
    args.output.mkdir(parents=True)
    with (args.output/'six_fold_index_differences.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys(),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    receipt=dict(protocol_id=config['protocol_id'],status='CONTRACT_AND_AUTHOR_ADAPTER_READY_FORMAL_SCOPE_PENDING',
                 config_sha256=hashlib.sha256(config_bytes).hexdigest(),real_data_reads=0,
                 historical_prediction_or_weight_reads=0,model_fits=0,optimizer_steps=0,foundation_inference=0,
                 transformer_loader_cases=transformer_cases,transformer_getitem_probes=probes,
                 classical_extracted_function_cases=classical_cases,metric_cases=4,
                 metric_max_error=max(metric_errors),dlinear_cases=cases,plan_rows=len(rows),
                 official_six_fold_completed=False,real_calendar_continuity_verified=False,
                 limitations=['Synthetic verification only; upstream classical functions extracted without unrelated imports.',
                              'No author training entrypoint run; external modules tested in this process, not vendored.',
                              'Unit/selection/aggregation and test-factory isolation covered by repository unit tests.'])
    (args.output/'synthetic_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='dlinear_cases'},indent=2))


if __name__=='__main__':
    main()
