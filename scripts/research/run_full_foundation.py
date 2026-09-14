"""Offline full-H foundation inference after all supervised selections are frozen."""
from __future__ import annotations

import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast.benchmark_contract import origins
from urbanev_forecast.full_benchmark_foundation import FullFoundationBackend
from urbanev_forecast.full_benchmark_data import FoldReader
from urbanev_forecast.full_benchmark_guard import verify_execution,consume_or_validate_claim


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def dump(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','data','private','model-dir'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--backend',choices=['chronos2','timesfm3'],required=True)
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'))
    verify_execution(c,ROOT);consume_or_validate_claim(a.private,a.config,'foundation')
    freeze=a.private/'supervised_prediction_freeze.json'
    if not freeze.exists():raise RuntimeError('Complete supervised prediction freeze first')
    info=json.loads(freeze.read_text())
    if info['config_sha256']!=sha(a.config):raise ValueError('Frozen config changed')
    root=a.private/a.backend;root.mkdir(exist_ok=True)
    finished=root/'foundation_complete.json'
    if finished.exists():
        record=json.loads(finished.read_text())
        if record['config_sha256']!=sha(a.config):raise ValueError('Config changed on completed foundation stage')
        for item in record['predictions']:
            if sha(a.private/item['file'])!=item['sha256']:raise ValueError('Completed prediction changed')
        print('Already complete; no new calls',flush=True);return
    backend=FullFoundationBackend(a.backend,a.model_dir)
    if backend.metadata!=c['foundation_models'][a.backend]['metadata']:
        raise ValueError('Installed foundation identity changed since preflight')
    records=[];origin_tasks=0;seconds=0.;ledger=[];all_alias=True
    for f in c['folds']:
        for h in c['horizons']:
            cell=root/f'fold{f}_h{h}';cell.mkdir(exist_ok=True)
            manifest=cell/'complete.json'
            if manifest.exists():
                saved=json.loads(manifest.read_text())
                if saved['config_sha256']!=sha(a.config):raise ValueError('Completed cell configuration changed')
                for item in saved['predictions']:
                    if sha(a.private/item['file'])!=item['sha256']:raise ValueError('Completed foundation cell changed')
                records.extend(saved['predictions']);origin_tasks+=saved['origin_tasks'];seconds+=saved['seconds']
                all_alias=all_alias and saved['point_equals_q05'];continue
            reader=FoldReader(a.data,c['data_identity'],f,h,ledger);reader.close_training()
            y,_,_=reader.read('predict');oo=origins(f,'test',168,h,contract='MATCHED_TERMINAL_168')
            cell_seconds=0.;points=[];medians=[]
            for index,o in enumerate(oo):
                target=cell/f'private_origin_{int(o)}.npz';receipt=cell/f'origin_{int(o)}.json'
                if receipt.exists():
                    rec=json.loads(receipt.read_text())
                    if rec['config_sha256']!=sha(a.config) or sha(target)!=rec['sha256']:raise ValueError('Origin resume identity mismatch')
                    with np.load(target,allow_pickle=False) as z:point=z['point'];median=z['q05']
                else:
                    start=time.perf_counter();point,median=backend.predict(y[o-168:o],h)
                    elapsed=time.perf_counter()-start
                    if target.exists():raise RuntimeError('Uncommitted origin output; reconcile interrupted call before retry')
                    np.savez(target,point=point,q05=median)
                    rec={'config_sha256':sha(a.config),'sha256':sha(target),'origin':int(o),'horizon':h,'seconds':elapsed}
                    dump(receipt,rec)
                points.append(point);medians.append(median);cell_seconds+=rec['seconds']
                if (index+1)%50==0:print(json.dumps({'backend':a.backend,'fold':f,'horizon':h,'origins_done':index+1,'origins_total':len(oo)}),flush=True)
            point=np.stack(points);median=np.stack(medians);cell_records=[]
            alias=bool(np.array_equal(point,median));all_alias=all_alias and alias
            for variant,q in [('POINT',point),('Q05',median)]:
                target=cell/f'private_test_{variant}.npy';np.save(target,q)
                cell_records.append({'file':target.relative_to(a.private).as_posix(),'sha256':sha(target),
                    'model':a.backend.upper()+'_'+variant,'seed':'pretrained','fold':f,'horizon':h,
                    'origins':len(oo),'origin_first':int(oo[0]),'origin_last':int(oo[-1]),
                    'information_track':'GLOBAL_O_168_PRETRAINED','variant':variant,'point_q05_alias':alias})
            dump(manifest,{'config_sha256':sha(a.config),'predictions':cell_records,'origin_tasks':len(oo),
                           'seconds':cell_seconds,'point_equals_q05':alias})
            records.extend(cell_records);origin_tasks+=len(oo);seconds+=cell_seconds
            print(json.dumps({'backend':a.backend,'fold':f,'horizon':h,'cell_complete':True,'origin_tasks_completed':origin_tasks}),flush=True)
    if origin_tasks!=5960:raise ValueError('Incomplete foundation request coverage')
    dump(finished,{'config_sha256':sha(a.config),'supervised_prediction_freeze_sha256':sha(freeze),
                   'predictions':records,'origin_tasks':origin_tasks,'seconds':seconds,
                   'point_q05_alias_for_all_outputs':all_alias,'metadata':backend.metadata,'reads':ledger,
                   'test_scores_seen':False})


if __name__=='__main__':main()
