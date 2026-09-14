"""One frozen contract and one claim across resumable execution phases."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def canonical_hash(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def verify_execution(config,project_root):
    if config.get('protocol_id')!='URBANEV_MATCHED_SIX_FOLD_V1_20260914' or config.get('freeze_status')!='FROZEN':
        raise RuntimeError('A frozen full-benchmark contract is required')
    if config.get('real_execution_authorized') is not True:
        raise RuntimeError('Contract is not executable')
    if config['folds']!=[1,2,3,4,5,6] or config['horizons']!=[3,6,9,12]:raise ValueError('Incomplete cell plan')
    if config['training']['epochs']!=20 or config['training']['seeds']!=[20260921,20260922]:
        raise ValueError('Unexpected fixed training budget')
    if config['training']['microbatch']!=8 or config['training']['batch_origins']!=16:
        raise ValueError('Unexpected frozen batch semantics')
    if len(config['neural_models'])!=9 or len({v['name'] for v in config['neural_models']})!=9:
        raise ValueError('Required neural configurations missing or duplicated')
    if any(v['learning_rates']!=[.001] for v in config['neural_models']):raise ValueError('Unregistered learning-rate search')
    if len(config['ridge_models'])!=3 or any(v['lambdas']!=[.01] for v in config['ridge_models']):
        raise ValueError('Unregistered ridge fitting budget')
    root=Path(project_root).resolve()
    if not config.get('execution_sources'):raise ValueError('Missing execution source identities')
    for rel,expected in config['execution_sources'].items():
        p=(root/rel).resolve()
        if root not in p.parents or canonical_hash(p)!=expected:
            raise ValueError(f'Frozen execution source changed: {rel}')


def consume_or_validate_claim(private,config_path,phase):
    root=Path(private);path=root/'execution_claim.json'
    digest=hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    if path.exists():
        claim=json.loads(path.read_text(encoding='utf-8'))
        if claim['config_sha256']!=digest or not claim['consumed']:
            raise ValueError('Claim/config mismatch')
        return claim
    if phase!='train':raise RuntimeError('Training must consume the claim before later phases')
    root.mkdir(parents=True,exist_ok=True)
    claim={'config_sha256':digest,'consumed':True,'created_utc':datetime.now(timezone.utc).isoformat(),
           'resume_policy':'same frozen configuration; saved optimizer/model/RNG; no best rerun selection'}
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(claim,f,indent=2)
    return claim
