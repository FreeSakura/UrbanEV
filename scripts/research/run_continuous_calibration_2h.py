"""One authorized H3/H12 calibration using V1 cache, with no model inference."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast.continuous_step import authorize_real_calibration
from urbanev_forecast.continuous_calibration_2h import execute_systems,explain_gates
from urbanev_forecast.data import load_rate_prefix
from urbanev_forecast.foundation import FoundationBackend
import residual_information as v1

AUTH=ROOT/'configs/research/CONTINUOUS_V2_CALIBRATION_2H_AUTHORIZATION.json'
MANIFEST=ROOT/'configs/research/CONTINUOUS_V2_CALIBRATION_2H_RUN.json'
REG=ROOT/'configs/research/RESIDUAL_CONTINUOUS_STEP_V2_REGISTRATION.json'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def text_digest(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def observe_identity(authorization):
    identity={}
    for key in authorization['accepted_code_identity']:
        if key=='continuous_step_except_authorizer_AST':
            tree=ast.parse((ROOT/'src/urbanev_forecast/continuous_step.py').read_text())
            tree.body=[node for node in tree.body if getattr(node,'name',None)!='authorize_real_calibration']
            identity[key]=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
        else:identity[key]=text_digest(ROOT/key)
    return {'accepted_solver_commit':authorization['accepted_solver_commit'],
            'solver_config_sha256':digest(ROOT/authorization['solver_config_path']),'code_identity':identity}


def opaque_prefix_hash(path,rows):
    with Path(path).open('rb') as stream:payload=b''.join(stream.readline() for _ in range(rows+1))
    return hashlib.sha256(payload).hexdigest(),len(payload)


def load_payload(args,manifest,reads):
    """Called only after stage admission and one-use execution claim."""
    old_path=args.cache/'result.json'
    if digest(old_path)!=manifest['v1_result_sha256']:raise ValueError('V1 result identity changed')
    old=json.loads(old_path.read_text())
    reads.append({'role':'V1 result','kind':'metadata','sha256':digest(old_path)})
    if np.__version__!=old['backend']['numpy_version']:raise ValueError('V1 NumPy version differs')
    caches={}
    for h in (3,12):
        rec=next(row for row in manifest['cache'] if row['horizon']==h)
        op=args.cache/f'private_origins_h{h}.npy';bp=args.cache/f'private_base_h{h}.npy'
        for path,key in ((op,'origins_sha256'),(bp,'prediction_sha256')):
            actual=digest(path)
            reads.append({'role':path.name,'kind':'opaque_byte_hash','bytes':path.stat().st_size,'sha256':actual})
            if actual!=rec[key]:raise ValueError('Cache hash changed')
        origins=np.load(op,allow_pickle=False)
        reads.append({'role':op.name,'kind':'origin_index_metadata','entries':int(origins.size),'prediction_values_read':False,'target_labels_read':False})
        if origins.ndim!=1 or len(np.unique(origins))!=len(origins):raise ValueError('Origin index malformed')
        needed=sorted({s for c in manifest['cells'] if c['horizon']==h for s in c['fit_origins']+c['calibration_origins']})
        if any(s+h>1560 for s in needed):raise ValueError('Requested cache target exceeds1560')
        owner={int(s):i for i,s in enumerate(origins)}
        if any(s not in owner for s in needed):raise ValueError('Required origin missing')
        caches[h]=(bp,origins,needed,owner)
    # Check all per-cell source prediction files before any label values are parsed.
    for row in manifest['direction_lineage']:
        path=args.cache/row['file']
        if Path(row['file']).name!=row['file']:raise ValueError('Invalid cache basename')
        actual=digest(path)
        reads.append({'role':row['file'],'kind':'opaque_byte_hash','bytes':path.stat().st_size,'sha256':actual})
        if actual!=row['sha256']:raise ValueError('Direction lineage file changed')
    for role,path in [('occupancy',args.csv),('duration',args.duration)]:
        actual,size=opaque_prefix_hash(path,1560)
        reads.append({'role':role,'kind':'opaque_prefix_hash','row_range':[0,1560],'bytes':size,'sha256':actual})
        if actual!=manifest['input_prefix_sha256'][role]:raise ValueError('Frozen input prefix changed')
    info_hash=digest(args.info)
    reads.append({'role':'info','kind':'opaque_static_file_hash','bytes':args.info.stat().st_size,'sha256':info_hash})
    if info_hash!=manifest['static_info_sha256']:raise ValueError('Static capacity data changed')
    o,source=load_rate_prefix(args.csv,1560)
    reads.append({'role':'occupancy','kind':'semantic_numeric_prefix','row_range':[0,1560],'dtype':str(o.dtype),'shape':list(o.shape)})
    if source['column_order_sha256']!=old['source']['column_order_sha256']:raise ValueError('Column order changed')
    d,dsource=v1.duration_prefix(args.duration,args.info,args.csv,1560)
    reads.extend([{'role':'duration','kind':'semantic_numeric_prefix','row_range':[0,1560],'dtype':str(d.dtype),'shape':list(d.shape)},
                  {'role':'info','kind':'static_capacity_parse','sha256':dsource['info_sha256']},
                  {'role':'occupancy header','kind':'metadata_only','data_rows':0}])
    numeric={}
    for h,(path,origins,needed,owner) in caches.items():
        mapped=np.load(path,mmap_mode='r',allow_pickle=False)
        if mapped.shape!=(len(origins),h,275) or mapped.dtype!=np.float32:raise ValueError('Baseline cache shape/dtype changed')
        selected=np.asarray(mapped[[owner[s] for s in needed]]).copy()
        if not np.isfinite(selected).all():raise ValueError('Nonfinite baseline')
        numeric[h]=(np.asarray(needed),selected)
        reads.append({'role':path.name,'kind':'numeric_selected_cache_rows','origins':needed,
                      'last_target':needed[-1]+h-1,'shape':list(selected.shape),'tail_numeric_rows_read':0})
    cells={s:[] for s in manifest['systems']};lineage=[]
    with threadpool_limits(limits=2):
        for entry in manifest['cells']:
            cut,h=entry['cut'],entry['horizon'];origin,base=numeric[h];index={int(s):i for i,s in enumerate(origin)}
            fit=np.asarray(entry['fit_origins']);cal=np.asarray(entry['calibration_origins'])
            tb=base[[index[int(s)] for s in fit]];vb=base[[index[int(s)] for s in cal]]
            x,td=v1.features(o,d,fit,tb);vx,vd=v1.features(o,d,cal,vb)
            target=o[fit[:,None]+np.arange(h)].transpose(0,2,1).reshape(-1,h)
            error=target-tb.transpose(0,2,1).reshape(-1,h)  # preserve V1 float32 subtraction before ridge
            corrections,diag=v1.fit_corrections(x,vx,td,vd,error)
            truth=o[cal[:,None]+np.arange(h)]
            for system in manifest['systems']:
                delta=corrections[system].reshape(len(cal),275,h).transpose(0,2,1)
                reference_path=args.cache/f'private_{cut}_h{h}_{system}_a1.npy'
                reference=np.load(reference_path,allow_pickle=False)
                matched=reference.shape==vb.shape and np.array_equal(vb+delta,reference)
                reads.append({'role':reference_path.name,'kind':'numeric_calibration_prediction_lineage','origin_range':[int(cal[0]),int(cal[-1])],'last_target':int(cal[-1]+h-1)})
                lineage.append({'cut':cut,'horizon':h,'system':system,'v1_alpha1_exact_numeric_equal':matched})
                if not matched:raise ValueError(f'V1 direction reconstruction mismatch: {system}/cut{cut}/h{h}')
                cells[system].append({'id':{'stage':'calibration','block':cut,'horizon':h},'p':vb,'delta':delta,'y':truth})
                np.save(args.output/f'private_direction_{cut}_h{h}_{system}.npy',delta)
            np.save(args.output/f'private_truth_{cut}_h{h}.npy',truth)
            print(json.dumps({'phase':'CPU_DIRECTION_REBUILT','cut':cut,'horizon':h,'fit_origins':len(fit),'calibration_origins':len(cal),'diagnostics':diag}),flush=True)
    return cells,source,dsource,lineage


def main():
    parser=argparse.ArgumentParser()
    for name in ('csv','duration','info','cache','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',default='calibrate_h3_h12');args=parser.parse_args()
    authorization=json.loads(AUTH.read_text());manifest=json.loads(MANIFEST.read_text());registration=json.loads(REG.read_text())
    config_path=ROOT/authorization['solver_config_path'];config=json.loads(config_path.read_text());observed=observe_identity(authorization)
    admission=authorize_real_calibration(registration,config,authorization=authorization,stage=args.stage,run_manifest=manifest,observed_identity=observed)
    if admission['status']!='AUTHORIZED':raise ValueError(admission)
    if set(manifest['entry_code_identity'])!={'scripts/research/run_continuous_calibration_2h.py','src/urbanev_forecast/continuous_calibration_2h.py'}:
        raise ValueError('Execution entry identities missing')
    if set(manifest['input_prefix_sha256'])!={'occupancy','duration'} or any(len(v)!=64 for v in manifest['input_prefix_sha256'].values()):
        raise ValueError('Input prefix identities were not frozen')
    for path,sha in manifest['entry_code_identity'].items():
        if text_digest(ROOT/path)!=sha:raise ValueError('Frozen execution entry changed')
    tracked=list(manifest['entry_code_identity'])+[AUTH.relative_to(ROOT).as_posix(),MANIFEST.relative_to(ROOT).as_posix()]
    subprocess.run(['git','ls-files','--error-unmatch',*tracked],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--exit-code','HEAD','--',*tracked],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    if args.output.exists():raise FileExistsError('Use a fresh run directory; no automatic retry')
    args.output.mkdir(parents=True)
    claim=ROOT.parent/'work/continuous-v2-stage-claims'/f"{manifest['run_id']}.json"
    claim.parent.mkdir(parents=True,exist_ok=True)
    with claim.open('x',encoding='utf-8') as stream:json.dump({'run_id':manifest['run_id'],'stage':args.stage,'status':'CLAIMED_ONCE'},stream)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    reads=[];calls={'foundation_constructor':0,'foundation_predict':0};start=time.perf_counter();execution=None
    receipt={'authorization':admission,'accepted_solver_commit':authorization['accepted_solver_commit'],'execution_commit':commit,
        'authorization_sha256':digest(AUTH),'manifest_sha256':digest(MANIFEST),'solver_config_sha256':digest(config_path),
        'observed_code_identity':observed,'read_ledger':reads,'model_call_monitor':calls,
        'automatic_stage_advance':False,'new_foundation_inference':0,'semantic_row_stop':1560,
        'h6_h9_run':False,'tail_scoring':False,'third_fold_validation_opened':False,'third_fold_test_opened':False,'quota_reset_used':False}
    def deny(name):
        def fail(*a,**kw):calls[name]+=1;raise RuntimeError('New foundation inference forbidden')
        return fail
    try:
        with patch.object(FoundationBackend,'__init__',deny('foundation_constructor')),patch.object(FoundationBackend,'predict',deny('foundation_predict')):
            cells,source,dsource,lineage=load_payload(args,manifest,reads)
            receipt.update(source=source,duration_source=dsource,direction_lineage=lineage)
            def record(system,result):
                (args.output/f'solver_{system}.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
                print(json.dumps({'phase':'SYSTEM_DONE','system':system,'status':result.get('status'),'alpha':result.get('alpha'),'reason':result.get('reason')}),flush=True)
            execution=execute_systems(cells,config,registration,on_result=record)
            for system,result in execution['systems'].items():
                if result.get('score',{}).get('status')=='OK':
                    for c in cells[system]:
                        ident=c['id'];pred=c['p']+result['alpha']*c['delta']
                        np.save(args.output/f"private_selected_{ident['block']}_h{ident['horizon']}_{system}.npy",pred)
            decision=explain_gates(execution,registration)
    except Exception as error:
        execution=execution or {'status':'CALIBRATION_2H_BLOCKED','systems':{}}
        execution.update(status='CALIBRATION_2H_BLOCKED',reason=f'{type(error).__name__}: {error}')
        decision={'stage_status':'CALIBRATION_2H_BLOCKED','information_gate':'NOT_RUN','structure_gate':'NOT_RUN','reason':execution['reason']}
    for system in registration['systems']:
        execution['systems'].setdefault(system,{'status':'NOT_RUN','alpha':None,'reason':'Stage stopped before this required system'})
    receipt['status']=execution['status'];receipt['elapsed_seconds']=time.perf_counter()-start
    (args.output/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    (args.output/'calibration_2h.json').write_text(json.dumps(execution,indent=2)+'\n',encoding='utf-8')
    (args.output/'gate_decision.json').write_text(json.dumps(decision,indent=2)+'\n',encoding='utf-8')
    with (args.output/'calibration_cells.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['system','outer_status','alpha','block','horizon','n','raw_rmse','raw_mae','rmse','mae']);writer.writeheader()
        for system,result in execution['systems'].items():
            rows=result.get('score',{}).get('cells',[])
            for row in rows:
                writer.writerow({'system':system,'outer_status':result['status'],'alpha':result['alpha'],
                    'block':row['id']['block'],'horizon':row['id']['horizon'],**{k:row[k] for k in ('n','raw_rmse','raw_mae','rmse','mae')}})
            if not rows:
                for cell in manifest['cells']:
                    writer.writerow({'system':system,'outer_status':result['status'],'alpha':'NOT_RUN',
                        'block':cell['cut'],'horizon':cell['horizon'],**{k:'NOT_RUN' for k in ('n','raw_rmse','raw_mae','rmse','mae')}})
    print(json.dumps({'status':execution['status'],'gate':decision,'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
