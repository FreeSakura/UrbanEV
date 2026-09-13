"""Run the single frozen real-development comparability bridge."""
import argparse,csv,hashlib,io,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from urbanev_forecast.bounded_predictor_csv import prefix_bytes,parse_prefix
from urbanev_forecast.benchmark_metrics import scoped_scores
from urbanev_forecast.comparability_bridge import build_inputs,baseline_predictions,zone_capacities


def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    p=argparse.ArgumentParser()
    for name in ['config','data-root','native-root','truth-root','private-output','public-output','claim']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();blob=a.config.read_bytes();c=json.loads(blob)
    if c['protocol_id']!='URBANEV_COMPARABILITY_BRIDGE_V2_20260913':raise ValueError('Wrong protocol')
    if c['prefix_rows']!={'occupancy.csv':1560,'duration.csv':1547} or c['cuts']!=[720,1056,1392] or c['horizons']!=[3,6,9,12]:raise ValueError('Scope mismatch')
    if a.private_output.exists() or a.public_output.exists():raise FileExistsError('New outputs required')
    for name,digest in c['code_sha256_lf'].items():
        if sha((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))!=digest:raise ValueError('Code changed after freeze')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    a.claim.parent.mkdir(parents=True,exist_ok=True)
    with a.claim.open('x',encoding='utf-8') as f:json.dump({'protocol_id':c['protocol_id'],'config_sha256':sha(blob),'code_commit':head,'status':'CONSUMED_ON_START'},f)
    a.private_output.mkdir(parents=True);a.public_output.mkdir(parents=True)
    started=time.perf_counter();ledger=[]
    try:
        payload={}
        for name,rows in c['prefix_rows'].items():
            payload[name]=prefix_bytes(a.data_root,name,rows,rows,set(c['prefix_rows']))
            if sha(payload[name])!=c['prefix_sha256'][name]:raise ValueError('Prefix hash mismatch')
            ledger.append({'file':name,'rows':[0,rows],'sha256':sha(payload[name])})
        columns=next(csv.reader([payload['occupancy.csv'].splitlines()[0].decode('utf-8-sig')]))[1:]
        if len(columns)!=275 or sha('\n'.join(columns).encode())!=c['column_order_sha256']:raise ValueError('Region order mismatch')
        info_path=(a.data_root/'inf.csv').resolve()
        if info_path.parent!=a.data_root.resolve():raise ValueError('Static path outside root')
        info_blob=info_path.read_bytes()
        if sha(info_blob)!=c['static_info_sha256']:raise ValueError('Static identity mismatch')
        info=pd.read_csv(io.BytesIO(info_blob),usecols=['station_id','TAZID','charge_count'],dtype={'TAZID':str})
        cap=zone_capacities(info,columns)
        raw={name:parse_prefix(payload[name],rows,columns,c['prefix_sha256'][name],nonnegative=True)[0] for name,rows in c['prefix_rows'].items()}
        y=(raw['occupancy.csv'].astype(np.float32)/cap.astype(np.float32)[None,:]).astype(np.float64)
        d=raw['duration.csv']/cap[None,:]
        if ((y<0)|(y>1)).any() or ((d<0)|(d>1)).any():raise ValueError('Input rate domain failed; no repair')
        origins=np.concatenate([np.arange(cut,cut+168,12) for cut in c['cuts']])
        pack=build_inputs(y,d,origins);pack['targets_H12']=y[origins[:,None]+np.arange(12)[None,:]];pack['capacity']=cap
        pack['hour_of_day']=origins%24;pack['hour_of_week']=origins%168
        package=a.private_output/'private_dynamic_inputs.npz';np.savez(package,**pack)
        cached={};lineage=0.;old_error=0.;decoded=0
        for entry in c['cache_files']:
            cut,h,s=entry['cut'],entry['horizon'],entry['system']
            if cut not in c['cuts'] or h not in (3,12) or s not in ('native','truth'):raise ValueError('Cache not whitelisted')
            root=a.truth_root if s=='truth' else a.native_root
            name=f'private_truth_{cut}_h{h}.npy' if s=='truth' else f'private_{cut}_h{h}_native_a1.npy'
            path=(root/name).resolve()
            if path.parent!=root.resolve() or name!=entry['file']:raise ValueError('Cache path mismatch')
            data=path.read_bytes()
            if sha(data)!=entry['sha256']:raise ValueError('Cache identity mismatch')
            values=np.load(io.BytesIO(data),allow_pickle=False)
            if values.shape!=(14,h,275) or not np.isfinite(values).all():raise ValueError('Cache shape/values failed')
            cached[cut,h,s]=values.astype(np.float64);decoded+=1
            if s=='truth':
                target=y[np.arange(cut,cut+168,12)[:,None]+np.arange(h)[None,:]]
                lineage=max(lineage,float(np.max(abs(target-values))))
        if len(cached)!=12 or lineage>c['identity_tolerance']:raise ValueError('Truth lineage mismatch or incomplete cache')
        rows=[];independent_error=0.;bound_counts=[]
        def score(system,cut,h,prediction,target,postprocess):
            nonlocal independent_error
            for scope in ('terminal_H','path_1_to_H'):
                result=scoped_scores(prediction,target,target_scope=scope,postprocess=postprocess)
                pp=np.clip(prediction,0,1) if postprocess=='clip_0_1' else prediction
                errors=pp-target
                if scope=='terminal_H':errors=errors[:,-1,:]
                direct_rmse=float(np.sqrt(np.sum(np.square(errors,dtype=np.float64))/errors.size));direct_mae=float(np.sum(abs(errors))/errors.size)
                independent_error=max(independent_error,abs(direct_rmse-result['rmse']),abs(direct_mae-result['mae']))
                rows.append({'system':system,'cut':cut,'horizon':h,'status':'AVAILABLE',**result})
        for cut in c['cuts']:
            oo=np.arange(cut,cut+168,12)
            for h in c['horizons']:
                target=y[oo[:,None]+np.arange(h)[None,:]]
                for name,forecast in baseline_predictions(y,oo,h).items():score(name,cut,h,forecast,target,'raw')
                if h in (3,12):
                    native=cached[cut,h,'native'];bound_counts.append({'cut':cut,'horizon':h,'values':native.size,'below_zero':int((native<0).sum()),'above_one':int((native>1).sum())})
                    for label,post in [('native_raw','raw'),('native_clip','clip_0_1')]:score(label,cut,h,native,target,post)
                    for old in c['native_reference_scores']:
                        if old['cut']==cut and old['horizon']==h:
                            for label,post in [('raw','raw'),('clipped','clip_0_1')]:
                                check=scoped_scores(native,target,target_scope='path_1_to_H',postprocess=post)
                                old_error=max(old_error,abs(check['rmse']-old[label]['rmse']),abs(check['mae']-old[label]['mae']))
                else:
                    for label,post in [('native_raw','raw'),('native_clip','clip_0_1')]:
                        for scope in ('terminal_H','path_1_to_H'):rows.append({'system':label,'cut':cut,'horizon':h,'status':'NOT_AVAILABLE','target_scope':scope,'postprocess':post,'scored_values':0,'rmse':None,'mae':None})
        if independent_error>1e-12 or old_error>c['score_tolerance']:raise ValueError('Score identity failed')
        summaries=[]
        for support,hs in [('H3_H12_COMMON',[3,12]),('H3_H6_H9_H12_BASELINES',[3,6,9,12])]:
            for system in (['last','day','week','native_raw','native_clip'] if len(hs)==2 else ['last','day','week']):
                for scope in ('terminal_H','path_1_to_H'):
                    rr=[r for r in rows if r['system']==system and r['target_scope']==scope and r['horizon'] in hs and r['status']=='AVAILABLE']
                    summaries.append({'support':support,'system':system,'target_scope':scope,'cells':len(rr),'macro_rmse':float(np.mean([v['rmse'] for v in rr])),'macro_mae':float(np.mean([v['mae'] for v in rr]))})
        keys=['system','cut','horizon','status','target_scope','postprocess','scored_values','rmse','mae']
        with (a.public_output/'comparison_table.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)
        dump(a.public_output/'summary.json',{'aggregation':'Equal mean of named window-horizon cell metrics, not pooled RMSE; comparisons only within same support','rows':summaries,'uncertainty':'Descriptive development estimates, no significance claim; three exposed windows are not independent tests'})
        dump(a.public_output/'alignment_checks.json',{'truth_source_max_difference':lineage,'legacy_native_path_max_difference':old_error,'independent_reduction_max_difference':independent_error,'capacity_rule':'sum station charge_count grouped by TAZID','native_output_domain':bound_counts,'future_inputs_used':False,'native_h6_h9':'NOT_AVAILABLE; no slicing or inference','old_candidates_reselected':False})
        dump(a.public_output/'private_package_manifest.json',{'file_role':'private_dynamic_inputs.npz','sha256':sha(package.read_bytes()),'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in pack.items()},'package_uploaded':False,'short_axes':['origin','region','feature'],'long_axes':['origin','history','region'],'short_features':['y[o-2]','y[o-1]','duration_per_capacity[o-2]'],'duration_history':'[o-169,o-1)','calendar':'hour offsets from dataset start; deterministic features only'})
        receipt={'status':'COMPARABILITY_BRIDGE_COMPLETE_REVIEW_REQUIRED','protocol_id':c['protocol_id'],'config_sha256':sha(blob),'frozen_code_commit':head,'prefixes':ledger,'static_numeric_columns':['station_id','TAZID','charge_count'],'cache_decodes':decoded,'origin_count':42,'source_max_target_index':1559,'duration_max_index':1546,'score_rows_available':sum(r['status']=='AVAILABLE' for r in rows),'score_rows_not_available':sum(r['status']=='NOT_AVAILABLE' for r in rows),'new_model_fits':0,'new_foundation_inference':0,'alpha_searches':0,'old_candidate_reselection':False,'executions':1,'elapsed_seconds':time.perf_counter()-started,'peak_memory':'NOT_MEASURED','official_paper_reproduction':False,'sota_claim':False,'automation_enabled':False,'human_review_required':True,'automatic_next_round':False}
        dump(a.public_output/'protocol_receipt.json',receipt);print(json.dumps({'receipt':receipt,'summary':summaries},indent=2))
    except Exception as e:
        dump(a.public_output/'protocol_receipt.json',{'status':'BRIDGE_BLOCKED','error_type':type(e).__name__,'error':str(e),'prefixes':ledger,'claim_consumed':True,'automatic_next_round':False})
        raise


if __name__=='__main__':main()
