"""Execute one fail-closed M7-C formal distillation cell."""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time, traceback
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

def sha(path):
 d=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): d.update(b)
 return d.hexdigest()
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write_json(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
def logit(x,eps=1e-6): x=x.clamp(eps,1-eps); return torch.log(x/(1-x))
def load_npz(path):
 with np.load(path,allow_pickle=False) as p:return {k:p[k] for k in p.files}
def validate_cache(row,p,path,rate,time_index,zone_ids,horizon):
 if sha(path)!=str(row.teacher_artifact_hash):raise RuntimeError("TERMINAL_PROVENANCE: cache file hash")
 checks=[ah(p["teacher_prediction"])==str(row.teacher_prediction_hash),ah(p["teacher_logit_residual"])==str(row.teacher_residual_hash),
         ah(p["target"])==str(row.target_hash),ah(p["target_index"].astype("<i8"))==str(row.target_index_hash),
         ah(p["zone_ids"].astype("U"))==str(row.zone_order_hash)]
 idx=p["target_index"].astype(np.int64);actual=rate[idx].astype(np.float32)
 checks.extend([np.allclose(p["target"],actual,atol=1e-7,rtol=0),np.array_equal(p["zone_ids"].astype(str),zone_ids.astype(str)),
                np.array_equal(p["forecast_origin_index"].astype(np.int64)+horizon,idx),
                np.array_equal(p["target_time"].astype(str),time_index[idx].astype(str).to_numpy())])
 if not all(checks):raise RuntimeError("TERMINAL_PROVENANCE: cache arrays/audited GT identity")
 p["target"]=actual
def metrics(pred,target):
 p=np.asarray(pred,np.float64); y=np.asarray(target,np.float64); e=p-y; a=np.abs(e); den=np.abs(p)+np.abs(y); m=den>1e-8
 return {"RMSE":float(np.sqrt(np.mean(e**2))),"MAE":float(a.mean()),"WAPE":float(a.sum()/np.abs(y).sum()),"sMAPE":float(np.mean(2*a[m]/den[m])),"RAE":float(a.sum()/np.abs(y-y.mean()).sum()),"output_min":float(p.min()),"output_max":float(p.max())}

class Data(Dataset):
 def __init__(self,rate,p,h,teacher): self.rate=rate.astype(np.float32,copy=False); self.idx=p["target_index"].astype(np.int64); self.anchor=p["caper_prediction"].astype(np.float32); self.y=p["target"].astype(np.float32); self.teacher=teacher.astype(np.float32); self.h=h
 def __len__(self):return len(self.idx)
 def __getitem__(self,i):
  t=int(self.idx[i]); o=t-self.h; x=np.ascontiguousarray(self.rate[o-167:o+1]);
  if x.shape!=(168,275):raise AssertionError("history")
  return torch.from_numpy(x),torch.from_numpy(self.anchor[i]),torch.from_numpy(self.y[i]),torch.from_numpy(self.teacher[i]),t
class MA(nn.Module):
 def __init__(self):super().__init__();self.pool=nn.AvgPool1d(25,1)
 def forward(self,x):return self.pool(torch.cat([x[:,:1].repeat(1,12,1),x,x[:,-1:].repeat(1,12,1)],1).transpose(1,2)).transpose(1,2)
class Model(nn.Module):
 def __init__(self):super().__init__();self.ma=MA();self.seasonal=nn.Linear(168,1);self.trend=nn.Linear(168,1)
 def forward(self,x):t=self.ma(x);return (self.seasonal((x-t).transpose(1,2))+self.trend(t.transpose(1,2))).squeeze(-1)

def concatenate(payloads):
 keys=("target_index","caper_prediction","target","teacher_logit_residual")
 return {k:np.concatenate([p[k] for p in payloads],axis=0) for k in keys}
def shuffled_residual(payloads,fold,horizon):
 combined=concatenate(payloads); residual=combined["teacher_logit_residual"]; blocks=[]; offset=0
 for source_id,p in enumerate(payloads):
  n=len(p["target_index"])
  for start in range(0,n,24): blocks.append({"source_cache_position":source_id,"source_start":offset+start,"length":min(24,n-start)})
  offset+=n
 rng=np.random.default_rng(20260829+100*fold+horizon); order=rng.permutation(len(blocks))
 if len(order)>1 and np.array_equal(order,np.arange(len(order))): order=np.roll(order,-1)
 if len(order)<=1 or np.array_equal(order,np.arange(len(order))): raise AssertionError("shuffle must be non-identity")
 pieces=[]; mapping=[]; dest=0
 for source_block in order:
  b=blocks[int(source_block)]; pieces.append(residual[b["source_start"]:b["source_start"]+b["length"]]); mapping.append({"destination_start":dest,"source_block":int(source_block),**b}); dest+=b["length"]
 shuffled=np.concatenate(pieces,axis=0)
 if shuffled.shape!=residual.shape or ah(np.sort(shuffled.reshape(-1)))!=ah(np.sort(residual.reshape(-1))): raise AssertionError("shuffle marginal identity")
 return shuffled,mapping,order.tolist()

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--project-root",type=Path,required=True);ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--run-id",required=True);ap.add_argument("--attempt",type=int,required=True);ap.add_argument("--source-root",type=Path,required=True);a=ap.parse_args()
 project=a.project_root.resolve();root=project/"experiments/07_residual_distillation";manifest_path=a.manifest.resolve(); rows=pd.read_csv(manifest_path,keep_default_na=False); sel=rows[rows.run_id==a.run_id]
 if len(sel)!=1 or a.attempt not in (1,2):raise RuntimeError("run identity/attempt")
 row=sel.iloc[0];run_root=root/"formal_runs"/a.run_id;success=run_root/"SUCCESS.json"
 if success.is_file():
  old=json.loads(success.read_text(encoding="utf-8"));
  if old.get("fingerprint")==row.fingerprint: print(json.dumps({"decision":"SKIPPED_DUPLICATE","run_id":a.run_id}));return
  raise RuntimeError("success fingerprint drift")
 attempt=run_root/f"attempt_{a.attempt:02d}";attempt.mkdir(parents=True,exist_ok=False);started=datetime.now().isoformat(timespec="seconds")
 try:
  config_path=root/"configs/FORMAL_STUDENT_CONFIG_V1.json";shuffle_path=root/"SHUFFLE_CONTROL_SPEC.json";auth_path=root/"M7C_AUTHORIZATION.md";cache_manifest_path=root/"TEACHER_CACHE_MANIFEST.csv";outer_path=root/"OUTER_FOLD_CACHE_INDEX.csv";audit_path=root/"TEACHER_CACHE_AUDIT.json"
  provenance={"formal_config_hash":sha(config_path),"shuffle_spec_hash":sha(shuffle_path),"authorization_hash":sha(auth_path),"cache_manifest_hash":sha(cache_manifest_path),"outer_index_hash":sha(outer_path),"cache_audit_hash":sha(audit_path),"runner_hash":sha(Path(__file__).resolve())}
  expected={"formal_config_hash":row.student_config_hash,"shuffle_spec_hash":row.shuffle_spec_hash,"authorization_hash":row.authorization_hash,"cache_manifest_hash":row.teacher_cache_hash,"outer_index_hash":row.outer_index_hash,"cache_audit_hash":row.cache_audit_hash,"runner_hash":row.runner_hash}
  if provenance!=expected:raise RuntimeError("TERMINAL_PROVENANCE: frozen input drift")
  cfg=json.loads(config_path.read_text(encoding="utf-8"));audit=json.loads(audit_path.read_text(encoding="utf-8"));
  if audit.get("overall_verdict")!="PASS" or cfg.get("formal_matrix_cells")!=72 or cfg.get("protected_test_access") is not False:raise RuntimeError("authorization drift")
  sys.path.insert(0,str(a.source_root.resolve()));from innovation.data import load_occupancy
  occupancy=load_occupancy(a.source_root.resolve()/"audited/data");rate=occupancy.rate;zones=np.asarray(occupancy.zone_ids).astype("U")
  cm=pd.read_csv(cache_manifest_path,keep_default_na=False); ids=str(row.train_cache_ids).split(";"); train_rows=cm[cm.cache_id.isin(ids)].sort_values(["outer_fold","target_index_start"])
  if train_rows.cache_id.astype(str).tolist()!=ids:raise AssertionError("train cache order")
  payloads=[]
  for source_row in train_rows.itertuples():
   source_path=Path(source_row.artifact_path);payload=load_npz(source_path);validate_cache(source_row,payload,source_path,rate,occupancy.time,zones,int(row.horizon));payloads.append(payload)
  train=concatenate(payloads); evaluation=cm[cm.cache_id==row.eval_cache_id]
  if len(evaluation)!=1:raise RuntimeError("TERMINAL_PROVENANCE: eval cache")
  eval_row=evaluation.iloc[0];eval_path=Path(eval_row.artifact_path);ev=load_npz(eval_path);validate_cache(eval_row,ev,eval_path,rate,occupancy.time,zones,int(row.horizon))
  if ah(ev["target"])!=row.target_hash or ah(ev["target_index"].astype("<i8"))!=row.target_index_hash or ah(zones)!=row.zone_order_hash:raise RuntimeError("TERMINAL_PROVENANCE: formal target identity")
  branch=str(row.branch); teacher=train["teacher_logit_residual"]; mapping=[]; order=[]
  if branch=="shuffled_teacher": teacher,mapping,order=shuffled_residual(payloads,int(row.outer_fold),int(row.horizon))
  elif branch not in ("GT_only","aligned_KD"):raise RuntimeError("branch")
  seed_all(42);device=torch.device("cuda");tr=Data(rate,train,int(row.horizon),teacher);te=Data(rate,ev,int(row.horizon),ev["teacher_logit_residual"])
  gen=torch.Generator().manual_seed(42);tl=DataLoader(tr,batch_size=32,shuffle=True,num_workers=0,pin_memory=True,generator=gen);el=DataLoader(te,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
  model=Model().to(device); 
  if sum(p.numel() for p in model.parameters())!=338:raise AssertionError("parameters")
  opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0);loss_fn=nn.HuberLoss(delta=1,reduction="mean");logs=[];clock=time.perf_counter()
  for epoch in range(1,51):
   model.train();total=gt_total=kd_total=0.;count=0
   for x,anchor,y,tres,_ in tl:
    x,anchor,y,tres=x.to(device),anchor.to(device),y.to(device),tres.to(device);r=model(x);pred=torch.sigmoid(logit(anchor)+r);gt=loss_fn(pred,y);kd=loss_fn(r,tres);loss=gt if branch=="GT_only" else gt+.5*kd
    opt.zero_grad(set_to_none=True);loss.backward();opt.step();n=len(x);total+=float(loss.detach())*n;gt_total+=float(gt.detach())*n;kd_total+=float(kd.detach())*n;count+=n
   logs.append({"epoch":epoch,"total_loss":total/count,"gt_loss":gt_total/count,"kd_loss":kd_total/count})
  model.eval();preds=[];resids=[];targets=[];indices=[]
  with torch.no_grad():
   for x,anchor,y,_,idx in el:
    r=model(x.to(device));pred=torch.sigmoid(logit(anchor.to(device))+r);preds.append(pred.cpu().numpy());resids.append(r.cpu().numpy());targets.append(y.numpy());indices.append(idx.numpy())
  pred=np.concatenate(preds);resid=np.concatenate(resids);target=np.concatenate(targets);target_index=np.concatenate(indices);origin=target_index-int(row.horizon)
  met=metrics(pred,target)
  if not np.isfinite(pred).all() or met["output_min"]<0 or met["output_max"]>1:raise AssertionError("output")
  cell_cfg={"run_id":a.run_id,"fingerprint":row.fingerprint,"branch":branch,"outer_fold":int(row.outer_fold),"horizon":int(row.horizon),"seed":42,"attempt":a.attempt,**provenance,"train_cache_ids":ids,"eval_cache_id":row.eval_cache_id,"protected_test_access":False}
  write_json(attempt/"config.json",cell_cfg);pd.DataFrame(logs).to_csv(attempt/"training_log.csv",index=False);pd.DataFrame(logs).to_csv(attempt/"loss_curve.csv",index=False)
  np.savez_compressed(attempt/"predictions.npz",prediction=pred,student_residual=resid);np.savez_compressed(attempt/"targets.npz",target=target,target_index=target_index)
  np.save(attempt/"forecast_origins.npy",origin);np.save(attempt/"target_timestamps.npy",np.asarray(occupancy.time[target_index].astype(str),dtype="U32"));np.save(attempt/"zone_order.npy",np.asarray(occupancy.zone_ids).astype("U"))
  torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"config":cell_cfg},attempt/"checkpoint.pt")
  if branch=="shuffled_teacher":write_json(attempt/"shuffle_mapping.json",{"seed":20260829+100*int(row.outer_fold)+int(row.horizon),"block_order":order,"mapping":mapping,"aligned_residual_hash":ah(train["teacher_logit_residual"]),"shuffled_residual_hash":ah(teacher)})
  extra={"teacher_prediction_hash":str(evaluation.iloc[0].teacher_prediction_hash),"teacher_residual_hash":str(evaluation.iloc[0].teacher_residual_hash),"train_teacher_residual_hash":ah(teacher),"teacher_student_residual_correlation":float(np.corrcoef(ev["teacher_logit_residual"].reshape(-1),resid.reshape(-1))[0,1])}
  write_json(attempt/"metrics.json",{**met,**extra,"train_runtime_seconds":time.perf_counter()-clock,"epoch_count":50,"NaN":False,"OOM":False})
  artifact_names=["config.json","training_log.csv","loss_curve.csv","predictions.npz","targets.npz","forecast_origins.npy","target_timestamps.npy","zone_order.npy","metrics.json","checkpoint.pt"]+(["shuffle_mapping.json"] if branch=="shuffled_teacher" else [])
  hashes={name:sha(attempt/name) for name in artifact_names};write_json(attempt/"artifact_hashes.json",hashes);hashes["artifact_hashes.json"]=sha(attempt/"artifact_hashes.json")
  receipt={"status":"success","run_id":a.run_id,"fingerprint":row.fingerprint,"attempt":a.attempt,"started_at":started,"finished_at":datetime.now().isoformat(timespec="seconds"),"provenance":provenance,"artifacts":hashes,"student_training_scope":"M7C_formal_seed42","protected_target_accessed":False}
  write_json(attempt/"run_receipt.json",receipt);write_json(success,{"status":"success","run_id":a.run_id,"fingerprint":row.fingerprint,"attempt":a.attempt,"receipt":str((attempt/"run_receipt.json").resolve()),"receipt_sha256":sha(attempt/"run_receipt.json")})
  print(json.dumps({"status":"success","run_id":a.run_id,"attempt":a.attempt}))
 except Exception as e:
  failure_class="terminal_provenance" if "TERMINAL_PROVENANCE" in str(e) or isinstance(e,AssertionError) else ("retryable_oom" if "out of memory" in str(e).lower() else ("retryable_io" if isinstance(e,OSError) else "terminal_other"))
  write_json(attempt/"FAILED.json",{"status":"failed","failure_class":failure_class,"run_id":a.run_id,"attempt":a.attempt,"error":repr(e),"traceback":traceback.format_exc(),"failed_at":datetime.now().isoformat(timespec="seconds")});raise
if __name__=="__main__":main()
