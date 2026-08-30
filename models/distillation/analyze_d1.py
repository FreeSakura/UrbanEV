"""Compute the preregistered M7-C Gate D1 only after a PASS matrix audit."""
from __future__ import annotations
import argparse,json
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
BR=("GT_only","aligned_KD","shuffled_teacher");MET=("RMSE","MAE","WAPE","sMAPE","RAE")
def met(p,y):
 p=np.asarray(p,np.float64);y=np.asarray(y,np.float64);e=p-y;a=np.abs(e);den=np.abs(p)+np.abs(y);m=den>1e-8
 return {"RMSE":float(np.sqrt(np.mean(e**2))),"MAE":float(a.mean()),"WAPE":float(a.sum()/np.abs(y).sum()),"sMAPE":float(np.mean(2*a[m]/den[m])),"RAE":float(a.sum()/np.abs(y-y.mean()).sum())}
def bootstrap(cells,a_branch,b_branch,n=5000,seed=20260829):
 rng=np.random.default_rng(seed);dist=[]
 pairs={(f,h):{} for f in range(1,7) for h in (3,6,9,12)}
 for item in cells:
  pairs[(item["fold"],item["horizon"])][item["branch"]]=item
 prepared=[]
 for key,pair in pairs.items():
  ya=pair[a_branch]["target"];yb=pair[b_branch]["target"]
  if not np.array_equal(ya,yb):raise AssertionError(f"paired target mismatch {key}")
  sse_a=np.sum((pair[a_branch]["prediction"].astype(np.float64)-ya.astype(np.float64))**2,axis=1)
  sse_b=np.sum((pair[b_branch]["prediction"].astype(np.float64)-yb.astype(np.float64))**2,axis=1)
  blocks=[np.arange(i,min(i+24,len(ya))) for i in range(0,len(ya),24)]
  prepared.append((sse_a,sse_b,blocks,len(ya),ya.shape[1]))
 for _ in range(n):
  ar=[];br=[]
  for sse_a,sse_b,blocks,n_rows,n_zones in prepared:
   chosen=rng.integers(0,len(blocks),size=len(blocks));idx=np.concatenate([blocks[i] for i in chosen])[:n_rows]
   ar.append(float(np.sqrt(sse_a[idx].sum()/(len(idx)*n_zones))));br.append(float(np.sqrt(sse_b[idx].sum()/(len(idx)*n_zones))))
  dist.append(100*(np.mean(br)-np.mean(ar))/np.mean(br))
 arr=np.asarray(dist);return {"iterations":n,"seed":seed,"gain_percent_mean":float(arr.mean()),"CI95_lower":float(np.quantile(arr,.025)),"CI95_upper":float(np.quantile(arr,.975))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--project-root",type=Path,required=True);a=ap.parse_args();root=a.project_root.resolve()/"experiments/07_residual_distillation";audit=json.loads((root/"outputs/FORMAL_MATRIX_AUDIT.json").read_text(encoding="utf-8"))
 if audit.get("overall_verdict")!="PASS" or audit.get("metrics_unlocked") is not True:raise RuntimeError("metrics locked")
 manifest=pd.read_csv(root/"FORMAL_MATRIX_MANIFEST.csv",keep_default_na=False);cells=[];rows=[]
 for row in manifest.itertuples():
  success=json.loads(Path(row.expected_output).read_text(encoding="utf-8"));attempt=Path(success["receipt"]).parent
  with np.load(attempt/"predictions.npz",allow_pickle=False) as p:pred=p["prediction"]
  with np.load(attempt/"targets.npz",allow_pickle=False) as p:y=p["target"]
  mm=met(pred,y);cells.append({"branch":row.branch,"fold":int(row.outer_fold),"horizon":int(row.horizon),"prediction":pred,"target":y});rows.append({"branch":row.branch,"fold":int(row.outer_fold),"horizon":int(row.horizon),**mm})
 frame=pd.DataFrame(rows);macro=frame.groupby("branch")[list(MET)].mean().reset_index();folds=frame.groupby(["branch","fold"])[list(MET)].mean().reset_index();horizons=frame.groupby(["branch","horizon"])[list(MET)].mean().reset_index()
 pooled=[]
 for b in BR:
  group=[x for x in cells if x["branch"]==b];pooled.append({"branch":b,**met(np.concatenate([x["prediction"].reshape(-1) for x in group]),np.concatenate([x["target"].reshape(-1) for x in group]))})
 def val(df,b,col="RMSE"):return float(df[df.branch==b][col].iloc[0])
 gt=val(macro,"GT_only");kd=val(macro,"aligned_KD");sh=val(macro,"shuffled_teacher");gain=100*(gt-kd)/gt;gain_sh=100*(sh-kd)/sh
 fold_wins=sum(val(folds[(folds.fold==f)],"aligned_KD")<val(folds[(folds.fold==f)],"GT_only") for f in range(1,7));cell_wins=int(sum(frame[frame.branch=="aligned_KD"].sort_values(["fold","horizon"]).RMSE.to_numpy()<frame[frame.branch=="GT_only"].sort_values(["fold","horizon"]).RMSE.to_numpy()))
 sh_fold_wins=sum(val(folds[(folds.fold==f)],"aligned_KD")<val(folds[(folds.fold==f)],"shuffled_teacher") for f in range(1,7))
 hdet=[]
 for h in (3,6,9,12):
  k=val(horizons[horizons.horizon==h],"aligned_KD");g=val(horizons[horizons.horizon==h],"GT_only");hdet.append({"horizon":h,"deterioration_percent":100*(k-g)/g,"protected":100*(k-g)/g<=1})
 boot_gt=bootstrap(cells,"aligned_KD","GT_only",seed=20260829);boot_sh=bootstrap(cells,"aligned_KD","shuffled_teacher",seed=20260830)
 checks={"macro_gain_at_least_1_percent":gain>=1,"fold_wins_at_least_5_of_6":fold_wins>=5,"cell_wins_at_least_18_of_24":cell_wins>=18,"paired_24h_block_CI_lower_gt_0":boot_gt["CI95_lower"]>0,"horizon_protection":all(x["protected"] for x in hdet),"aligned_beats_shuffled_macro":kd<sh,"aligned_beats_shuffled_5_of_6_folds":sh_fold_wins>=5,"aligned_vs_shuffled_CI_lower_gt_0":boot_sh["CI95_lower"]>0}
 d1=all(checks.values());summary={"status":"complete","generated_at":datetime.now().isoformat(timespec="seconds"),"primary_endpoint":"equal_mean_of_24_cell_RMSE","macro_RMSE":{"GT_only":gt,"aligned_KD":kd,"shuffled_teacher":sh},"aligned_gain_vs_GT_percent":gain,"aligned_gain_vs_shuffled_percent":gain_sh,"fold_wins_vs_GT":int(fold_wins),"cell_wins_vs_GT":cell_wins,"fold_wins_vs_shuffled":int(sh_fold_wins),"horizon_deterioration":hdet,"bootstrap_vs_GT":boot_gt,"bootstrap_vs_shuffled":boot_sh,"checks":checks,"D1_pass":d1,"decision":"AUTHORIZE_D2" if d1 else "TERMINATE_DISTILL_V1","protected_test_access":False}
 out=root/"outputs";frame.to_csv(out/"D1_CELLS.csv",index=False,encoding="utf-8-sig");macro.to_csv(out/"D1_MACRO.csv",index=False,encoding="utf-8-sig");folds.to_csv(out/"D1_FOLDS.csv",index=False,encoding="utf-8-sig");horizons.to_csv(out/"D1_HORIZONS.csv",index=False,encoding="utf-8-sig");pd.DataFrame(pooled).to_csv(out/"D1_POOLED.csv",index=False,encoding="utf-8-sig");(out/"D1_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
