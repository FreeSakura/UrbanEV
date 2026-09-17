"""Frozen AP1/AP2 cover validation. No fitting or prediction generation.

The solver represents ALL unary and two-point covers by exact finite constraint
generation, not just the one diagnostic cover per label. A returned optimum is
accepted only after a complete separation scan finds no violated cover. Unary-
redundant covers follow from box nonnegativity. Auxiliary empirical endpoints
are numerical solver results, not claims of symbolic optimality.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import itertools
import math

import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from urbanev_audit.event_cover_geometry import EventCoverGeometry
from urbanev_audit.persistent_events import event_labels, paired_bounds

SEPARATION_TOL=1e-9


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


class AllCovers:
    def __init__(self,g):
        self.g=g;self.a=g.ambiguous;self.m=len(self.a)
        self.rows=set();self.specs=[];self.unary_count=0;self.two_count=0
        for j,origin in enumerate(self.a):
            _,left,right=g.witnesses(origin)
            order=np.argsort(left,kind='stable');left=left[order];right=right[order]
            first=int(left.max());last=int(right.min())
            il=int(np.searchsorted(self.a,first));ir=int(np.searchsorted(self.a,last,side='right'))
            for i in range(il,ir):
                if i!=j:self.rows.add((j,i));self.unary_count+=1
            # An irreducible two-cover must be on opposite sides of the whole
            # implication interval. For each left point i, every witness not
            # hit by i must extend to k. Hence k <= min(right[left > i]).
            starts=np.arange(np.searchsorted(self.a,left.min()),il,dtype=int)
            suffix=np.minimum.accumulate(right[::-1])[::-1]
            pos=np.searchsorted(left,self.a[starts],side='right')
            ends=np.searchsorted(self.a,suffix[pos],side='right')-1
            keep=ends>=ir;starts=starts[keep];ends=ends[keep]
            if len(starts):
                self.specs.append((j,starts,ir,ends))
                self.two_count+=int(np.sum(ends-ir+1))
        self.last_max_violation=0.

    def matrix(self):
        rows=sorted(self.rows);rr=[];cc=[];vv=[]
        for q,row in enumerate(rows):
            rr.extend([q]*len(row));cc.extend(row);vv.extend([1.]+[-1.]*(len(row)-1))
        return coo_matrix((vv,(rr,cc)),shape=(len(rows),self.m)).tocsc()

    def separate(self,x):
        out=[];maxv=0.
        for j,ii,start,ends in self.specs:
            values=x[start:int(ends.max())+1]
            mins=np.minimum.accumulate(values)
            # Stable leftmost argmin for each prefix.
            changed=np.r_[True,values[1:]<mins[:-1]]
            arg=np.maximum.accumulate(np.where(changed,np.arange(len(values)),-1))
            rhs=x[ii]+mins[ends-start];q=int(np.argmin(rhs))
            violation=float(x[j]-rhs[q]);maxv=max(maxv,violation)
            if violation>SEPARATION_TOL:
                out.append((j,int(ii[q]),start+int(arg[ends[q]-start])))
        self.last_max_violation=maxv
        return out

    def optimize(self,c,integer=False):
        c=np.asarray(c,float);started=time.perf_counter()
        if not self.m:return {'value':0.,'x':np.empty(0),'rounds':0,'status':'NO_AMBIGUITY','max_violation':0.,'gap':0.}
        for iteration in range(1000):
            matrix=self.matrix()
            if integer:
                result=milp(c,integrality=np.ones(self.m),bounds=Bounds(0,1),
                    constraints=LinearConstraint(matrix,-np.inf,0) if matrix.shape[0] else None,
                    options={'mip_rel_gap':0.,'time_limit':120.})
            else:
                result=linprog(c,A_ub=matrix if matrix.shape[0] else None,
                    b_ub=np.zeros(matrix.shape[0]) if matrix.shape[0] else None,
                    bounds=(0,1),method='highs',options={'time_limit':120.,
                    'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            if not result.success:raise RuntimeError(f'{"ILP" if integer else "LP"}: {result.message}')
            violated=self.separate(result.x)
            if not violated:
                primal=float(np.max(matrix@result.x,initial=0.))
                if primal>2e-8:raise AssertionError('Explicit row violation')
                if integer:
                    if np.max(np.abs(result.x-np.rint(result.x)))>1e-7:raise AssertionError('Noninteger MILP solution')
                    gap=float(result.fun-result.mip_dual_bound)
                else:
                    # A valid Lagrangian box lower bound for all original covers:
                    # any nonnegative multipliers on the generated subset work.
                    lam=np.maximum(0.,-result.ineqlin.marginals)
                    residual=c+matrix.T@lam
                    dual=float(np.minimum(residual,0).sum())
                    gap=float(result.fun-dual)
                return {'value':float(result.fun),'x':result.x,'rounds':iteration+1,
                    'status':'ALL_COVERS_OPTIMAL_NUMERIC','max_violation':max(primal,self.last_max_violation),
                    'gap':gap,'seconds':time.perf_counter()-started}
            before=len(self.rows);self.rows.update(violated)
            if len(self.rows)==before:raise RuntimeError('Separation stalled on existing rows')
        raise RuntimeError('Constraint-generation iteration limit')


def brute(o,k,l):
    missing=np.flatnonzero(o==-1);out=set()
    for assignment in itertools.product((0,1),repeat=len(missing)):
        z=o.copy();z[missing]=assignment
        out.add(tuple(int(any(all(z[r:r+l]) for r in range(s,s+k-l+1))) for s in range(len(o)-k+1)))
    return out


def selfcheck(output):
    rng=np.random.default_rng(17092026);counts={'layouts':0,'objectives':0,'integer_patterns':0}
    cases=[(np.full(10,-1,np.int8),6,3),(np.array([-1,1,1,-1,0,-1,1,1,-1],np.int8),5,3)]
    for _ in range(100):
        l=int(rng.integers(1,4));k=int(rng.integers(l,2*l+3));n=int(rng.integers(1,7))
        o=rng.choice([-1,0,1],size=k+n-1).astype(np.int8)
        if (o==-1).sum()<=9:cases.append((o,k,l))
    for o,k,l in cases:
        g=EventCoverGeometry(o,k,l);solver=AllCovers(g);m=solver.m
        explicit=[];coverlist=[]
        for j in range(m):
            for size in (1,2):
                for cov in itertools.combinations([i for i in range(m) if i!=j],size):
                    if g.cover_valid(int(solver.a[j]),solver.a[list(cov)]):
                        row=np.zeros(m);row[j]=1;row[list(cov)]-=1;explicit.append(row);coverlist.append((j,*cov))
        for x in itertools.product((0.,.5,1.),repeat=m):
            x=np.asarray(x);actual=max([0.]+[float(np.dot(row,x)) for row in explicit])
            solver.separate(x)
            unary=max([0.]+[x[row[0]]-x[row[1]] for row in solver.rows if len(row)==2])
            assert abs(max(unary,solver.last_max_violation)-max(0.,actual))<1e-12
            counts['integer_patterns']+=int(np.all((x==0)|(x==1)))
        if not m:counts['layouts']+=1;continue
        for _ in range(3):
            c=rng.normal(size=m)
            dense=linprog(c,A_ub=np.array(explicit) if explicit else None,b_ub=np.zeros(len(explicit)) if explicit else None,bounds=(0,1),method='highs')
            lp=solver.optimize(c);ip=solver.optimize(c,True)
            assert abs(lp['value']-dense.fun)<1e-8
            if k>=2*l-1:
                patterns=brute(o,k,l);truth=min(np.dot(c,np.array(y)[solver.a]) for y in patterns)
                assert abs(ip['value']-truth)<1e-8
            counts['objectives']+=1
        counts['layouts']+=1
    g=EventCoverGeometry(np.full(10,-1,np.int8),6,3);s=AllCovers(g);c=np.array([1,-1,1,-1,1.])
    lp=s.optimize(c);ip=s.optimize(c,True)
    assert abs(lp['value']+.5)<1e-10 and abs(ip['value'])<1e-10
    counter={'K':6,'L':3,'T':10,'binary_patterns':len(brute(g.o,6,3)),
        'all_irreducible_two_covers':s.two_count,'LP_min_linear':lp['value'],'ILP_min_linear':ip['value'],
        'LP_min_Brier':(.25+lp['value'])/5,'ILP_min_Brier':(.25+ip['value'])/5,
        'LP_x':lp['x'].tolist(),'meaning':'Relaxation loses strict direction; it does not certify the opposite model.'}
    dump(output/'selfcheck.json',{'status':'PASS',**counts,'counterexample':counter})
    print(json.dumps({'selfcheck':'PASS',**counts}),flush=True)


def direction(low,high):
    return 'A' if high< -1e-10 else ('B' if low>1e-10 else 'UNRESOLVED')


def run(output,data_root,private_dirs,result_dirs,strict_frozen_hashes=False,limit=None):
    started=time.perf_counter()
    spec=importlib.util.spec_from_file_location('pilot',REPO/'scripts/research/run_shared_missing_p0.py')
    pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
    configs={p:json.loads((REPO/f'configs/research/SHARED_MISSING_EVENTS_{p}_202609{16 if p=="AP1" else 17}.json').read_text()) for p in ['AP1','AP2']}
    data,_,sources=pilot.read_data(data_root,configs['AP2'])
    dump(output/'data_sources.json',sources)
    source_paths=[REPO/'src/urbanev_audit'/n for n in ['event_cover_geometry.py','persistent_events.py']]
    source_paths +=[Path(__file__),REPO/'PROOF_PACKAGE.md']
    source_paths +=[REPO/f'configs/research/SHARED_MISSING_EVENTS_{p}_202609{16 if p=="AP1" else 17}.json' for p in configs]
    for phase in configs:
        source_paths.append(result_dirs[phase]/'new_year_hierarchy.csv')
        source_paths.extend(sorted(private_dirs[phase].glob('*_predictions.npz')))
    hashes={str(p):sha(p) for p in source_paths};dump(output/'input_hashes_before.json',hashes)
    allrows=[];panelrows=[];witness_checks=0;predcache={};max_cache_error=0.;geometry_old=pd.read_csv(REPO/'artifacts/summaries/persistent_event_structure_20260917/frozen_panel_geometry.csv')
    keys=['dataset','series','K','L','scope','origin_start','N']
    old=geometry_old.set_index(['phase',*keys])
    for phase,cfg in configs.items():
        frame=pd.read_csv(result_dirs[phase]/'new_year_hierarchy.csv')
        frame=frame[frame['mask']=='natural']
        witnesses={}
        if strict_frozen_hashes:
            for line in (REPO/f'artifacts/summaries/shared_missing_events_ap2/{phase.lower()}_completion_witnesses.jsonl').read_text().splitlines():
                w=json.loads(line);witnesses[(w['comparison_id'],w['endpoint'])]=w
        for ident,group in frame.groupby(keys,sort=True):
            if limit is not None and len(panelrows)>=limit:break
            dataset,series,k,l,scope,start,n=ident;k,l,start,n=map(int,(k,l,start,n))
            meta=dict(zip(keys,ident));meta.update(phase=phase,K=k,L=l,origin_start=start,N=n)
            o=pilot.observed_bits(data[dataset,series][0][start:start+n+k-1],cfg['datasets'][dataset]['threshold'])
            assert len(o)==n+k-1
            geom=EventCoverGeometry(o,k,l);diagnosis=geom.diagnose();assert diagnosis['long_window_regime']
            assert diagnosis['status']==old.loc[(phase,*ident)].geometry_status
            solver=AllCovers(geom);m=solver.m;clock=time.perf_counter()
            cachekey=phase,dataset,series,k
            if cachekey not in predcache:
                path=private_dirs[phase]/f'{dataset}_{series}_K{k}_predictions.npz'
                with np.load(path) as pack:predcache[cachekey]={key:pack[key] for key in pack.files}
            offset=start-cfg['datasets'][dataset]['calibration_end']
            assert offset>=0
            pred={key:array[offset:offset+n] for key,array in predcache[cachekey].items()}
            for _,row in group.iterrows():
                a,b=pred[row.model_A],pred[row.model_B]
                assert len(a)==n and len(b)==n and np.isfinite(a).all() and np.isfinite(b).all()
                assert ((a>=0)&(a<=1)&(b>=0)&(b<=1)).all()
                coef=2*(b-a);base=float(np.sum(a*a-b*b+coef*geom.lo));w=coef[geom.ambiguous]
                rec={**meta,'model_A':row.model_A,'model_B':row.model_B,'ambiguous_labels':m,'geometry_status':diagnosis['status']}
                independent=[(base+np.minimum(w,0).sum())/n,(base+np.maximum(w,0).sum())/n]
                if m:
                    dp=paired_bounds(o,a,b,k,l);dp_values=[dp['lower_sum']/n,dp['upper_sum']/n]
                else:dp_values=[base/n,base/n]
                for s,side in enumerate(['lower','upper']):
                    rec['independent_'+side]=float(independent[s]);rec['implication_'+side]=float(row['pairwise_'+side]);rec['DP_'+side]=float(dp_values[s])
                    cache_error=max(abs(dp_values[s]-row['joint_'+side]),abs(independent[s]-row['independent_'+side]))
                    max_cache_error=max(max_cache_error,cache_error)
                    if cache_error>1e-8:raise AssertionError(f'Cache alignment failed {meta}: {cache_error}')
                    if not m:
                        for method in ['cover_LP','cover_ILP']:rec[method+'_'+side]=base/n
                        continue
                    rid=f'{phase}_{dataset}_{series}_K{k}_{scope}_{start}_{row.model_A}_{row.model_B}'
                    if strict_frozen_hashes:
                        oldw=witnesses[(rid,side)]
                        assert oldw['observed_bits_sha256']==hashlib.sha256(o.tobytes()).hexdigest()
                        assert oldw['predictions_sha256']==hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest()
                        witness_checks+=1
                    objective=w if side=='lower' else -w
                    for method,integer in [('cover_LP',False),('cover_ILP',True)]:
                        result=solver.optimize(objective,integer)
                        value=(base+(result['value'] if side=='lower' else -result['value']))/n
                        rec[method+'_'+side]=float(value)
                        rec[method+'_'+side+'_rounds']=result['rounds']
                        rec[method+'_'+side+'_max_violation']=result['max_violation']
                        rec[method+'_'+side+'_primal_dual_gap_per_N']=result['gap']/n
                        fractional=float(np.max(np.abs(result['x']-np.rint(result['x'])),initial=0.))
                        rec[method+'_'+side+'_fractionality']=fractional
                        if integer:
                            labels=geom.lo.copy();labels[geom.ambiguous]=np.rint(result['x']).astype(np.int8)
                            z=geom.realize(labels)
                            assert z is not None and np.array_equal(z[o!=-1],o[o!=-1])
                            assert np.array_equal(event_labels(z,k,l),labels)
                            check=float(np.mean(a*a-b*b+coef*labels))
                            if abs(check-value)>1e-8 or abs(value-dp_values[s])>1e-8:raise AssertionError('ILP replay/DP mismatch')
                for method in ['independent','implication','cover_LP','cover_ILP','DP']:
                    rec[method+'_direction']=direction(rec[method+'_lower'],rec[method+'_upper'])
                rec['cover_LP_width_gap']=(rec['cover_LP_upper']-rec['cover_LP_lower'])-(rec['DP_upper']-rec['DP_lower'])
                rec['cover_LP_endpoint_gap']=max(rec['DP_lower']-rec['cover_LP_lower'],rec['cover_LP_upper']-rec['DP_upper'])
                rec['implication_width_gap']=(rec['implication_upper']-rec['implication_lower'])-(rec['DP_upper']-rec['DP_lower'])
                rec['cover_ILP_DP_max_error']=max(abs(rec['cover_ILP_'+s]-rec['DP_'+s]) for s in ['lower','upper'])
                for low,high in [('independent','implication'),('implication','cover_LP'),('cover_LP','cover_ILP')]:
                    assert rec[low+'_lower']<=rec[high+'_lower']+1e-8 and rec[low+'_upper']>=rec[high+'_upper']-1e-8
                allrows.append(rec)
            panelrows.append({**meta,'ambiguous_labels':m,'geometry_status':diagnosis['status'],
                'unary_constraints':solver.unary_count,'irreducible_two_covers':solver.two_count,
                'generated_constraints':len(solver.rows),'seconds':time.perf_counter()-clock})
            if len(panelrows)%25==0:
                pd.DataFrame(allrows).to_csv(output/'comparison_results.csv',index=False)
                pd.DataFrame(panelrows).to_csv(output/'panel_results.csv',index=False)
                print(json.dumps({'panels':len(panelrows),'comparisons':len(allrows),'last':meta,'seconds':round(time.perf_counter()-started,1)}),flush=True)
        print(json.dumps({'phase_complete':phase,'panels':len(panelrows)}),flush=True)
    f=pd.DataFrame(allrows);p=pd.DataFrame(panelrows)
    f.to_csv(output/'comparison_results.csv',index=False);p.to_csv(output/'panel_results.csv',index=False)
    hashes_after={str(path):sha(path) for path in source_paths};assert hashes==hashes_after
    dump(output/'input_hashes_after.json',hashes_after)
    summary={'status':'COMPLETE' if limit is None else 'PILOT_ONLY','panels':len(p),'comparisons':len(f),
        'ambiguous_panels':int((p.ambiguous_labels>0).sum()),'ambiguous_comparisons':int((f.ambiguous_labels>0).sum()),
        'all_cover_ILP_DP_max_error':float(f.cover_ILP_DP_max_error.max()),'cached_DP_independent_max_error':max_cache_error,
        'strict_frozen_hashes':strict_frozen_hashes,'old_witness_hash_matches':witness_checks,'input_files_unchanged':len(hashes),'seconds':time.perf_counter()-started,
        'training_runs':0,'new_predictions':0,'analysis_role':'Post-hoc exhaustive frozen natural-panel diagnostic; not new independent data',
        'cover_family':'All unary and two-point covers, generated to complete separation; redundant covers implied by unary and nonnegativity',
        'solver_tolerances':{'separation':SEPARATION_TOL,'LP_primal_dual':1e-9,'MIP_relative_gap':0.,'direction':1e-10},
        'gap_counts':{str(t):{'LP_DP_comparisons':int((f.cover_LP_width_gap>t).sum()),
            'implication_DP_comparisons':int((f.implication_width_gap>t).sum())} for t in [1e-10,1e-9,1e-8]},
        'LP_DP_direction_differences':int((f.cover_LP_direction!=f.DP_direction).sum()),
        'ILP_DP_direction_differences':int((f.cover_ILP_direction!=f.DP_direction).sum()),
        'implication_DP_direction_differences':int((f.implication_direction!=f.DP_direction).sum())}
    dump(output/'STRUCTURE_VALIDATION.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selfcheck',action='store_true');parser.add_argument('--limit',type=int)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--ap1-private',type=Path);parser.add_argument('--ap2-private',type=Path)
    parser.add_argument('--ap1-results',type=Path,default=REPO/'artifacts/summaries/shared_missing_events_ap1')
    parser.add_argument('--ap2-results',type=Path,default=REPO/'artifacts/summaries/shared_missing_events_ap2')
    parser.add_argument('--strict-frozen-hashes',action='store_true',help='Require the original archived forecast/observation endpoint hashes')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):parser.error('Choose a new empty output directory; frozen evidence is not overwritten')
    if not args.selfcheck and any(x is None for x in [args.data,args.ap1_private,args.ap2_private]):parser.error('--data, --ap1-private and --ap2-private are required for natural validation')
    args.output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        if args.selfcheck:selfcheck(args.output)
        else:run(args.output,args.data,{'AP1':args.ap1_private,'AP2':args.ap2_private},
            {'AP1':args.ap1_results,'AP2':args.ap2_results},args.strict_frozen_hashes,args.limit)
