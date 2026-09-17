"""AP2 complete denominators, information/structure diagnostics and case evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_audit.event_comparison_ap1 import sign
from urbanev_audit.event_witnesses import pairwise_pattern,pattern_feasible
from urbanev_audit.event_relaxations import PairwiseLP
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,coo_matrix,vstack


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/research'/name)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','private','results'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=a.results;pilot=load('run_shared_missing_p0.py');oldsummary=load('summarize_shared_missing_ap1.py')
    f=pd.read_csv(r/'new_year_hierarchy.csv');nat=f[f['mask']=='natural'];art=f[f['mask']!='natural']
    groupkeys=['dataset','K','scope','mask']
    pd.DataFrame([{**dict(zip(groupkeys,ident)),**oldsummary.summary(g)} for ident,g in f.groupby(groupkeys)]).to_csv(r/'scope_summary.csv',index=False)
    nat[nat.joint_new].to_csv(r/'new_natural_decisions.csv',index=False)
    sets=pd.read_csv(r/'new_year_candidate_sets.csv').fillna('');ns=sets[sets['mask']=='natural']
    changes=ns[ns.joint_set_changed];changes.to_csv(r/'new_candidate_changes.csv',index=False)
    best=ns[(ns.independent_best=='')&(ns.joint_best!='')];advantages=[]
    for _,s in best.iterrows():
        group=nat[(nat.dataset==s.dataset)&(nat.series==s.series)&(nat.K==s.K)&(nat.scope==s.scope)&(nat.origin_start==s.origin_start)]
        for _,v in group.iterrows():
            if s.joint_best not in (v.model_A,v.model_B):continue
            gain=-v.joint_enclosure_upper if v.model_A==s.joint_best else v.joint_enclosure_lower
            if gain<=1e-10:raise AssertionError('Unique best lacks direct robust pairwise advantage')
            advantages.append({'series':s.series,'K':int(s.K),'scope':s.scope,'origin_start':int(s.origin_start),
                'winner':s.joint_best,'opponent':v.model_B if v.model_A==s.joint_best else v.model_A,'worst_advantage':gain})
    pd.DataFrame(advantages).to_csv(r/'unique_best_advantages.csv',index=False)
    pooled=[]
    for ident,g in nat[nat.scope=='annual'].groupby(['dataset','K','model_A','model_B']):
        item=dict(zip(['dataset','K','model_A','model_B'],ident));item['N']=int(g.N.sum())
        for method in ('independent','pairwise','joint'):
            lo='joint_enclosure_lower' if method=='joint' else method+'_lower';hi='joint_enclosure_upper' if method=='joint' else method+'_upper'
            item[method+'_lower']=float((g[lo]*g.N).sum()/g.N.sum());item[method+'_upper']=float((g[hi]*g.N).sum()/g.N.sum())
            item[method+'_direction']=sign(item[method+'_lower'],item[method+'_upper'])
        pooled.append(item)
    pf=pd.DataFrame(pooled);pf.to_csv(r/'pooled_annual_bounds.csv',index=False)
    artificial=[]
    for ident,g in art.groupby(['dataset','K','mask']):
        for method in ('independent','pairwise','joint','locf','identified'):
            if method in ('locf','identified'):
                score=pd.to_numeric(g[method+'_delta'],errors='coerce');directed=score.abs()>1e-10
                wrong=directed&(score*g.truth_delta<0)&(g.truth_delta.abs()>1e-10)
                contained=None
            else:
                direction=g[method+'_direction'];directed=direction.isin(['A','B'])
                wrong=directed&(((direction=='A')&(g.truth_delta>1e-10))|((direction=='B')&(g.truth_delta< -1e-10)))
                lo='joint_enclosure_lower' if method=='joint' else method+'_lower';hi='joint_enclosure_upper' if method=='joint' else method+'_upper'
                contained=int(((g.truth_delta>=g[lo]-1e-10)&(g.truth_delta<=g[hi]+1e-10)).sum())
            artificial.append(dict(zip(['dataset','K','mask'],ident),method=method,comparisons=len(g),directed=int(directed.sum()),
                wrong_direction=int(wrong.sum()),undirected=int((~directed).sum()),truth_contained=contained))
    af=pd.DataFrame(artificial);af.to_csv(r/'artificial_direction_tradeoff.csv',index=False)
    c=json.loads((ROOT/'configs/research/SHARED_MISSING_EVENTS_AP2_20260917.json').read_text());data,_,_=pilot.read_data(a.data,c)
    cases=[]
    for _,row in nat[nat.higher_order_new].iterrows():
        k,ell,s,n=map(int,(row.K,row.L,row.origin_start,row.N));dc=c['datasets'][row.dataset]
        o=pilot.observed_bits(data[row.dataset,row.series][0][s:s+n+k-1],dc['threshold']);lp=PairwiseLP(o,k,ell)
        with np.load(a.private/f'{row.dataset}_{row.series}_K{k}_predictions.npz') as z:
            offset=s-dc['calibration_end'];pa=z[row.model_A][offset:offset+n];pb=z[row.model_B][offset:offset+n]
        side='upper' if row.joint_direction=='A' else 'lower';pattern=pairwise_pattern(lp,2*(pb-pa),side)
        required=np.full(n,-1,np.int8);required[lp.ambiguous]=pattern[lp.ambiguous]
        assert not pattern_feasible(o,required,k,ell)
        for i in lp.ambiguous:
            prior=required[i];required[i]=-1
            if pattern_feasible(o,required,k,ell):required[i]=prior
        core=[{'origin_offset':int(i),'required_label':int(required[i])} for i in np.flatnonzero(required!=-1)]
        proper_pairs=True
        from itertools import combinations
        for pair in combinations(core,2):
            two=np.full(n,-1,np.int8)
            for item in pair:two[item['origin_offset']]=item['required_label']
            proper_pairs &= pattern_feasible(o,two,k,ell)
        positive=[v['origin_offset'] for v in core if v['required_label']==1]
        negative=[v['origin_offset'] for v in core if v['required_label']==0]
        cover=False;cut_bound=None
        if len(positive)==1 and len(negative)==2:
            m=k-ell+1;possible=set(range(positive[0],positive[0]+m))
            covered=set().union(*(set(range(v,v+m)) for v in negative));cover=possible<=covered
            if cover:
                # This is an exploratory single-case explanatory cut, not a new baseline.
                active=len(lp.lp_indices);mvars=len(lp.ambiguous)
                column_map=np.r_[lp.lp_indices,np.arange(mvars,mvars+lp.variable_count-active)]
                old=lp.matrix.tocoo();size=mvars+lp.variable_count-active
                expanded=coo_matrix((old.data,(old.row,column_map[old.col])),shape=(old.shape[0],size)).tocsr()
                cut=np.zeros(size)
                for origin,value in [(positive[0],1)]+[(v,-1) for v in negative]:
                    q=int(np.searchsorted(lp.ambiguous,origin));assert lp.ambiguous[q]==origin
                    cut[q]=value
                matrix=vstack([expanded,csr_matrix(cut.reshape(1,-1))]);coef=2*(pb-pa);objective=np.zeros(size)
                objective[:mvars]=-coef[lp.ambiguous]
                sol=linprog(objective,A_ub=matrix,b_ub=np.zeros(matrix.shape[0]),bounds=(0,1),method='highs')
                assert sol.success
                cut_bound=float((np.sum(pa*pa-pb*pb+coef*lp.lo)-sol.fun)/n)
        missing=np.flatnonzero(o==-1);runs=np.split(missing,np.flatnonzero(np.diff(missing)>1)+1)
        cases.append({'dataset':row.dataset,'series':row.series,'K':k,'L':ell,'origin_start':s,'time_start':row.time_start,'N':n,
            'model_A':row.model_A,'model_B':row.model_B,'missing_records':len(missing),'support_records':len(o),
            'longest_missing_run':max(map(len,runs)),'independent':[row.independent_lower,row.independent_upper],
            'pairwise':[row.pairwise_lower,row.pairwise_upper],'joint':[row.joint_lower,row.joint_upper],
            'endpoint':side,'deletion_minimal_core':core,'minimum_cardinality_proved':len(core)==3 and proper_pairs,
            'all_pairs_feasible':bool(proper_pairs),'three_window_run_cover':cover,
            'single_explanatory_cut_upper_numeric':cut_bound,'single_cut_is_posthoc_diagnostic':True})
    pilot.dump(r/'higher_order_case.json',cases)
    struct=[]
    for phase in ('ap1','ap2'):
        sp=pd.read_csv(r/f'{phase}_panel_structure.csv');sp=sp[sp.components>0]
        for name,g in sp.groupby('structure_status'):struct.append({'phase':phase,'status':name,'panels':len(g),'ambiguous_panels':len(sp)})
    pd.DataFrame(struct).to_csv(r/'ambiguous_panel_structure_summary.csv',index=False)
    summary={'natural':oldsummary.summary(nat),'weekly':oldsummary.summary(nat[nat.scope=='origin_week']),
        'annual':oldsummary.summary(nat[nat.scope=='annual']),'artificial':oldsummary.summary(art),
        'candidate_changes':len(changes),'new_unique_best':len(best),'higher_order_candidate_changes':int(ns.higher_order_set_changed.sum()),
        'new_decision_time_clusters':int(nat[nat.joint_new&(nat.scope=='origin_week')].cluster.nunique()),
        'pooled_annual_new_directions':int((~pf.independent_direction.isin(['A','B'])&pf.joint_direction.isin(['A','B'])).sum()),
        'artificial_by_method':af.groupby('method')[['comparisons','directed','wrong_direction','undirected']].sum().to_dict('index')}
    pilot.dump(r/'summary.json',summary)
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    x=np.arange(4);width=.35
    for offset,phase in [(-width/2,'ap1'),(width/2,'ap2')]:
        frame=pd.read_csv(ROOT/f'artifacts/summaries/shared_missing_events_{phase}/new_year_hierarchy.csv')
        g=frame[(frame['mask']=='natural')&(frame.scope=='origin_week')].groupby(['dataset','K'])
        counts=[int(g.get_group(key).joint_new.sum()) for key in [('beijing',24),('beijing',72),('household',60),('household',180)]]
        ax[0].bar(x+offset,counts,width,label=phase.upper())
    ax[0].set_xticks(x,['Beijing 24','Beijing 72','Power 60','Power 180']);ax[0].set_ylabel('New strict weekly comparisons');ax[0].legend()
    totals=af.groupby('method')[['directed','wrong_direction']].sum().reindex(['independent','pairwise','joint','locf','identified'])
    ax[1].bar(np.arange(5),totals.directed,label='Directed');ax[1].bar(np.arange(5),totals.wrong_direction,label='Wrong direction',color='#b84735')
    ax[1].set_xticks(np.arange(5),['Independent','Pairwise','Joint','LOCF','Identified'],rotation=20);ax[1].set_ylabel('Artificial comparisons (234 total)');ax[1].legend()
    fig.tight_layout();fig.savefig(ROOT/'docs/reports/figures/shared_missing_ap2_replication.png',dpi=180);plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
