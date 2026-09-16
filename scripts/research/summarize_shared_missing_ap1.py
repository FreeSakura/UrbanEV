"""Produce AP1 denominators, candidate decisions, concentration and morphology."""
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
from urbanev_audit.event_comparison_ap1 import sign,origin_blocks


def summary(group):
    n=len(group);u=int(group.independent_undirected.sum());j=int(group.joint_new.sum())
    return {'comparisons':n,'ambiguous_comparisons':int((group.ambiguous_windows>0).sum()),
            'independent_undirected':u,'pairwise_new':int(group.pairwise_new.sum()),'joint_new':j,
            'higher_order_new':int(group.higher_order_new.sum()),'joint_new_fraction_all':j/n,
            'joint_new_fraction_independent_undirected':j/u if u else None,
            'higher_order_tighter':int(((group.pairwise_upper-group.pairwise_lower)-(group.joint_upper-group.joint_lower)>1e-10).sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--results',type=Path,default=ROOT/'artifacts/summaries/shared_missing_events_ap1');a=p.parse_args()
    r=a.results
    old=pd.read_csv(r/'ap0_hierarchy.csv');new=pd.read_csv(r/'new_year_hierarchy.csv');allrows=pd.concat([old,new],ignore_index=True)
    grouped=[]
    for ident,group in allrows.groupby(['phase','dataset','K','scope','mask']):
        grouped.append(dict(zip(['phase','dataset','K','scope','mask'],ident),**summary(group)))
    pd.DataFrame(grouped).to_csv(r/'scope_summary.csv',index=False)
    by_quarter=[]
    for ident,group in new[new.scope=='origin_week'].groupby(['dataset','K','quarter']):
        by_quarter.append(dict(zip(['dataset','K','quarter'],ident),**summary(group)))
    pd.DataFrame(by_quarter).to_csv(r/'quarter_summary.csv',index=False)
    categories=allrows.groupby(['phase','dataset','K','scope','category']).size().reset_index(name='comparisons')
    categories.to_csv(r/'decision_categories.csv',index=False)
    new[(new['mask']=='natural') & new.joint_new].to_csv(r/'new_natural_decisions.csv',index=False)
    costs=new.groupby(['dataset','K','scope','mask','series','origin_start']).agg(
        pair_prepare=('pair_preprocess_seconds','first'),pair_solve=('pair_solve_seconds','sum'),joint=('joint_seconds','sum')).reset_index()
    costs['pair_total']=costs.pair_prepare+costs.pair_solve
    costs.to_csv(r/'panel_algorithm_costs.csv',index=False)
    costs[costs['mask']=='natural'].groupby(['dataset','K','scope']).agg(
        pair_median_seconds=('pair_total','median'),joint_median_seconds=('joint','median'),panels=('joint','size')).to_csv(r/'algorithm_cost_summary.csv')
    sets=pd.concat([pd.read_csv(r/'ap0_candidate_sets.csv'),pd.read_csv(r/'new_year_candidate_sets.csv')],ignore_index=True).fillna('')
    candidate_summary=[]
    for ident,group in sets.groupby(['phase','dataset','K','scope','mask']):
        candidate_summary.append(dict(zip(['phase','dataset','K','scope','mask'],ident),panels=len(group),
          pairwise_set_changes=int(group.pairwise_set_changed.sum()),joint_set_changes=int(group.joint_set_changed.sum()),
          higher_order_set_changes=int(group.higher_order_set_changed.sum()),
          newly_confirmed_best=int(((group.independent_best=='')&(group.joint_best!='')).sum())))
    pd.DataFrame(candidate_summary).to_csv(r/'candidate_summary.csv',index=False)
    pooled=[]
    for ident,group in new[new.scope=='annual'].groupby(['dataset','K','model_A','model_B']):
        record=dict(zip(['dataset','K','model_A','model_B'],ident),N=int(group.N.sum()),series_count=len(group))
        for method in ['independent','pairwise','joint']:
            lo='joint_enclosure_lower' if method=='joint' else method+'_lower'
            hi='joint_enclosure_upper' if method=='joint' else method+'_upper'
            record[method+'_lower']=float((group[lo]*group.N).sum()/group.N.sum())
            record[method+'_upper']=float((group[hi]*group.N).sum()/group.N.sum())
            record[method+'_direction']=sign(record[method+'_lower'],record[method+'_upper'])
        pooled.append(record)
    pd.DataFrame(pooled).to_csv(r/'pooled_annual_bounds.csv',index=False)
    artificial=[]
    for ident,group in new[new.scope=='artificial'].groupby(['dataset','K','mask']):
        artificial.append(dict(zip(['dataset','K','mask'],ident),comparisons=len(group),
           independent_truth_violations=int(((group.truth_delta<group.independent_lower-1e-10)|(group.truth_delta>group.independent_upper+1e-10)).sum()),
           pairwise_truth_violations=int(((group.truth_delta<group.pairwise_lower-1e-10)|(group.truth_delta>group.pairwise_upper+1e-10)).sum()),
           joint_truth_violations=int(group.truth_contained.eq(False).sum()),
           locf_wrong_directions=int(group.locf_wrong.eq(True).sum()),identified_wrong_directions=int(group.identified_wrong.eq(True).sum()),
           identified_evaluable=int(group.identified_delta.notna().sum()),joint_abstentions=int((~group.joint_direction.isin(['A','B'])).sum())))
    pd.DataFrame(artificial).to_csv(r/'artificial_comparison.csv',index=False)
    spec=importlib.util.spec_from_file_location('ap0',ROOT/'scripts/research/run_shared_missing_p0.py')
    ap0=importlib.util.module_from_spec(spec);spec.loader.exec_module(ap0)
    config=json.loads((ROOT/'configs/research/SHARED_MISSING_EVENTS_AP1_20260916.json').read_text())
    series,_,_=ap0.read_data(a.data,config);morph=[];shared=[];masks={}
    for (dataset,key),(values,dates) in series.items():
        dc=config['datasets'][dataset];begin=dc['calibration_end'];mask=np.isnan(values[begin:]);masks[dataset,key]=mask
        boundaries=np.diff(np.r_[False,mask,False].astype(int));lengths=np.flatnonzero(boundaries==-1)-np.flatnonzero(boundaries==1)
        morph.append({'dataset':dataset,'series':key,'evaluation_records':len(mask),'missing':int(mask.sum()),'missing_rate':float(mask.mean()),
                      'missing_runs':len(lengths),'max_missing_run':int(lengths.max()) if len(lengths) else 0,
                      'median_missing_run':float(np.median(lengths)) if len(lengths) else 0.})
        for k in dc['windows']:
            blocks=list(origin_blocks(begin,len(values),k,dc['panel_length']))
            for i,(s,b,end) in enumerate(blocks):
                overlap=0 if i+1==len(blocks) else int(np.isnan(values[blocks[i+1][0]:end]).sum())
                shared.append({'dataset':dataset,'series':key,'K':k,'origin_start':s,'N':b-s,'support_end_exclusive':end,
                               'shared_missing_with_next_block':overlap})
    pd.DataFrame(morph).to_csv(r/'missing_morphology.csv',index=False);pd.DataFrame(shared).to_csv(r/'origin_support_and_overlap.csv',index=False)
    beijing=np.stack([m for (dataset,key),m in sorted(masks.items()) if dataset=='beijing'])
    values,counts=np.unique(beijing.sum(axis=0),return_counts=True)
    pd.DataFrame({'simultaneously_missing_stations':values,'hour_count':counts}).to_csv(r/'beijing_missing_cooccurrence.csv',index=False)
    clusters=[]
    for ident,group in new[(new['mask']=='natural')&(new.scope=='origin_week')].groupby(['dataset','time_start']):
        clusters.append({'dataset':ident[0],'time_start':ident[1],'comparisons':len(group),'joint_new':int(group.joint_new.sum()),
                         'affected_series':';'.join(sorted(set(group.loc[group.joint_new,'series']))),
                         'higher_order_new':int(group.higher_order_new.sum())})
    pd.DataFrame(clusters).to_csv(r/'time_clusters.csv',index=False)
    natural=new[new['mask']=='natural'];local=natural[natural.scope=='origin_week'];annual=natural[natural.scope=='annual']
    receipt={'status':'COMPLETE','ap0_natural':summary(old),'ap1_natural':summary(natural),'ap1_local':summary(local),'ap1_annual':summary(annual),
             'ap1_artificial':summary(new[new.scope=='artificial']),
             'new_local_time_clusters':sum(x['joint_new']>0 for x in clusters),'local_new_candidate_sets':int(sets[(sets.phase=='AP1')&(sets.scope=='origin_week')].joint_set_changed.sum()),
             'generic_compilation_state_identity':'See compiler_benchmark.csv and language tests',
             'source_annual_pooling':'Weighted station annual endpoints, never sums of weekly endpoints',
             'ap2_scope_used':False}
    (r/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    figs=ROOT/'docs/reports/figures'
    labels=['Beijing K24','Beijing K72','Power K60','Power K180'];groups=[local[(local.dataset==d)&(local.K==k)] for d,k in [('beijing',24),('beijing',72),('household',60),('household',180)]]
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4)
    pvals=[g.pairwise_new.sum() for g in groups];jvals=[g.higher_order_new.sum() for g in groups]
    ax.bar(x,pvals,color='#247b87',label='New decisions from pairwise LP');ax.bar(x,jvals,bottom=pvals,color='#de873b',label='Additional decisions from full DP')
    for i,g in enumerate(groups):ax.text(i,pvals[i]+.15,f"{int(pvals[i])}/{len(g)}\nundirected: {int(g.independent_undirected.sum())}",ha='center',fontsize=9)
    ax.set_xticks(x,labels);ax.set_ylim(0,7);ax.set_ylabel('Additional strict pairwise directions');ax.set_title('AP1 natural missingness: all fixed origin-week panels');ax.legend(loc='upper left',fontsize=8)
    fig.tight_layout();fig.savefig(figs/'shared_missing_ap1_decision_hierarchy.png',dpi=180);plt.close(fig)
    benchmark=pd.read_csv(r/'algorithm_benchmark.csv');benchmark['ratio']=benchmark.generic_dfa_same_kernel_seconds/benchmark.specialized_state_kernel_seconds
    fig,ax=plt.subplots(figsize=(7,4));group=benchmark.groupby('K').ratio
    middle=group.median();ax.plot(middle.index,middle,'o-',color='#247b87');ax.fill_between(middle.index,group.min(),group.max(),alpha=.2,color='#247b87')
    ax.axhline(1,color='black',ls='--',lw=1);ax.set_xscale('log');ax.set_xticks(middle.index,[str(x) for x in middle.index]);ax.set_xlabel('Window K');ax.set_ylabel('Generic / specialized warm-kernel time')
    ax.set_title('Identical minimal graphs, same compiled kernel\nBand: range across L, T and layouts; not a confidence interval');fig.tight_layout();fig.savefig(figs/'shared_missing_ap1_generic_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
