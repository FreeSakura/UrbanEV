"""Summarize all pilot comparisons and examine the component sign criterion."""
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
from urbanev_audit.persistent_events import component_sign_certificate, endpoint_certificate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    results=ROOT/'artifacts/summaries/shared_missing_events_ap0'
    spec=importlib.util.spec_from_file_location('pilot',ROOT/'scripts/research/run_shared_missing_p0.py')
    pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
    rows=[];conflicts=[]
    for extension in (False,True):
        cfg='SHARED_MISSING_EVENTS_AP0_POWER_EXTENSION_20260916.json' if extension else 'SHARED_MISSING_EVENTS_AP0_20260916.json'
        config=json.loads((ROOT/'configs/research'/cfg).read_text(encoding='utf-8'))
        series,_,_=pilot.read_data(a.work/'data',config)
        directory=results/'power_extension' if extension else results
        private=a.work/('run_power_extension' if extension else 'run')
        frame=pd.read_csv(directory/'paired_bounds.csv')
        cache={}
        for _,row in frame[frame['mask']=='natural'].iterrows():
            dataset,key=row['dataset'],row['series'];k=int(row.K);ell=int(row.L);start=int(row.panel_start);n=int(row.windows)
            dc=config['datasets'][dataset];ident=(dataset,key,k)
            if ident not in cache:
                with np.load(private/f'{dataset}_{key}_K{k}_predictions.npz') as package:cache[ident]={name:package[name] for name in package.files}
            offset=start-dc['calibration_end'];pred=cache[ident]
            coefficient=2*(pred[row.model_B][offset:offset+n]-pred[row.model_A][offset:offset+n])
            observed=pilot.observed_bits(series[dataset,key][0][start:start+n+k-1],dc['threshold'])
            cert=component_sign_certificate(observed,coefficient,k,ell)
            if row.new_strict_decision:
                side='lower' if row.joint_winner=='B' else 'upper'
                conflict=endpoint_certificate(observed,coefficient,k,ell,side,explain=True)
                if conflict['independent_endpoint_attainable']:raise AssertionError('Decision-changing endpoint should be unattainable')
                conflicts.append({'phase':'extension' if extension else 'initial','dataset':dataset,'series':key,'K':k,
                                  'panel_start':start,'model_A':row.model_A,'model_B':row.model_B,**conflict})
            if cert['independent_bounds_provably_exact'] and row.width_reduction>1e-10:
                raise AssertionError('Component theorem contradicted by a saved result')
            rows.append({'phase':'extension' if extension else 'initial','dataset':dataset,'series':key,'K':k,'scope':row.scope,
                         'model_A':row.model_A,'model_B':row.model_B,'panel_start':start,'independent_width':row.independent_width,
                         'width_reduction':row.width_reduction,**{key:cert[key] for key in cert if key!='component_details'}})
    pd.DataFrame(rows).to_csv(results/'component_sign_diagnostics.csv',index=False)
    diag=pd.DataFrame(rows)
    summary=[]
    for (phase,scope),group in diag.groupby(['phase','scope']):
        amb=group[group.independent_width>1e-12]
        summary.append({'phase':phase,'scope':scope,'comparisons':len(group),'ambiguous_comparisons':len(amb),
                        'provably_exact_among_ambiguous':int(amb.independent_bounds_provably_exact.sum()),
                        'criterion_violations':0})
    (results/'component_sign_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (results/'independent_endpoint_conflicts.json').write_text(json.dumps(conflicts,indent=2)+'\n',encoding='utf-8')
    figdir=ROOT/'docs/reports/figures';figdir.mkdir(exist_ok=True)
    frame=pd.read_csv(results/'paired_bounds.csv');cases=frame[(frame['mask']=='natural') & frame.new_strict_decision]
    fig,ax=plt.subplots(figsize=(9,4.7));names={'constant':'Constant','logistic':'Logistic','hist_gradient_boosting':'GB'}
    for i,(_,r) in enumerate(cases.iterrows()):
        ax.plot([r.independent_lower,r.independent_upper],[i+.13]*2,color='#e3a33c',lw=4,label='Independent bounds' if i==0 else None)
        ax.plot([r.joint_lower,r.joint_upper],[i-.13]*2,color='#247b87',lw=4,label='Shared-observation bounds' if i==0 else None)
    labels=[f"{r['series']}: {names[r.model_A]} vs {names[r.model_B]}" for _,r in cases.iterrows()]
    ax.set_yticks(range(len(cases)),labels);ax.axvline(0,color='#666',ls='--',lw=1)
    ax.set_xlabel('Mean paired Brier difference (A - B); positive favors B')
    ax.set_title('All six new natural-missingness decisions (fixed weekly panels)')
    ax.legend(loc='upper right',fontsize=8);ax.grid(axis='x',alpha=.2);fig.tight_layout()
    fig.savefig(figdir/'shared_missing_natural_decisions.png',dpi=180);plt.close(fig)
    times=pd.read_csv(results/'algorithm_timings.csv')
    fig,ax=plt.subplots(figsize=(7,4.2))
    ax.plot(times.K,times.compressed_median_seconds*1000,'o-',label='Compressed DP',color='#247b87')
    measured=times[times.suffix_status=='MEASURED']
    ax.plot(measured.K,measured.suffix_median_seconds*1000,'s-',label='Suffix DP (same compiler)',color='#e3a33c')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks(times.K,[str(k) for k in times.K]);ax.set_xlabel('Window K')
    ax.set_ylabel('Median kernel time (milliseconds)');ax.set_title('T=2048, L=3, 30% missing; 3 repeats; JIT excluded')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(figdir/'shared_missing_scaling.png',dpi=180);plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
